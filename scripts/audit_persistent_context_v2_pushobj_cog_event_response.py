#!/usr/bin/env python3
"""Independent artifact audit for the P3b PushObj CoG event-response study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPRESENTATIONS = (
    "C10_aggregate",
    "S100_state",
    "S100_state_geometry",
    "S100_state_geometry_impulse",
    "P100_true_contact",
)
PRIMARY_FIRST = "C10_aggregate"
PRIMARY_SECOND = "S100_state_geometry_impulse"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def feature_map(values, contexts, mean, scale):
    normalized = (np.asarray(values, dtype=np.float64) - mean) / scale
    context = np.asarray(contexts, dtype=np.float64).reshape(-1, 1) / 30.0
    return np.concatenate([context, np.square(context), context * normalized, np.square(context) * normalized], axis=1)


def response_errors(prediction, target, target_scale):
    difference = (np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)) / target_scale
    return np.sqrt(np.mean(np.square(difference), axis=1))


def grouped(values, segment_ids):
    unique = np.unique(segment_ids)
    return np.asarray([np.asarray(values)[segment_ids == segment].mean() for segment in unique], dtype=np.float64)


def bootstrap(first, second, segment_ids, design):
    delta = grouped(first, segment_ids) - grouped(second, segment_ids)
    rng = np.random.default_rng(int(design["bootstrap_seed"]))
    indexes = rng.integers(0, len(delta), size=(int(design["bootstrap_resamples"]), len(delta)))
    means = delta[indexes].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "positive": int(np.sum(delta > 1e-12)),
        "tie": int(np.sum(np.abs(delta) <= 1e-12)),
        "negative": int(np.sum(delta < -1e-12)),
        "n_segments": int(len(delta)),
    }


def close(first, second, tolerance=1e-8):
    return bool(np.allclose(np.asarray(first), np.asarray(second), rtol=0, atol=tolerance, equal_nan=True))


def complete_factor_blocks(data, factors):
    rows = zip(data["segment_ids"], data["variant_ids"], data["control_ids"], data["substep_ids"], data["contexts"])
    groups = {}
    for segment, variant, control, substep, factor in rows:
        key = (int(segment), int(variant), int(control), int(substep))
        groups.setdefault(key, set()).add(float(factor))
    expected = set(map(float, factors))
    return bool(groups) and all(values == expected for values in groups.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((args.output_dir / "data_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((args.output_dir / "selection_summary.json").read_text(encoding="utf-8"))
    summary = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    train = dict(np.load(args.output_dir / "train_data.npz"))
    evaluation = dict(np.load(args.output_dir / "eval_data.npz"))
    models = dict(np.load(args.output_dir / "frozen_models.npz"))
    stored = dict(np.load(args.output_dir / "predictions_and_errors.npz"))
    failures = []

    def require(condition, label):
        if not condition:
            failures.append(label)

    require(manifest.get("exit_status") == 0, "runner_exit_status")
    require(summary.get("valid") is True, "runner_valid")
    require(manifest.get("smoke") is False, "not_smoke")
    require(manifest.get("design_sha256") == sha256(args.design), "design_hash")
    require(manifest.get("contract_sha256") == sha256(args.contract), "contract_hash")
    require(manifest.get("source_snapshot_sha256") == sha256(args.source_snapshot), "source_snapshot_hash")
    require(manifest.get("cuda_visible_devices") in (None, ""), "cpu_only_allocation")
    require((args.output_dir / "gpu_monitor.csv").is_file(), "gpu_monitor_exists")
    require((args.output_dir / "gpu_monitor.csv").stat().st_size > 0, "gpu_monitor_nonempty")

    expected_train = set(map(int, design["train_segment_indices"]))
    expected_eval = set(map(int, design["eval_segment_indices"]))
    forbidden = set(map(int, design["forbidden_formal_segment_indices"]))
    actual_train = set(map(int, np.unique(train["segment_ids"])))
    actual_eval = set(map(int, np.unique(evaluation["segment_ids"])))
    require(actual_train == expected_train, "train_segments")
    require(actual_eval == expected_eval, "eval_segments")
    require(actual_train.isdisjoint(actual_eval), "train_eval_disjoint")
    require((actual_train | actual_eval).isdisjoint(forbidden), "formal_segment_leak")
    require(complete_factor_blocks(train, design["train_factors_cog_x"]), "train_factor_blocks")
    require(complete_factor_blocks(evaluation, design["eval_factors_cog_x"]), "eval_factor_blocks")
    require(set(map(int, np.unique(train["variant_ids"]))) == set(range(design["variants_per_segment"])), "train_variants")
    require(set(map(int, np.unique(evaluation["variant_ids"]))) == set(range(design["variants_per_segment"])), "eval_variants")
    require(np.all(train["nominal_point_count"] > 0) and np.all(evaluation["nominal_point_count"] > 0), "nominal_contact_rule")
    require(np.all(train["nominal_block_wall_point_count"] == 0) and np.all(evaluation["nominal_block_wall_point_count"] == 0), "nominal_wall_rule")
    require(all(value > 0 for value in data_manifest["train_nominal_events_by_segment"].values()), "train_segment_events")
    require(all(value > 0 for value in data_manifest["eval_nominal_events_by_segment"].values()), "eval_segment_events")
    require(data_manifest["train_identity_max_abs"] <= 1e-6 and data_manifest["eval_identity_max_abs"] <= 1e-6, "rollout_identity")
    require(data_manifest["deterministic_repeat"]["equal"], "deterministic_repeat")
    require(data_manifest["zero_context_target_max_abs"] == 0.0, "zero_context_target")
    require(selection["locked_unix"] <= data_manifest["eval_generated_unix"], "model_locked_before_eval")
    require(data_manifest["selection_summary_sha256_before_eval"] == sha256(args.output_dir / "selection_summary.json"), "selection_unchanged")
    require(data_manifest["frozen_models_sha256_before_eval"] == sha256(args.output_dir / "frozen_models.npz"), "models_unchanged")

    nonzero = train["contexts"] != 0
    replay_scale = np.maximum(
        np.sqrt(np.mean(np.square(np.asarray(train["targets"], dtype=np.float64)[nonzero]), axis=0)),
        float(design["target_rms_floor"]),
    )
    target_scale = np.asarray(models["target_scale"], dtype=np.float64)
    require(close(replay_scale, target_scale), "target_scale_train_only")
    require(close(selection["target_scale"], target_scale), "selection_target_scale")
    require(selection["target_scale_sha256"] == array_sha256(target_scale), "target_scale_hash")

    predictions = {"zero_response": np.zeros_like(evaluation["targets"], dtype=np.float32)}
    for name in REPRESENTATIONS:
        mean = models[f"{name}_mean"]
        scale = models[f"{name}_scale"]
        beta = models[f"{name}_beta"]
        prediction_scaled = feature_map(evaluation[f"inputs_{name}"], evaluation["contexts"], mean, scale) @ beta
        predictions[name] = (prediction_scaled * target_scale).astype(np.float32)
        zero = feature_map(evaluation[f"inputs_{name}"][:3], np.zeros(3), mean, scale) @ beta
        require(np.max(np.abs(zero)) == 0.0, f"zero_context_model:{name}")
        require(selection["models"][name]["mean_sha256"] == array_sha256(mean), f"mean_hash:{name}")
        require(selection["models"][name]["scale_sha256"] == array_sha256(scale), f"scale_hash:{name}")
        require(selection["models"][name]["beta_sha256"] == array_sha256(beta), f"beta_hash:{name}")

    errors = {}
    for name, prediction in predictions.items():
        require(close(prediction, stored[f"prediction_{name}"], tolerance=1e-6), f"stored_prediction:{name}")
        errors[name] = response_errors(prediction, evaluation["targets"], target_scale)
        require(close(errors[name], stored[f"error_{name}"], tolerance=1e-6), f"stored_error:{name}")
        require(close(errors[name].mean(), summary["models"][name]["mean_event_error"]), f"event_mean:{name}")
        require(close(grouped(errors[name], evaluation["segment_ids"]).mean(), summary["models"][name]["mean_segment_error"]), f"segment_mean:{name}")

    comparisons = (
        (PRIMARY_FIRST, PRIMARY_SECOND),
        ("S100_state", "S100_state_geometry"),
        ("S100_state_geometry", "S100_state_geometry_impulse"),
        ("S100_state_geometry_impulse", "P100_true_contact"),
        ("zero_response", "S100_state_geometry_impulse"),
    )
    replay_comparisons = {}
    for first, second in comparisons:
        key = f"{first}_minus_{second}"
        replay = bootstrap(errors[first], errors[second], evaluation["segment_ids"], design)
        replay_comparisons[key] = replay
        stored_comparison = summary["comparisons"][key]
        require(close(replay["mean_delta"], stored_comparison["mean_delta"]), f"comparison_mean:{key}")
        require(close(replay["ci95"], stored_comparison["ci95"]), f"comparison_ci:{key}")
        require(all(replay[field] == stored_comparison[field] for field in ("positive", "tie", "negative", "n_segments")), f"comparison_counts:{key}")

    require(summary["primary_comparison"] == design["primary_comparison"], "primary_comparison")
    require(all(summary["structural_checks"].values()), "structural_checks")
    artifact_names = (
        "manifest.json", "data_manifest.json", "selection_summary.json", "frozen_models.npz",
        "train_data.npz", "eval_data.npz", "runner_summary.json", "predictions_and_errors.npz", "gpu_monitor.csv",
    )
    result = {
        "valid": not failures,
        "failures": failures,
        "contract_id": summary["contract_id"],
        "recomputed_models": {name: float(grouped(error, evaluation["segment_ids"]).mean()) for name, error in errors.items()},
        "recomputed_comparisons": replay_comparisons,
        "artifact_hashes": {name: sha256(args.output_dir / name) for name in artifact_names},
        "design_sha256": sha256(args.design),
        "contract_sha256": sha256(args.contract),
        "source_snapshot_sha256": sha256(args.source_snapshot),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
