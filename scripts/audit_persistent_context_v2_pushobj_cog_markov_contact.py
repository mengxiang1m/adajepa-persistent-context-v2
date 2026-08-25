#!/usr/bin/env python3
"""Independent artifact audit for the PushObj CoG Markov/contact development study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ACTION_COUNT = 10
ANGLE_SCALE = math.pi / 9.0
MODEL_NAMES = (
    "v1_film",
    "R0_legacy",
    "R1_markov",
    "R2_nominal_agent_block_contact",
    "v1_plus_C1_markov",
    "v1_plus_C2_markov_contact",
)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_angle_delta(first, second):
    return np.arctan2(np.sin(np.asarray(first) - np.asarray(second)), np.cos(np.asarray(first) - np.asarray(second)))


def trajectory_errors(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, ACTION_COUNT, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, ACTION_COUNT, 3)
    position = np.linalg.norm(prediction[:, :, :2] - target[:, :, :2], axis=2)
    angle = np.abs(signed_angle_delta(prediction[:, :, 2] * ANGLE_SCALE, target[:, :, 2] * ANGLE_SCALE)) / ANGLE_SCALE
    return (position + angle).mean(axis=1)


def grouped(values, segment_ids):
    unique = np.unique(segment_ids)
    return np.asarray([np.asarray(values)[segment_ids == segment].mean() for segment in unique])


def bootstrap(first, second, segment_ids, seed, resamples):
    delta = grouped(first, segment_ids) - grouped(second, segment_ids)
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, len(delta), size=(int(resamples), len(delta)))
    means = delta[index].mean(axis=1)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((args.output_dir / "data_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    train = dict(np.load(args.output_dir / "train_data.npz"))
    evaluation = dict(np.load(args.output_dir / "eval_data.npz"))
    predictions = dict(np.load(args.output_dir / "predictions_and_errors.npz"))
    failures = []

    def require(condition, label):
        if not condition:
            failures.append(label)

    require(manifest.get("exit_status") == 0, "runner_exit_status")
    require(summary.get("valid") is True, "runner_valid")
    require(manifest.get("design_sha256") == sha256(args.design), "design_hash")
    require(manifest.get("contract_sha256") == sha256(args.contract), "contract_hash")
    require(manifest.get("v1_checkpoint_sha256") == sha256(args.checkpoint), "checkpoint_hash")
    require(manifest.get("source_snapshot_sha256") == sha256(args.source_snapshot), "source_snapshot_hash")
    require(manifest.get("cuda_visible_devices") == "0", "physical_gpu_allocation")
    require((args.output_dir / "gpu_monitor.csv").stat().st_size > 0, "gpu_monitor")

    expected_train = len(design["train_segment_indices"]) * design["variants_per_segment"] * len(design["train_factors_cog_x"])
    expected_eval = len(design["eval_segment_indices"]) * design["variants_per_segment"] * len(design["eval_factors_cog_x"])
    require(len(train["targets"]) == expected_train == 480, "train_count")
    require(len(evaluation["targets"]) == expected_eval == 256, "eval_count")
    require(set(map(int, np.unique(train["segment_ids"]))) == set(design["train_segment_indices"]), "train_segments")
    require(set(map(int, np.unique(evaluation["segment_ids"]))) == set(design["eval_segment_indices"]), "eval_segments")
    used = set(map(int, np.unique(train["segment_ids"]))) | set(map(int, np.unique(evaluation["segment_ids"])))
    require(used.isdisjoint(set(design["forbidden_formal_segment_indices"])), "formal_segment_leak")
    require(float(train["identity_max_abs"]) <= 1e-6 and float(evaluation["identity_max_abs"]) <= 1e-6, "state_identity")
    repeat = data_manifest["deterministic_repeat"]
    require(repeat["state_equal"] and repeat["contact_equal"], "deterministic_repeat")
    require(repeat["state_sha256_first"] == repeat["state_sha256_second"], "state_repeat_hash")
    require(repeat["contact_sha256_first"] == repeat["contact_sha256_second"], "contact_repeat_hash")
    zero_context = train["contexts"] == 0
    require(np.max(np.abs(train["targets"][zero_context])) == 0.0, "zero_context_target_identity")

    errors = {}
    for name in MODEL_NAMES:
        prediction = predictions[f"prediction_{name}"]
        require(prediction.shape == evaluation["targets"].shape, f"prediction_shape:{name}")
        require(bool(np.isfinite(prediction).all()), f"prediction_finite:{name}")
        errors[name] = trajectory_errors(prediction, evaluation["targets"])
        stored_error = predictions[f"error_{name}"]
        require(close(errors[name], stored_error), f"stored_error:{name}")
        model = summary["models"][name]
        require(close(errors[name].mean(), model["mean_trajectory_pose_error"]), f"trajectory_mean:{name}")
        require(close(grouped(errors[name], evaluation["segment_ids"]).mean(), model["mean_segment_pose_error"]), f"segment_mean:{name}")
        mse = np.square(prediction - evaluation["targets"]).mean()
        require(close(mse, model["normalized_residual_mse"]), f"mse:{name}")

    for key, stored in summary["comparisons"].items():
        first, second = key.split("_minus_", 1)
        replay = bootstrap(errors[first], errors[second], evaluation["segment_ids"], design["bootstrap_seed"], design["bootstrap_resamples"])
        require(close(replay["mean_delta"], stored["mean_delta"]), f"comparison_mean:{key}")
        require(close(replay["ci95"], stored["ci95"]), f"comparison_ci:{key}")
        require(all(replay[field] == stored[field] for field in ("positive", "tie", "negative", "n_segments")), f"comparison_counts:{key}")

    require(np.max(np.abs(evaluation["true_markov"][:, :, 7:10])) > 0, "markov_fields_nontrivial")
    require(np.max(np.abs(evaluation["true_contacts"])) > 0, "contact_fields_nontrivial")
    require(all(summary["structural_checks"].values()), "structural_checks")
    nominal_contact = evaluation["inputs_R2_nominal_agent_block_contact"][:, -170:].reshape(-1, ACTION_COUNT, 17)
    true_contact = evaluation["true_contacts"][:, :, 0, :]
    nominal_event = nominal_contact[:, :, 1] > 0
    true_event = true_contact[:, :, 1] > 0
    event_mismatch = nominal_event != true_event
    any_event_mismatch = event_mismatch.any(axis=1)
    nominal_first = np.where(nominal_event.any(axis=1), nominal_event.argmax(axis=1), ACTION_COUNT)
    true_first = np.where(true_event.any(axis=1), true_event.argmax(axis=1), ACTION_COUNT)
    impulse_difference = np.abs(true_contact[:, :, 7] - nominal_contact[:, :, 7]).sum(axis=1)
    rank_error = np.argsort(np.argsort(errors["v1_film"], kind="mergesort"), kind="mergesort")
    rank_impulse = np.argsort(np.argsort(impulse_difference, kind="mergesort"), kind="mergesort")
    contact_divergence = {
        "label": "post-hoc descriptive; not pre-registered as a primary metric",
        "step_event_mismatch_fraction": float(event_mismatch.mean()),
        "trajectory_any_event_mismatch": int(any_event_mismatch.sum()),
        "n_trajectories": int(len(any_event_mismatch)),
        "mean_v1_error_any_mismatch": float(errors["v1_film"][any_event_mismatch].mean()),
        "mean_v1_error_no_mismatch": float(errors["v1_film"][~any_event_mismatch].mean()),
        "first_contact_step_diff_nonzero": int(np.sum(nominal_first != true_first)),
        "mean_abs_first_contact_step_diff": float(np.abs(nominal_first - true_first).mean()),
        "mean_abs_impulse_difference": float(impulse_difference.mean()),
        "spearman_v1_error_impulse_difference": float(np.corrcoef(rank_error, rank_impulse)[0, 1]),
    }
    artifact_hashes = {name: sha256(args.output_dir / name) for name in (
        "manifest.json", "data_manifest.json", "runner_summary.json", "train_data.npz", "eval_data.npz", "predictions_and_errors.npz", "gpu_monitor.csv"
    )}
    result = {
        "valid": not failures,
        "failures": failures,
        "contract_id": summary["contract_id"],
        "recomputed_models": {name: float(errors[name].mean()) for name in MODEL_NAMES},
        "recomputed_comparisons": summary["comparisons"],
        "posthoc_contact_divergence": contact_divergence,
        "artifact_hashes": artifact_hashes,
        "design_sha256": sha256(args.design),
        "contract_sha256": sha256(args.contract),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_snapshot_sha256": sha256(args.source_snapshot),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
