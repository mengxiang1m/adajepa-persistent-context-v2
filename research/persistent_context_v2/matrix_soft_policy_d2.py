"""Frozen low-capacity D2 soft-context policy feasibility analysis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.persistent_context_v2.pushobj_rotation_stage0 import dump_json, read_jsonl, sha256


D2_ID = "persistent-context-v2-matrix-soft-policy-d2-exploratory-v1"


def load_design(path: Path) -> dict:
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("design_id") != D2_ID:
        raise RuntimeError("D2 design id mismatch")
    if [float(value) for value in design["alphas"]] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("D2 alpha grid mismatch")
    return design


def load_joined(d0_dir: Path, d1_dir: Path, design: dict) -> dict[str, list[dict]]:
    feature_rows = [row for row in read_jsonl(d0_dir / "features.jsonl") if row.get("record_type") == "d0_exploratory_feature"]
    outcome_rows = [row for row in read_jsonl(d1_dir / "raw.jsonl") if row.get("record_type") == "d1_alpha_treatment"]
    feature_lookup = {(str(row["split"]), int(row["sequence_id"])): row for row in feature_rows}
    outcome_lookup = {(str(row["split"]), int(row["sequence_id"]), float(row["alpha"])): row for row in outcome_rows}
    if len(feature_lookup) != 96 or len(feature_rows) != 96:
        raise RuntimeError("D0 features must contain 96 unique sequences")
    if len(outcome_lookup) != 480 or len(outcome_rows) != 480:
        raise RuntimeError("D1 outcomes must contain 480 unique treatments")
    result = {split: [] for split in ("train", "dev", "formal")}
    for key in sorted(feature_lookup):
        split, sequence_id = key
        feature = feature_lookup[key]
        names = list(feature["feature_names"])
        if len(names) != 18 or len(feature["features"]) != 18:
            raise RuntimeError("unexpected D0 feature schema")
        costs = []
        factor_indexes = set()
        for alpha in [float(value) for value in design["alphas"]]:
            row = outcome_lookup.get((split, sequence_id, alpha))
            if row is None:
                raise RuntimeError(f"missing D1 outcome {(split, sequence_id, alpha)}")
            costs.append(float(row["metrics"]["pose_auc10"]))
            factor_indexes.add(int(row["factor_index"]))
        if factor_indexes != {int(feature["factor_index"])}:
            raise RuntimeError("D0/D1 factor mismatch")
        result[split].append({
            "split": split,
            "sequence_id": sequence_id,
            "factor_index": int(feature["factor_index"]),
            "feature_names": names,
            "features": np.asarray(feature["features"], dtype=np.float64),
            "costs": np.asarray(costs, dtype=np.float64),
        })
    if any(len(result[split]) != 32 for split in result):
        raise RuntimeError("each D2 split must contain 32 sequences")
    return result


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.zeros(x.shape[1], dtype=np.float64)
    scale = np.ones(x.shape[1], dtype=np.float64)
    if x.shape[1] > 1:
        mean[1:] = x[:, 1:].mean(axis=0)
        scale[1:] = x[:, 1:].std(axis=0)
        scale[1:][scale[1:] < 1e-12] = 1.0
    return (x - mean) / scale, mean, scale


def dose_matrix(x: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if len(x) != len(alphas):
        raise ValueError("x and alphas length mismatch")
    return np.concatenate([alphas[:, None] * x, (alphas * (1.0 - alphas))[:, None] * x], axis=1)


def fit_model(rows: list[dict], feature_count: int, ridge_alpha: float) -> dict:
    x = np.stack([row["features"][:feature_count] for row in rows])
    x_scaled, mean, scale = standardize_fit(x)
    nonzero_alphas = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    tiled_x = np.repeat(x_scaled, len(nonzero_alphas), axis=0)
    tiled_alpha = np.tile(nonzero_alphas, len(rows))
    phi = dose_matrix(tiled_x, tiled_alpha)
    y = np.concatenate([row["costs"][0] - row["costs"][1:] for row in rows])
    penalty = np.eye(phi.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    penalty[feature_count, feature_count] = 0.0
    beta = np.linalg.solve(phi.T @ phi + float(ridge_alpha) * penalty, phi.T @ y)
    return {"mean": mean, "scale": scale, "beta": beta, "ridge_alpha": float(ridge_alpha)}


def predict(model: dict, rows: list[dict], feature_count: int, alphas: list[float]) -> np.ndarray:
    x = np.stack([row["features"][:feature_count] for row in rows])
    x = (x - model["mean"]) / model["scale"]
    result = np.zeros((len(rows), len(alphas)), dtype=np.float64)
    for index, alpha in enumerate(alphas):
        result[:, index] = dose_matrix(x, np.full(len(rows), float(alpha))) @ model["beta"]
    return result


def decisions(predictions: np.ndarray) -> np.ndarray:
    # np.argmax returns the first maximum; alphas are ascending, so ties use less context.
    return np.argmax(predictions, axis=1)


def policy_cost(rows: list[dict], selected: np.ndarray) -> np.ndarray:
    costs = np.stack([row["costs"] for row in rows])
    return costs[np.arange(len(rows)), np.asarray(selected, dtype=int)]


def bootstrap_ci(values: np.ndarray, seed: int, resamples: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    indexes = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def effect(population: np.ndarray, treatment: np.ndarray, design: dict, stream: int) -> dict:
    delta = np.asarray(population) - np.asarray(treatment)
    return {
        "mean_cost": float(np.mean(treatment)),
        "mean_delta_vs_population": float(delta.mean()),
        "relative_improvement_vs_population": float(delta.mean() / np.mean(population)),
        "bootstrap_ci95_delta": bootstrap_ci(delta, int(design["bootstrap_seed"]) + stream, int(design["bootstrap_resamples"])),
        "harm_fraction": float(np.mean(delta < -1e-12)),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "unit_deltas": delta.tolist(),
    }


def fit_evaluate(train_rows: list[dict], dev_rows: list[dict], test_rows: list[dict], feature_count: int, design: dict, stream: int) -> dict:
    alphas = [float(value) for value in design["alphas"]]
    candidates = []
    for ridge_alpha in [float(value) for value in design["ridge_alphas"]]:
        model = fit_model(train_rows, feature_count, ridge_alpha)
        prediction = predict(model, dev_rows, feature_count, alphas)
        selected = decisions(prediction)
        dev_cost = policy_cost(dev_rows, selected)
        population = np.asarray([row["costs"][0] for row in dev_rows])
        candidates.append({
            "ridge_alpha": ridge_alpha,
            "dev_mean_delta": float(np.mean(population - dev_cost)),
            "dev_mean_cost": float(np.mean(dev_cost)),
            "dev_selection_counts": {str(alpha): int(np.sum(selected == index)) for index, alpha in enumerate(alphas)},
        })
    selected_candidate = max(candidates, key=lambda row: (row["dev_mean_delta"], row["ridge_alpha"]))
    model = fit_model(train_rows + dev_rows, feature_count, selected_candidate["ridge_alpha"])
    predictions = predict(model, test_rows, feature_count, alphas)
    selected = decisions(predictions)
    learned_cost = policy_cost(test_rows, selected)
    costs = np.stack([row["costs"] for row in test_rows])
    population = costs[:, 0]
    fixed_half, fixed_three_quarters, full = costs[:, 2], costs[:, 3], costs[:, 4]
    best = costs.min(axis=1)
    fixed_minus_learned = fixed_three_quarters - learned_cost
    return {
        "selected_ridge_alpha": float(selected_candidate["ridge_alpha"]),
        "ridge_candidates": candidates,
        "test_predictions": predictions.tolist(),
        "test_selected_alpha_indexes": selected.tolist(),
        "test_selected_alphas": [alphas[index] for index in selected],
        "test_selection_counts": {str(alpha): int(np.sum(selected == index)) for index, alpha in enumerate(alphas)},
        "policies": {
            "fixed_alpha_0.5": effect(population, fixed_half, design, stream + 1),
            "fixed_alpha_0.75": effect(population, fixed_three_quarters, design, stream + 2),
            "full_context": effect(population, full, design, stream + 3),
            "learned_soft_policy": effect(population, learned_cost, design, stream + 4),
            "per_sequence_best_alpha_ceiling": effect(population, best, design, stream + 5),
        },
        "learned_vs_fixed_alpha_0.75": {
            "mean_delta_fixed_minus_learned": float(fixed_minus_learned.mean()),
            "bootstrap_ci95": bootstrap_ci(fixed_minus_learned, int(design["bootstrap_seed"]) + stream + 6, int(design["bootstrap_resamples"])),
            "learned_minus_fixed_harm_fraction": float(np.mean((population - learned_cost) < -1e-12) - np.mean((population - fixed_three_quarters) < -1e-12)),
            "unit_deltas": fixed_minus_learned.tolist(),
        },
        "model": {
            "feature_count": feature_count,
            "mean": model["mean"].tolist(),
            "scale": model["scale"].tolist(),
            "beta": model["beta"].tolist(),
        },
    }


def analyze(args) -> dict:
    started = time.time()
    design = load_design(args.design)
    joined = load_joined(args.d0_dir, args.d1_dir, design)
    result = {
        "design_id": D2_ID,
        "evidence_level": design["evidence_level"],
        "feature_names": joined["train"][0]["feature_names"],
        "rotations": [],
    }
    for rotation_index, (train_split, dev_split, test_split) in enumerate(design["rotations"]):
        rotation = {"train_split": train_split, "dev_split": dev_split, "test_split": test_split, "models": {}}
        for model_index, (name, feature_count) in enumerate(design["feature_sets"].items()):
            rotation["models"][name] = fit_evaluate(
                joined[train_split], joined[dev_split], joined[test_split], int(feature_count), design,
                1000 * rotation_index + 100 * model_index,
            )
        f0 = rotation["models"]["F0_factor_only"]
        f2 = rotation["models"]["F2_task_interaction"]
        f0_cost = np.asarray([joined[test_split][index]["costs"][f0["test_selected_alpha_indexes"][index]] for index in range(32)])
        f2_cost = np.asarray([joined[test_split][index]["costs"][f2["test_selected_alpha_indexes"][index]] for index in range(32)])
        rotation["F2_vs_F0"] = {
            "mean_delta_F0_minus_F2": float(np.mean(f0_cost - f2_cost)),
            "bootstrap_ci95": bootstrap_ci(f0_cost - f2_cost, int(design["bootstrap_seed"]) + 5000 + rotation_index, int(design["bootstrap_resamples"])),
        }
        result["rotations"].append(rotation)
    result["direction_summary"] = {
        name: {
            "F2_vs_fixed_alpha_0.75_mean_deltas": [float(rotation["models"][name]["learned_vs_fixed_alpha_0.75"]["mean_delta_fixed_minus_learned"]) for rotation in result["rotations"]],
            "F2_policy_mean_deltas_vs_population": [float(rotation["models"][name]["policies"]["learned_soft_policy"]["mean_delta_vs_population"]) for rotation in result["rotations"]],
        }
        for name in design["feature_sets"]
    }
    dump_json(args.output_dir / "analysis.json", result)
    manifest = {
        "design_id": D2_ID,
        "command": " ".join(__import__("sys").argv),
        "design": str(args.design),
        "design_sha256": sha256(args.design),
        "source_snapshot": str(args.source_snapshot),
        "source_snapshot_sha256": sha256(args.source_snapshot),
        "d0_features": str(args.d0_dir / "features.jsonl"),
        "d0_features_sha256": sha256(args.d0_dir / "features.jsonl"),
        "d1_raw": str(args.d1_dir / "raw.jsonl"),
        "d1_raw_sha256": sha256(args.d1_dir / "raw.jsonl"),
        "analysis_sha256": sha256(args.output_dir / "analysis.json"),
        "started_unix": started,
        "finished_unix": time.time(),
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_task_interaction_d0_exploration_v1"))
    parser.add_argument("--d1-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_soft_context_d1_exploration_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_soft_policy_d2_design.json"))
    parser.add_argument("--source-snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(args)
    print(json.dumps(result["direction_summary"], indent=2))


if __name__ == "__main__":
    main()
