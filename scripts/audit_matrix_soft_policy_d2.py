#!/usr/bin/env python3
"""Independent replay of D2 without importing the D2 analysis module."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def joined(d0: Path, d1: Path, alphas: list[float]) -> dict[str, list[dict]]:
    features = [row for row in read_jsonl(d0) if row.get("record_type") == "d0_exploratory_feature"]
    outcomes = [row for row in read_jsonl(d1) if row.get("record_type") == "d1_alpha_treatment"]
    f_lookup = {(row["split"], int(row["sequence_id"])): row for row in features}
    o_lookup = {(row["split"], int(row["sequence_id"]), float(row["alpha"])): row for row in outcomes}
    result = {split: [] for split in ("train", "dev", "formal")}
    for (split, sequence), feature in sorted(f_lookup.items()):
        result[split].append({
            "x": np.asarray(feature["features"], dtype=np.float64),
            "costs": np.asarray([o_lookup[(split, sequence, alpha)]["metrics"]["pose_auc10"] for alpha in alphas]),
        })
    return result


def fit(rows: list[dict], count: int, ridge: float) -> dict:
    x = np.stack([row["x"][:count] for row in rows])
    mean, scale = np.zeros(count), np.ones(count)
    mean[1:] = x[:, 1:].mean(0)
    scale[1:] = x[:, 1:].std(0)
    scale[1:][scale[1:] < 1e-12] = 1.0
    x = (x - mean) / scale
    nonzero = np.asarray([0.25, 0.5, 0.75, 1.0])
    tiled_x = np.repeat(x, 4, axis=0)
    a = np.tile(nonzero, len(rows))
    phi = np.concatenate([a[:, None] * tiled_x, (a * (1 - a))[:, None] * tiled_x], axis=1)
    y = np.concatenate([row["costs"][0] - row["costs"][1:] for row in rows])
    penalty = np.eye(2 * count)
    penalty[0, 0] = penalty[count, count] = 0.0
    beta = np.linalg.solve(phi.T @ phi + ridge * penalty, phi.T @ y)
    return {"mean": mean, "scale": scale, "beta": beta}


def predict(model: dict, rows: list[dict], count: int, alphas: list[float]) -> np.ndarray:
    x = (np.stack([row["x"][:count] for row in rows]) - model["mean"]) / model["scale"]
    predictions = np.zeros((len(rows), len(alphas)))
    for index, alpha in enumerate(alphas):
        phi = np.concatenate([alpha * x, alpha * (1 - alpha) * x], axis=1)
        predictions[:, index] = phi @ model["beta"]
    return predictions


def replay(train: list[dict], dev: list[dict], test: list[dict], count: int, design: dict) -> dict:
    alphas = [float(value) for value in design["alphas"]]
    candidates = []
    for ridge in [float(value) for value in design["ridge_alphas"]]:
        model = fit(train, count, ridge)
        selected = np.argmax(predict(model, dev, count, alphas), axis=1)
        costs = np.stack([row["costs"] for row in dev])
        mean_delta = float(np.mean(costs[:, 0] - costs[np.arange(len(dev)), selected]))
        candidates.append((mean_delta, ridge))
    selected_ridge = max(candidates, key=lambda value: (value[0], value[1]))[1]
    model = fit(train + dev, count, selected_ridge)
    predictions = predict(model, test, count, alphas)
    selected = np.argmax(predictions, axis=1)
    costs = np.stack([row["costs"] for row in test])
    learned = costs[np.arange(len(test)), selected]
    return {
        "ridge": selected_ridge,
        "predictions": predictions,
        "selected": selected,
        "learned_delta": costs[:, 0] - learned,
        "fixed_minus_learned": costs[:, 3] - learned,
    }


def audit(args) -> dict:
    design = json.loads(args.design.read_text(encoding="utf-8"))
    analysis = json.loads((args.output_dir / "analysis.json").read_text(encoding="utf-8"))
    data = joined(args.d0_dir / "features.jsonl", args.d1_dir / "raw.jsonl", [float(value) for value in design["alphas"]])
    failures, rotation_results = [], []
    max_prediction_error = max_beta_error = max_summary_error = 0.0
    decision_mismatches = ridge_mismatches = 0
    for rotation_index, (train_name, dev_name, test_name) in enumerate(design["rotations"]):
        observed_rotation = analysis["rotations"][rotation_index]
        replayed_models = {}
        for name, count in design["feature_sets"].items():
            replayed = replay(data[train_name], data[dev_name], data[test_name], int(count), design)
            observed = observed_rotation["models"][name]
            ridge_mismatches += int(float(observed["selected_ridge_alpha"]) != replayed["ridge"])
            max_prediction_error = max(max_prediction_error, float(np.max(np.abs(replayed["predictions"] - np.asarray(observed["test_predictions"])))))
            decision_mismatches += int(np.sum(replayed["selected"] != np.asarray(observed["test_selected_alpha_indexes"])))
            expected_mean = float(np.mean(replayed["learned_delta"]))
            observed_mean = float(observed["policies"]["learned_soft_policy"]["mean_delta_vs_population"])
            expected_fixed = float(np.mean(replayed["fixed_minus_learned"]))
            observed_fixed = float(observed["learned_vs_fixed_alpha_0.75"]["mean_delta_fixed_minus_learned"])
            max_summary_error = max(max_summary_error, abs(expected_mean - observed_mean), abs(expected_fixed - observed_fixed))
            model = fit(data[train_name] + data[dev_name], int(count), replayed["ridge"])
            max_beta_error = max(max_beta_error, float(np.max(np.abs(model["beta"] - np.asarray(observed["model"]["beta"])))))
            replayed_models[name] = replayed
        f0, f2 = replayed_models["F0_factor_only"], replayed_models["F2_task_interaction"]
        test_costs = np.stack([row["costs"] for row in data[test_name]])
        f0_cost = test_costs[np.arange(32), f0["selected"]]
        f2_cost = test_costs[np.arange(32), f2["selected"]]
        expected_f2_f0 = float(np.mean(f0_cost - f2_cost))
        observed_f2_f0 = float(observed_rotation["F2_vs_F0"]["mean_delta_F0_minus_F2"])
        max_summary_error = max(max_summary_error, abs(expected_f2_f0 - observed_f2_f0))
        rotation_results.append({"train": train_name, "dev": dev_name, "test": test_name, "F2_vs_fixed_0.75": float(np.mean(f2["fixed_minus_learned"]))})
    if ridge_mismatches:
        failures.append(f"ridge mismatches: {ridge_mismatches}")
    if decision_mismatches:
        failures.append(f"decision mismatches: {decision_mismatches}")
    if max_prediction_error > 1e-12 or max_beta_error > 1e-12 or max_summary_error > 1e-12:
        failures.append("numeric replay mismatch")
    result = {
        "design_id": design["design_id"],
        "valid": not failures,
        "failures": failures,
        "ridge_mismatches": ridge_mismatches,
        "decision_mismatches": decision_mismatches,
        "prediction_replay_max_abs": max_prediction_error,
        "beta_replay_max_abs": max_beta_error,
        "summary_replay_max_abs": max_summary_error,
        "rotation_results": rotation_results,
        "design_sha256": sha256(args.design),
        "d0_features_sha256": sha256(args.d0_dir / "features.jsonl"),
        "d1_raw_sha256": sha256(args.d1_dir / "raw.jsonl"),
        "analysis_sha256": sha256(args.output_dir / "analysis.json"),
    }
    args.audit_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d0-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_task_interaction_d0_exploration_v1"))
    parser.add_argument("--d1-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_soft_context_d1_exploration_v1"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_soft_policy_d2_design.json"))
    parser.add_argument("--audit-output", type=Path, required=True)
    result = audit(parser.parse_args())
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
