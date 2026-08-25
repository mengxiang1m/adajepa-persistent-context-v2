#!/usr/bin/env python3
"""Independent end-to-end audit of the prospective F0 formal experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def feature_from_posterior(row: dict, design: dict) -> np.ndarray:
    z = np.asarray(row["posterior"]["mean_z"], dtype=np.float64)
    gain = float(np.linalg.norm(z))
    rotation = math.degrees(math.atan2(z[1], z[0]))
    g = (gain - float(design["gain_center"])) / float(design["gain_scale"])
    r = rotation / float(design["rotation_scale_degrees"])
    return np.asarray([1.0, g, r, g * g, g * r, r * r])


def posterior_from_observations(observations) -> dict:
    rotations = (-30.0, -15.0, 0.0, 15.0, 30.0)
    gains = (0.75, 1.0, 1.25)
    samples = np.asarray([[gain * math.cos(math.radians(theta)), gain * math.sin(math.radians(theta))]
                          for theta in rotations for gain in gains], dtype=np.float64)
    prior_mean = samples.mean(axis=0)
    prior_cov = np.cov(samples.T, bias=True) + 1e-4 * np.eye(2)
    precision = np.linalg.inv(prior_cov)
    information = precision @ prior_mean
    obs = np.asarray(observations, dtype=np.float64).reshape(-1, 2)
    if len(obs):
        precision = precision + len(obs) * 10000.0 * np.eye(2)
        information = information + 10000.0 * obs.sum(axis=0)
    covariance = np.linalg.inv(precision)
    mean_z = np.linalg.solve(precision, information)
    mean_matrix = np.asarray([[mean_z[0], -mean_z[1]], [mean_z[1], mean_z[0]]])
    return {"precision": precision, "information": information, "covariance": covariance,
            "mean_z": mean_z, "mean_matrix": mean_matrix, "observation_count": len(obs)}


def model_predict(model: dict, x: np.ndarray, alphas: list[float]) -> np.ndarray:
    normalized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    beta = np.asarray(model["beta"])
    return np.asarray([np.concatenate([alpha * normalized, alpha * (1.0 - alpha) * normalized]) @ beta
                       for alpha in alphas])


def pose_auc10(treatment: dict) -> float:
    states = np.asarray(treatment["states"], dtype=np.float64)
    goal = np.asarray(treatment["goal_state"], dtype=np.float64)
    post = states[1:11]
    position = np.linalg.norm(post[:, 2:4] - goal[None, 2:4], axis=1)
    delta = np.abs(post[:, 4] - goal[4]) % (2.0 * np.pi)
    angle = np.minimum(delta, 2.0 * np.pi - delta)
    return float(np.mean(position / 20.0 + angle / (np.pi / 9.0)))


def bootstrap_ci(values: np.ndarray, design: dict, stream: int) -> np.ndarray:
    rng = np.random.default_rng(int(design["bootstrap_seed"]) + stream)
    indexes = rng.integers(0, len(values), size=(int(design["bootstrap_resamples"]), len(values)))
    return np.quantile(values[indexes].mean(axis=1), [.025, .975])


def audit(args) -> dict:
    design = json.loads(args.design.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    model = json.loads(args.locked_model.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    alphas = [float(value) for value in design["alphas"]]
    failures: list[str] = []
    max_feature_error = max_posterior_error = max_prediction_error = max_metric_error = 0.0
    decision_mismatches = execution_order_mismatches = 0
    all_rows: dict[str, list[dict]] = {}
    source_hashes = set()
    for split, expected in (("train", 64), ("dev", 32), ("formal", 96)):
        raw_path = args.output_dir / split / "raw.jsonl"
        manifest_path = args.output_dir / split / "manifest.json"
        rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "f0_soft_policy_sequence"]
        all_rows[split] = rows
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hashes.add(manifest.get("source_snapshot_sha256"))
        if len(rows) != expected or manifest.get("rows") != expected:
            failures.append(f"{split} row count mismatch")
        if manifest.get("raw_sha256") != sha256(raw_path):
            failures.append(f"{split} raw hash mismatch")
        if not manifest.get("model_state_unchanged"):
            failures.append(f"{split} model state changed")
        if split == "formal" and manifest.get("locked_model_sha256") != sha256(args.locked_model):
            failures.append("formal locked model hash mismatch")
        if sorted(int(row["sequence_id"]) for row in rows) != list(range(expected)):
            failures.append(f"{split} sequence ids mismatch")
        for row in rows:
            sequence_id = int(row["sequence_id"])
            if int(row["e1"]["segment_index"]) != int(selection[split][2 * sequence_id]):
                failures.append(f"{split} E1 selection mismatch")
            if int(row["e2_meta"]["segment_index"]) != int(selection[split][2 * sequence_id + 1]):
                failures.append(f"{split} E2 selection mismatch")
            expected_factor = design["factors"][sequence_id % 8]
            if float(row["rotation_degrees"]) != float(expected_factor["rotation_degrees"]) or float(row["gain"]) != float(expected_factor["gain"]):
                failures.append(f"{split} factor schedule mismatch")
            replay = posterior_from_observations(row["e1"]["matrix_observations"])
            observed = row["posterior"]
            for key in ("precision", "information", "covariance", "mean_z", "mean_matrix"):
                max_posterior_error = max(max_posterior_error, float(np.max(np.abs(replay[key] - np.asarray(observed[key])))))
            if int(replay["observation_count"]) != int(observed["observation_count"]):
                failures.append(f"{split} posterior count mismatch")
            x = feature_from_posterior(row, design)
            max_feature_error = max(max_feature_error, float(np.max(np.abs(x - np.asarray(row["features"])))))
            if set(row["treatments"]) != {str(alpha) for alpha in alphas}:
                failures.append(f"{split} incomplete alpha grid")
            for alpha in alphas:
                treatment = row["treatments"][str(alpha)]
                max_metric_error = max(max_metric_error, abs(pose_auc10(treatment) - float(treatment["metrics"]["pose_auc10"])))
            if split == "formal":
                prediction = model_predict(model, x, alphas)
                max_prediction_error = max(max_prediction_error, float(np.max(np.abs(prediction - np.asarray(row["predicted_benefits"])))))
                expected_index = int(np.argmax(prediction))
                decision_mismatches += int(float(row["selected_alpha"]) != alphas[expected_index])
                execution_order_mismatches += int(float(row["execution_order"][0]) != float(row["selected_alpha"]))
                execution_order_mismatches += int(int(row["treatments"][str(float(row["selected_alpha"]))]["execution_index"]) != 0)
    if len(source_hashes) != 1:
        failures.append("source snapshot differs across splits")
    if max_posterior_error > 1e-12: failures.append("posterior replay mismatch")
    if max_feature_error > 1e-12: failures.append("feature replay mismatch")
    if max_prediction_error > 1e-12: failures.append("prediction replay mismatch")
    if max_metric_error > 1e-12: failures.append("metric replay mismatch")
    if decision_mismatches: failures.append(f"decision mismatches: {decision_mismatches}")
    if execution_order_mismatches: failures.append(f"execution order mismatches: {execution_order_mismatches}")

    rows = all_rows["formal"]
    costs = np.asarray([[row["treatments"][str(alpha)]["metrics"]["pose_auc10"] for alpha in alphas] for row in rows])
    selected = np.asarray([alphas.index(float(row["selected_alpha"])) for row in rows])
    learned = costs[np.arange(len(rows)), selected]
    primary = costs[:, 3] - learned
    summary_error = max(abs(float(summary["primary"]["mean_delta_fixed075_minus_learned"]) - float(primary.mean())),
                        float(np.max(np.abs(np.asarray(summary["primary"]["unit_deltas"]) - primary))),
                        float(np.max(np.abs(np.asarray(summary["primary"]["bootstrap_ci95"]) - bootstrap_ci(primary, design, 100)))))
    expected_counts = {str(alpha): int(np.sum(selected == index)) for index, alpha in enumerate(alphas)}
    if summary["selection_counts"] != expected_counts: failures.append("summary selection counts mismatch")
    if summary_error > 1e-12: failures.append("summary replay mismatch")
    result = {"contract_id": design["contract_id"], "valid": not failures, "failures": failures,
              "row_counts": {key: len(value) for key, value in all_rows.items()},
              "source_snapshot_sha256": next(iter(source_hashes)) if len(source_hashes) == 1 else sorted(str(x) for x in source_hashes),
              "posterior_replay_max_abs": max_posterior_error, "feature_replay_max_abs": max_feature_error,
              "prediction_replay_max_abs": max_prediction_error, "metric_replay_max_abs": max_metric_error,
              "decision_mismatches": decision_mismatches, "execution_order_mismatches": execution_order_mismatches,
              "summary_replay_max_abs": summary_error, "locked_model_sha256": sha256(args.locked_model),
              "formal_raw_sha256": sha256(args.output_dir / "formal/raw.jsonl"), "formal_summary_sha256": sha256(args.summary)}
    args.audit_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_f0_soft_policy_design.json"))
    parser.add_argument("--selection", type=Path, default=Path("docs/research/persistent_context_v2_matrix_f0_soft_policy_selection.json"))
    parser.add_argument("--locked-model", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    result = audit(parser.parse_args())
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
