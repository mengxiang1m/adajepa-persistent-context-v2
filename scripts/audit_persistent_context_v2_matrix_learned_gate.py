#!/usr/bin/env python3
"""Independent replay audit for matrix learned surrogate gate; imports no experiment implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np


EXPECTED_DESIGN_SHA256 = "607d5e943635e34c10883ac16b37162c212e1b0e30fd075bcb1f7e6136f3d756"
CONTRACT_ID = "persistent-context-v2-matrix-learned-surrogate-gate-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prior_parameters() -> tuple[np.ndarray, np.ndarray]:
    rotations = (-30.0, -15.0, 0.0, 15.0, 30.0)
    gains = (0.75, 1.0, 1.25)
    samples = []
    for rotation in rotations:
        theta = math.radians(rotation)
        for gain in gains:
            samples.append([gain * math.cos(theta), gain * math.sin(theta)])
    values = np.asarray(samples, dtype=np.float64)
    return values.mean(axis=0), np.cov(values.T, bias=True) + 1e-4 * np.eye(2)


def replay_posterior(observations) -> tuple[np.ndarray, np.ndarray]:
    mean, covariance = prior_parameters()
    precision = np.linalg.inv(covariance)
    information = precision @ mean
    observations = np.asarray(observations, dtype=np.float64).reshape(-1, 2)
    precision = precision + len(observations) * 10000.0 * np.eye(2)
    information = information + 10000.0 * observations.sum(axis=0)
    return np.linalg.solve(precision, information), np.linalg.inv(precision)


def feature(row: dict, design: dict) -> np.ndarray:
    z = np.asarray(row["posterior"]["mean_z"], dtype=np.float64)
    gain = float(np.linalg.norm(z))
    rotation = float(math.degrees(math.atan2(z[1], z[0])))
    g = (gain - float(design["gain_center"])) / float(design["gain_scale"])
    r = rotation / float(design["rotation_scale_degrees"])
    return np.asarray([1.0, g, r, g * g, g * r, r * r], dtype=np.float64)


def arrays(rows: list[dict], design: dict):
    x = np.stack([feature(row, design) for row in rows])
    population = np.asarray([row["e2"]["population"]["metrics"]["pose_auc10"] for row in rows], dtype=np.float64)
    context = np.asarray([row["e2"]["context"]["metrics"]["pose_auc10"] for row in rows], dtype=np.float64)
    return x, population, context


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + float(alpha) * penalty, x.T @ y)


def bootstrap(values: np.ndarray, design: dict, stream: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(design["bootstrap_seed"]) + int(stream))
    indexes = rng.integers(0, len(values), size=(int(design["bootstrap_resamples"]), len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def effect(population: np.ndarray, treatment: np.ndarray, design: dict, stream: int) -> dict:
    delta = np.asarray(population, dtype=np.float64) - np.asarray(treatment, dtype=np.float64)
    return {
        "population_mean": float(np.mean(population)),
        "treatment_mean": float(np.mean(treatment)),
        "mean_delta": float(np.mean(delta)),
        "relative_improvement": float(np.mean(delta) / np.mean(population)),
        "bootstrap_ci95_delta": bootstrap(delta, design, stream),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)),
        "harm_fraction": float(np.mean(delta < -1e-12)),
    }


def pose_auc10(states, goal_state) -> float:
    states = np.asarray(states, dtype=np.float64)
    goal = np.asarray(goal_state, dtype=np.float64)
    post = states[1:11]
    position = np.linalg.norm(post[:, 2:4] - goal[None, 2:4], axis=1)
    angle_delta = np.abs(post[:, 4] - goal[4]) % (2.0 * np.pi)
    angle = np.minimum(angle_delta, 2.0 * np.pi - angle_delta)
    return float(np.mean(position / 20.0 + angle / (np.pi / 9.0)))


def max_numeric_difference(left, right) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        common = set(left) & set(right)
        return max((max_numeric_difference(left[key], right[key]) for key in common), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        return max((max_numeric_difference(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return abs(float(left) - float(right))
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate/evaluation"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_learned_gate_design.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    args = parser.parse_args()
    failures = []
    if file_sha256(args.design) != EXPECTED_DESIGN_SHA256:
        failures.append("design_hash")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    if design.get("contract_id") != CONTRACT_ID:
        failures.append("contract_id")
    manifest = json.loads((args.evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
    model = json.loads((args.evaluation_dir / "gate_model.json").read_text(encoding="utf-8"))
    reported = json.loads((args.evaluation_dir / "runner_summary.json").read_text(encoding="utf-8"))
    rows = {}
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    checkpoint_hash = file_sha256(args.checkpoint)
    data_hash = file_sha256(args.data)
    all_segments = []
    posterior_max = 0.0
    covariance_max = 0.0
    pairing_failures = 0
    seed_failures = 0
    factor_failures = 0
    eligibility_failures = 0
    metric_replay_max = 0.0
    for split in ("train", "dev", "formal"):
        raw_path = args.data_dir / split / "raw.jsonl"
        collection_manifest = json.loads((args.data_dir / split / "manifest.json").read_text(encoding="utf-8"))
        if collection_manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256:
            failures.append(f"{split}_manifest_design")
        if collection_manifest.get("checkpoint_sha256") != checkpoint_hash:
            failures.append(f"{split}_checkpoint_hash")
        if collection_manifest.get("data_sha256") != data_hash:
            failures.append(f"{split}_data_hash")
        if collection_manifest.get("raw_sha256") != file_sha256(raw_path):
            failures.append(f"{split}_manifest_raw_hash")
        if file_sha256(raw_path) != manifest["raw_sha256"][split]:
            failures.append(f"{split}_raw_hash")
        split_rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "paired_sequence"]
        split_rows.sort(key=lambda row: int(row["sequence_id"]))
        if len(split_rows) != 32 or [int(row["sequence_id"]) for row in split_rows] != list(range(32)):
            failures.append(f"{split}_count")
        rows[split] = split_rows
        expected_indices = [int(value) for value in design["splits"][split]["segment_indices"]]
        actual_indices = []
        for row in split_rows:
            sequence_id = int(row["sequence_id"])
            factor_index = sequence_id % 8
            if int(row["factor_index"]) != factor_index:
                factor_failures += 1
            expected_factor = design["formal_factors"][factor_index]
            if abs(float(row["gain"]) - float(expected_factor["gain"])) > 1e-12 or abs(float(row["rotation_degrees"]) - float(expected_factor["rotation_degrees"])) > 1e-12:
                factor_failures += 1
            for episode_name, episode_index in (("e1", 0), ("e2", 1)):
                episode = row[episode_name]
                ordinal = sequence_id * 2 + episode_index
                actual_indices.append(int(episode["segment_index"]))
                segment_states = np.asarray(segments[int(episode["segment_index"])]["states"], dtype=np.float64)
                displacement = float(np.linalg.norm(segment_states[10, 2:4] - segment_states[0, 2:4]))
                if displacement < float(design["segment_pool"]["minimum_nominal_block_displacement_at_10"]):
                    eligibility_failures += 1
                expected_env = int(design["splits"][split]["env_seed_base"]) + sequence_id * 100 + episode_index
                expected_cem = int(design["splits"][split]["cem_seed_base"]) + sequence_id * 100 + episode_index
                if int(episode["env_seed"]) != expected_env or int(episode["cem_seed"]) != expected_cem:
                    seed_failures += 1
            if not np.array_equal(np.asarray(row["e2"]["population"]["initial_state"]), np.asarray(row["e2"]["context"]["initial_state"])):
                pairing_failures += 1
            if not np.array_equal(np.asarray(row["e2"]["population"]["goal_state"]), np.asarray(row["e2"]["context"]["goal_state"])):
                pairing_failures += 1
            for policy in ("population", "context"):
                replayed_metric = pose_auc10(row["e2"][policy]["states"], row["e2"][policy]["goal_state"])
                metric_replay_max = max(metric_replay_max, abs(replayed_metric - float(row["e2"][policy]["metrics"]["pose_auc10"])))
            mean, covariance = replay_posterior(row["e1"]["matrix_observations"])
            posterior_max = max(posterior_max, float(np.max(np.abs(mean - np.asarray(row["posterior"]["mean_z"], dtype=np.float64)))))
            covariance_max = max(covariance_max, float(np.max(np.abs(covariance - np.asarray(row["posterior"]["covariance"], dtype=np.float64)))))
        if actual_indices != expected_indices:
            failures.append(f"{split}_segment_schedule")
        all_segments.extend(actual_indices)
    if len(set(all_segments)) != 192:
        failures.append("cross_split_segment_overlap")
    if set(all_segments) & set(int(value) for value in design["segment_pool"]["excluded_matrix_stage1_segments"]):
        failures.append("matrix_stage1_overlap")
    if pairing_failures:
        failures.append("pairing")
    if seed_failures:
        failures.append("seeds")
    if factor_failures:
        failures.append("factors")
    if eligibility_failures:
        failures.append("segment_eligibility")
    if metric_replay_max > 1e-12:
        failures.append("metric_replay")
    if posterior_max > 1e-12 or covariance_max > 1e-12:
        failures.append("posterior_replay")

    train_x, train_population, train_context = arrays(rows["train"], design)
    dev_x, dev_population, dev_context = arrays(rows["dev"], design)
    candidate_rows = []
    for alpha in design["ridge_alphas"]:
        beta = ridge(train_x, train_population - train_context, float(alpha))
        prediction = dev_x @ beta
        decision = prediction > 0.0
        outcome = np.where(decision, dev_context, dev_population)
        candidate_rows.append((float(np.mean(dev_population - outcome)), float(alpha)))
    selected_alpha = max(candidate_rows)[1]
    if selected_alpha != float(model["selected_alpha"]) or selected_alpha != float(reported["selected_alpha"]):
        failures.append("alpha_selection")
    fit_x = np.concatenate([train_x, dev_x], axis=0)
    fit_y = np.concatenate([train_population - train_context, dev_population - dev_context])
    beta = ridge(fit_x, fit_y, selected_alpha)
    beta_error = float(np.max(np.abs(beta - np.asarray(model["beta"], dtype=np.float64))))
    if beta_error > 1e-12:
        failures.append("model_refit")

    formal_x, population, context = arrays(rows["formal"], design)
    predictions = formal_x @ beta
    learned = predictions > 0.0
    functional = np.asarray([
        (np.linalg.norm(np.asarray(row["posterior"]["mean_z"], dtype=np.float64)) < float(design["gain_center"]))
        or abs(math.degrees(math.atan2(float(row["posterior"]["mean_z"][1]), float(row["posterior"]["mean_z"][0])))) >= 15.0
        for row in rows["formal"]
    ], dtype=bool)
    random = np.zeros(32, dtype=bool)
    order = np.random.default_rng(int(design["random_control_seed"])).permutation(32)
    random[order[:int(np.sum(learned))]] = True
    outcomes = {
        "always_context": context,
        "learned_gate": np.where(learned, context, population),
        "functional_gate": np.where(functional, context, population),
        "inverted_learned_gate": np.where(~learned, context, population),
        "selection_matched_random_gate": np.where(random, context, population),
        "best_of_two_behavior_ceiling": np.minimum(population, context),
    }
    replay_policies = {name: effect(population, value, design, 100 + index) for index, (name, value) in enumerate(outcomes.items())}
    summary_error = max_numeric_difference(replay_policies, reported["policies"])
    if summary_error > 1e-12:
        failures.append("summary_replay")
    decisions = read_jsonl(args.evaluation_dir / "formal_decisions.jsonl")
    decision_mismatches = 0
    for index, decision in enumerate(decisions):
        decision_mismatches += int(bool(decision["learned_use_context"]) != bool(learned[index]))
        decision_mismatches += int(bool(decision["functional_use_context"]) != bool(functional[index]))
        decision_mismatches += int(bool(decision["random_use_context"]) != bool(random[index]))
        if abs(float(decision["predicted_delta"]) - float(predictions[index])) > 1e-12:
            decision_mismatches += 1
    if len(decisions) != 32 or decision_mismatches:
        failures.append("formal_decisions")
    audit = {
        "contract_id": CONTRACT_ID,
        "valid": not failures,
        "failures": failures,
        "unit_counts": {split: len(rows[split]) for split in rows},
        "all_segment_count": len(all_segments),
        "all_segments_unique": len(set(all_segments)) == len(all_segments),
        "pairing_failure_count": pairing_failures,
        "seed_failure_count": seed_failures,
        "factor_failure_count": factor_failures,
        "eligibility_failure_count": eligibility_failures,
        "metric_replay_max_abs": metric_replay_max,
        "posterior_replay_max_abs": posterior_max,
        "posterior_covariance_replay_max_abs": covariance_max,
        "selected_alpha": selected_alpha,
        "beta_replay_max_abs": beta_error,
        "formal_decision_mismatch_count": decision_mismatches,
        "summary_replay_max_abs": summary_error,
        "design_sha256": file_sha256(args.design),
        "raw_sha256": {split: file_sha256(args.data_dir / split / "raw.jsonl") for split in rows},
        "checkpoint_sha256": checkpoint_hash,
        "data_sha256": data_hash,
    }
    (args.evaluation_dir / "independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
