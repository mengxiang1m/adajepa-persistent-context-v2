#!/usr/bin/env python3
"""Independent audit for the development-only discrete-delay history smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pd_coefficients() -> tuple[float, float, float]:
    def simulate(position: float, velocity: float, target: float) -> tuple[float, float]:
        for _ in range(10):
            velocity += (100.0 * (target - position) - 20.0 * velocity) * 0.01
            position += velocity * 0.01
        return position, velocity
    target, _ = simulate(0.0, 0.0, 1.0)
    initial, _ = simulate(1.0, 0.0, 0.0)
    velocity, _ = simulate(0.0, 1.0, 0.0)
    return initial, velocity, target


def infer(commands, states) -> np.ndarray:
    commands, states = np.asarray(commands, float), np.asarray(states, float)
    pc, vc, tc = pd_coefficients()
    values = []
    for index in range(len(commands)):
        p0, p1, v0 = states[index, :2], states[index + 1, :2], states[index, 5:7]
        target = (p1 - pc * p0 - vc * v0) / tc
        values.append((target - p0) / 100.0)
    return np.asarray(values)


def delayed(commands, delay: int) -> np.ndarray:
    commands = np.asarray(commands)
    result = np.zeros_like(commands)
    if delay == 0:
        return commands.copy()
    result[delay:] = commands[:-delay]
    return result


def posterior(commands, states, design: dict) -> dict:
    candidates = design["candidate_delay_steps"]
    observed = infer(commands, states)
    sigma = float(design["estimator"]["observation_noise_std"])
    logp = np.asarray([-np.sum((observed - delayed(commands, delay)) ** 2) / (2 * sigma**2) for delay in candidates])
    probability = np.exp(logp - logp.max())
    probability /= probability.sum()
    index = int(np.argmax(probability))
    return {
        "probability": probability,
        "map": int(candidates[index]),
        "entropy": float(-np.sum(probability * np.log(np.maximum(probability, 1e-300)))),
        "high": float(sum(probability[candidates.index(delay)] for delay in (3, 4))),
    }


def pose_auc10(states, goal) -> float:
    states, goal = np.asarray(states, float), np.asarray(goal, float)
    post = states[1:11]
    position = np.linalg.norm(post[:, 2:4] - goal[None, 2:4], axis=1)
    delta = np.abs(post[:, 4] - goal[4]) % (2 * np.pi)
    angle = np.minimum(delta, 2 * np.pi - delta)
    return float(np.mean(position / 20.0 + angle / (np.pi / 9.0)))


def maximum_error(expected, observed) -> float:
    return float(np.max(np.abs(np.asarray(expected, float) - np.asarray(observed, float))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    manifest = json.loads((args.smoke_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.smoke_dir / "runner_summary.json").read_text(encoding="utf-8"))
    evidence_rows = read_jsonl(args.smoke_dir / "e1_evidence_raw.jsonl")
    evaluation_rows = read_jsonl(args.smoke_dir / "e2_evaluation_raw.jsonl")
    failures: list[str] = []
    max_estimator_error = max_fifo_error = max_metric_error = max_summary_error = 0.0

    if manifest["design_sha256"] != sha256(args.design): failures.append("design hash mismatch")
    if manifest["contract_sha256"] != sha256(args.contract): failures.append("contract hash mismatch")
    if manifest["source_snapshot_sha256"] != sha256(args.source_snapshot): failures.append("source snapshot hash mismatch")
    if manifest["e1_raw_sha256"] != sha256(args.smoke_dir / "e1_evidence_raw.jsonl"): failures.append("E1 raw hash mismatch")
    if manifest["e2_raw_sha256"] != sha256(args.smoke_dir / "e2_evaluation_raw.jsonl"): failures.append("E2 raw hash mismatch")
    if not manifest.get("model_state_unchanged"): failures.append("world-model state changed")
    if len(evidence_rows) != 4 or len(evaluation_rows) != 8: failures.append("incomplete smoke rows")

    evidence = {int(row["sequence_id"]): row for row in evidence_rows}
    expected_factors = design["e1_factor_steps"]
    segments = design["segment_selection"]["smoke_segment_indices_by_sequence_e1_e2"]
    for sequence_id, row in evidence.items():
        if int(row["factor_steps"]) != int(expected_factors[sequence_id]): failures.append("E1 factor schedule mismatch")
        if int(row["segment_index"]) != int(segments[sequence_id][0]): failures.append("E1 segment mismatch")
        replay = posterior(row["commands"], row["states"], design)
        stored = row["estimator"]
        max_estimator_error = max(
            max_estimator_error,
            maximum_error(replay["probability"], stored["posterior_probability"]),
            abs(replay["entropy"] - float(stored["entropy"])),
            abs(replay["high"] - float(stored["probability_high_delay"])),
        )
        if replay["map"] != int(stored["map_delay_steps"]) or replay["map"] != int(row["factor_steps"]): failures.append("E1 MAP replay mismatch")
        expected_effective = delayed(row["commands"], int(row["factor_steps"]))
        max_fifo_error = max(max_fifo_error, maximum_error(expected_effective, row["effective_actions"]))

    rows = {(row["condition"], int(row["sequence_id"])): row for row in evaluation_rows}
    expected_contexts: dict[tuple[str, int, str], int] = {}
    for condition in design["conditions"]:
        for sequence_id in range(4):
            row = rows.get((condition, sequence_id))
            if row is None:
                continue
            if int(row["segment_index"]) != int(segments[sequence_id][1]): failures.append("E2 segment mismatch")
            true_index = sequence_id if condition == "persistent" else (sequence_id + int(design["no_persistence_e2_factor_index_shift"])) % 4
            true_factor = int(expected_factors[true_index])
            if int(row["factor_steps"]) != true_factor: failures.append("E2 factor schedule mismatch")
            own = posterior(evidence[sequence_id]["commands"], evidence[sequence_id]["states"], design)
            wrong_id = (sequence_id + 1) % 4
            wrong = posterior(evidence[wrong_id]["commands"], evidence[wrong_id]["states"], design)
            rng = np.random.default_rng(np.random.SeedSequence([int(design["shuffled_command_seed"]), sequence_id]))
            order = rng.permutation(len(evidence[sequence_id]["commands"]))
            if np.array_equal(order, np.arange(len(order))): order = np.roll(order, 1)
            shuffled = posterior(np.asarray(evidence[sequence_id]["commands"])[order], evidence[sequence_id]["states"], design)
            gate = own["map"] if own["high"] >= float(design["estimator"]["high_delay_gate_probability"]) else int(design["population_prior_steps"])
            expected = {
                "population_prior": 2,
                "current_only": 2,
                "correct_history_map": own["map"],
                "correct_history_high_delay_gate": gate,
                "shuffled_history": shuffled["map"],
                "wrong_sequence_history": wrong["map"],
                "true_factor_oracle": true_factor,
            }
            for policy, context in expected.items():
                value = row["policies"][policy]
                expected_contexts[(condition, sequence_id, policy)] = context
                if int(value["context_delay_steps"]) != context: failures.append(f"context mismatch {condition}/{sequence_id}/{policy}")
                if int(value["current_episode_evidence_count"]) != 0: failures.append("current E2 evidence leak")
                max_fifo_error = max(max_fifo_error, maximum_error(delayed(value["commands"], true_factor), value["effective_actions"]))
                max_metric_error = max(max_metric_error, abs(pose_auc10(value["states"], row["goal_state"]) - float(value["metrics"]["pose_auc10"])))
            if row["policies"]["population_prior"]["state_sha256"] != row["policies"]["current_only"]["state_sha256"]: failures.append("population/current identity failure")
            other = rows.get(("no_persistence" if condition == "persistent" else "persistent", sequence_id))
            if other is not None:
                if int(other["segment_index"]) != int(row["segment_index"]): failures.append("condition scene mismatch")
                if not np.array_equal(other["initial_state"], row["initial_state"]) or not np.array_equal(other["goal_state"], row["goal_state"]): failures.append("condition state mismatch")

    for condition in design["conditions"]:
        condition_rows = [rows[(condition, sequence_id)] for sequence_id in range(4)]
        for policy in design["policies"]:
            replay_mean = float(np.mean([row["policies"][policy]["metrics"]["pose_auc10"] for row in condition_rows]))
            max_summary_error = max(max_summary_error, abs(replay_mean - float(summary["policy_means"][condition][policy])))
            if policy != "current_only":
                current = np.asarray([row["policies"]["current_only"]["metrics"]["pose_auc10"] for row in condition_rows])
                target = np.asarray([row["policies"][policy]["metrics"]["pose_auc10"] for row in condition_rows])
                max_summary_error = max(max_summary_error, abs(float(np.mean(current - target)) - float(summary["effects"][condition][policy]["mean_delta"])))

    tolerance = 1e-9
    if max_estimator_error > tolerance: failures.append("estimator replay error")
    if max_fifo_error > 1e-6: failures.append("FIFO replay error")
    if max_metric_error > tolerance: failures.append("metric replay error")
    if max_summary_error > tolerance: failures.append("summary replay error")
    if not summary.get("valid"): failures.append("runner summary invalid")
    audit = {
        "contract_id": design["contract_id"],
        "evidence_level": "development_smoke_not_formal",
        "valid": not failures,
        "failures": failures,
        "e1_rows": len(evidence_rows),
        "e2_rows": len(evaluation_rows),
        "estimator_replay_max_abs": max_estimator_error,
        "fifo_replay_max_abs": max_fifo_error,
        "metric_replay_max_abs": max_metric_error,
        "summary_replay_max_abs": max_summary_error,
        "design_sha256": sha256(args.design),
        "contract_sha256": sha256(args.contract),
        "source_snapshot_sha256": sha256(args.source_snapshot),
        "e1_raw_sha256": sha256(args.smoke_dir / "e1_evidence_raw.jsonl"),
        "e2_raw_sha256": sha256(args.smoke_dir / "e2_evaluation_raw.jsonl"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
