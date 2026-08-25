#!/usr/bin/env python3
"""Independent raw replay audit for the frozen delay-history formal/smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FACTORS = (0, 1, 3, 4)
CONDITIONS = ("persistent", "no_persistence")
POLICIES = (
    "population_prior", "current_only", "correct_history_map",
    "correct_history_high_delay_gate", "shuffled_history",
    "wrong_sequence_history", "true_factor_oracle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pd_coefficients() -> tuple[float, float, float]:
    def simulate(position, velocity, target):
        for _ in range(10):
            velocity += (100.0 * (target - position) - 20.0 * velocity) * 0.01
            position += velocity * 0.01
        return position
    return simulate(1.0, 0.0, 0.0), simulate(0.0, 1.0, 0.0), simulate(0.0, 0.0, 1.0)


def infer(commands, states) -> np.ndarray:
    commands, states = np.asarray(commands, float), np.asarray(states, float)
    pc, vc, tc = pd_coefficients()
    result = []
    for index in range(len(commands)):
        p0, p1, v0 = states[index, :2], states[index + 1, :2], states[index, 5:7]
        target = (p1 - pc * p0 - vc * v0) / tc
        result.append((target - p0) / 100.0)
    return np.asarray(result)


def delayed(commands, delay: int) -> np.ndarray:
    commands = np.asarray(commands)
    if delay == 0: return commands.copy()
    result = np.zeros_like(commands); result[delay:] = commands[:-delay]
    return result


def posterior(commands, states, design: dict) -> dict:
    candidates = design["candidate_delay_steps"]
    observed = infer(commands, states)
    sigma = float(design["estimator"]["observation_noise_std"])
    logp = np.asarray([-np.sum((observed - delayed(commands, delay)) ** 2) / (2 * sigma**2) for delay in candidates])
    probability = np.exp(logp - logp.max()); probability /= probability.sum()
    return {
        "probability": probability, "map": int(candidates[int(np.argmax(probability))]),
        "entropy": float(-np.sum(probability * np.log(np.maximum(probability, 1e-300)))),
        "high": float(probability[candidates.index(3)] + probability[candidates.index(4)]),
    }


def pose_auc10(states, goal) -> float:
    states, goal = np.asarray(states, float), np.asarray(goal, float)
    post = states[1:11]
    position = np.linalg.norm(post[:, 2:4] - goal[None, 2:4], axis=1)
    delta = np.abs(post[:, 4] - goal[4]) % (2 * np.pi)
    angle = np.minimum(delta, 2 * np.pi - delta)
    return float(np.mean(position / 20.0 + angle / (np.pi / 9.0)))


def bootstrap(values, design: dict, stream: int) -> list[float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(int(design["bootstrap_seed"]) + stream)
    indexes = rng.integers(0, len(values), size=(int(design["bootstrap_resamples"]), len(values)))
    return np.quantile(values[indexes].mean(axis=1), [0.025, 0.975]).tolist()


def replay_effect(rows, policy, design, stream) -> dict:
    current = np.asarray([row["policies"]["current_only"]["metrics"]["pose_auc10"] for row in rows])
    target = np.asarray([row["policies"][policy]["metrics"]["pose_auc10"] for row in rows])
    delta = current - target
    return {
        "current_mean": float(current.mean()), "treatment_mean": float(target.mean()),
        "mean": float(delta.mean()), "relative_improvement": float(delta.mean() / current.mean()),
        "bootstrap_ci95": bootstrap(delta, design, stream),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)), "sequence_deltas": delta,
    }


def numeric_error(expected, observed) -> float:
    return float(np.max(np.abs(np.asarray(expected, float) - np.asarray(observed, float))))


def factor_for(condition, sequence_id, episode_index):
    base = sequence_id % 4
    if condition == "persistent" or episode_index == 0: return FACTORS[base]
    return FACTORS[(base + 1) % 4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "formal"), required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    pool = json.loads(args.pool_audit.read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run_dir / "runner_summary.json").read_text(encoding="utf-8"))
    e1_rows = read_jsonl(args.run_dir / "e1_evidence_raw.jsonl")
    e2_rows = read_jsonl(args.run_dir / "e2_evaluation_raw.jsonl")
    pairs = selection[f"{args.split}_segment_indices_by_sequence_e1_e2"]
    n = 4 if args.split == "smoke" else 32
    failures = []
    estimator_error = fifo_error = metric_error = summary_error = 0.0

    hashes = {
        "design_sha256": sha256(args.design), "selection_sha256": sha256(args.selection),
        "contract_sha256": sha256(args.contract), "pool_audit_sha256": sha256(args.pool_audit),
        "source_snapshot_sha256": sha256(args.source_snapshot),
        "e1_raw_sha256": sha256(args.run_dir / "e1_evidence_raw.jsonl"),
        "e2_raw_sha256": sha256(args.run_dir / "e2_evaluation_raw.jsonl"),
    }
    for key, value in hashes.items():
        if manifest.get(key) != value: failures.append(f"{key} mismatch")
    if not manifest.get("model_state_unchanged"): failures.append("world model changed")
    if not pool.get("valid") or pool.get("selection_sha256") != hashes["selection_sha256"]: failures.append("pool audit invalid")
    if len(e1_rows) != n or len(e2_rows) != 2 * n: failures.append("row count mismatch")
    pool_group = {int(row["segment_index"]): row for row in pool["groups"][args.split]}
    evidence = {int(row["sequence_id"]): row for row in e1_rows}
    for sequence_id in range(n):
        row = evidence[sequence_id]
        if int(row["segment_index"]) != int(pairs[sequence_id][0]): failures.append("E1 selection mismatch")
        if row["segment_sha256"] != pool_group[int(row["segment_index"])]["segment_sha256"]: failures.append("E1 segment hash mismatch")
        if int(row["factor_steps"]) != factor_for("persistent", sequence_id, 0): failures.append("E1 factor mismatch")
        replay = posterior(row["commands"], row["states"], design); stored = row["estimator"]
        estimator_error = max(estimator_error, numeric_error(replay["probability"], stored["posterior_probability"]), abs(replay["entropy"] - stored["entropy"]), abs(replay["high"] - stored["probability_high_delay"]))
        if replay["map"] != int(stored["map_delay_steps"]): failures.append("E1 MAP replay mismatch")
        fifo_error = max(fifo_error, numeric_error(delayed(row["commands"], int(row["factor_steps"])), row["effective_actions"]))

    lookup = {(row["condition"], int(row["sequence_id"])): row for row in e2_rows}
    for condition in CONDITIONS:
        for sequence_id in range(n):
            row = lookup[(condition, sequence_id)]
            if int(row["segment_index"]) != int(pairs[sequence_id][1]): failures.append("E2 selection mismatch")
            if row["segment_sha256"] != pool_group[int(row["segment_index"])]["segment_sha256"]: failures.append("E2 segment hash mismatch")
            true_factor = factor_for(condition, sequence_id, 1)
            if int(row["factor_steps"]) != true_factor: failures.append("E2 factor mismatch")
            own = posterior(evidence[sequence_id]["commands"], evidence[sequence_id]["states"], design)
            donor = (sequence_id + int(design["wrong_donor_sequence_offset"])) % n
            wrong = posterior(evidence[donor]["commands"], evidence[donor]["states"], design)
            order = np.random.default_rng(np.random.SeedSequence([int(design["shuffled_command_seed"]), sequence_id])).permutation(len(evidence[sequence_id]["commands"]))
            if np.array_equal(order, np.arange(len(order))): order = np.roll(order, 1)
            shuffled = posterior(np.asarray(evidence[sequence_id]["commands"])[order], evidence[sequence_id]["states"], design)
            gate = own["map"] if own["high"] >= float(design["estimator"]["high_delay_gate_probability"]) else 2
            contexts = {
                "population_prior": 2, "current_only": 2, "correct_history_map": own["map"],
                "correct_history_high_delay_gate": gate, "shuffled_history": shuffled["map"],
                "wrong_sequence_history": wrong["map"], "true_factor_oracle": true_factor,
            }
            if not row.get("decision_pre_e2_execution"): failures.append("decision order missing")
            if row.get("policy_evaluation_order") != list(POLICIES): failures.append("policy order mismatch")
            for policy, context in contexts.items():
                value = row["policies"][policy]
                if int(value["context_delay_steps"]) != context or int(row["decision_contexts"][policy]) != context: failures.append(f"context replay mismatch {condition}/{sequence_id}/{policy}")
                if int(value["current_episode_evidence_count"]) != 0: failures.append("E2 evidence leak")
                fifo_error = max(fifo_error, numeric_error(delayed(value["commands"], true_factor), value["effective_actions"]))
                metric_error = max(metric_error, abs(pose_auc10(value["states"], row["goal_state"]) - float(value["metrics"]["pose_auc10"])))
            wrong_factor = int(row["policies"]["wrong_sequence_history"]["donors"][0]["factor_steps"])
            if wrong_factor == true_factor: failures.append("wrong donor factor collision")
            if row["policies"]["population_prior"]["state_sha256"] != row["policies"]["current_only"]["state_sha256"]: failures.append("population/current identity")
            other = lookup[("no_persistence" if condition == "persistent" else "persistent", sequence_id)]
            if row["segment_index"] != other["segment_index"] or not np.array_equal(row["initial_state"], other["initial_state"]) or not np.array_equal(row["goal_state"], other["goal_state"]): failures.append("condition pairing mismatch")

    replay_effects = {}
    for condition_index, condition in enumerate(CONDITIONS):
        rows = [lookup[(condition, sequence_id)] for sequence_id in range(n)]
        replay_effects[condition] = {}
        for policy_index, policy in enumerate(POLICIES):
            policy_mean = float(np.mean([row["policies"][policy]["metrics"]["pose_auc10"] for row in rows]))
            summary_error = max(summary_error, abs(policy_mean - float(summary["policy_means"][condition][policy])))
            deadline = float(np.mean([row["policies"][policy]["deadline_success"] for row in rows]))
            summary_error = max(summary_error, abs(deadline - float(summary["deadline_success"][condition][policy])))
            if policy == "current_only": continue
            result = replay_effect(rows, policy, design, 1000 * condition_index + policy_index)
            replay_effects[condition][policy] = result
            stored = summary["effects"][condition][policy]
            for key in ("current_mean", "treatment_mean", "mean", "relative_improvement", "positive_fraction", "tie_fraction", "negative_fraction", "bootstrap_ci95", "sequence_deltas"):
                summary_error = max(summary_error, numeric_error(result[key], stored[key]))
        for factor in FACTORS:
            subset = [row for row in rows if int(row["factor_steps"]) == factor]
            result = replay_effect(subset, "correct_history_map", design, 10_000 + 100 * condition_index + factor)
            stored = summary["by_current_factor"][condition][str(factor)]
            for key in ("mean", "bootstrap_ci95", "sequence_deltas"):
                summary_error = max(summary_error, numeric_error(result[key], stored[key]))
    persistent = replay_effects["persistent"]["correct_history_map"]["sequence_deltas"]
    no_persistence = replay_effects["no_persistence"]["correct_history_map"]["sequence_deltas"]
    did_values = persistent - no_persistence
    did = summary["did_correct_history_map"]
    summary_error = max(summary_error, abs(float(did_values.mean()) - float(did["mean"])), numeric_error(bootstrap(did_values, design, 9000), did["bootstrap_ci95"]), numeric_error(did_values, did["sequence_deltas"]))

    if estimator_error > 1e-9: failures.append("estimator replay error")
    if fifo_error > 1e-6: failures.append("FIFO replay error")
    if metric_error > 1e-9: failures.append("metric replay error")
    if summary_error > 1e-9: failures.append("summary replay error")
    if not summary.get("valid"): failures.append("runner summary invalid")
    audit = {
        "contract_id": design["contract_id"], "split": args.split,
        "evidence_level": "formal" if args.split == "formal" else "formal_pipeline_smoke",
        "valid": not failures, "failures": failures, "e1_rows": len(e1_rows), "e2_rows": len(e2_rows),
        "estimator_replay_max_abs": estimator_error, "fifo_replay_max_abs": fifo_error,
        "metric_replay_max_abs": metric_error, "summary_replay_max_abs": summary_error, **hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
