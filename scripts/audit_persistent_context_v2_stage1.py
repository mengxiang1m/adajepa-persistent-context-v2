#!/usr/bin/env python3
"""Independent evidence reconstruction for V2 Stage 1 raw artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


POLICIES = (
    "population_prior", "current_only", "correct_history", "shuffled_history",
    "wrong_sequence_history", "true_factor_oracle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int, stream: int, resamples: int) -> tuple:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    parts = []
    remaining = resamples
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        parts.append(values[indices].mean(axis=1))
        remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(parts), [0.025, 0.975]))


def bootstrap_relative_ci(current: np.ndarray, treatment: np.ndarray, seed: int, stream: int, resamples: int) -> tuple:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    parts = []
    remaining = resamples
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(current), size=(count, len(current)))
        current_mean = current[indices].mean(axis=1)
        treatment_mean = treatment[indices].mean(axis=1)
        parts.append((current_mean - treatment_mean) / current_mean)
        remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(parts), [0.025, 0.975]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    raw_path = args.run_dir / "raw_results.csv"
    summary_path = args.run_dir / "summary.json"
    manifest_path = args.run_dir / "run_manifest.json"
    with raw_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = summary["config"]
    n, episodes = int(config["n_sequences"]), int(config["n_episodes"])
    grouped = defaultdict(list)
    episode_one = defaultdict(dict)
    row_lookup = {}
    for row in rows:
        condition, sequence_id, episode_id, policy = row["condition"], int(row["sequence_id"]), int(row["episode_id"]), row["policy"]
        row_lookup[(condition, policy, sequence_id, episode_id)] = row
        if episode_id >= 2:
            grouped[(condition, sequence_id, policy)].append(row)
        elif policy in ("current_only", "correct_history"):
            episode_one[(condition, sequence_id)][policy] = row
    identity_failures = []
    identity_fields = ("scenario_hash", "factor_id", "target", "noise", "action", "response", "early_task_cost", "unsafe")
    for key, pair in episode_one.items():
        if set(pair) != {"current_only", "correct_history"} or any(pair["current_only"][field] != pair["correct_history"][field] for field in identity_fields):
            identity_failures.append(str(key))
    factor_failures = []
    cross_condition_pairing_failures = []
    for sequence_id in range(n):
        factors = {row["factor_id"] for row in rows if row["condition"] == "persistent" and row["policy"] == "current_only" and int(row["sequence_id"]) == sequence_id}
        if len(factors) != 1:
            factor_failures.append(sequence_id)
        for episode_id in range(1, episodes + 1):
            left = row_lookup[("persistent", "current_only", sequence_id, episode_id)]
            right = row_lookup[("no_persistence", "current_only", sequence_id, episode_id)]
            if left["target"] != right["target"] or left["noise"] != right["noise"]:
                cross_condition_pairing_failures.append(f"{sequence_id}:{episode_id}")
    donor_failures = [
        f"{row['condition']}:{row['sequence_id']}:{row['episode_id']}:{row['policy']}"
        for row in rows
        if int(row["episode_id"]) >= 2 and row["policy"] in ("shuffled_history", "wrong_sequence_history") and row["donor_sequence_id"] == row["sequence_id"]
    ]
    count_failures = []
    for row in rows:
        episode_id, policy = int(row["episode_id"]), row["policy"]
        before, after = int(row["history_count_before"]), int(row["history_count_after"])
        if policy in ("correct_history", "shuffled_history", "wrong_sequence_history") and (before != episode_id - 1 or after != episode_id):
            count_failures.append(f"{row['condition']}:{row['sequence_id']}:{episode_id}:{policy}")
        if policy == "current_only" and (before != 0 or after != 1):
            count_failures.append(f"{row['condition']}:{row['sequence_id']}:{episode_id}:{policy}")
    values = {}
    for condition in ("persistent", "no_persistence"):
        values[condition] = {}
        for policy in POLICIES:
            values[condition][policy] = np.asarray([
                np.mean([float(row["early_task_cost"]) for row in grouped[(condition, sequence_id, policy)]])
                for sequence_id in range(n)
            ])
    p_current = values["persistent"]["current_only"]
    p_correct = values["persistent"]["correct_history"]
    np_current = values["no_persistence"]["current_only"]
    np_correct = values["no_persistence"]["correct_history"]
    p_delta = p_current - p_correct
    np_delta = np_current - np_correct
    did = p_delta - np_delta
    p_true = values["persistent"]["true_factor_oracle"]
    p_shuffled = values["persistent"]["shuffled_history"]
    p_wrong = values["persistent"]["wrong_sequence_history"]
    current_mean = float(p_current.mean())
    true_relative = float((p_current.mean() - p_true.mean()) / p_current.mean())
    correct_relative = float(p_delta.mean() / p_current.mean())
    no_persist_relative = float(np_delta.mean() / np_current.mean())
    shuffled_relative = float((p_current - p_shuffled).mean() / p_current.mean())
    wrong_relative = float((p_current - p_wrong).mean() / p_current.mean())
    true_ratio_ci = bootstrap_relative_ci(p_current, p_true, int(config["bootstrap_seed"]), 950, int(config["bootstrap_resamples"]))
    p_delta_ci = bootstrap_ci(p_delta, int(config["bootstrap_seed"]), 100, int(config["bootstrap_resamples"]))
    did_ci = bootstrap_ci(did, int(config["bootstrap_seed"]), 900, int(config["bootstrap_resamples"]))
    true_gap = float(p_current.mean() - p_true.mean())
    recovery = float(p_delta.mean() / true_gap)
    criteria = {
        "true_oracle_relative_improvement_at_least_25pct_and_ci_lower_20pct": true_relative >= 0.25 and true_ratio_ci[0] >= 0.20,
        "persistent_correct_relative_at_least_30pct_and_ci_lower_above_20pct_current": correct_relative >= 0.30 and p_delta_ci[0] > 0.20 * current_mean,
        "did_ci_lower_above_15pct_current": did_ci[0] > 0.15 * current_mean,
        "history_recovers_at_least_half_true_gap": true_gap > 0 and recovery >= 0.50,
        "persistent_positive_sequence_fraction_at_least_75pct": float(np.mean(p_delta > 0)) >= 0.75,
        "no_persistence_relative_improvement_at_most_5pct": no_persist_relative <= 0.05,
        "shuffled_and_wrong_not_comparable": shuffled_relative <= 0.05 and wrong_relative <= 0.05 and shuffled_relative < 0.5 * correct_relative and wrong_relative < 0.5 * correct_relative,
    }
    structural = {
        "raw_row_count": len(rows) == 2 * n * episodes * len(POLICIES),
        "episode1_exact_identity": len(episode_one) == 2 * n and not identity_failures,
        "persistent_factor_lifetime": not factor_failures,
        "cross_condition_nuisance_noise_pairing": not cross_condition_pairing_failures,
        "donor_sequence_isolation": not donor_failures,
        "history_count_budget": not count_failures,
        "raw_hash": sha256(raw_path) == manifest["raw_results_sha256"],
        "summary_hash": sha256(summary_path) == manifest["summary_sha256"],
    }
    criteria["all_sequences_and_audits_valid"] = all(structural.values())
    verdict = "HISTORY_VALUE_SUPPORTED" if all(criteria.values()) else "HISTORY_VALUE_NOT_ESTABLISHED"
    audit = {
        "schema": "persistent-context-v2-stage1-independent-audit-v1",
        "passed": all(structural.values()) and criteria == summary["criteria"] and verdict == summary["decision"]["verdict"],
        "structural_checks": structural,
        "criteria": criteria,
        "recomputed": {
            "persistent_current_mean_cost": current_mean,
            "persistent_correct_mean_cost": float(p_correct.mean()),
            "persistent_true_mean_cost": float(p_true.mean()),
            "persistent_correct_relative_improvement": correct_relative,
            "persistent_correct_difference_ci95": p_delta_ci,
            "no_persistence_correct_relative_improvement": no_persist_relative,
            "shuffled_relative_improvement": shuffled_relative,
            "wrong_relative_improvement": wrong_relative,
            "did_mean": float(did.mean()),
            "did_ci95": did_ci,
            "gap_recovery": recovery,
            "positive_sequence_fraction": float(np.mean(p_delta > 0)),
            "true_oracle_relative_improvement": true_relative,
            "true_oracle_bootstrap_relative_ci95": true_ratio_ci,
        },
        "failures": {"identity": identity_failures, "factor": factor_failures, "cross_condition_pairing": cross_condition_pairing_failures, "donor": donor_failures, "count": count_failures},
        "recomputed_verdict": verdict,
        "reported_decision": summary["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
