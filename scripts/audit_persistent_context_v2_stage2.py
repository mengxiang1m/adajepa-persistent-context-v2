#!/usr/bin/env python3
"""Independent raw audit, including reconstruction of every RLS estimate."""

from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

POLICIES = ("population_prior", "current_only_rls", "persistent_rls", "shuffled_rls", "wrong_sequence_rls", "categorical_history_oracle", "true_factor_oracle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int, stream: int, resamples: int) -> tuple:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    parts, remaining = [], resamples
    while remaining:
        count = min(1000, remaining); indices = rng.integers(0, len(values), size=(count, len(values))); parts.append(values[indices].mean(axis=1)); remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(parts), [0.025, 0.975]))


def rls_mean(history, prior_mean: float, prior_variance: float, noise_std: float) -> float:
    sum_u2 = sum(float(row["action"]) ** 2 for row in history)
    sum_uy = sum(float(row["action"]) * float(row["response"]) for row in history)
    prior_precision, obs_precision = 1.0 / prior_variance, 1.0 / noise_std ** 2
    return float((prior_precision * prior_mean + obs_precision * sum_uy) / (prior_precision + obs_precision * sum_u2))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite: {args.output}")
    raw_path, summary_path, manifest_path = args.run_dir / "raw_results.csv", args.run_dir / "summary.json", args.run_dir / "run_manifest.json"
    with raw_path.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8")); manifest = json.loads(manifest_path.read_text(encoding="utf-8")); config = summary["config"]
    n, episodes = int(config["n_sequences"]), int(config["n_episodes"])
    lookup = {(row["condition"], row["policy"], int(row["sequence_id"]), int(row["episode_id"])): row for row in rows}
    identity_failures, scenario_failures, factor_failures, pairing_failures, donor_failures, estimate_failures, count_failures = [], [], [], [], [], [], []
    for condition in ("persistent", "no_persistence"):
        for sequence_id in range(n):
            left, right = lookup[(condition, "current_only_rls", sequence_id, 1)], lookup[(condition, "persistent_rls", sequence_id, 1)]
            if any(left[key] != right[key] for key in ("scenario_hash", "action", "response", "early_task_cost", "unsafe")): identity_failures.append(f"{condition}:{sequence_id}")
            if condition == "persistent" and len({lookup[(condition, "current_only_rls", sequence_id, episode)]["factor_id"] for episode in range(1, episodes + 1)}) != 1: factor_failures.append(sequence_id)
            for episode in range(1, episodes + 1):
                hashes = {lookup[(condition, policy, sequence_id, episode)]["scenario_hash"] for policy in POLICIES}
                if len(hashes) != 1: scenario_failures.append(f"{condition}:{sequence_id}:{episode}")
    for sequence_id in range(n):
        for episode in range(1, episodes + 1):
            left = lookup[("persistent", "current_only_rls", sequence_id, episode)]
            right = lookup[("no_persistence", "current_only_rls", sequence_id, episode)]
            if left["target"] != right["target"] or left["noise"] != right["noise"]: pairing_failures.append(f"{sequence_id}:{episode}")
    prior_mean, prior_variance, noise_std = float(summary["population_prior_mean"]), float(summary["population_prior_variance"]), float(config["noise_std"])
    for row in rows:
        condition, policy, sequence_id, episode = row["condition"], row["policy"], int(row["sequence_id"]), int(row["episode_id"])
        if policy not in ("current_only_rls", "persistent_rls", "shuffled_rls", "wrong_sequence_rls"): continue
        donor = int(row["donor_sequence_id"])
        if episode >= 2 and policy in ("shuffled_rls", "wrong_sequence_rls") and donor == sequence_id: donor_failures.append(f"{condition}:{policy}:{sequence_id}:{episode}")
        source = sequence_id if policy in ("current_only_rls", "persistent_rls") else donor
        history = [] if policy == "current_only_rls" else [lookup[(condition, policy, source, previous)] for previous in range(1, episode)]
        expected = rls_mean(history, prior_mean, prior_variance, noise_std)
        if not np.isclose(expected, float(row["estimate_before"]), rtol=1e-12, atol=1e-12): estimate_failures.append(f"{condition}:{policy}:{sequence_id}:{episode}")
        if int(row["history_count_before"]) != len(history): count_failures.append(f"{condition}:{policy}:{sequence_id}:{episode}")
    buckets = defaultdict(list)
    for row in rows:
        if int(row["episode_id"]) >= 2: buckets[(row["condition"], row["policy"], int(row["sequence_id"]))].append(float(row["early_task_cost"]))
    values = {condition: {policy: np.asarray([np.mean(buckets[(condition, policy, sequence_id)]) for sequence_id in range(n)]) for policy in POLICIES} for condition in ("persistent", "no_persistence")}
    pc, pr = values["persistent"]["current_only_rls"], values["persistent"]["persistent_rls"]
    nc, nr = values["no_persistence"]["current_only_rls"], values["no_persistence"]["persistent_rls"]
    p_delta, n_delta = pc - pr, nc - nr; did = p_delta - n_delta; current_mean = float(pc.mean())
    categorical_delta = pc - values["persistent"]["categorical_history_oracle"]; true_delta = pc - values["persistent"]["true_factor_oracle"]
    shuffled_relative = float((pc - values["persistent"]["shuffled_rls"]).mean() / current_mean); wrong_relative = float((pc - values["persistent"]["wrong_sequence_rls"]).mean() / current_mean)
    relative, no_relative = float(p_delta.mean() / current_mean), float(n_delta.mean() / nc.mean()); seed, resamples = int(config["bootstrap_seed"]), int(config["bootstrap_resamples"])
    structural = {"raw_row_count": len(rows) == 2 * n * episodes * len(POLICIES), "episode1_identity": not identity_failures, "paired_scenarios": not scenario_failures, "cross_condition_nuisance_noise_pairing": not pairing_failures, "persistent_factor_lifetime": not factor_failures, "donor_isolation": not donor_failures, "rls_estimates_reconstructed_from_past_only": not estimate_failures, "history_count_budget": not count_failures, "raw_hash": sha256(raw_path) == manifest["raw_results_sha256"], "summary_hash": sha256(summary_path) == manifest["summary_sha256"]}
    criteria = {
        "persistent_rls_relative_at_least_30pct_and_ci_lower_above_20pct_current": relative >= .30 and bootstrap_ci(p_delta, seed, 100, resamples)[0] > .20 * current_mean,
        "rls_did_ci_lower_above_15pct_current": bootstrap_ci(did, seed, 900, resamples)[0] > .15 * current_mean,
        "rls_recovers_80pct_categorical_and_50pct_true_gap": p_delta.mean() / categorical_delta.mean() >= .80 and p_delta.mean() / true_delta.mean() >= .50,
        "positive_sequence_fraction_at_least_75pct": float(np.mean(p_delta > 0)) >= .75,
        "no_persistence_relative_improvement_at_most_5pct": no_relative <= .05,
        "shuffled_and_wrong_not_comparable": shuffled_relative <= .05 and wrong_relative <= .05 and shuffled_relative < .5 * relative and wrong_relative < .5 * relative,
        "categorical_oracle_relative_improvement_at_least_30pct": float(categorical_delta.mean() / current_mean) >= .30,
        "all_sequences_and_audits_valid": all(structural.values()),
    }
    criteria = {key: bool(value) for key, value in criteria.items()}; verdict = "EXPLICIT_CONTEXT_SUPPORTED" if all(criteria.values()) else "EXPLICIT_CONTEXT_NOT_ESTABLISHED"
    audit = {"schema": "persistent-context-v2-stage2-independent-audit-v1", "passed": all(structural.values()) and criteria == summary["criteria"] and verdict == summary["decision"]["verdict"], "structural_checks": structural, "criteria": criteria, "recomputed_verdict": verdict, "recomputed": {"persistent_current_mean": current_mean, "persistent_rls_mean": float(pr.mean()), "relative_improvement": relative, "difference_ci95": bootstrap_ci(p_delta, seed, 100, resamples), "did_mean": float(did.mean()), "did_ci95": bootstrap_ci(did, seed, 900, resamples), "no_persistence_relative": no_relative, "shuffled_relative": shuffled_relative, "wrong_relative": wrong_relative, "categorical_gap_recovery": float(p_delta.mean() / categorical_delta.mean()), "true_gap_recovery": float(p_delta.mean() / true_delta.mean()), "positive_sequence_fraction": float(np.mean(p_delta > 0))}, "failure_counts": {"identity": len(identity_failures), "scenario": len(scenario_failures), "cross_condition_pairing": len(pairing_failures), "factor": len(factor_failures), "donor": len(donor_failures), "estimate": len(estimate_failures), "count": len(count_failures)}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(audit, indent=2, sort_keys=True)); return 0 if audit["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
