#!/usr/bin/env python3
"""Independent raw-artifact audit for persistent-context V2 Stage 0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    n = int(summary["config"]["n_sequences"])
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(float(row["candidate_tolerance"]), int(row["sequence_id"]))][row["policy"]] = row
    identity_failures = []
    recomputed = []
    for tolerance in (0.20, 0.15, 0.10):
        prior, oracle, prior_unsafe = [], [], []
        for sequence_id in range(n):
            pair = grouped[(tolerance, sequence_id)]
            if set(pair) != {"population_prior", "true_factor_oracle"}:
                identity_failures.append(f"{tolerance}:{sequence_id}:missing_pair")
                continue
            left, right = pair["population_prior"], pair["true_factor_oracle"]
            if left["scenario_hash"] != right["scenario_hash"]:
                identity_failures.append(f"{tolerance}:{sequence_id}:scenario_mismatch")
            prior.append(float(left["early_task_cost"]))
            oracle.append(float(right["early_task_cost"]))
            prior_unsafe.append(int(left["unsafe"]))
        prior_array = np.asarray(prior)
        oracle_array = np.asarray(oracle)
        recomputed.append({
            "tolerance": tolerance,
            "population_prior_mean_cost": float(prior_array.mean()),
            "true_factor_oracle_mean_cost": float(oracle_array.mean()),
            "population_prior_unsafe_fraction": float(np.mean(prior_unsafe)),
            "relative_cost_improvement": float((prior_array.mean() - oracle_array.mean()) / prior_array.mean()),
        })
    reported = summary["candidates"]
    numeric_checks = []
    for observed, expected in zip(recomputed, reported):
        numeric_checks.append(all(np.isclose(observed[key], expected[key]) for key in (
            "tolerance", "population_prior_mean_cost", "true_factor_oracle_mean_cost",
            "population_prior_unsafe_fraction", "relative_cost_improvement",
        )))
    selected = next((item["tolerance"] for item in reported if item["qualified"]), None)
    checks = {
        "row_count": len(rows) == 3 * 2 * n,
        "paired_identity": not identity_failures,
        "numeric_recompute": all(numeric_checks),
        "raw_hash": sha256(raw_path) == manifest["raw_results_sha256"],
        "summary_hash": sha256(summary_path) == manifest["summary_sha256"],
        "formal_outcomes_not_generated": summary["formal_outcomes_generated"] is False,
        "mechanical_selection": selected == summary["decision"]["selected_tolerance"],
    }
    audit = {
        "schema": "persistent-context-v2-stage0-independent-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "identity_failures": identity_failures,
        "recomputed": recomputed,
        "reported_decision": summary["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
