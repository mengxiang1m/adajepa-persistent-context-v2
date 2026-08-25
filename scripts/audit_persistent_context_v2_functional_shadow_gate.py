"""Independent raw-to-decision audit for functional shadow gate v1."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.persistent_context_v2.pushobj_rotation_stage0 import dump_json, read_jsonl, sha256


CONTRACT_ID = "persistent-context-v2-functional-shadow-gate-v1"
DESIGN_SHA = "384a2ae480abe3987dfd7c227a0c6aaf01a10902e17e4aea3c34cd2aa2f80271"
RAW_HASHES = {
    "deadzone": "05a64892344d3a7f5f2b8f1351716ebb2039d7fd3177cfed63d7d7a5dd7024c4",
    "delay": "02d27c5fe51d4932dd7d1f8ce67d893be0fbf55b6e3cb77255915d3eec43368d",
    "matrix": "47f6afd6b0b5d3739e126f9ef697d9cee739de82d93659d0c1ccb7a8532d5868",
}
BOOTSTRAP_SEED = 1_370_300


def close(a, b, atol=1e-10):
    return abs(float(a) - float(b)) <= atol


def ci(values, offset):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    indexes = rng.integers(0, len(values), size=(20_000, len(values)))
    return np.quantile(values[indexes].mean(axis=1), [0.025, 0.975])


def matrix_decision(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    gain = math.hypot(matrix[0, 0], matrix[1, 0])
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    return gain < 0.9327804920294028 or abs(rotation) >= 15.0


def expected_deadzone(path):
    grouped = defaultdict(list)
    for row in read_jsonl(path):
        if row.get("record_type") == "evaluation_episode" and int(row["episode_index"]) > 0:
            grouped[int(row["sequence_id"])].append(row)
    expected = {}
    for sequence_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["episode_index"]))
        population = np.asarray([row["policies"]["current_only"]["metrics"]["pose_auc10"] for row in rows])
        context = np.asarray([row["policies"]["correct_history"]["metrics"]["pose_auc10"] for row in rows])
        decisions = np.asarray([float(row["policies"]["correct_history"]["context"]) > 0.1 for row in rows])
        expected[("deadzone", str(sequence_id))] = (population, context, decisions)
    return expected


def expected_delay(path):
    expected = {}
    for row in read_jsonl(path):
        if row.get("record_type") == "paired_delay":
            factor = int(row["factor_steps"])
            expected[("delay", str(row["ordinal"]))] = (
                np.asarray([row["prior"]["metrics"]["pose_auc10"]]),
                np.asarray([row["oracle"]["metrics"]["pose_auc10"]]),
                np.asarray([factor > 2]),
            )
    return expected


def expected_matrix(path):
    expected = {}
    for row in read_jsonl(path):
        if row.get("record_type") == "evaluation_episode" and int(row["episode_index"]) == 1:
            policy = row["policies"]["correct_history"]
            expected[("matrix", str(row["sequence_id"]))] = (
                np.asarray([row["policies"]["current_only"]["metrics"]["pose_auc10"]]),
                np.asarray([policy["metrics"]["pose_auc10"]]),
                np.asarray([matrix_decision(policy["context_matrix"])]),
            )
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_functional_shadow_gate_design.json"))
    parser.add_argument("--deadzone-raw", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_deadzone_stage1/persistent_raw.jsonl"))
    parser.add_argument("--delay-raw", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_delay_stage0/raw.jsonl"))
    parser.add_argument("--matrix-raw", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_matrix_stage1/persistent_raw.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_functional_shadow_gate"))
    args = parser.parse_args()
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    paths = {"deadzone": args.deadzone_raw, "delay": args.delay_raw, "matrix": args.matrix_raw}
    check(sha256(args.design) == DESIGN_SHA, "design hash")
    for task, path in paths.items():
        check(sha256(path) == RAW_HASHES[task], f"{task} raw hash")
    expected = {}
    expected.update(expected_deadzone(args.deadzone_raw))
    expected.update(expected_delay(args.delay_raw))
    expected.update(expected_matrix(args.matrix_raw))
    units = read_jsonl(args.output_dir / "shadow_units.jsonl")
    check(len(units) == 96 and len(expected) == 96, "unit count")
    max_value_error = 0.0
    decision_mismatch = 0
    for unit in units:
        key = (unit["task"], unit["unit_id"])
        if key not in expected:
            failures.append(f"unknown unit {key}")
            continue
        population, context, decisions = expected[key]
        gated = np.where(decisions, context, population)
        inverted = np.where(~decisions, context, population)
        best = np.minimum(population, context)
        values = {
            "population": population.mean(), "always_context": context.mean(), "functional_gate": gated.mean(),
            "inverted_gate": inverted.mean(), "best_of_two_behavior_ceiling": best.mean(),
        }
        max_value_error = max(max_value_error, max(abs(float(unit[name]) - float(value)) for name, value in values.items()))
        decision_mismatch += int(list(decisions.astype(bool)) != unit["atomic_decisions"])
    check(max_value_error <= 1e-12, "unit value replay")
    check(decision_mismatch == 0, "decision replay")
    runner = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    policies = ("always_context", "functional_gate", "inverted_gate", "best_of_two_behavior_ceiling")
    summary_max_error = 0.0
    macro = defaultdict(list)
    for task_index, task in enumerate(("deadzone", "delay", "matrix")):
        selected = [unit for unit in units if unit["task"] == task]
        population = np.asarray([unit["population"] for unit in selected])
        check(len(selected) == 32, f"{task} count")
        check(close(np.mean([unit["context_selection_fraction"] for unit in selected]), runner["tasks"][task]["context_selection_rate"]), f"{task} selection rate")
        for policy_index, policy in enumerate(policies):
            treatment = np.asarray([unit[policy] for unit in selected])
            delta = population - treatment
            observed = runner["tasks"][task]["policies"][policy]
            expected_values = {
                "population_mean": population.mean(), "treatment_mean": treatment.mean(), "mean_delta": delta.mean(),
                "relative_improvement": delta.mean() / population.mean(), "positive_fraction": np.mean(delta > 1e-12),
                "tie_fraction": np.mean(np.abs(delta) <= 1e-12), "negative_fraction": np.mean(delta < -1e-12),
                "harm_fraction": np.mean(treatment > population + 1e-12),
            }
            summary_max_error = max(summary_max_error, max(abs(float(observed[name]) - float(value)) for name, value in expected_values.items()))
            expected_ci = ci(delta, task_index * 100 + policy_index)
            summary_max_error = max(summary_max_error, float(np.max(np.abs(expected_ci - np.asarray(observed["bootstrap_ci95_delta"])))))
            macro[policy].append(float(expected_values["relative_improvement"]))
    for policy in policies:
        summary_max_error = max(summary_max_error, abs(float(np.mean(macro[policy])) - float(runner["macro_relative_improvement"][policy])))
    check(summary_max_error <= 1e-12, "summary replay")
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["shadow_units_sha256"] == sha256(args.output_dir / "shadow_units.jsonl"), "unit hash")
    audit = {
        "contract_id": CONTRACT_ID,
        "valid": not failures,
        "failures": failures,
        "unit_count": len(units),
        "decision_mismatch_count": decision_mismatch,
        "unit_value_replay_max_abs": max_value_error,
        "summary_replay_max_abs": summary_max_error,
        "design_sha256": sha256(args.design),
        "raw_sha256": {task: sha256(path) for task, path in paths.items()},
        "shadow_units_sha256": sha256(args.output_dir / "shadow_units.jsonl"),
    }
    dump_json(args.output_dir / "independent_audit.json", audit)
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["valid"] else 1)


if __name__ == "__main__":
    main()
