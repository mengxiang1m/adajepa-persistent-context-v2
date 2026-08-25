"""Retrospective paired evaluation of auditable functional context gates."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl,
    dump_json,
    git_revision,
    read_jsonl,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-functional-shadow-gate-v1"
EXPECTED_DESIGN_SHA256 = "384a2ae480abe3987dfd7c227a0c6aaf01a10902e17e4aea3c34cd2aa2f80271"
EXPECTED_RAW_HASHES = {
    "deadzone": "05a64892344d3a7f5f2b8f1351716ebb2039d7fd3177cfed63d7d7a5dd7024c4",
    "delay": "02d27c5fe51d4932dd7d1f8ce67d893be0fbf55b6e3cb77255915d3eec43368d",
    "matrix": "47f6afd6b0b5d3739e126f9ef697d9cee739de82d93659d0c1ccb7a8532d5868",
}
DEADZONE_PRIOR = 0.10
DELAY_PRIOR = 2
MATRIX_PRIOR_GAIN = 0.9327804920294028
MATRIX_ROTATION_CUTOFF = 15.0
BOOTSTRAP_SEED = 1_370_300
BOOTSTRAP_RESAMPLES = 20_000


def deadzone_gate(estimate: float) -> bool:
    return float(estimate) > DEADZONE_PRIOR


def delay_gate(delay_steps: int) -> bool:
    return int(delay_steps) > DELAY_PRIOR


def matrix_parameters(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    gain = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    return gain, rotation


def matrix_gate(matrix) -> bool:
    gain, rotation = matrix_parameters(matrix)
    return gain < MATRIX_PRIOR_GAIN or abs(rotation) >= MATRIX_ROTATION_CUTOFF


def _unit(task, unit_id, factor, population, context, decisions, metadata):
    decisions = np.asarray(decisions, dtype=bool)
    population = np.asarray(population, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    if not (len(decisions) == len(population) == len(context)):
        raise ValueError("atomic decision arrays must align")
    gated = np.where(decisions, context, population)
    inverted = np.where(~decisions, context, population)
    best = np.minimum(population, context)
    return {
        "task": task,
        "unit_id": str(unit_id),
        "factor": str(factor),
        "atomic_count": len(population),
        "context_selection_fraction": float(decisions.mean()),
        "population": float(population.mean()),
        "always_context": float(context.mean()),
        "functional_gate": float(gated.mean()),
        "inverted_gate": float(inverted.mean()),
        "best_of_two_behavior_ceiling": float(best.mean()),
        "atomic_decisions": decisions.tolist(),
        "metadata": metadata,
    }


def extract_deadzone(path: Path):
    grouped = defaultdict(list)
    for row in read_jsonl(path):
        if row.get("record_type") == "evaluation_episode" and int(row["episode_index"]) > 0:
            grouped[int(row["sequence_id"])].append(row)
    units = []
    for sequence_id in sorted(grouped):
        rows = sorted(grouped[sequence_id], key=lambda row: int(row["episode_index"]))
        factor = float(rows[0]["factor"])
        estimates = [float(row["policies"]["correct_history"]["context"]) for row in rows]
        units.append(_unit(
            "deadzone", sequence_id, factor,
            [row["policies"]["current_only"]["metrics"]["pose_auc10"] for row in rows],
            [row["policies"]["correct_history"]["metrics"]["pose_auc10"] for row in rows],
            [deadzone_gate(estimate) for estimate in estimates],
            {"estimated_deadzone": estimates, "episode_indices": [int(row["episode_index"]) for row in rows]},
        ))
    return units


def extract_delay(path: Path):
    units = []
    for row in read_jsonl(path):
        if row.get("record_type") != "paired_delay":
            continue
        factor = int(row["factor_steps"])
        units.append(_unit(
            "delay", int(row["ordinal"]), factor,
            [row["prior"]["metrics"]["pose_auc10"]],
            [row["oracle"]["metrics"]["pose_auc10"]],
            [delay_gate(factor)],
            {"true_delay_steps": factor, "functional_ceiling_only": True},
        ))
    return units


def extract_matrix(path: Path):
    units = []
    for row in read_jsonl(path):
        if row.get("record_type") != "evaluation_episode" or int(row["episode_index"]) != 1:
            continue
        matrix = row["policies"]["correct_history"]["context_matrix"]
        gain, rotation = matrix_parameters(matrix)
        factor = f"theta={float(row['rotation_degrees']):+g},gain={float(row['gain']):g}"
        units.append(_unit(
            "matrix", int(row["sequence_id"]), factor,
            [row["policies"]["current_only"]["metrics"]["pose_auc10"]],
            [row["policies"]["correct_history"]["metrics"]["pose_auc10"]],
            [matrix_gate(matrix)],
            {"estimated_gain": gain, "estimated_rotation_degrees": rotation, "context_matrix": matrix},
        ))
    return units


def bootstrap_ci(values, seed_offset=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    return [float(value) for value in np.quantile(values[indexes].mean(axis=1), [0.025, 0.975])]


def effect(population, treatment, seed_offset=0):
    population = np.asarray(population, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    delta = population - treatment
    return {
        "population_mean": float(population.mean()),
        "treatment_mean": float(treatment.mean()),
        "mean_delta": float(delta.mean()),
        "relative_improvement": float(delta.mean() / population.mean()),
        "bootstrap_ci95_delta": bootstrap_ci(delta, seed_offset),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)),
        "harm_fraction": float(np.mean(treatment > population + 1e-12)),
        "unit_deltas": delta.tolist(),
    }


def summarize(units):
    policies = ("always_context", "functional_gate", "inverted_gate", "best_of_two_behavior_ceiling")
    result = {"contract_id": CONTRACT_ID, "evaluation_type": "retrospective_paired_shadow_on_frozen_raw", "tasks": {}, "macro_relative_improvement": {}}
    macro = defaultdict(list)
    for task_index, task in enumerate(("deadzone", "delay", "matrix")):
        selected = [unit for unit in units if unit["task"] == task]
        population = np.asarray([unit["population"] for unit in selected])
        task_result = {
            "n_units": len(selected),
            "context_selection_rate": float(np.mean([unit["context_selection_fraction"] for unit in selected])),
            "policies": {},
            "functional_vs_always_context": {},
            "best_of_two_opportunity_recovery": None,
            "by_factor": {},
        }
        for policy_index, policy in enumerate(policies):
            treatment = np.asarray([unit[policy] for unit in selected])
            task_result["policies"][policy] = effect(population, treatment, task_index * 100 + policy_index)
            macro[policy].append(task_result["policies"][policy]["relative_improvement"])
        always = np.asarray([unit["always_context"] for unit in selected])
        gated = np.asarray([unit["functional_gate"] for unit in selected])
        improvement = always - gated
        task_result["functional_vs_always_context"] = {
            "mean_delta_always_minus_gate": float(improvement.mean()),
            "relative_improvement_over_always": float(improvement.mean() / always.mean()),
            "bootstrap_ci95": bootstrap_ci(improvement, task_index * 100 + 50),
            "gate_better_fraction": float(np.mean(improvement > 1e-12)),
        }
        gate_delta = task_result["policies"]["functional_gate"]["mean_delta"]
        best_delta = task_result["policies"]["best_of_two_behavior_ceiling"]["mean_delta"]
        task_result["best_of_two_opportunity_recovery"] = float(gate_delta / best_delta) if abs(best_delta) > 1e-12 else math.nan
        for factor in sorted({unit["factor"] for unit in selected}):
            group = [unit for unit in selected if unit["factor"] == factor]
            task_result["by_factor"][factor] = {
                "n": len(group),
                "context_selection_rate": float(np.mean([unit["context_selection_fraction"] for unit in group])),
                "population_mean": float(np.mean([unit["population"] for unit in group])),
                "always_context_mean": float(np.mean([unit["always_context"] for unit in group])),
                "functional_gate_mean": float(np.mean([unit["functional_gate"] for unit in group])),
                "functional_gate_relative_improvement": float((np.mean([unit["population"] for unit in group]) - np.mean([unit["functional_gate"] for unit in group])) / np.mean([unit["population"] for unit in group])),
            }
        result["tasks"][task] = task_result
    result["macro_relative_improvement"] = {policy: float(np.mean(values)) for policy, values in macro.items()}
    result["structural_checks"] = {
        "unit_counts": {task: sum(unit["task"] == task for unit in units) for task in ("deadzone", "delay", "matrix")},
        "all_finite": bool(all(np.isfinite(unit[policy]) for unit in units for policy in ("population",) + policies)),
        "binary_decisions": bool(all(all(isinstance(value, bool) for value in unit["atomic_decisions"]) for unit in units)),
    }
    result["valid"] = result["structural_checks"]["unit_counts"] == {"deadzone": 32, "delay": 32, "matrix": 32} and result["structural_checks"]["all_finite"] and result["structural_checks"]["binary_decisions"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_functional_shadow_gate_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_functional_shadow_gate_contract_zh.md"))
    parser.add_argument("--deadzone-raw", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_deadzone_stage1/persistent_raw.jsonl"))
    parser.add_argument("--delay-raw", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_delay_stage0/raw.jsonl"))
    parser.add_argument("--matrix-raw", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_matrix_stage1/persistent_raw.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_functional_shadow_gate"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sha256(args.design) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen shadow-gate design hash mismatch")
    paths = {"deadzone": args.deadzone_raw, "delay": args.delay_raw, "matrix": args.matrix_raw}
    hashes = {task: sha256(path) for task, path in paths.items()}
    if hashes != EXPECTED_RAW_HASHES:
        raise RuntimeError(f"frozen raw hash mismatch: {hashes}")
    started = time.time()
    units = extract_deadzone(args.deadzone_raw) + extract_delay(args.delay_raw) + extract_matrix(args.matrix_raw)
    unit_path = args.output_dir / "shadow_units.jsonl"
    if unit_path.exists():
        unit_path.unlink()
    for unit in units:
        append_jsonl(unit_path, unit)
    result = summarize(units)
    dump_json(args.output_dir / "runner_summary.json", result)
    manifest = {
        "contract_id": CONTRACT_ID,
        "git_revision": git_revision(),
        "design_path": str(args.design),
        "design_sha256": sha256(args.design),
        "contract_path": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "raw_paths": {task: str(path) for task, path in paths.items()},
        "raw_sha256": hashes,
        "shadow_units_sha256": sha256(unit_path),
        "started_unix": started,
        "finished_unix": time.time(),
        "command": " ".join(__import__("sys").argv),
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
