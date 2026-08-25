#!/usr/bin/env python3
"""Independent raw-artifact audit for the discrete-delay Stage 0."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.persistent_context_v2.pushobj_delay_stage0 import (
    CONTRACT_ID,
    EXPECTED_DESIGN_SHA256,
    FACTORS,
    SEGMENT_INDICES,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def read_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def delayed(commands, delay):
    commands = np.asarray(commands, dtype=np.float32)
    effective = np.zeros_like(commands)
    if int(delay) == 0:
        return commands
    if int(delay) < len(commands):
        effective[int(delay):] = commands[:-int(delay)]
    return effective


def pose_metrics(states, goal, window=10):
    states = np.asarray(states, dtype=np.float64)[1 : window + 1]
    goal = np.asarray(goal, dtype=np.float64)
    position = np.linalg.norm(states[:, 2:4] - goal[None, 2:4], axis=1)
    angle = np.abs(states[:, 4] - goal[4]) % (2 * np.pi)
    angle = np.minimum(angle, 2 * np.pi - angle)
    pose = position / 20.0 + angle / (np.pi / 9.0)
    return {
        "pose_auc10": float(np.mean(pose)),
        "position_auc10": float(np.mean(position)),
        "angle_auc10": float(np.mean(angle)),
        "position_end10": float(position[-1]),
        "angle_end10": float(angle[-1]),
    }


def deadline_success(states, goal):
    states = np.asarray(states)
    goal = np.asarray(goal)
    position = float(np.linalg.norm(states[-1, 2:4] - goal[2:4]))
    angle = abs(float(states[-1, 4] - goal[4])) % (2 * np.pi)
    angle = min(angle, 2 * np.pi - angle)
    return bool(position < 20.0 and angle < np.pi / 9.0)


def bootstrap(values, seed=960101, count=20000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(count, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def recompute_summary(rows, identity):
    prior = np.asarray([row["prior"]["metrics"]["pose_auc10"] for row in rows])
    oracle = np.asarray([row["oracle"]["metrics"]["pose_auc10"] for row in rows])
    delta = prior - oracle
    summary = {
        "contract_id": CONTRACT_ID,
        "n_pairs": len(rows),
        "primary_metric": "pose_auc10_to_waypoint",
        "prior_mean": float(prior.mean()),
        "oracle_mean": float(oracle.mean()),
        "mean_delta": float(delta.mean()),
        "relative_improvement": float(delta.mean() / prior.mean()),
        "bootstrap_ci95_delta": bootstrap(delta),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)),
        "prior_deadline_success": float(np.mean([row["prior"]["deadline_success"] for row in rows])),
        "oracle_deadline_success": float(np.mean([row["oracle"]["deadline_success"] for row in rows])),
        "plan_changed_fraction": float(np.mean([row["prior"]["command_sha256"] != row["oracle"]["command_sha256"] for row in rows])),
        "identity_audit": identity,
        "by_factor": {},
    }
    for factor in FACTORS:
        mask = np.asarray([int(row["factor_steps"]) == factor for row in rows])
        factor_delta = delta[mask]
        matching = [row for row in rows if int(row["factor_steps"]) == factor]
        summary["by_factor"][str(factor)] = {
            "n": int(mask.sum()),
            "prior_mean": float(prior[mask].mean()),
            "oracle_mean": float(oracle[mask].mean()),
            "mean_delta": float(factor_delta.mean()),
            "relative_improvement": float(factor_delta.mean() / prior[mask].mean()),
            "positive_fraction": float(np.mean(factor_delta > 1e-12)),
            "prior_deadline_success": float(np.mean([row["prior"]["deadline_success"] for row in matching])),
            "oracle_deadline_success": float(np.mean([row["oracle"]["deadline_success"] for row in matching])),
        }
    checks = {
        "complete": len(rows) == 32,
        "unique_segments": len({row["segment_index"] for row in rows}) == 32,
        "waypoint_displacement": all(row["nominal_block_displacement_at_10"] >= 10 for row in rows),
        "identity": max(identity.values(), default=np.inf) <= 1e-6,
        "plan_changed": summary["plan_changed_fraction"] > 0,
    }
    summary["structural_checks"] = checks
    summary["valid"] = all(checks.values())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_delay_stage0"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    runner = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    rows = [row for row in read_rows(args.output_dir / "raw.jsonl") if row.get("record_type") == "paired_delay"]
    failures = {key: [] for key in ("manifest", "count", "scenario", "metric", "hash", "fifo", "pairing")}
    if manifest.get("contract_id") != CONTRACT_ID or manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("contract_or_design")
    design_path = Path(manifest.get("design_path", ""))
    if not design_path.exists() or sha256(design_path) != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("design_file")
    if max(manifest.get("identity_audit", {}).values(), default=np.inf) > 1e-6:
        failures["manifest"].append("identity")
    if len(rows) != 32 or len({row["ordinal"] for row in rows}) != 32:
        failures["count"].append("rows")
    factor_counts = {factor: 0 for factor in FACTORS}
    for row in rows:
        ordinal = int(row["ordinal"])
        factor = int(row["factor_steps"])
        factor_counts[factor] = factor_counts.get(factor, 0) + 1
        if ordinal >= len(SEGMENT_INDICES) or row["segment_index"] != SEGMENT_INDICES[ordinal] or row["nominal_block_displacement_at_10"] < 10:
            failures["scenario"].append(str(ordinal))
        for policy in ("prior", "oracle"):
            payload = row[policy]
            metrics = pose_metrics(payload["states"], row["goal_state"])
            if any(not np.isclose(metrics[key], payload["metrics"][key], rtol=1e-8, atol=1e-9) for key in metrics):
                failures["metric"].append(f"{ordinal}:{policy}:pose")
            if bool(payload["deadline_success"]) != deadline_success(payload["states"], row["goal_state"]):
                failures["metric"].append(f"{ordinal}:{policy}:success")
            commands = np.asarray(payload["commands"], dtype=np.float32)
            effective = np.asarray(payload["effective_actions"], dtype=np.float32)
            expected = delayed(commands, factor)
            if not np.array_equal(effective, expected):
                failures["fifo"].append(f"{ordinal}:{policy}")
            if payload["command_sha256"] != array_sha256(commands) or payload["effective_action_sha256"] != array_sha256(effective) or payload["state_sha256"] != array_sha256(np.asarray(payload["states"], dtype=np.float32)):
                failures["hash"].append(f"{ordinal}:{policy}")
        if row["prior"]["context_delay_steps"] != 2 or row["oracle"]["context_delay_steps"] != factor:
            failures["pairing"].append(f"{ordinal}:context")
        if len(row["prior"]["planner"]["trace"]) != len(row["oracle"]["planner"]["trace"]):
            failures["pairing"].append(f"{ordinal}:budget")
    if any(factor_counts.get(factor) != 8 for factor in FACTORS):
        failures["scenario"].append("factor_balance")
    recomputed = recompute_summary(rows, manifest["identity_audit"])
    checks = {
        "raw_recomputes_runner_summary_exactly": recomputed == runner,
        "all_failure_counts_zero": all(not values for values in failures.values()),
        "runner_valid": bool(recomputed.get("valid")),
    }
    result = {
        "schema": "persistent-context-v2-pushobj-delay-stage0-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failure_counts": {key: len(value) for key, value in failures.items()},
        "failure_examples": {key: value[:20] for key, value in failures.items() if value},
        "recomputed_summary": recomputed,
    }
    (args.output_dir / "independent_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
