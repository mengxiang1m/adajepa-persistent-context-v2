#!/usr/bin/env python3
"""Independent raw audit for horizontal-CoG simulator oracle Stage 0."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.persistent_context_v2.pushobj_cog_stage0 import (
    COG_Y,
    CONTRACT_ID,
    EXPECTED_DESIGN_SHA256,
    FACTORS,
    OPT_STEPS,
    POPULATION_PRIOR_COG_X,
    SEGMENT_INDICES,
    array_sha256,
    read_jsonl,
    sha256,
    summarize,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import deadline_success
from research.persistent_context_v2.pushobj_rotation_stage0 import make_env, pose_metrics


def independent_rollout(env, shape, initial_state, env_seed, commands, cog_x):
    from pymunk.vec2d import Vec2d

    env.seed(int(env_seed))
    env.shape = shape
    env._setup()
    env.block.center_of_gravity = (float(cog_x), COG_Y)
    env._set_state(np.asarray(initial_state, dtype=np.float32))
    states = [env._get_obs().copy()]
    for command in np.asarray(commands):
        action = np.array(command) * env.action_scale
        target = env.agent.position + action
        dt = 1.0 / env.sim_hz
        for _ in range(env.sim_hz // env.control_hz):
            acceleration = env.k_p * (target - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
            env.agent.velocity += acceleration * dt
            env.space.step(dt)
        states.append(env._get_obs().copy())
    return np.asarray(states, dtype=np.float32)


def close(left, right):
    return bool(np.isclose(float(left), float(right), rtol=1e-8, atol=1e-9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_stage0"))
    parser.add_argument("--output-name", default="independent_audit.json")
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    runner = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    rows = [row for row in read_jsonl(args.output_dir / "raw.jsonl") if row.get("record_type") == "paired_cog"]
    failures = {key: [] for key in ("manifest", "count", "scenario", "identity", "physics", "metric", "hash", "pairing", "planner")}
    if manifest.get("contract_id") != CONTRACT_ID or manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("contract_or_design")
    design_path = Path(manifest.get("design_path", ""))
    if not design_path.exists() or sha256(design_path) != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("design_file")
    if max(manifest.get("identity_audit", {}).values(), default=np.inf) > 1e-6:
        failures["identity"].append("manifest_identity")
    if len(rows) != 32 or len({int(row["ordinal"]) for row in rows}) != 32:
        failures["count"].append("rows")
    env = make_env()
    factor_counts = {factor: 0 for factor in FACTORS}
    for row in rows:
        ordinal = int(row["ordinal"])
        factor = float(row["factor_cog_x"])
        factor_counts[factor] = factor_counts.get(factor, 0) + 1
        if ordinal >= len(SEGMENT_INDICES) or int(row["segment_index"]) != SEGMENT_INDICES[ordinal] or float(row["nominal_block_displacement_at_10"]) < 10:
            failures["scenario"].append(str(ordinal))
        nominal_commands = np.asarray(row["nominal_commands"], dtype=np.float32)
        nominal = independent_rollout(env, row["shape"], row["initial_state"], row["env_seed"], nominal_commands, POPULATION_PRIOR_COG_X)
        if not np.array_equal(nominal, np.asarray(row["nominal_states"], dtype=np.float32)) or not np.array_equal(nominal[-1], np.asarray(row["goal_state"], dtype=np.float32)):
            failures["physics"].append(f"{ordinal}:nominal")
        for policy in ("prior", "oracle"):
            payload = row[policy]
            commands = np.asarray(payload["commands"], dtype=np.float32)
            predicted = independent_rollout(env, row["shape"], row["initial_state"], row["env_seed"], commands, payload["context_cog_x"])
            execution = independent_rollout(env, row["shape"], row["initial_state"], row["env_seed"], commands, factor)
            if not np.array_equal(predicted, np.asarray(payload["predicted_states"], dtype=np.float32)) or not np.array_equal(execution, np.asarray(payload["states"], dtype=np.float32)):
                failures["physics"].append(f"{ordinal}:{policy}")
            metrics = pose_metrics(execution, np.asarray(row["goal_state"]), 10)
            if any(not close(metrics[key], payload["metrics"][key]) for key in metrics) or bool(payload["deadline_success"]) != deadline_success(execution, np.asarray(row["goal_state"])):
                failures["metric"].append(f"{ordinal}:{policy}")
            if payload["command_sha256"] != array_sha256(commands) or payload["predicted_state_sha256"] != array_sha256(predicted) or payload["state_sha256"] != array_sha256(execution):
                failures["hash"].append(f"{ordinal}:{policy}")
            trace = payload["planner"]["trace"]
            if len(trace) != OPT_STEPS or any(item["iteration"] != index or item["candidate_count"] != 128 or item["topk"] != 16 for index, item in enumerate(trace)):
                failures["planner"].append(f"{ordinal}:{policy}")
        if float(row["prior"]["context_cog_x"]) != POPULATION_PRIOR_COG_X or float(row["oracle"]["context_cog_x"]) != factor:
            failures["pairing"].append(f"{ordinal}:context")
        if float(row["oracle"]["prediction_execution_max_abs"]) > 1e-6:
            failures["pairing"].append(f"{ordinal}:oracle_prediction")
    if any(factor_counts.get(factor) != 8 for factor in FACTORS):
        failures["scenario"].append("factor_balance")
    recomputed = summarize(args.output_dir / "raw.jsonl", manifest["identity_audit"])
    checks = {"raw_recomputes_runner_summary_exactly": recomputed == runner, "all_failure_counts_zero": all(not values for values in failures.values()), "runner_valid": bool(recomputed.get("valid"))}
    result = {"schema": "persistent-context-v2-pushobj-cog-stage0-audit-v1", "passed": all(checks.values()), "checks": checks, "failure_counts": {key: len(value) for key, value in failures.items()}, "failure_examples": {key: value[:20] for key, value in failures.items() if value}, "recomputed_summary": recomputed}
    (args.output_dir / args.output_name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
