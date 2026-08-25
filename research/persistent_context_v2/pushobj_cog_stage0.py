"""Ground-truth simulator oracle for a hidden horizontal PushObj center of gravity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    WINDOW,
    deadline_success,
    nominal_block_displacement_at_10,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl,
    dump_json,
    git_revision,
    make_env,
    pad_state,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-pushobj-horizontal-cog-simulator-oracle-stage0-v1"
EXPECTED_DESIGN_SHA256 = "619eccbcb45d9e4f5e9a5d525692ffc7524b6dd34a53297cf8a27b7c20b36edd"
FACTORS = (-22.5, -7.5, 7.5, 22.5)
POPULATION_PRIOR_COG_X = 0.0
COG_Y = 45.0
SEGMENT_INDICES = (
    442, 307, 436, 216, 421, 446, 194, 458,
    137, 200, 464, 24, 329, 10, 113, 210,
    373, 363, 337, 361, 491, 140, 267, 277,
    291, 483, 69, 448, 439, 195, 226, 344,
)
ACTION_COUNT = 10
NUM_SAMPLES = 128
TOPK = 16
OPT_STEPS = 5
INITIAL_SIGMA = 0.2
BOOTSTRAP_SEED = 1_070_301
BOOTSTRAP_RESAMPLES = 20_000


def array_sha256(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _setup_physics(env, shape, initial_state, env_seed, cog_x):
    env.seed(int(env_seed))
    env.shape = shape
    env._setup()
    env.block.center_of_gravity = (float(cog_x), COG_Y)
    env._set_state(np.asarray(initial_state, dtype=np.float32))


def rollout_physics(env, shape, initial_state, env_seed, commands, cog_x):
    """Exact PushTEnv.step dynamics without rendering or reward geometry."""
    from pymunk.vec2d import Vec2d

    _setup_physics(env, shape, initial_state, env_seed, cog_x)
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


def standard_rollout(env, shape, initial_state, env_seed, commands, cog_x):
    env.shape = shape
    env.block_cog = (float(cog_x), COG_Y)
    _, state = env.prepare(int(env_seed), initial_state)
    states = [np.asarray(state, dtype=np.float32)]
    for command in np.asarray(commands, dtype=np.float32):
        _, _, _, info = env.step(command)
        states.append(np.asarray(info["state"], dtype=np.float32))
    return np.stack(states)


def pose_auc10(states, goal_state):
    return float(pose_metrics(np.asarray(states), np.asarray(goal_state), WINDOW)["pose_auc10"])


def plan_physics_cem(env, shape, initial_state, env_seed, nominal_commands, goal_state, context_cog_x, cem_seed):
    generator = np.random.default_rng(int(cem_seed))
    mu = np.asarray(nominal_commands, dtype=np.float64).copy()
    sigma = np.full_like(mu, INITIAL_SIGMA)
    trace = []
    for iteration in range(OPT_STEPS):
        candidates = generator.normal(mu, sigma, size=(NUM_SAMPLES, ACTION_COUNT, 2))
        candidates[0] = mu
        losses = np.empty(NUM_SAMPLES, dtype=np.float64)
        for index, commands in enumerate(candidates):
            predicted = rollout_physics(env, shape, initial_state, env_seed, commands, context_cog_x)
            losses[index] = pose_auc10(predicted, goal_state)
        elite_index = np.argsort(losses)[:TOPK]
        elite = candidates[elite_index]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0, ddof=1), 1e-4)
        trace.append({"iteration": iteration, "best_loss": float(losses[elite_index[0]]), "mean_loss": float(losses.mean()), "mu_sha256": array_sha256(mu), "sigma_sha256": array_sha256(sigma), "candidate_count": NUM_SAMPLES, "topk": TOPK})
    commands = mu.astype(np.float32)
    predicted_states = rollout_physics(env, shape, initial_state, env_seed, commands, context_cog_x)
    return commands, predicted_states, {"context_cog_x": float(context_cog_x), "trace": trace}


def scenarios(segments):
    rows = []
    for ordinal, segment_index in enumerate(SEGMENT_INDICES):
        displacement = nominal_block_displacement_at_10(segments[segment_index])
        if displacement < 10:
            raise RuntimeError("invalid early-waypoint segment")
        rows.append({"ordinal": ordinal, "factor_cog_x": FACTORS[ordinal // 8], "within_factor": ordinal % 8, "segment_index": int(segment_index), "nominal_block_displacement_at_10": displacement, "env_seed": 1_080_000 + ordinal, "cem_seed": 1_090_000 + ordinal})
    return rows


def prepare_waypoint_physics(env, segment, env_seed):
    shape = segment.get("shape", "T")
    initial_state = pad_state(np.asarray(segment["states"])[0])
    nominal_commands = np.asarray(segment["actions"], dtype=np.float32)[:ACTION_COUNT] / 100.0
    if len(nominal_commands) != ACTION_COUNT:
        raise ValueError("segment has insufficient actions")
    nominal_states = rollout_physics(env, shape, initial_state, env_seed, nominal_commands, POPULATION_PRIOR_COG_X)
    return shape, initial_state, nominal_commands, nominal_states


def identity_audit(env, segment, env_seed):
    shape, initial_state, commands, _ = prepare_waypoint_physics(env, segment, env_seed)
    manual = rollout_physics(env, shape, initial_state, env_seed, commands, POPULATION_PRIOR_COG_X)
    standard = standard_rollout(env, shape, initial_state, env_seed, commands, POPULATION_PRIOR_COG_X)
    visuals, proprios, states = [], [], []
    for factor in FACTORS:
        env.shape = shape
        env.block_cog = (float(factor), COG_Y)
        observation, state = env.prepare(int(env_seed), initial_state)
        visuals.append(np.asarray(observation["visual"]))
        proprios.append(np.asarray(observation["proprio"]))
        states.append(np.asarray(state))
    return {"manual_standard_state_max_abs": float(np.max(np.abs(manual - standard))), "initial_visual_max_abs": float(max(np.max(np.abs(value.astype(np.int16) - visuals[0].astype(np.int16))) for value in visuals)), "initial_proprio_max_abs": float(max(np.max(np.abs(value - proprios[0])) for value in proprios)), "initial_state_max_abs": float(max(np.max(np.abs(value - states[0])) for value in states))}


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(raw_path, identity):
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "paired_cog"]
    prior = np.asarray([row["prior"]["metrics"]["pose_auc10"] for row in rows])
    oracle = np.asarray([row["oracle"]["metrics"]["pose_auc10"] for row in rows])
    delta = prior - oracle
    result = {"contract_id": CONTRACT_ID, "n_pairs": len(rows), "primary_metric": "pose_auc10_to_waypoint", "prior_mean": float(prior.mean()), "oracle_mean": float(oracle.mean()), "mean_delta": float(delta.mean()), "relative_improvement": float(delta.mean() / prior.mean()), "bootstrap_ci95_delta": bootstrap_ci(delta), "positive_fraction": float(np.mean(delta > 1e-12)), "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)), "negative_fraction": float(np.mean(delta < -1e-12)), "prior_deadline_success": float(np.mean([row["prior"]["deadline_success"] for row in rows])), "oracle_deadline_success": float(np.mean([row["oracle"]["deadline_success"] for row in rows])), "plan_changed_fraction": float(np.mean([row["prior"]["command_sha256"] != row["oracle"]["command_sha256"] for row in rows])), "oracle_prediction_execution_max_abs": float(max(row["oracle"]["prediction_execution_max_abs"] for row in rows)) if rows else math.inf, "identity_audit": identity, "by_factor": {}}
    for factor in FACTORS:
        mask = np.asarray([float(row["factor_cog_x"]) == factor for row in rows])
        d = delta[mask]
        result["by_factor"][str(factor)] = {"n": int(mask.sum()), "prior_mean": float(prior[mask].mean()), "oracle_mean": float(oracle[mask].mean()), "mean_delta": float(d.mean()), "relative_improvement": float(d.mean() / prior[mask].mean()), "positive_fraction": float(np.mean(d > 1e-12))}
    checks = {"complete": len(rows) == 32, "unique_segments": len({row["segment_index"] for row in rows}) == 32, "factor_balance": all(sum(float(row["factor_cog_x"]) == factor for row in rows) == 8 for factor in FACTORS), "waypoint_displacement": all(row["nominal_block_displacement_at_10"] >= 10 for row in rows), "identity": max(identity.values(), default=math.inf) <= 1e-6, "oracle_prediction_execution": result["oracle_prediction_execution_max_abs"] <= 1e-6, "plan_changed": result["plan_changed_fraction"] > 0}
    result["structural_checks"] = checks
    result["valid"] = all(checks.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_stage0_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_stage0_contract_zh.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_stage0"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    selection = scenarios(segments)
    if args.mode == "inspect":
        payload = {"contract_id": CONTRACT_ID, "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract), "scenarios": selection}
        dump_json(args.output_dir / "selection_audit.json", payload)
        print(json.dumps(payload, indent=2))
        return
    manifest_path = args.output_dir / "manifest.json"
    if args.mode == "summarize":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = summarize(args.output_dir / "raw.jsonl", manifest["identity_audit"])
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
        return
    if sha256(args.design) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen design hash mismatch")
    seed_all(1_070_401)
    env = make_env()
    device = torch.device("cpu")
    if not manifest_path.exists():
        manifest = {"contract_id": CONTRACT_ID, "git_revision": git_revision(), "design_path": str(args.design), "design_sha256": sha256(args.design), "contract_path": str(args.contract), "contract_sha256": sha256(args.contract), "data": str(args.data), "data_sha256": sha256(args.data), "command": " ".join(__import__("sys").argv), "started_unix": time.time(), "resource_start": resource_snapshot(device)}
        dump_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "identity_audit" not in manifest:
        first = selection[0]
        manifest["identity_audit"] = identity_audit(env, segments[first["segment_index"]], first["env_seed"])
        dump_json(manifest_path, manifest)
    raw_path = args.output_dir / "raw.jsonl"
    completed = {int(row["ordinal"]) for row in read_jsonl(raw_path) if row.get("record_type") == "paired_cog"}
    for meta in selection[: args.limit]:
        if meta["ordinal"] in completed:
            continue
        started = time.perf_counter()
        segment = segments[meta["segment_index"]]
        shape, initial_state, nominal_commands, nominal_states = prepare_waypoint_physics(env, segment, meta["env_seed"])
        goal_state = nominal_states[-1]
        policies = {}
        for name, context_cog_x in (("prior", POPULATION_PRIOR_COG_X), ("oracle", meta["factor_cog_x"])):
            commands, predicted_states, planner = plan_physics_cem(env, shape, initial_state, meta["env_seed"], nominal_commands, goal_state, context_cog_x, meta["cem_seed"])
            execution_states = rollout_physics(env, shape, initial_state, meta["env_seed"], commands, meta["factor_cog_x"])
            policies[name] = {"context_cog_x": float(context_cog_x), "commands": commands, "predicted_states": predicted_states, "states": execution_states, "metrics": pose_metrics(execution_states, goal_state, WINDOW), "deadline_success": deadline_success(execution_states, goal_state), "command_sha256": array_sha256(commands), "predicted_state_sha256": array_sha256(predicted_states), "state_sha256": array_sha256(execution_states), "prediction_execution_max_abs": float(np.max(np.abs(predicted_states - execution_states))), "planner": planner}
        row = {"record_type": "paired_cog", "contract_id": CONTRACT_ID, **meta, "shape": shape, "initial_state": initial_state, "goal_state": goal_state, "nominal_commands": nominal_commands, "nominal_states": nominal_states, "prior": policies["prior"], "oracle": policies["oracle"], "elapsed_s": time.perf_counter() - started, "resource": resource_snapshot(device)}
        append_jsonl(raw_path, row)
        print(f"DONE {meta['ordinal'] + 1}/32 cog_x={meta['factor_cog_x']:+g} prior={policies['prior']['metrics']['pose_auc10']:.4f} oracle={policies['oracle']['metrics']['pose_auc10']:.4f} elapsed={row['elapsed_s']:.2f}s", flush=True)
    if args.limit == 32:
        result = summarize(raw_path, manifest["identity_audit"])
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(device)
    dump_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
