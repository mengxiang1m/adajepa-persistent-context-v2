"""PushObj rotation oracle experiment with a genuine 10-action waypoint deadline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from einops import repeat

from research.persistent_context_v2.pushobj_rotation_stage0 import (
    ACTION_DIM,
    FRAMESKIP,
    POPULATION_PRIOR_DEG,
    RotationWorldModel,
    append_jsonl,
    bootstrap_ci as _unused_stage0_bootstrap,
    dump_json,
    execute,
    git_revision,
    identity_audit,
    load_world_model,
    make_env,
    make_preprocessor,
    obs_batch,
    pad_state,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-pushobj-rotation-early-waypoint-stage0-v1"
EXPECTED_DESIGN_SHA256 = "b8552f29ff47a64ea8fd021bef799692ba035f35df9468ff2afeac49cd0e3c37"
FACTORS_DEG = (-22.5, -7.5, 7.5, 22.5)
SEGMENT_INDICES = (
    139, 249, 245, 124, 258, 288, 173, 156,
    295, 98, 476, 419, 34, 372, 313, 88,
    30, 172, 444, 293, 362, 105, 0, 274,
    425, 237, 492, 119, 296, 355, 190, 345,
)
MODEL_HORIZON = 2
ACTION_COUNT = MODEL_HORIZON * FRAMESKIP
NUM_SAMPLES = 200
TOPK = 30
OPT_STEPS = 10
WINDOW = 10
BOOTSTRAP_SEED = 820001
BOOTSTRAP_RESAMPLES = 20_000


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def nominal_block_displacement_at_10(segment: Dict) -> float:
    states = np.asarray(segment["states"], dtype=np.float64)
    if len(states) <= WINDOW:
        raise ValueError("segment is shorter than the waypoint deadline")
    return float(np.linalg.norm(states[WINDOW, 2:4] - states[0, 2:4]))


def scenario_rows(segments: Sequence[Dict]) -> list:
    rows = []
    for ordinal, segment_index in enumerate(SEGMENT_INDICES):
        displacement = nominal_block_displacement_at_10(segments[segment_index])
        if displacement < 10.0:
            raise RuntimeError(f"segment {segment_index} violates waypoint displacement")
        rows.append(
            {
                "ordinal": ordinal,
                "factor_deg": FACTORS_DEG[ordinal // 8],
                "within_factor": ordinal % 8,
                "segment_index": int(segment_index),
                "nominal_block_displacement_at_10": displacement,
                "env_seed": int(830_000 + ordinal),
                "cem_seed": int(840_000 + ordinal),
            }
        )
    return rows


def prepare_waypoint(env, segment: Dict, env_seed: int):
    env.shape = segment.get("shape", "T")
    initial_state = pad_state(np.asarray(segment["states"])[0])
    _, state = env.prepare(env_seed, initial_state)
    states = [np.asarray(state, dtype=np.float32)]
    actions = np.asarray(segment["actions"], dtype=np.float32)[:ACTION_COUNT] / 100.0
    if len(actions) != ACTION_COUNT:
        raise ValueError("segment has insufficient waypoint actions")
    waypoint_obs = None
    for action in actions:
        waypoint_obs, _, _, info = env.step(action)
        states.append(np.asarray(info["state"], dtype=np.float32))
    return initial_state, waypoint_obs, np.stack(states), actions


def plan_waypoint_cem(
    wm: RotationWorldModel,
    preprocessor,
    obs_0: Dict[str, np.ndarray],
    obs_g: Dict[str, np.ndarray],
    context_degrees: float,
    rng_seed: int,
) -> Tuple[np.ndarray, Dict]:
    from planning.objectives import create_objective_fn
    from utils import move_to_device

    device = next(wm.parameters()).device
    wm.set_context(context_degrees)
    transformed_0 = move_to_device(preprocessor.transform_obs(obs_batch(obs_0)), device)
    transformed_g = move_to_device(preprocessor.transform_obs(obs_batch(obs_g)), device)
    with torch.no_grad():
        goal_z = wm.encode_obs(transformed_g)
    repeated_0 = {
        key: repeat(value, "1 ... -> n ...", n=NUM_SAMPLES)
        for key, value in transformed_0.items()
    }
    repeated_goal = {
        key: repeat(value, "1 ... -> n ...", n=NUM_SAMPLES) for key, value in goal_z.items()
    }
    objective = create_objective_fn(alpha=1, base=2, mode="staged")
    generator = torch.Generator(device=device).manual_seed(int(rng_seed))
    mu = torch.zeros(MODEL_HORIZON, FRAMESKIP * ACTION_DIM, device=device)
    sigma = torch.ones_like(mu)
    trace = []
    for iteration in range(OPT_STEPS):
        actions = (
            torch.randn(
                NUM_SAMPLES,
                MODEL_HORIZON,
                FRAMESKIP * ACTION_DIM,
                generator=generator,
                device=device,
            )
            * sigma
            + mu
        )
        actions[0] = mu
        with torch.no_grad():
            prediction, _ = wm.rollout(repeated_0, actions)
            loss = objective(prediction, repeated_goal, step=0)
        elite_idx = torch.argsort(loss)[:TOPK]
        elite = actions[elite_idx]
        mu = elite.mean(0)
        sigma = elite.std(0).clamp_min(1e-4)
        trace.append(
            {
                "iteration": iteration,
                "best_loss": float(loss[elite_idx[0]].item()),
                "mu_sha256": array_sha256(mu.detach().cpu().numpy()),
            }
        )
    normalized = mu.detach().cpu().reshape(ACTION_COUNT, ACTION_DIM)
    physical = preprocessor.denormalize_actions(normalized).numpy().astype(np.float32)
    return physical, {"context_degrees": float(context_degrees), "trace": trace}


def deadline_success(states: np.ndarray, goal_state: np.ndarray) -> bool:
    final = np.asarray(states)[-1]
    goal = np.asarray(goal_state)
    position = float(np.linalg.norm(final[2:4] - goal[2:4]))
    angle = abs(float(final[4] - goal[4])) % (2.0 * np.pi)
    angle = min(angle, 2.0 * np.pi - angle)
    return bool(position < 20.0 and angle < np.pi / 9.0)


def bootstrap_ci(values: np.ndarray) -> list:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(raw_path: Path, identity: Dict) -> Dict:
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "paired_waypoint"]
    prior = np.asarray([row["prior"]["metrics"]["pose_auc10"] for row in rows])
    oracle = np.asarray([row["oracle"]["metrics"]["pose_auc10"] for row in rows])
    delta = prior - oracle
    result = {
        "contract_id": CONTRACT_ID,
        "n_pairs": len(rows),
        "primary_metric": "pose_auc10_to_waypoint",
        "prior_mean": float(prior.mean()),
        "oracle_mean": float(oracle.mean()),
        "mean_delta": float(delta.mean()),
        "relative_improvement": float(delta.mean() / prior.mean()),
        "bootstrap_ci95_delta": bootstrap_ci(delta),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)),
        "prior_deadline_success": float(np.mean([row["prior"]["deadline_success"] for row in rows])),
        "oracle_deadline_success": float(np.mean([row["oracle"]["deadline_success"] for row in rows])),
        "plan_changed_fraction": float(np.mean([row["prior"]["command_sha256"] != row["oracle"]["command_sha256"] for row in rows])),
        "nominal_waypoint_displacement_mean": float(np.mean([row["nominal_block_displacement_at_10"] for row in rows])),
        "identity_audit": identity,
        "by_factor": {},
    }
    for factor in FACTORS_DEG:
        indexes = np.asarray([float(row["factor_deg"]) == factor for row in rows])
        factor_delta = delta[indexes]
        result["by_factor"][str(factor)] = {
            "n": int(indexes.sum()),
            "prior_mean": float(prior[indexes].mean()),
            "oracle_mean": float(oracle[indexes].mean()),
            "mean_delta": float(factor_delta.mean()),
            "relative_improvement": float(factor_delta.mean() / prior[indexes].mean()),
            "positive_fraction": float(np.mean(factor_delta > 1e-12)),
        }
    checks = {
        "complete_32_pairs": len(rows) == 32,
        "unique_segments": len({int(row["segment_index"]) for row in rows}) == 32,
        "waypoint_displacement": all(float(row["nominal_block_displacement_at_10"]) >= 10.0 for row in rows),
        "identity": max(identity.values(), default=math.inf) <= 1e-6,
        "intervention_reached_planner": result["plan_changed_fraction"] > 0.0,
    }
    result["structural_checks"] = checks
    result["valid"] = all(checks.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"),
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"),
    )
    parser.add_argument(
        "--design", type=Path,
        default=Path("docs/research/persistent_context_v2_pushobj_rotation_early_waypoint_stage0_design.json"),
    )
    parser.add_argument(
        "--contract", type=Path,
        default=Path("docs/research/persistent_context_v2_pushobj_rotation_early_waypoint_stage0_contract_zh.md"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage0"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    selections = scenario_rows(segments)
    if args.mode == "inspect":
        payload = {
            "contract_id": CONTRACT_ID,
            "design_sha256": sha256(args.design),
            "contract_sha256": sha256(args.contract),
            "selections": selections,
        }
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
    seed_all(9201)
    device = torch.device(args.device)
    if not manifest_path.exists():
        manifest = {
            "contract_id": CONTRACT_ID,
            "git_revision": git_revision(),
            "design_path": str(args.design),
            "design_sha256": sha256(args.design),
            "contract_path": str(args.contract),
            "contract_sha256": sha256(args.contract),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "data": str(args.data),
            "data_sha256": sha256(args.data),
            "command": " ".join(__import__("sys").argv),
            "started_unix": time.time(),
            "resource_start": resource_snapshot(device),
        }
        dump_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base, wrapper, _ = load_world_model(args.checkpoint, device)
    preprocessor = make_preprocessor()
    env = make_env()
    raw_path = args.output_dir / "raw.jsonl"
    completed = {
        int(row["ordinal"]) for row in read_jsonl(raw_path)
        if row.get("record_type") == "paired_waypoint"
    }
    for selection in selections[: args.limit]:
        if selection["ordinal"] in completed:
            continue
        started = time.perf_counter()
        segment = segments[selection["segment_index"]]
        initial_state, waypoint_obs, nominal_states, nominal_actions = prepare_waypoint(
            env, segment, selection["env_seed"]
        )
        start_obs, _ = env.prepare(selection["env_seed"], initial_state)
        if "identity_audit" not in manifest:
            manifest["identity_audit"] = identity_audit(base, wrapper, preprocessor, start_obs)
            dump_json(manifest_path, manifest)
        plans = {}
        for label, context in (("prior", 0.0), ("oracle", selection["factor_deg"])):
            commands, planner = plan_waypoint_cem(
                wrapper, preprocessor, start_obs, waypoint_obs, context, selection["cem_seed"]
            )
            states, effective, contacts, coverage = execute(
                env, initial_state, selection["env_seed"], commands, selection["factor_deg"]
            )
            plans[label] = {
                "context_degrees": context,
                "commands": commands,
                "effective_actions": effective,
                "states": states,
                "contacts": contacts,
                "coverages": coverage,
                "metrics": pose_metrics(states, nominal_states[-1], WINDOW),
                "deadline_success": deadline_success(states, nominal_states[-1]),
                "command_sha256": array_sha256(commands),
                "state_sha256": array_sha256(states),
                "planner": planner,
            }
        row = {
            "record_type": "paired_waypoint",
            "contract_id": CONTRACT_ID,
            **selection,
            "initial_state": initial_state,
            "goal_state": nominal_states[-1],
            "nominal_actions": nominal_actions,
            "nominal_states": nominal_states,
            "prior": plans["prior"],
            "oracle": plans["oracle"],
            "elapsed_s": time.perf_counter() - started,
            "resource": resource_snapshot(device),
        }
        append_jsonl(raw_path, row)
        print(
            f"DONE {selection['ordinal']+1}/32 factor={selection['factor_deg']:+g} "
            f"segment={selection['segment_index']} prior={plans['prior']['metrics']['pose_auc10']:.4f} "
            f"oracle={plans['oracle']['metrics']['pose_auc10']:.4f} elapsed={row['elapsed_s']:.2f}s",
            flush=True,
        )
    if args.limit == 32:
        result = summarize(raw_path, manifest["identity_audit"])
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(device)
    dump_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
