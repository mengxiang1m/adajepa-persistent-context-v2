"""True-matrix oracle for joint PushObj action rotation and gain."""

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
from einops import repeat

from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    ACTION_COUNT,
    MODEL_HORIZON,
    NUM_SAMPLES,
    OPT_STEPS,
    TOPK,
    WINDOW,
    deadline_success,
    nominal_block_displacement_at_10,
    prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    ACTION_DIM,
    FRAMESKIP,
    append_jsonl,
    dump_json,
    git_revision,
    load_world_model,
    make_env,
    make_preprocessor,
    obs_batch,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-pushobj-rotation-gain-matrix-stage0-v1"
EXPECTED_DESIGN_SHA256 = "4daf9de6f0415c2fb4eb1434683a50b420240813ab82b07e9639babf9820a73e"
PRIOR_SCALE = 0.9327804920294028
POPULATION_PRIOR_MATRIX = np.eye(2, dtype=np.float64) * PRIOR_SCALE
FACTORS = (
    (-22.5, 0.85), (-22.5, 1.15),
    (-7.5, 0.85), (-7.5, 1.15),
    (7.5, 0.85), (7.5, 1.15),
    (22.5, 0.85), (22.5, 1.15),
)
SEGMENT_INDICES = (
    326, 233, 86, 50, 180, 460, 147, 95,
    283, 487, 208, 129, 240, 339, 89, 396,
    449, 397, 327, 31, 121, 338, 280, 282,
    5, 285, 19, 134, 232, 370, 116, 456,
)
BOOTSTRAP_SEED = 1_000_101
BOOTSTRAP_RESAMPLES = 20_000


def array_sha256(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def factor_matrix(rotation_degrees: float, gain: float, dtype=np.float32) -> np.ndarray:
    theta = math.radians(float(rotation_degrees))
    return float(gain) * np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=dtype,
    )


def apply_action_matrix(actions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    matrix = np.asarray(matrix, dtype=actions.dtype)
    if matrix.shape != (ACTION_DIM, ACTION_DIM):
        raise ValueError(f"expected 2x2 matrix, got {matrix.shape}")
    if np.array_equal(matrix, np.eye(ACTION_DIM, dtype=matrix.dtype)):
        return actions
    return actions @ matrix.T


class MatrixWorldModel(torch.nn.Module):
    def __init__(self, base, action_mean, action_std):
        super().__init__()
        self.base = base
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float())
        self.context_matrix = np.eye(ACTION_DIM, dtype=np.float64)

    def set_context(self, matrix) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (ACTION_DIM, ACTION_DIM) or not np.all(np.isfinite(matrix)):
            raise ValueError("context matrix must be finite and 2x2")
        self.context_matrix = matrix.copy()

    def encode_obs(self, obs):
        return self.base.encode_obs(obs)

    def effective_normalized_actions(self, action):
        if np.array_equal(self.context_matrix, np.eye(ACTION_DIM)):
            return action
        shape = action.shape
        physical = action.reshape(*shape[:-1], FRAMESKIP, ACTION_DIM)
        physical = physical * self.action_std + self.action_mean
        matrix = physical.new_tensor(self.context_matrix)
        effective = physical @ matrix.T
        return ((effective - self.action_mean) / self.action_std).reshape(shape)

    def rollout(self, obs_0, act):
        return self.base.rollout(obs_0, self.effective_normalized_actions(act))


def load_matrix_world_model(checkpoint: Path, device: torch.device):
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD

    base, _, cfg = load_world_model(checkpoint, device)
    wrapper = MatrixWorldModel(base, ACTION_MEAN, ACTION_STD).to(device).eval()
    return base, wrapper, cfg


def plan_matrix_waypoint(wm, preprocessor, obs_0, obs_g, context_matrix, rng_seed):
    from planning.objectives import create_objective_fn
    from utils import move_to_device

    device = next(wm.parameters()).device
    wm.set_context(context_matrix)
    transformed_0 = move_to_device(preprocessor.transform_obs(obs_batch(obs_0)), device)
    transformed_g = move_to_device(preprocessor.transform_obs(obs_batch(obs_g)), device)
    with torch.no_grad():
        goal_z = wm.encode_obs(transformed_g)
    repeated_0 = {key: repeat(value, "1 ... -> n ...", n=NUM_SAMPLES) for key, value in transformed_0.items()}
    repeated_goal = {key: repeat(value, "1 ... -> n ...", n=NUM_SAMPLES) for key, value in goal_z.items()}
    objective = create_objective_fn(alpha=1, base=2, mode="staged")
    generator = torch.Generator(device=device).manual_seed(int(rng_seed))
    mu = torch.zeros(MODEL_HORIZON, FRAMESKIP * ACTION_DIM, device=device)
    sigma = torch.ones_like(mu)
    trace = []
    for iteration in range(OPT_STEPS):
        actions = torch.randn(NUM_SAMPLES, MODEL_HORIZON, FRAMESKIP * ACTION_DIM, generator=generator, device=device) * sigma + mu
        actions[0] = mu
        with torch.no_grad():
            prediction, _ = wm.rollout(repeated_0, actions)
            loss = objective(prediction, repeated_goal, step=0)
        elite_index = torch.argsort(loss)[:TOPK]
        elite = actions[elite_index]
        mu = elite.mean(0)
        sigma = elite.std(0).clamp_min(1e-4)
        trace.append({"iteration": iteration, "best_loss": float(loss[elite_index[0]].item()), "mu_sha256": array_sha256(mu.detach().cpu().numpy())})
    normalized = mu.detach().cpu().reshape(ACTION_COUNT, ACTION_DIM)
    commands = preprocessor.denormalize_actions(normalized).numpy().astype(np.float32)
    return commands, {"context_matrix": np.asarray(context_matrix), "trace": trace}


def execute_matrix(env, initial_state, env_seed, commands, true_matrix):
    _, state = env.prepare(env_seed, initial_state)
    states = [np.asarray(state, dtype=np.float32)]
    effective = apply_action_matrix(np.asarray(commands, dtype=np.float32), np.asarray(true_matrix, dtype=np.float32)).astype(np.float32)
    contacts, coverages = [], []
    for action in effective:
        _, _, _, info = env.step(action)
        states.append(np.asarray(info["state"], dtype=np.float32))
        contacts.append(int(info["n_contacts"]))
        coverages.append(float(info["final_coverage"]))
    return np.stack(states), effective, contacts, coverages


def identity_audit(base, wrapper, preprocessor, obs):
    from utils import move_to_device

    device = next(base.parameters()).device
    transformed = move_to_device(preprocessor.transform_obs(obs_batch(obs)), device)
    repeated = {key: repeat(value, "1 ... -> n ...", n=2) for key, value in transformed.items()}
    generator = torch.Generator(device=device).manual_seed(1_000_781)
    actions = torch.randn(2, MODEL_HORIZON, FRAMESKIP * ACTION_DIM, generator=generator, device=device)
    wrapper.set_context(np.eye(2))
    with torch.no_grad():
        base_prediction, _ = base.rollout(repeated, actions)
        wrapped_prediction, _ = wrapper.rollout(repeated, actions)
    rollout_max = max((float((base_prediction[key] - wrapped_prediction[key]).abs().max().item()) for key in base_prediction), default=0.0)
    action_max = float((actions - wrapper.effective_normalized_actions(actions)).abs().max().item())
    return {"action_max_abs": action_max, "rollout_max_abs": rollout_max}


def scenarios(segments):
    result = []
    for ordinal, segment_index in enumerate(SEGMENT_INDICES):
        rotation, gain = FACTORS[ordinal // 4]
        displacement = nominal_block_displacement_at_10(segments[segment_index])
        if displacement < 10:
            raise RuntimeError("invalid waypoint segment")
        result.append({"ordinal": ordinal, "rotation_degrees": rotation, "gain": gain, "factor_index": ordinal // 4, "within_factor": ordinal % 4, "true_matrix": factor_matrix(rotation, gain), "segment_index": int(segment_index), "nominal_block_displacement_at_10": displacement, "env_seed": 1_010_000 + ordinal, "cem_seed": 1_020_000 + ordinal})
    return result


def bootstrap_ci(values, stream=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(stream))
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def effect(prior, oracle, stream=0):
    prior, oracle = np.asarray(prior), np.asarray(oracle)
    delta = prior - oracle
    return {"n": len(delta), "prior_mean": float(prior.mean()), "oracle_mean": float(oracle.mean()), "mean_delta": float(delta.mean()), "relative_improvement": float(delta.mean() / prior.mean()), "bootstrap_ci95_delta": bootstrap_ci(delta, stream), "positive_fraction": float(np.mean(delta > 1e-12)), "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)), "negative_fraction": float(np.mean(delta < -1e-12))}


def summarize(raw_path, identity):
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "paired_matrix"]
    prior = np.asarray([row["prior"]["metrics"]["pose_auc10"] for row in rows])
    oracle = np.asarray([row["oracle"]["metrics"]["pose_auc10"] for row in rows])
    result = {"contract_id": CONTRACT_ID, "primary_metric": "pose_auc10_to_waypoint", **effect(prior, oracle), "prior_deadline_success": float(np.mean([row["prior"]["deadline_success"] for row in rows])), "oracle_deadline_success": float(np.mean([row["oracle"]["deadline_success"] for row in rows])), "plan_changed_fraction": float(np.mean([row["prior"]["command_sha256"] != row["oracle"]["command_sha256"] for row in rows])), "identity_audit": identity, "by_factor": {}, "by_gain": {}, "by_rotation": {}}
    for factor_index, (rotation, gain) in enumerate(FACTORS):
        mask = np.asarray([int(row["factor_index"]) == factor_index for row in rows])
        result["by_factor"][f"theta={rotation:+g},gain={gain:g}"] = effect(prior[mask], oracle[mask], 10 + factor_index)
    for index, gain in enumerate((0.85, 1.15)):
        mask = np.asarray([float(row["gain"]) == gain for row in rows])
        result["by_gain"][str(gain)] = effect(prior[mask], oracle[mask], 30 + index)
    for index, rotation in enumerate((-22.5, -7.5, 7.5, 22.5)):
        mask = np.asarray([float(row["rotation_degrees"]) == rotation for row in rows])
        result["by_rotation"][str(rotation)] = effect(prior[mask], oracle[mask], 40 + index)
    checks = {"complete": len(rows) == 32, "unique_segments": len({row["segment_index"] for row in rows}) == 32, "factor_balance": all(sum(int(row["factor_index"]) == index for row in rows) == 4 for index in range(8)), "waypoint_displacement": all(row["nominal_block_displacement_at_10"] >= 10 for row in rows), "identity": max(identity.values(), default=math.inf) <= 1e-6, "plan_changed": result["plan_changed_fraction"] > 0}
    result["structural_checks"] = checks
    result["valid"] = all(checks.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_matrix_stage0_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_matrix_stage0_contract_zh.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_matrix_stage0"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    selection = scenarios(segments)
    if args.mode == "inspect":
        payload = {"contract_id": CONTRACT_ID, "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract), "population_prior_matrix": POPULATION_PRIOR_MATRIX, "scenarios": selection}
        dump_json(args.output_dir / "selection_audit.json", payload)
        print(json.dumps(payload, indent=2, default=lambda x: np.asarray(x).tolist()))
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
    seed_all(1_030_001)
    device = torch.device(args.device)
    if not manifest_path.exists():
        manifest = {"contract_id": CONTRACT_ID, "git_revision": git_revision(), "design_path": str(args.design), "design_sha256": sha256(args.design), "contract_path": str(args.contract), "contract_sha256": sha256(args.contract), "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "data": str(args.data), "data_sha256": sha256(args.data), "command": " ".join(__import__("sys").argv), "started_unix": time.time(), "resource_start": resource_snapshot(device)}
        dump_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base, wrapper, _ = load_matrix_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    raw_path = args.output_dir / "raw.jsonl"
    completed = {int(row["ordinal"]) for row in read_jsonl(raw_path) if row.get("record_type") == "paired_matrix"}
    for meta in selection[: args.limit]:
        if meta["ordinal"] in completed:
            continue
        started = time.perf_counter()
        initial, goal_obs, nominal_states, nominal_actions = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
        start_obs, _ = env.prepare(meta["env_seed"], initial)
        if "identity_audit" not in manifest:
            manifest["identity_audit"] = identity_audit(base, wrapper, preprocessor, start_obs)
            dump_json(manifest_path, manifest)
        policies = {}
        for name, context in (("prior", POPULATION_PRIOR_MATRIX), ("oracle", meta["true_matrix"])):
            commands, planner = plan_matrix_waypoint(wrapper, preprocessor, start_obs, goal_obs, context, meta["cem_seed"])
            states, effective, contacts, coverages = execute_matrix(env, initial, meta["env_seed"], commands, meta["true_matrix"])
            policies[name] = {"context_matrix": context, "commands": commands, "effective_actions": effective, "states": states, "contacts": contacts, "coverages": coverages, "metrics": pose_metrics(states, nominal_states[-1], WINDOW), "deadline_success": deadline_success(states, nominal_states[-1]), "command_sha256": array_sha256(commands), "effective_action_sha256": array_sha256(effective), "state_sha256": array_sha256(states), "planner": planner}
        row = {"record_type": "paired_matrix", "contract_id": CONTRACT_ID, **meta, "initial_state": initial, "goal_state": nominal_states[-1], "nominal_actions": nominal_actions, "prior": policies["prior"], "oracle": policies["oracle"], "elapsed_s": time.perf_counter() - started, "resource": resource_snapshot(device)}
        append_jsonl(raw_path, row)
        print(f"DONE {meta['ordinal'] + 1}/32 theta={meta['rotation_degrees']:+g} gain={meta['gain']:.2f} prior={policies['prior']['metrics']['pose_auc10']:.4f} oracle={policies['oracle']['metrics']['pose_auc10']:.4f}", flush=True)
    if args.limit == 32:
        result = summarize(raw_path, manifest["identity_audit"])
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(device)
    dump_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
