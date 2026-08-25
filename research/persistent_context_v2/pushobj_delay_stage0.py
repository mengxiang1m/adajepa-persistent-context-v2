"""True-factor oracle for an episode-reset discrete action delay in PushObj."""

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
    plan_waypoint_cem,
    prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    ACTION_DIM,
    FRAMESKIP,
    append_jsonl,
    dump_json,
    git_revision,
    identity_audit,
    load_world_model,
    make_env,
    make_preprocessor,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-pushobj-discrete-delay-stage0-v1"
EXPECTED_DESIGN_SHA256 = "12fbc8f89e298144a4b2f7fe724b86790db3dfebe0154416acc4f86e6fe2c72e"
FACTORS = (0, 1, 3, 4)
POPULATION_PRIOR = 2
SEGMENT_INDICES = (
    61, 369, 485, 468, 102, 189, 495, 202,
    473, 14, 218, 426, 305, 122, 433, 157,
    286, 141, 37, 430, 441, 214, 65, 306,
    244, 11, 422, 264, 64, 80, 261, 74,
)
BOOTSTRAP_SEED = 960101
BOOTSTRAP_RESAMPLES = 20_000


def array_sha256(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _validate_delay(delay_steps: int) -> int:
    delay = int(delay_steps)
    if delay != delay_steps or delay < 0:
        raise ValueError(f"delay must be a non-negative integer, got {delay_steps}")
    return delay


def apply_discrete_delay(actions: np.ndarray, delay_steps: int) -> np.ndarray:
    """Apply an episode-reset physical zero-filled FIFO to low-level actions."""
    actions = np.asarray(actions)
    delay = _validate_delay(delay_steps)
    if delay == 0:
        return actions
    if actions.ndim < 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected [..., time, {ACTION_DIM}] actions, got {actions.shape}")
    effective = np.zeros_like(actions)
    if delay < actions.shape[-2]:
        effective[..., delay:, :] = actions[..., :-delay, :]
    return effective


class DelayWorldModel(torch.nn.Module):
    """Frozen base model with a delay transform over the full rollout sequence."""

    def __init__(self, base, action_mean, action_std):
        super().__init__()
        self.base = base
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float())
        self.context_delay_steps = 0

    def set_context(self, delay_steps: int) -> None:
        self.context_delay_steps = _validate_delay(delay_steps)

    def encode_obs(self, obs):
        return self.base.encode_obs(obs)

    def effective_normalized_actions(self, action: torch.Tensor) -> torch.Tensor:
        delay = self.context_delay_steps
        if delay == 0:
            return action
        if action.ndim < 2 or action.shape[-1] != FRAMESKIP * ACTION_DIM:
            raise ValueError(f"unexpected model action shape {tuple(action.shape)}")
        original_shape = action.shape
        physical = action.reshape(*original_shape[:-1], FRAMESKIP, ACTION_DIM)
        physical = physical * self.action_std + self.action_mean
        # Flatten model horizon and frameskip so the FIFO does not reset at a
        # model-step boundary. Any leading batch/sample dimensions are retained.
        leading = physical.shape[:-3]
        low_level_count = physical.shape[-3] * physical.shape[-2]
        flat = physical.reshape(*leading, low_level_count, ACTION_DIM)
        effective = torch.zeros_like(flat)
        if delay < low_level_count:
            effective[..., delay:, :] = flat[..., :-delay, :]
        normalized = (effective - self.action_mean) / self.action_std
        return normalized.reshape(original_shape)

    def rollout(self, obs_0, act):
        return self.base.rollout(obs_0, self.effective_normalized_actions(act))


def load_delay_world_model(checkpoint: Path, device: torch.device):
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD

    base, _, cfg = load_world_model(checkpoint, device)
    wrapper = DelayWorldModel(base, ACTION_MEAN, ACTION_STD).to(device).eval()
    return base, wrapper, cfg


def execute_delay(env, initial_state, env_seed, commands, delay_steps):
    _, state = env.prepare(env_seed, initial_state)
    states = [np.asarray(state, dtype=np.float32)]
    commands = np.asarray(commands, dtype=np.float32)
    effective = apply_discrete_delay(commands, delay_steps).astype(np.float32)
    contacts, coverages = [], []
    for action in effective:
        _, _, _, info = env.step(action)
        states.append(np.asarray(info["state"], dtype=np.float32))
        contacts.append(int(info["n_contacts"]))
        coverages.append(float(info["final_coverage"]))
    return np.stack(states), effective, contacts, coverages


def scenarios(segments):
    rows = []
    for ordinal, segment_index in enumerate(SEGMENT_INDICES):
        displacement = nominal_block_displacement_at_10(segments[segment_index])
        if displacement < 10.0:
            raise RuntimeError("invalid waypoint segment")
        rows.append(
            {
                "ordinal": ordinal,
                "factor_steps": FACTORS[ordinal // 8],
                "within_factor": ordinal % 8,
                "segment_index": int(segment_index),
                "nominal_block_displacement_at_10": displacement,
                "env_seed": 970_000 + ordinal,
                "cem_seed": 980_000 + ordinal,
            }
        )
    return rows


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(raw_path, identity):
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "paired_delay"]
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
        "identity_audit": identity,
        "by_factor": {},
    }
    for factor in FACTORS:
        mask = np.asarray([int(row["factor_steps"]) == factor for row in rows])
        factor_delta = delta[mask]
        result["by_factor"][str(factor)] = {
            "n": int(mask.sum()),
            "prior_mean": float(prior[mask].mean()),
            "oracle_mean": float(oracle[mask].mean()),
            "mean_delta": float(factor_delta.mean()),
            "relative_improvement": float(factor_delta.mean() / prior[mask].mean()),
            "positive_fraction": float(np.mean(factor_delta > 1e-12)),
            "prior_deadline_success": float(np.mean([row["prior"]["deadline_success"] for row in rows if int(row["factor_steps"]) == factor])),
            "oracle_deadline_success": float(np.mean([row["oracle"]["deadline_success"] for row in rows if int(row["factor_steps"]) == factor])),
        }
    checks = {
        "complete": len(rows) == 32,
        "unique_segments": len({row["segment_index"] for row in rows}) == 32,
        "waypoint_displacement": all(row["nominal_block_displacement_at_10"] >= 10 for row in rows),
        "identity": max(identity.values(), default=math.inf) <= 1e-6,
        "plan_changed": result["plan_changed_fraction"] > 0,
    }
    result["structural_checks"] = checks
    result["valid"] = all(checks.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_delay_stage0_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_delay_stage0_contract_zh.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_delay_stage0"))
    parser.add_argument("--device", default="cuda:0")
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
    seed_all(9901)
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
    base, wrapper, _ = load_delay_world_model(args.checkpoint, device)
    preprocessor = make_preprocessor()
    env = make_env()
    raw_path = args.output_dir / "raw.jsonl"
    completed = {int(row["ordinal"]) for row in read_jsonl(raw_path) if row.get("record_type") == "paired_delay"}
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
        for name, context in (("prior", POPULATION_PRIOR), ("oracle", meta["factor_steps"])):
            commands, planner = plan_waypoint_cem(wrapper, preprocessor, start_obs, goal_obs, context, meta["cem_seed"])
            planner["context_delay_steps"] = int(context)
            states, effective, contacts, coverages = execute_delay(env, initial, meta["env_seed"], commands, meta["factor_steps"])
            policies[name] = {
                "context_delay_steps": int(context),
                "commands": commands,
                "effective_actions": effective,
                "states": states,
                "contacts": contacts,
                "coverages": coverages,
                "metrics": pose_metrics(states, nominal_states[-1], WINDOW),
                "deadline_success": deadline_success(states, nominal_states[-1]),
                "command_sha256": array_sha256(commands),
                "effective_action_sha256": array_sha256(effective),
                "state_sha256": array_sha256(states),
                "planner": planner,
            }
        row = {
            "record_type": "paired_delay",
            "contract_id": CONTRACT_ID,
            **meta,
            "initial_state": initial,
            "goal_state": nominal_states[-1],
            "nominal_actions": nominal_actions,
            "prior": policies["prior"],
            "oracle": policies["oracle"],
            "elapsed_s": time.perf_counter() - started,
            "resource": resource_snapshot(device),
        }
        append_jsonl(raw_path, row)
        print(
            f"DONE {meta['ordinal'] + 1}/32 delay={meta['factor_steps']} "
            f"prior={policies['prior']['metrics']['pose_auc10']:.4f} "
            f"oracle={policies['oracle']['metrics']['pose_auc10']:.4f}",
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
