"""True-factor oracle for a persistent radial action dead zone in PushObj."""

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
    WINDOW, deadline_success, nominal_block_displacement_at_10,
    plan_waypoint_cem, prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    ACTION_DIM, FRAMESKIP, append_jsonl, dump_json, git_revision, identity_audit,
    load_world_model, make_env, make_preprocessor, pose_metrics, read_jsonl,
    resource_snapshot, seed_all, sha256,
)


CONTRACT_ID = "persistent-context-v2-pushobj-radial-deadzone-stage0-v1"
EXPECTED_DESIGN_SHA256 = "7cff872d4d5c9d57cba167a6d67dbfdb3ecd00db6433059b7b706857b434edcc"
FACTORS = (0.025, 0.075, 0.125, 0.175)
POPULATION_PRIOR = 0.10
SEGMENT_INDICES = (
    142, 343, 32, 428, 199, 380, 148, 429,
    292, 407, 385, 376, 35, 149, 413, 321,
    360, 347, 398, 357, 179, 18, 152, 163,
    330, 193, 294, 322, 359, 138, 336, 469,
)
BOOTSTRAP_SEED = 900101
BOOTSTRAP_RESAMPLES = 20_000


def array_sha256(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def apply_radial_deadzone(actions: np.ndarray, radius: float) -> np.ndarray:
    actions = np.asarray(actions)
    if float(radius) == 0.0:
        return actions
    norms = np.linalg.norm(actions, axis=-1, keepdims=True)
    scale = np.maximum(norms - float(radius), 0.0) / np.maximum(norms, 1e-12)
    return actions * scale


class DeadZoneWorldModel(torch.nn.Module):
    def __init__(self, base, action_mean, action_std):
        super().__init__()
        self.base = base
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float())
        self.context_radius = 0.0

    def set_context(self, radius: float):
        self.context_radius = float(radius)

    def encode_obs(self, obs):
        return self.base.encode_obs(obs)

    def effective_normalized_actions(self, action):
        if self.context_radius == 0.0:
            return action
        shape = action.shape
        physical = action.reshape(*shape[:-1], FRAMESKIP, ACTION_DIM)
        physical = physical * self.action_std + self.action_mean
        norms = torch.linalg.vector_norm(physical, dim=-1, keepdim=True)
        scale = torch.clamp(norms - self.context_radius, min=0.0) / torch.clamp(norms, min=1e-12)
        effective = physical * scale
        return ((effective - self.action_mean) / self.action_std).reshape(shape)

    def rollout(self, obs_0, act):
        return self.base.rollout(obs_0, self.effective_normalized_actions(act))


def load_deadzone_world_model(checkpoint: Path, device: torch.device):
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
    base, _, cfg = load_world_model(checkpoint, device)
    wrapper = DeadZoneWorldModel(base, ACTION_MEAN, ACTION_STD).to(device).eval()
    return base, wrapper, cfg


def execute_deadzone(env, initial_state, env_seed, commands, radius):
    _, state = env.prepare(env_seed, initial_state)
    states = [np.asarray(state, dtype=np.float32)]
    effective = apply_radial_deadzone(np.asarray(commands, dtype=np.float32), radius).astype(np.float32)
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
        if displacement < 10.0: raise RuntimeError("invalid waypoint segment")
        rows.append({"ordinal": ordinal, "factor": FACTORS[ordinal // 8], "within_factor": ordinal % 8,
                     "segment_index": int(segment_index), "nominal_block_displacement_at_10": displacement,
                     "env_seed": 910000 + ordinal, "cem_seed": 920000 + ordinal})
    return rows


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def summarize(raw_path, identity):
    rows = [r for r in read_jsonl(raw_path) if r.get("record_type") == "paired_deadzone"]
    prior = np.asarray([r["prior"]["metrics"]["pose_auc10"] for r in rows]); oracle = np.asarray([r["oracle"]["metrics"]["pose_auc10"] for r in rows]); delta = prior - oracle
    result = {"contract_id": CONTRACT_ID, "n_pairs": len(rows), "primary_metric": "pose_auc10_to_waypoint",
              "prior_mean": float(prior.mean()), "oracle_mean": float(oracle.mean()), "mean_delta": float(delta.mean()),
              "relative_improvement": float(delta.mean() / prior.mean()), "bootstrap_ci95_delta": bootstrap_ci(delta),
              "positive_fraction": float(np.mean(delta > 1e-12)), "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
              "negative_fraction": float(np.mean(delta < -1e-12)),
              "prior_deadline_success": float(np.mean([r["prior"]["deadline_success"] for r in rows])),
              "oracle_deadline_success": float(np.mean([r["oracle"]["deadline_success"] for r in rows])),
              "prior_zero_action_fraction": float(np.mean([r["prior"]["zero_action_fraction"] for r in rows])),
              "oracle_zero_action_fraction": float(np.mean([r["oracle"]["zero_action_fraction"] for r in rows])),
              "plan_changed_fraction": float(np.mean([r["prior"]["command_sha256"] != r["oracle"]["command_sha256"] for r in rows])),
              "identity_audit": identity, "by_factor": {}}
    for factor in FACTORS:
        mask = np.asarray([float(r["factor"]) == factor for r in rows]); d = delta[mask]
        result["by_factor"][str(factor)] = {"n": int(mask.sum()), "prior_mean": float(prior[mask].mean()), "oracle_mean": float(oracle[mask].mean()),
                                                    "mean_delta": float(d.mean()), "relative_improvement": float(d.mean() / prior[mask].mean()), "positive_fraction": float(np.mean(d > 1e-12))}
    checks = {"complete": len(rows) == 32, "unique_segments": len({r["segment_index"] for r in rows}) == 32,
              "waypoint_displacement": all(r["nominal_block_displacement_at_10"] >= 10 for r in rows),
              "identity": max(identity.values(), default=math.inf) <= 1e-6, "plan_changed": result["plan_changed_fraction"] > 0}
    result["structural_checks"] = checks; result["valid"] = all(checks.values()); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("inspect", "run", "summarize")); parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_deadzone_stage0_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_deadzone_stage0_contract_zh.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_deadzone_stage0")); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as h: segments = pickle.load(h)["segments"]
    selection = scenarios(segments)
    if args.mode == "inspect":
        p = {"contract_id": CONTRACT_ID, "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract), "scenarios": selection}; dump_json(args.output_dir / "selection_audit.json", p); print(json.dumps(p, indent=2)); return
    manifest_path = args.output_dir / "manifest.json"
    if args.mode == "summarize":
        m = json.loads(manifest_path.read_text(encoding="utf-8")); r = summarize(args.output_dir / "raw.jsonl", m["identity_audit"]); dump_json(args.output_dir / "runner_summary.json", r); print(json.dumps(r, indent=2)); return
    if sha256(args.design) != EXPECTED_DESIGN_SHA256: raise RuntimeError("frozen design hash mismatch")
    seed_all(9401); device = torch.device(args.device)
    if not manifest_path.exists():
        manifest = {"contract_id": CONTRACT_ID, "git_revision": git_revision(), "design_path": str(args.design), "design_sha256": sha256(args.design),
                    "contract_path": str(args.contract), "contract_sha256": sha256(args.contract), "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                    "data": str(args.data), "data_sha256": sha256(args.data), "command": " ".join(__import__("sys").argv), "started_unix": time.time(), "resource_start": resource_snapshot(device)}; dump_json(manifest_path, manifest)
    else: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base, wrapper, _ = load_deadzone_world_model(args.checkpoint, device); preprocessor = make_preprocessor(); env = make_env(); raw_path = args.output_dir / "raw.jsonl"
    completed = {int(r["ordinal"]) for r in read_jsonl(raw_path) if r.get("record_type") == "paired_deadzone"}
    for meta in selection[:args.limit]:
        if meta["ordinal"] in completed: continue
        started = time.perf_counter(); initial, goal_obs, nominal_states, nominal_actions = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"]); start_obs, _ = env.prepare(meta["env_seed"], initial)
        if "identity_audit" not in manifest: manifest["identity_audit"] = identity_audit(base, wrapper, preprocessor, start_obs); dump_json(manifest_path, manifest)
        policies = {}
        for name, context in (("prior", POPULATION_PRIOR), ("oracle", meta["factor"])):
            commands, planner = plan_waypoint_cem(wrapper, preprocessor, start_obs, goal_obs, context, meta["cem_seed"])
            states, effective, contacts, coverages = execute_deadzone(env, initial, meta["env_seed"], commands, meta["factor"])
            policies[name] = {"context": context, "commands": commands, "effective_actions": effective, "states": states, "contacts": contacts, "coverages": coverages,
                              "metrics": pose_metrics(states, nominal_states[-1], WINDOW), "deadline_success": deadline_success(states, nominal_states[-1]),
                              "zero_action_fraction": float(np.mean(np.linalg.norm(effective, axis=1) <= 1e-12)), "command_sha256": array_sha256(commands), "state_sha256": array_sha256(states), "planner": planner}
        row = {"record_type": "paired_deadzone", "contract_id": CONTRACT_ID, **meta, "initial_state": initial, "goal_state": nominal_states[-1], "nominal_actions": nominal_actions,
               "prior": policies["prior"], "oracle": policies["oracle"], "elapsed_s": time.perf_counter()-started, "resource": resource_snapshot(device)}; append_jsonl(raw_path, row)
        print(f"DONE {meta['ordinal']+1}/32 d={meta['factor']:.3f} prior={policies['prior']['metrics']['pose_auc10']:.4f} oracle={policies['oracle']['metrics']['pose_auc10']:.4f}", flush=True)
    if args.limit == 32:
        r = summarize(raw_path, manifest["identity_audit"]); dump_json(args.output_dir / "runner_summary.json", r); print(json.dumps(r, indent=2))
    manifest["finished_unix"] = time.time(); manifest["resource_end"] = resource_snapshot(device); dump_json(manifest_path, manifest)


if __name__ == "__main__": main()
