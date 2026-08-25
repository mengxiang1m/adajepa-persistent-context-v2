"""Pre-registered real PushObj Stage-0 test for a persistent tool-frame rotation.

This runner intentionally contains no history estimator.  It can only compare
the population-prior context (zero degrees) with the known true factor.  The
separate audit program recomputes descriptive evidence from append-only raw rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from einops import repeat

try:  # ``resource`` is POSIX-only; pure-function tests also run on Windows.
    import resource
except ImportError:  # pragma: no cover - exercised by the Windows test host
    resource = None


DESIGN_ID = "persistent-context-v2-pushobj-tool-rotation-stage0-v1"
EXPECTED_DESIGN_SHA256 = "a1c1f077890d2ec591871eab86e0c26cab263557840d2ce8871f69f65a8aa299"
DEV_FACTORS_DEG = (-22.5, -7.5, 7.5, 22.5)
POPULATION_PRIOR_DEG = 0.0
FRAMESKIP = 5
ACTION_DIM = 2
MODEL_HORIZON = 5
ACTION_COUNT = FRAMESKIP * MODEL_HORIZON
NUM_SAMPLES = 200
TOPK = 30
OPT_STEPS = 10
SCENARIOS_PER_FACTOR = 8

CANDIDATES = {
    "A_released": {"selection_seed": 410000, "window": 25},
    "B_early_contact": {"selection_seed": 420000, "window": 10},
}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rotation_matrix(degrees: float, dtype=np.float32) -> np.ndarray:
    theta = math.radians(float(degrees))
    return np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=dtype,
    )


def rotate_actions(actions: np.ndarray, degrees: float) -> np.ndarray:
    actions = np.asarray(actions)
    return actions @ rotation_matrix(degrees, dtype=actions.dtype).T


def wrapped_angle_error(angle: np.ndarray, target: float) -> np.ndarray:
    delta = np.abs(np.asarray(angle) - float(target)) % (2.0 * np.pi)
    return np.minimum(delta, 2.0 * np.pi - delta)


def pose_metrics(states: np.ndarray, goal_state: np.ndarray, window: int) -> Dict[str, float]:
    states = np.asarray(states, dtype=np.float64)
    goal_state = np.asarray(goal_state, dtype=np.float64)
    end = min(int(window), len(states) - 1)
    if end < 1:
        raise ValueError("execution contains no post-action state")
    post = states[1 : end + 1]
    position = np.linalg.norm(post[:, 2:4] - goal_state[None, 2:4], axis=1)
    angle = wrapped_angle_error(post[:, 4], goal_state[4])
    pose = position / 20.0 + angle / (np.pi / 9.0)
    return {
        f"pose_auc{window}": float(np.mean(pose)),
        f"position_auc{window}": float(np.mean(position)),
        f"angle_auc{window}": float(np.mean(angle)),
        f"position_end{window}": float(position[-1]),
        f"angle_end{window}": float(angle[-1]),
    }


def early_contact_pool(segments: Sequence[Dict]) -> np.ndarray:
    chosen = []
    for index, segment in enumerate(segments[:500]):
        states = np.asarray(segment["states"])
        if len(states) > 10:
            displacement = np.linalg.norm(states[10, 2:4] - states[0, 2:4])
            if displacement >= 10.0:
                chosen.append(index)
    return np.asarray(chosen, dtype=np.int64)


def select_scenarios(segments: Sequence[Dict], candidate: str) -> List[Dict]:
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    if candidate == "A_released":
        pool = np.arange(min(500, len(segments)), dtype=np.int64)
    else:
        pool = early_contact_pool(segments)
    required = len(DEV_FACTORS_DEG) * SCENARIOS_PER_FACTOR
    if len(pool) < required:
        raise RuntimeError(f"candidate pool too small: {len(pool)} < {required}")
    rng = np.random.default_rng(CANDIDATES[candidate]["selection_seed"])
    selected = rng.permutation(pool)[:required]
    rows = []
    ordinal = 0
    for factor in DEV_FACTORS_DEG:
        for within_factor in range(SCENARIOS_PER_FACTOR):
            segment_index = int(selected[ordinal])
            rows.append(
                {
                    "candidate": candidate,
                    "ordinal": ordinal,
                    "within_factor": within_factor,
                    "factor_deg": float(factor),
                    "segment_index": segment_index,
                    "env_seed": int(510000 + CANDIDATES[candidate]["selection_seed"] + ordinal),
                    "cem_seed": int(CANDIDATES[candidate]["selection_seed"] + ordinal),
                }
            )
            ordinal += 1
    return rows


def pad_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] == 7:
        return state.copy()
    if state.shape[-1] != 5:
        raise ValueError(f"unexpected PushObj state dimension {state.shape[-1]}")
    padded = np.zeros(7, dtype=np.float32)
    padded[:5] = state
    return padded


def obs_batch(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.expand_dims(np.expand_dims(value, 0), 0) for key, value in obs.items()}


class RotationWorldModel(torch.nn.Module):
    """Frozen base checkpoint with an exact persistent action-coordinate map."""

    def __init__(self, base: torch.nn.Module, action_mean: torch.Tensor, action_std: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float())
        self.context_degrees = 0.0

    def set_context(self, degrees: float) -> None:
        self.context_degrees = float(degrees)

    def encode_obs(self, obs):
        return self.base.encode_obs(obs)

    def effective_normalized_actions(self, action: torch.Tensor) -> torch.Tensor:
        # Preserve exact identity for the population prior and identity audit.
        if self.context_degrees == 0.0:
            return action
        shape = action.shape
        physical = action.reshape(*shape[:-1], FRAMESKIP, ACTION_DIM)
        physical = physical * self.action_std + self.action_mean
        theta = math.radians(self.context_degrees)
        matrix = physical.new_tensor(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        effective = physical @ matrix.T
        normalized = (effective - self.action_mean) / self.action_std
        return normalized.reshape(shape)

    def rollout(self, obs_0, act):
        return self.base.rollout(obs_0, self.effective_normalized_actions(act))


def make_preprocessor():
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import (
        ACTION_MEAN,
        ACTION_STD,
        PROPRIO_MEAN,
        PROPRIO_STD,
        STATE_MEAN,
        STATE_STD,
    )
    from preprocessor import Preprocessor

    return Preprocessor(
        action_mean=ACTION_MEAN,
        action_std=ACTION_STD,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        proprio_mean=PROPRIO_MEAN,
        proprio_std=PROPRIO_STD,
        transform=default_transform(224),
    )


def load_world_model(checkpoint: Path, device: torch.device):
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
    from omegaconf import OmegaConf
    from plan import load_model

    model_dir = checkpoint.parent.parent
    cfg = OmegaConf.load(model_dir / "hydra.yaml")
    base = load_model(checkpoint, cfg, int(cfg.num_action_repeat), device)
    base.to(device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    wrapper = RotationWorldModel(base, ACTION_MEAN, ACTION_STD).to(device).eval()
    return base, wrapper, cfg


def plan_cem(
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
                "mu_sha256": hashlib.sha256(mu.detach().cpu().numpy().tobytes()).hexdigest(),
            }
        )
    normalized = mu.detach().cpu().reshape(ACTION_COUNT, ACTION_DIM)
    # Deliberately no clipping: this matches released PushObj evaluation.
    physical = preprocessor.denormalize_actions(normalized).numpy().astype(np.float32)
    return physical, {"context_degrees": context_degrees, "trace": trace}


def make_env():
    # Import registers the gym id.
    import env  # noqa: F401
    import gym

    wrapped = gym.make("pushobj")
    return wrapped.unwrapped


def prepare_target(env, segment: Dict, env_seed: int):
    env.shape = segment.get("shape", "T")
    initial_state = pad_state(np.asarray(segment["states"])[0])
    obs, state = env.prepare(env_seed, initial_state)
    states = [np.asarray(state, dtype=np.float32)]
    actions = np.asarray(segment["actions"], dtype=np.float32) / 100.0
    if len(actions) != ACTION_COUNT:
        raise ValueError(f"expected {ACTION_COUNT} segment actions, found {len(actions)}")
    for action in actions:
        obs, _, _, info = env.step(action)
        states.append(np.asarray(info["state"], dtype=np.float32))
    return initial_state, obs, np.stack(states), actions


def execute(env, initial_state, env_seed: int, commands: np.ndarray, factor_deg: float):
    _, state = env.prepare(env_seed, initial_state)
    states = [np.asarray(state, dtype=np.float32)]
    effective = rotate_actions(commands, factor_deg).astype(np.float32)
    contacts = []
    coverages = []
    for action in effective:
        _, _, _, info = env.step(action)
        states.append(np.asarray(info["state"], dtype=np.float32))
        contacts.append(int(info["n_contacts"]))
        coverages.append(float(info["final_coverage"]))
    return np.stack(states), effective, contacts, coverages


def identity_audit(
    base: torch.nn.Module,
    wrapper: RotationWorldModel,
    preprocessor,
    obs: Dict[str, np.ndarray],
) -> Dict[str, float]:
    from utils import move_to_device

    device = next(base.parameters()).device
    transformed = move_to_device(preprocessor.transform_obs(obs_batch(obs)), device)
    repeated = {key: repeat(value, "1 ... -> n ...", n=2) for key, value in transformed.items()}
    generator = torch.Generator(device=device).manual_seed(7781)
    actions = torch.randn(2, MODEL_HORIZON, FRAMESKIP * ACTION_DIM, generator=generator, device=device)
    wrapper.set_context(0.0)
    with torch.no_grad():
        base_prediction, _ = base.rollout(repeated, actions)
        wrapped_prediction, _ = wrapper.rollout(repeated, actions)
    maxima = []
    for key in base_prediction:
        maxima.append(float((base_prediction[key] - wrapped_prediction[key]).abs().max().item()))
    action_max = float((actions - wrapper.effective_normalized_actions(actions)).abs().max().item())
    return {"action_max_abs": action_max, "rollout_max_abs": max(maxima, default=0.0)}


def resource_snapshot(device: torch.device) -> Dict:
    payload = {"time_unix": time.time()}
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        payload.update(
            {
                "max_rss_kb": int(usage.ru_maxrss),
                "user_cpu_s": float(usage.ru_utime),
                "system_cpu_s": float(usage.ru_stime),
            }
        )
    if device.type == "cuda":
        payload.update(
            {
                "cuda_name": torch.cuda.get_device_name(device),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    return payload


def completed_keys(raw_path: Path) -> set:
    return {
        (row["candidate"], float(row["factor_deg"]), int(row["segment_index"]))
        for row in read_jsonl(raw_path)
        if row.get("record_type") == "paired_scenario"
    }


def run_candidate(args, candidate: str, base, wrapper, preprocessor, segments, env, manifest):
    raw_path = args.output_dir / f"{candidate}_raw.jsonl"
    existing = completed_keys(raw_path)
    selections = select_scenarios(segments, candidate)
    window = CANDIDATES[candidate]["window"]
    for selection in selections:
        key = (candidate, selection["factor_deg"], selection["segment_index"])
        if key in existing:
            print(f"SKIP completed {key}", flush=True)
            continue
        started = time.perf_counter()
        segment = segments[selection["segment_index"]]
        initial_state, goal_obs, nominal_states, nominal_actions = prepare_target(
            env, segment, selection["env_seed"]
        )
        start_obs, _ = env.prepare(selection["env_seed"], initial_state)
        if not manifest.get("identity_audit"):
            manifest["identity_audit"] = identity_audit(base, wrapper, preprocessor, start_obs)
            dump_json(args.output_dir / "manifest.json", manifest)

        prior_commands, prior_trace = plan_cem(
            wrapper,
            preprocessor,
            start_obs,
            goal_obs,
            POPULATION_PRIOR_DEG,
            selection["cem_seed"],
        )
        oracle_commands, oracle_trace = plan_cem(
            wrapper,
            preprocessor,
            start_obs,
            goal_obs,
            selection["factor_deg"],
            selection["cem_seed"],
        )
        prior_states, prior_effective, prior_contacts, prior_coverage = execute(
            env, initial_state, selection["env_seed"], prior_commands, selection["factor_deg"]
        )
        oracle_states, oracle_effective, oracle_contacts, oracle_coverage = execute(
            env, initial_state, selection["env_seed"], oracle_commands, selection["factor_deg"]
        )
        goal_state = nominal_states[-1]
        prior_metrics = pose_metrics(prior_states, goal_state, window)
        oracle_metrics = pose_metrics(oracle_states, goal_state, window)
        plan_l2 = float(np.linalg.norm(prior_commands - oracle_commands))
        row = {
            "record_type": "paired_scenario",
            "design_id": DESIGN_ID,
            **selection,
            "window": window,
            "goal_state": goal_state,
            "initial_state": initial_state,
            "nominal_actions": nominal_actions,
            "nominal_states": nominal_states,
            "prior": {
                "commands": prior_commands,
                "effective_actions": prior_effective,
                "states": prior_states,
                "contacts": prior_contacts,
                "coverages": prior_coverage,
                "metrics": prior_metrics,
                "planner": prior_trace,
            },
            "oracle": {
                "commands": oracle_commands,
                "effective_actions": oracle_effective,
                "states": oracle_states,
                "contacts": oracle_contacts,
                "coverages": oracle_coverage,
                "metrics": oracle_metrics,
                "planner": oracle_trace,
            },
            "plan_command_l2": plan_l2,
            "elapsed_s": time.perf_counter() - started,
            "resource": resource_snapshot(next(wrapper.parameters()).device),
        }
        append_jsonl(raw_path, row)
        metric = f"pose_auc{window}"
        print(
            f"DONE {candidate} {selection['ordinal'] + 1}/32 factor={selection['factor_deg']:+g} "
            f"segment={selection['segment_index']} prior={prior_metrics[metric]:.4f} "
            f"oracle={oracle_metrics[metric]:.4f} l2={plan_l2:.4f} elapsed={row['elapsed_s']:.1f}s",
            flush=True,
        )
    return raw_path


def bootstrap_ci(deltas: np.ndarray, seed: int = 6401, n: int = 20000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_raw(raw_path: Path, identity: Dict) -> Dict:
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "paired_scenario"]
    if not rows:
        raise RuntimeError(f"no rows in {raw_path}")
    windows = {int(row["window"]) for row in rows}
    if len(windows) != 1:
        raise RuntimeError(f"mixed windows: {windows}")
    window = windows.pop()
    metric = f"pose_auc{window}"
    prior = np.asarray([row["prior"]["metrics"][metric] for row in rows], dtype=np.float64)
    oracle = np.asarray([row["oracle"]["metrics"][metric] for row in rows], dtype=np.float64)
    delta = prior - oracle
    ci = bootstrap_ci(delta)
    plan_changed = float(np.mean([float(row["plan_command_l2"]) > 1e-6 for row in rows]))
    result = {
        "candidate": rows[0]["candidate"],
        "n_pairs": len(rows),
        "primary_metric": metric,
        "prior_mean": float(prior.mean()),
        "oracle_mean": float(oracle.mean()),
        "mean_delta": float(delta.mean()),
        "relative_improvement": float(delta.mean() / prior.mean()),
        "bootstrap_ci95_delta": list(ci),
        "direction_fraction": float(np.mean(delta > 0.0)),
        "plan_changed_fraction": plan_changed,
        "identity_audit": identity,
        "by_factor": {},
    }
    for factor in DEV_FACTORS_DEG:
        indexes = [i for i, row in enumerate(rows) if float(row["factor_deg"]) == factor]
        result["by_factor"][str(factor)] = {
            "n": len(indexes),
            "prior_mean": float(prior[indexes].mean()),
            "oracle_mean": float(oracle[indexes].mean()),
            "mean_delta": float(delta[indexes].mean()),
            "direction_fraction": float(np.mean(delta[indexes] > 0.0)),
        }
    validity_audits = {
        "complete_32_pairs": len(rows) == 32,
        "identity": max(identity.get("action_max_abs", math.inf), identity.get("rollout_max_abs", math.inf))
        <= 1e-6,
        "intervention_reached_planner": plan_changed > 0.0,
    }
    result["validity_audits"] = validity_audits
    result["valid"] = all(validity_audits.values())
    result["descriptive_assessment"] = {
        "mean_effect_direction": "positive" if delta.mean() > 0 else "negative" if delta.mean() < 0 else "zero",
        "ci_relation_to_zero": "above_zero" if ci[0] > 0 else "below_zero" if ci[1] < 0 else "includes_zero",
        "majority_of_pairs_positive": bool(np.mean(delta > 0.0) > 0.5),
    }
    return result


def make_manifest(args, checkpoint: Path, data_path: Path, design_path: Path, device) -> Dict:
    design_hash = sha256(design_path)
    if design_hash != EXPECTED_DESIGN_SHA256:
        raise RuntimeError(
            f"design hash mismatch: expected {EXPECTED_DESIGN_SHA256}, got {design_hash}"
        )
    return {
        "design_id": DESIGN_ID,
        "git_revision": git_revision(),
        "design_path": str(design_path),
        "design_sha256": design_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "command": " ".join(os.sys.argv),
        "started_unix": time.time(),
        "resource_start": resource_snapshot(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "sequence", "summarize"))
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), default="A_released")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("docs/research/persistent_context_v2_pushobj_rotation_stage0_design.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repro_outputs/persistent_context_v2_pushobj_rotation_stage0"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.data.open("rb") as handle:
        data = pickle.load(handle)
    segments = data["segments"]
    selections = {name: select_scenarios(segments, name) for name in CANDIDATES}
    if args.mode == "inspect":
        payload = {
            "design_sha256": sha256(args.design),
            "n_segments": len(segments),
            "early_contact_pool_size": len(early_contact_pool(segments)),
            "selections": selections,
        }
        dump_json(args.output_dir / "selection_audit.json", payload)
        print(json.dumps(payload, indent=2))
        return

    if args.mode == "summarize":
        manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        result = summarize_raw(
            args.output_dir / f"{args.candidate}_raw.jsonl", manifest["identity_audit"]
        )
        dump_json(args.output_dir / f"{args.candidate}_runner_summary.json", result)
        print(json.dumps(result, indent=2))
        return

    seed_all(9001)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but unavailable")
    device = torch.device(args.device)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["design_sha256"] != EXPECTED_DESIGN_SHA256:
            raise RuntimeError("existing manifest has a different design hash")
    else:
        manifest = make_manifest(args, args.checkpoint, args.data, args.design, device)
        dump_json(manifest_path, manifest)
    base, wrapper, _ = load_world_model(args.checkpoint, device)
    preprocessor = make_preprocessor()
    env = make_env()

    candidates = [args.candidate] if args.mode == "run" else ["A_released", "B_early_contact"]
    assessments = {}
    for candidate in candidates:
        raw_path = run_candidate(
            args, candidate, base, wrapper, preprocessor, segments, env, manifest
        )
        result = summarize_raw(raw_path, manifest["identity_audit"])
        dump_json(args.output_dir / f"{candidate}_runner_summary.json", result)
        assessments[candidate] = result["descriptive_assessment"]
        print(json.dumps(result, indent=2), flush=True)
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(device)
    manifest["runner_assessments"] = assessments
    dump_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
