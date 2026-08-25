"""Persistent action-calibration context on the released PointMaze AdaJEPA model.

The module deliberately lives outside the production planner.  It exposes one
physical intervention only: a scalar command-to-executed-action gain.  The
released world model is frozen; a tiny FiLM action adapter converts commanded
actions and an explicit scalar context into the nominal executed-action space
understood by the checkpoint.

Modes are intentionally separated so development calibration cannot inspect
formal history results:

* ``audit``: fit/check the nominal state-transition regression.
* ``train-adapter``: supervised factor-diverse FiLM training.
* ``dev``: population-prior versus analytic true-context task calibration.
* ``audit-dev``: independent Stage-0 gate recomputation from raw artifacts.
* ``formal``: frozen paired persistent/no-persistence evaluation (guarded by a
  separately frozen contract; not authorized when the Stage-0 gate fails).
* ``audit-formal``: independent recomputation from raw JSONL artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf


TRAIN_FACTORS = (0.50, 0.75, 1.00, 1.25, 1.50)
DEV_FACTORS = (0.60, 0.90, 1.10, 1.40)
FORMAL_FACTORS = (0.65, 0.85, 1.15, 1.35)
POPULATION_PRIOR = float(np.mean(TRAIN_FACTORS))
FRAMESKIP = 5
ACTION_DIM = 2

HARD_CELLS = np.asarray(
    [
        (1, 1), (1, 2), (1, 5), (1, 6),
        (2, 1), (2, 2), (2, 4), (2, 5), (2, 6),
        (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2), (4, 4), (4, 5), (4, 6),
        (5, 1), (5, 3), (5, 4), (5, 6),
        (6, 1), (6, 2), (6, 3), (6, 5), (6, 6),
    ],
    dtype=np.float32,
)
QVEL_RANGE = np.asarray([[-5.2262554, 5.2262554]] * 2, dtype=np.float32)


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def seed_all(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def masked_mean_std(data: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = [data[i, : int(lengths[i])] for i in range(len(lengths))]
    flat = torch.cat(rows, dim=0).float()
    return flat.mean(0), flat.std(0)


@dataclass
class DataStats:
    action_mean: torch.Tensor
    action_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor


def load_data_arrays(data_dir: Path):
    actions = torch.load(data_dir / "actions.pth", map_location="cpu").float()
    states = torch.load(data_dir / "states.pth", map_location="cpu").float()
    lengths = torch.load(data_dir / "seq_lengths.pth", map_location="cpu").long()
    action_mean, action_std = masked_mean_std(actions, lengths)
    state_mean, state_std = masked_mean_std(states, lengths)
    return actions, states, lengths, DataStats(action_mean, action_std, state_mean, state_std)


class ActionFiLM(torch.nn.Module):
    """Six-parameter FiLM: per-axis affine gain/bias generated from scalar z."""

    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.Linear(1, ACTION_DIM)
        self.beta = torch.nn.Linear(1, ACTION_DIM)

    def forward_preclip(self, command: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(context, dtype=command.dtype, device=command.device)
        while z.ndim < command.ndim:
            z = z.unsqueeze(-1)
        if z.shape[-1] != 1:
            z = z.unsqueeze(-1)
        gamma = self.gamma(z)
        beta = self.beta(z)
        return gamma * command + beta

    def forward(self, command: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.forward_preclip(command.clamp(-1.0, 1.0), context).clamp(-1.0, 1.0)


class AnalyticCalibration(torch.nn.Module):
    def forward(self, command: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(context, dtype=command.dtype, device=command.device)
        while z.ndim < command.ndim:
            z = z.unsqueeze(-1)
        return (command.clamp(-1.0, 1.0) * z).clamp(-1.0, 1.0)


class CalibratedWorldModel(torch.nn.Module):
    """Frozen AdaJEPA with an explicit, auditable action-context interface."""

    def __init__(self, base, adapter, stats: DataStats):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.register_buffer("action_mean", stats.action_mean.float())
        self.register_buffer("action_std", stats.action_std.float())
        self.context = POPULATION_PRIOR

    def set_context(self, context: float) -> None:
        self.context = float(context)

    def encode_obs(self, obs):
        return self.base.encode_obs(obs)

    def calibrated_actions(self, action: torch.Tensor) -> torch.Tensor:
        shape = action.shape
        physical = action.reshape(*shape[:-1], FRAMESKIP, ACTION_DIM)
        physical = physical * self.action_std + self.action_mean
        effective = self.adapter(physical, torch.as_tensor(self.context, device=action.device))
        normalized = (effective - self.action_mean) / self.action_std
        return normalized.reshape(shape)

    def rollout(self, obs_0, act):
        return self.base.rollout(obs_0, self.calibrated_actions(act))


class NullWandb:
    def log(self, *args, **kwargs):
        return None


def load_world_model(checkpoint_dir: Path, stats: DataStats, adapter, device: torch.device):
    from plan import load_model

    cfg = OmegaConf.load(checkpoint_dir / "hydra.yaml")
    checkpoint = checkpoint_dir / "checkpoints" / "model_latest.pth"
    base = load_model(checkpoint, cfg, int(cfg.num_action_repeat), device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    wrapper = CalibratedWorldModel(base, adapter.to(device), stats).to(device).eval()
    return wrapper, cfg, checkpoint


def make_preprocessor(stats: DataStats):
    from datasets.img_transforms import default_transform
    from preprocessor import Preprocessor

    return Preprocessor(
        action_mean=stats.action_mean,
        action_std=stats.action_std,
        state_mean=stats.state_mean,
        state_std=stats.state_std,
        proprio_mean=stats.state_mean,
        proprio_std=stats.state_std,
        transform=default_transform(224),
    )


def hard_start_goal(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(seed)
    rs.random()  # parity with the released hard-goal protocol
    while True:
        i, j = rs.choice(len(HARD_CELLS), size=2, replace=False)
        if np.linalg.norm(HARD_CELLS[i] - HARD_CELLS[j]) >= 3.0:
            break

    def sample(cell):
        qpos = cell + rs.uniform(-0.25, 0.25, size=2).astype(np.float32)
        qvel = np.asarray(
            [rs.uniform(*QVEL_RANGE[0]), rs.uniform(*QVEL_RANGE[1])], dtype=np.float32
        )
        return np.concatenate([qpos, qvel]).astype(np.float32)

    return sample(HARD_CELLS[i]), sample(HARD_CELLS[j])


def obs_batch(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.expand_dims(np.expand_dims(value, 0), 0) for key, value in obs.items()}


def plan_cem(
    wm: CalibratedWorldModel,
    preprocessor,
    obs_0: Dict[str, np.ndarray],
    obs_g: Dict[str, np.ndarray],
    context: float,
    rng_seed: int,
    num_samples: int,
    topk: int,
    opt_steps: int,
    horizon: int,
) -> Tuple[np.ndarray, Dict]:
    from planning.objectives import create_objective_fn
    from utils import move_to_device

    device = next(wm.parameters()).device
    wm.set_context(context)
    transformed_0 = move_to_device(preprocessor.transform_obs(obs_batch(obs_0)), device)
    transformed_g = move_to_device(preprocessor.transform_obs(obs_batch(obs_g)), device)
    with torch.no_grad():
        goal_z = wm.encode_obs(transformed_g)
    repeated_0 = {
        key: repeat(value, "1 ... -> n ...", n=num_samples) for key, value in transformed_0.items()
    }
    repeated_goal = {
        key: repeat(value, "1 ... -> n ...", n=num_samples) for key, value in goal_z.items()
    }
    objective = create_objective_fn(alpha=0, base=2, mode="all")
    generator = torch.Generator(device=device).manual_seed(int(rng_seed))
    mu = torch.zeros(horizon, FRAMESKIP * ACTION_DIM, device=device)
    sigma = torch.ones_like(mu)
    trace = []
    for iteration in range(opt_steps):
        actions = torch.randn(
            num_samples,
            horizon,
            FRAMESKIP * ACTION_DIM,
            generator=generator,
            device=device,
        ) * sigma + mu
        actions[0] = mu
        with torch.no_grad():
            pred, _ = wm.rollout(repeated_0, actions)
            loss = objective(pred, repeated_goal)
        elite_idx = torch.argsort(loss)[:topk]
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
    normalized = mu.detach().cpu().reshape(horizon * FRAMESKIP, ACTION_DIM)
    physical = preprocessor.denormalize_actions(normalized).numpy()
    physical = np.clip(physical, -1.0, 1.0)
    return physical.astype(np.float32), {"trace": trace, "context": float(context)}


def execute_calibrated(env, seed: int, init_state: np.ndarray, commands: np.ndarray, true_gain: float):
    obs, state = env.prepare(seed, init_state)
    env.return_value = "state"  # state logging after the one visual observation; avoids needless rendering
    states = [np.asarray(state, dtype=np.float32)]
    executed = np.clip(float(true_gain) * np.clip(commands, -1.0, 1.0), -1.0, 1.0)
    for action in executed:
        _, _, _, info = env.step(action)
        states.append(np.asarray(info["state"], dtype=np.float32))
    return obs, np.stack(states), executed.astype(np.float32)


def position_costs(states: np.ndarray, goal: np.ndarray) -> Dict[str, float]:
    dist = np.linalg.norm(states[:, :2] - goal[None, :2], axis=1)
    out = {
        "initial_pos_dist": float(dist[0]),
        "cost_k5": float(dist[min(5, len(dist) - 1)]),
        "cost_k10": float(dist[min(10, len(dist) - 1)]),
        "cost_k25": float(dist[min(25, len(dist) - 1)]),
        "auc_k5": float(np.mean(dist[: min(6, len(dist))])),
        "auc_k25": float(np.mean(dist[: min(26, len(dist))])),
    }
    out["success_k25"] = float(dist[min(25, len(dist) - 1)] < 0.5)
    return out


def fit_nominal_regression(states: torch.Tensor, actions: torch.Tensor, train_rollouts: int = 3600):
    s = states[:train_rollouts].numpy()
    a = actions[:train_rollouts].numpy()
    x = np.concatenate(
        [
            np.ones((s.shape[0] * (s.shape[1] - 1), 1), dtype=np.float32),
            s[:, :-1, 2:].reshape(-1, 2),
            a[:, :-1].reshape(-1, 2),
        ],
        axis=1,
    )
    y = s[:, 1:, 2:].reshape(-1, 2)
    weight, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ weight
    rmse = np.sqrt(np.mean(np.square(prediction - y), axis=0))
    r2 = 1.0 - np.sum(np.square(prediction - y), axis=0) / np.sum(
        np.square(y - y.mean(axis=0)), axis=0
    )
    return weight.astype(np.float64), {"rmse": rmse, "r2": r2, "n": len(x)}


@dataclass
class ScalarRLS:
    prior_mean: float = POPULATION_PRIOR
    prior_precision: float = 2.0
    numerator: float = 0.0
    denominator: float = 0.0
    transition_count: int = 0

    def __post_init__(self):
        if self.denominator == 0.0:
            self.denominator = float(self.prior_precision)
            self.numerator = float(self.prior_precision * self.prior_mean)

    @property
    def mean(self) -> float:
        return float(np.clip(self.numerator / self.denominator, 0.35, 1.65))

    def update(self, states: np.ndarray, commands: np.ndarray, regression: np.ndarray) -> None:
        for t, command in enumerate(commands):
            passive_features = np.concatenate([[1.0], states[t, 2:]])
            passive = passive_features @ regression[:3]
            x = np.asarray(command) @ regression[3:5]
            y = states[t + 1, 2:] - passive
            energy = float(x @ x)
            if energy < 1e-5:
                continue
            # Huber-like residual cap protects the scalar sufficient statistic at wall contacts.
            residual_projection = float(x @ y)
            cap = 3.0 * energy
            self.numerator += float(np.clip(residual_projection, -cap, cap))
            self.denominator += energy
            self.transition_count += 1

    def snapshot(self) -> Dict:
        return {
            "posterior_mean": self.mean,
            "posterior_precision": self.denominator,
            "numerator": self.numerator,
            "transition_count": self.transition_count,
        }


def train_adapter(args, actions: torch.Tensor, output_dir: Path) -> None:
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ActionFiLM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.adapter_lr)
    train_commands = actions[:3600].reshape(-1, ACTION_DIM)
    val_commands = actions[3600:].reshape(-1, ACTION_DIM)
    generator = torch.Generator().manual_seed(args.seed)
    records = []
    for step in range(args.adapter_steps):
        idx = torch.randint(len(train_commands), (args.adapter_batch,), generator=generator)
        command = train_commands[idx].to(device)
        factor_idx = torch.randint(len(TRAIN_FACTORS), (args.adapter_batch,), generator=generator)
        factor = torch.as_tensor(TRAIN_FACTORS, dtype=torch.float32)[factor_idx].to(device)
        target = factor.unsqueeze(-1) * command.clamp(-1.0, 1.0)
        prediction = model.forward_preclip(command.clamp(-1.0, 1.0), factor)
        loss = torch.mean(torch.square(prediction - target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, args.adapter_steps - 1} or (step + 1) % 100 == 0:
            records.append({"step": step + 1, "loss": float(loss.item())})

    def evaluate(factors: Sequence[float]):
        command = val_commands[:: max(1, len(val_commands) // 10000)].to(device)
        per_factor = {}
        with torch.no_grad():
            for factor in factors:
                z = torch.full((len(command),), float(factor), device=device)
                target = (z.unsqueeze(-1) * command.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)
                pred = model(command, z)
                per_factor[str(factor)] = float(torch.mean(torch.square(pred - target)).item())
        return per_factor

    ckpt = output_dir / "film_adapter.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "train_factors": TRAIN_FACTORS}, ckpt)
    summary = {
        "mode": "train-adapter",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_factors": TRAIN_FACTORS,
        "dev_factors": DEV_FACTORS,
        "formal_factors": FORMAL_FACTORS,
        "population_prior": POPULATION_PRIOR,
        "records": records,
        "train_mse": evaluate(TRAIN_FACTORS),
        "heldout_dev_mse": evaluate(DEV_FACTORS),
        "heldout_formal_mse": evaluate(FORMAL_FACTORS),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    dump_json(output_dir / "adapter_summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def load_adapter(path: Path, device: torch.device) -> ActionFiLM:
    adapter = ActionFiLM().to(device)
    payload = torch.load(path, map_location=device)
    adapter.load_state_dict(payload["state_dict"])
    return adapter.eval()


def create_env():
    import gym
    import env  # noqa: F401 - registers project environments

    # The project vector workers invoke custom ``prepare`` directly on the base
    # env.  Using Gym's TimeLimit wrapper here would leave OrderEnforcing's
    # private reset flag unset even though ``prepare`` has reset the simulator.
    return gym.make("point_maze_medium").unwrapped


def run_development(args, stats: DataStats, output_dir: Path) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    adapter = AnalyticCalibration()
    wm, _, checkpoint = load_world_model(Path(args.checkpoint_dir), stats, adapter, device)
    preprocessor = make_preprocessor(stats)
    env = create_env()
    raw_path = output_dir / "dev_raw.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    start = time.time()
    for factor_index, factor in enumerate(DEV_FACTORS):
        for sequence_index in range(args.dev_sequences_per_factor):
            scenario_seed = args.dev_seed + 1000 * factor_index + sequence_index
            init_state, goal_state = hard_start_goal(scenario_seed)
            obs_0, _ = env.prepare(scenario_seed, init_state)
            obs_g, _ = env.prepare(scenario_seed, goal_state)
            for policy in ("population_prior", "true_context"):
                context = POPULATION_PRIOR if policy == "population_prior" else factor
                plan_seed = args.cem_seed + 1000 * factor_index + sequence_index
                commands, plan_info = plan_cem(
                    wm, preprocessor, obs_0, obs_g, context, plan_seed,
                    args.num_samples, args.topk, args.opt_steps, args.horizon,
                )
                _, states, executed = execute_calibrated(
                    env, scenario_seed, init_state, commands, factor
                )
                append_jsonl(
                    raw_path,
                    {
                        "factor": factor,
                        "factor_index": factor_index,
                        "sequence_index": sequence_index,
                        "scenario_seed": scenario_seed,
                        "plan_seed": plan_seed,
                        "policy": policy,
                        "context": context,
                        "init_state": init_state,
                        "goal_state": goal_state,
                        "commands": commands,
                        "executed_actions": executed,
                        "states": states,
                        "metrics": position_costs(states, goal_state),
                        "plan": plan_info,
                    },
                )
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    summary = summarize_dev(records)
    summary.update(
        {
            "mode": "dev",
            "checkpoint_sha256": sha256(checkpoint),
            "script_git_revision": git_revision(),
            "elapsed_s": time.time() - start,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
    )
    dump_json(output_dir / "dev_summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def summarize_dev(records: List[Dict]) -> Dict:
    by_key = {(r["factor_index"], r["sequence_index"], r["policy"]): r for r in records}
    effects = {metric: [] for metric in ("cost_k5", "cost_k10", "cost_k25", "auc_k5", "auc_k25")}
    action_diffs = []
    for factor_index in range(len(DEV_FACTORS)):
        sequence_ids = sorted(
            {r["sequence_index"] for r in records if r["factor_index"] == factor_index}
        )
        for sequence_index in sequence_ids:
            pop = by_key[(factor_index, sequence_index, "population_prior")]
            true = by_key[(factor_index, sequence_index, "true_context")]
            for metric in effects:
                effects[metric].append(pop["metrics"][metric] - true["metrics"][metric])
            action_diffs.append(
                float(np.mean(np.abs(np.asarray(pop["commands"]) - np.asarray(true["commands"]))))
            )
    return {
        "n_pairs": len(action_diffs),
        "mean_population": {
            metric: float(np.mean([r["metrics"][metric] for r in records if r["policy"] == "population_prior"]))
            for metric in effects
        },
        "mean_true": {
            metric: float(np.mean([r["metrics"][metric] for r in records if r["policy"] == "true_context"]))
            for metric in effects
        },
        "paired_pop_minus_true": {
            metric: {"mean": float(np.mean(values)), "median": float(np.median(values))}
            for metric, values in effects.items()
        },
        "mean_abs_command_difference": float(np.mean(action_diffs)),
        "command_difference_nonzero_fraction": float(np.mean(np.asarray(action_diffs) > 1e-6)),
    }


def audit_development(args, output_dir: Path) -> None:
    """Recompute the real-benchmark parent gate without loading the model."""
    raw_path = output_dir / "dev_raw.jsonl"
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    pairs = {}
    for record in records:
        identity = (record["factor_index"], record["sequence_index"])
        pairs.setdefault(identity, {})[record["policy"]] = record
    expected_pairs = len(DEV_FACTORS) * int(args.dev_sequences_per_factor)
    identity_ok = len(pairs) == expected_pairs and all(
        set(value) == {"population_prior", "true_context"} for value in pairs.values()
    )
    paired_nuisance_ok = all(
        value["population_prior"]["scenario_seed"] == value["true_context"]["scenario_seed"]
        and value["population_prior"]["plan_seed"] == value["true_context"]["plan_seed"]
        and value["population_prior"]["init_state"] == value["true_context"]["init_state"]
        and value["population_prior"]["goal_state"] == value["true_context"]["goal_state"]
        for value in pairs.values()
    )
    metrics = ("cost_k5", "cost_k10", "cost_k25", "auc_k5", "auc_k25")
    recomputed = {}
    for metric in metrics:
        pop = np.asarray([value["population_prior"]["metrics"][metric] for value in pairs.values()])
        true = np.asarray([value["true_context"]["metrics"][metric] for value in pairs.values()])
        effect = pop - true
        recomputed[metric] = {
            "population_mean": float(pop.mean()),
            "true_context_mean": float(true.mean()),
            "pop_minus_true_mean": float(effect.mean()),
            "pop_minus_true_ci95": bootstrap_ci(effect, seed=4401),
            "relative_improvement": float(effect.mean() / pop.mean()),
            "direction_fraction": float(np.mean(effect > 0)),
        }
    command_differences = [
        float(
            np.mean(
                np.abs(
                    np.asarray(value["population_prior"]["commands"])
                    - np.asarray(value["true_context"]["commands"])
                )
            )
        )
        for value in pairs.values()
    ]
    primary = "auc_k25"
    primary_result = recomputed[primary]
    gates = {
        "identity_complete": bool(identity_ok),
        "paired_nuisance_exact": bool(paired_nuisance_ok),
        "context_changes_all_action_plans": bool(np.all(np.asarray(command_differences) > 1e-6)),
        "true_context_relative_improvement_at_least_25pct": primary_result["relative_improvement"] >= 0.25,
        "true_context_paired_ci_positive": primary_result["pop_minus_true_ci95"][0] > 0,
    }
    decision = (
        "GO_FREEZE_POINTMAZE_FORMAL_CONTRACT"
        if all(gates.values())
        else "NO_GO_POINTMAZE_ACTION_CALIBRATION_TASK"
    )
    payload = {
        "decision": decision,
        "stage": "real_pointmaze_stage0_task_qualification",
        "primary_metric": primary,
        "n_pairs": len(pairs),
        "expected_pairs": expected_pairs,
        "metrics": recomputed,
        "mean_abs_command_difference": float(np.mean(command_differences)),
        "gates": gates,
        "raw_sha256": sha256(raw_path),
        "formal_contract_authorized": decision.startswith("GO_"),
    }
    dump_json(output_dir / "dev_audit_summary.json", payload)
    print(json.dumps(payload, indent=2, default=_json_default))


def bootstrap_ci(values: Sequence[float], seed: int = 9041, draws: int = 20000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        means[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def policy_context(
    policy: str,
    true_factor: float,
    own_rls: ScalarRLS,
    sequence_index: int,
    episode_index: int,
    history_bank: Dict[int, ScalarRLS],
    shuffle_map: Dict[int, int],
) -> float:
    if policy in {"population_prior", "current_only"}:
        return POPULATION_PRIOR
    if policy == "true_context":
        return float(true_factor)
    if policy == "correct_history":
        return own_rls.mean
    if policy == "wrong_sequence":
        return history_bank[(sequence_index + 1) % len(history_bank)].mean
    if policy == "shuffled_history":
        return history_bank[shuffle_map[sequence_index]].mean
    raise ValueError(policy)


def run_formal(args, actions: torch.Tensor, states_data: torch.Tensor, stats: DataStats, output_dir: Path):
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "FROZEN_BEFORE_FORMAL":
        raise ValueError("formal run requires a frozen contract")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    adapter = load_adapter(Path(args.adapter), device)
    wm, _, checkpoint = load_world_model(Path(args.checkpoint_dir), stats, adapter, device)
    preprocessor = make_preprocessor(stats)
    regression, regression_metrics = fit_nominal_regression(states_data, actions)
    env = create_env()
    raw_path = output_dir / "formal_raw.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    policies = contract["policies"]
    n_sequences = int(contract["n_sequences"])
    n_episodes = int(contract["episodes_per_sequence"])
    factors = [FORMAL_FACTORS[i % len(FORMAL_FACTORS)] for i in range(n_sequences)]
    shuffle_rng = np.random.RandomState(int(contract["shuffle_seed"]))
    permutation = shuffle_rng.permutation(n_sequences).tolist()
    if any(i == permutation[i] for i in range(n_sequences)):
        permutation = permutation[1:] + permutation[:1]
    shuffle_map = {i: int(permutation[i]) for i in range(n_sequences)}
    start_time = time.time()
    for persistence in ("persistent", "no_persistence"):
        policy_histories = {
            policy: {i: ScalarRLS(prior_precision=float(contract["rls_prior_precision"])) for i in range(n_sequences)}
            for policy in policies
        }
        for episode_index in range(n_episodes):
            for sequence_index in range(n_sequences):
                base_factor = factors[sequence_index]
                if persistence == "persistent":
                    true_factor = base_factor
                else:
                    true_factor = FORMAL_FACTORS[(sequence_index + episode_index) % len(FORMAL_FACTORS)]
                scenario_seed = int(contract["scenario_seed"]) + 10000 * episode_index + sequence_index
                init_state, goal_state = hard_start_goal(scenario_seed)
                obs_0, _ = env.prepare(scenario_seed, init_state)
                obs_g, _ = env.prepare(scenario_seed, goal_state)
                for policy_index, policy in enumerate(policies):
                    own_rls = policy_histories[policy][sequence_index]
                    context = policy_context(
                        policy, true_factor, own_rls, sequence_index, episode_index,
                        policy_histories[policy], shuffle_map,
                    )
                    before = own_rls.snapshot()
                    plan_seed = int(contract["cem_seed"]) + 10000 * episode_index + sequence_index
                    commands, plan_info = plan_cem(
                        wm, preprocessor, obs_0, obs_g, context, plan_seed,
                        int(contract["cem"]["num_samples"]), int(contract["cem"]["topk"]),
                        int(contract["cem"]["opt_steps"]), int(contract["cem"]["horizon"]),
                    )
                    _, states, executed = execute_calibrated(
                        env, scenario_seed, init_state, commands, true_factor
                    )
                    own_rls.update(states, commands, regression)
                    append_jsonl(
                        raw_path,
                        {
                            "persistence": persistence,
                            "sequence_index": sequence_index,
                            "episode_index": episode_index,
                            "policy": policy,
                            "base_factor_hash": hashlib.sha256(str(base_factor).encode()).hexdigest(),
                            "true_factor_audit": true_factor,
                            "scenario_seed": scenario_seed,
                            "plan_seed": plan_seed,
                            "context_before": context,
                            "belief_before": before,
                            "belief_after": own_rls.snapshot(),
                            "init_state": init_state,
                            "goal_state": goal_state,
                            "commands": commands,
                            "executed_actions": executed,
                            "states": states,
                            "metrics": position_costs(states, goal_state),
                            "plan": plan_info,
                        },
                    )
    run_manifest = {
        "mode": "formal",
        "contract_sha256": sha256(contract_path),
        "adapter_sha256": sha256(Path(args.adapter)),
        "checkpoint_sha256": sha256(checkpoint),
        "code_sha256": sha256(Path(__file__)),
        "git_revision": git_revision(),
        "regression": regression,
        "regression_metrics": regression_metrics,
        "shuffle_map": shuffle_map,
        "elapsed_s": time.time() - start_time,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    dump_json(output_dir / "formal_run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, default=_json_default))


def audit_formal(args, output_dir: Path):
    contract = json.loads(Path(args.contract).read_text())
    raw_path = output_dir / "formal_raw.jsonl"
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    expected = 2 * int(contract["n_sequences"]) * int(contract["episodes_per_sequence"]) * len(contract["policies"])
    identities = len(records) == expected
    key = {
        (r["persistence"], r["sequence_index"], r["episode_index"], r["policy"]): r
        for r in records
    }
    episode1_equal = True
    for persistence in ("persistent", "no_persistence"):
        for seq in range(int(contract["n_sequences"])):
            a = key[(persistence, seq, 0, "current_only")]
            b = key[(persistence, seq, 0, "correct_history")]
            episode1_equal &= a["commands"] == b["commands"] and a["states"] == b["states"]

    metric = contract["primary_metric"]
    per_sequence = []
    for seq in range(int(contract["n_sequences"])):
        row = {"sequence_index": seq}
        for persistence in ("persistent", "no_persistence"):
            current = []
            history = []
            true = []
            for ep in range(1, int(contract["episodes_per_sequence"])):
                current.append(key[(persistence, seq, ep, "current_only")]["metrics"][metric])
                history.append(key[(persistence, seq, ep, "correct_history")]["metrics"][metric])
                true.append(key[(persistence, seq, ep, "true_context")]["metrics"][metric])
            row[f"delta_{persistence}"] = float(np.mean(current) - np.mean(history))
            row[f"oracle_gap_{persistence}"] = float(np.mean(current) - np.mean(true))
        row["did"] = row["delta_persistent"] - row["delta_no_persistence"]
        per_sequence.append(row)
    deltas = np.asarray([row["delta_persistent"] for row in per_sequence])
    dids = np.asarray([row["did"] for row in per_sequence])
    oracle_gaps = np.asarray([row["oracle_gap_persistent"] for row in per_sequence])
    recovered = float(np.mean(deltas) / np.mean(oracle_gaps)) if np.mean(oracle_gaps) > 0 else float("nan")

    controls = {}
    for policy in ("shuffled_history", "wrong_sequence"):
        effects = []
        for seq in range(int(contract["n_sequences"])):
            current = []
            control = []
            for ep in range(1, int(contract["episodes_per_sequence"])):
                current.append(key[("persistent", seq, ep, "current_only")]["metrics"][metric])
                control.append(key[("persistent", seq, ep, policy)]["metrics"][metric])
            effects.append(float(np.mean(current) - np.mean(control)))
        controls[policy] = {"mean_effect": float(np.mean(effects)), "ci95": bootstrap_ci(effects)}

    gates = {
        "identity_count": identities,
        "episode1_exact": bool(episode1_equal),
        "persistent_ci_positive": bootstrap_ci(deltas)[0] > 0,
        "did_ci_above_mde": bootstrap_ci(dids)[0] > float(contract["minimum_did"]),
        "recovery_at_least": recovered >= float(contract["minimum_oracle_recovery"]),
        "direction_fraction": float(np.mean(deltas > 0)) >= float(contract["minimum_direction_fraction"]),
        "shuffled_not_similar": controls["shuffled_history"]["mean_effect"] < float(contract["control_max_fraction"]) * float(np.mean(deltas)),
        "wrong_not_similar": controls["wrong_sequence"]["mean_effect"] < float(contract["control_max_fraction"]) * float(np.mean(deltas)),
    }
    decision = "GO_REAL_POINTMAZE_CONTEXT" if all(gates.values()) else "NO_GO_REAL_POINTMAZE_CONTEXT"
    summary = {
        "decision": decision,
        "primary_metric": metric,
        "n_raw_records": len(records),
        "expected_records": expected,
        "persistent_effect_mean": float(np.mean(deltas)),
        "persistent_effect_ci95": bootstrap_ci(deltas),
        "did_mean": float(np.mean(dids)),
        "did_ci95": bootstrap_ci(dids),
        "oracle_gap_mean": float(np.mean(oracle_gaps)),
        "oracle_recovery": recovered,
        "direction_fraction": float(np.mean(deltas > 0)),
        "controls": controls,
        "gates": gates,
        "per_sequence": per_sequence,
        "raw_sha256": sha256(raw_path),
    }
    dump_json(output_dir / "formal_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def run_audit(args, actions: torch.Tensor, states: torch.Tensor, output_dir: Path):
    regression, metrics = fit_nominal_regression(states, actions)
    payload = {
        "mode": "audit",
        "data_dir": str(Path(args.data_dir).resolve()),
        "actions_shape": list(actions.shape),
        "states_shape": list(states.shape),
        "train_rollouts": 3600,
        "heldout_rollouts": 400,
        "regression": regression,
        "metrics": metrics,
        "train_factors": TRAIN_FACTORS,
        "dev_factors": DEV_FACTORS,
        "formal_factors": FORMAL_FACTORS,
        "population_prior": POPULATION_PRIOR,
    }
    dump_json(output_dir / "asset_audit.json", payload)
    print(json.dumps(payload, indent=2, default=_json_default))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=["audit", "train-adapter", "dev", "audit-dev", "formal", "audit-formal"],
    )
    p.add_argument("--data-dir", default="data/point_maze_medium")
    p.add_argument("--checkpoint-dir", default="/home/zhaoqing/adajepa/checkpoints/mediummaze_dynamics_shift")
    p.add_argument("--output-dir", default="persistent_context_v2_outputs/pointmaze_transfer/v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=4301)
    p.add_argument("--adapter", default="persistent_context_v2_outputs/pointmaze_transfer/v1/film_adapter.pth")
    p.add_argument("--adapter-lr", type=float, default=0.02)
    p.add_argument("--adapter-steps", type=int, default=1000)
    p.add_argument("--adapter-batch", type=int, default=4096)
    p.add_argument("--dev-seed", type=int, default=70000)
    p.add_argument("--cem-seed", type=int, default=91000)
    p.add_argument("--dev-sequences-per-factor", type=int, default=3)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--opt-steps", type=int, default=4)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--contract", default="docs/research/persistent_context_v2_pointmaze_contract.json")
    return p


def main():
    args = parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions, states, _, stats = load_data_arrays(Path(args.data_dir))
    if args.mode == "audit":
        run_audit(args, actions, states, output_dir)
    elif args.mode == "train-adapter":
        train_adapter(args, actions, output_dir)
    elif args.mode == "dev":
        run_development(args, stats, output_dir)
    elif args.mode == "audit-dev":
        audit_development(args, output_dir)
    elif args.mode == "formal":
        run_formal(args, actions, states, stats, output_dir)
    elif args.mode == "audit-formal":
        audit_formal(args, output_dir)


if __name__ == "__main__":
    main()
