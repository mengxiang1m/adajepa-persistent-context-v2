"""Frozen Stage-0 parent gate for persistent actuator lag on PointMaze.

This module may only compare a population-prior context with the directly
observed true factor.  It does not contain or execute a history estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from .pointmaze_transfer import (
    ACTION_DIM,
    FRAMESKIP,
    HARD_CELLS,
    DataStats,
    append_jsonl,
    bootstrap_ci,
    create_env,
    dump_json,
    git_revision,
    hard_start_goal,
    load_data_arrays,
    make_preprocessor,
    plan_cem,
    sha256,
)


CONTRACT_SHA256 = "ff173c47bf78746b01adce8dea0779e526ca548522823be94f4aafe75c45de2f"
DEV_FACTORS = (0.10, 0.30, 0.50, 0.70)
POPULATION_PRIOR = 0.40


class LagWorldModel(torch.nn.Module):
    """Map commanded action sequences to lag-filtered executed actions."""

    def __init__(self, base, stats: DataStats):
        super().__init__()
        self.base = base
        self.register_buffer("action_mean", stats.action_mean.float())
        self.register_buffer("action_std", stats.action_std.float())
        self.context = POPULATION_PRIOR

    def set_context(self, context: float) -> None:
        self.context = float(context)

    def encode_obs(self, obs):
        return self.base.encode_obs(obs)

    def lag_actions(self, action: torch.Tensor) -> torch.Tensor:
        original_shape = action.shape
        command = action.reshape(*original_shape[:-1], FRAMESKIP, ACTION_DIM)
        command = (command * self.action_std + self.action_mean).clamp(-1.0, 1.0)
        batch_shape = command.shape[:-3]
        flat = command.reshape(*batch_shape, -1, ACTION_DIM)
        previous = torch.zeros(*batch_shape, ACTION_DIM, dtype=flat.dtype, device=flat.device)
        executed = []
        rho = float(self.context)
        for index in range(flat.shape[-2]):
            previous = rho * previous + (1.0 - rho) * flat[..., index, :]
            executed.append(previous)
        filtered = torch.stack(executed, dim=-2).reshape(command.shape)
        normalized = (filtered - self.action_mean) / self.action_std
        return normalized.reshape(original_shape)

    def rollout(self, obs_0, act):
        return self.base.rollout(obs_0, self.lag_actions(act))


def load_lag_world_model(checkpoint_dir: Path, stats: DataStats, device: torch.device):
    from plan import load_model

    cfg = OmegaConf.load(checkpoint_dir / "hydra.yaml")
    checkpoint = checkpoint_dir / "checkpoints" / "model_latest.pth"
    base = load_model(checkpoint, cfg, int(cfg.num_action_repeat), device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return LagWorldModel(base, stats).to(device).eval(), checkpoint


def execute_lag(env, seed: int, init_state: np.ndarray, commands: np.ndarray, rho: float):
    obs, state = env.prepare(seed, init_state)
    env.return_value = "state"
    states = [np.asarray(state, dtype=np.float32)]
    executed = []
    previous = np.zeros(ACTION_DIM, dtype=np.float32)
    for command in np.clip(commands, -1.0, 1.0):
        previous = float(rho) * previous + (1.0 - float(rho)) * command
        _, _, _, info = env.step(previous)
        executed.append(previous.copy())
        states.append(np.asarray(info["state"], dtype=np.float32))
    return obs, np.stack(states), np.stack(executed)


def lag_metrics(states: np.ndarray, goal: np.ndarray) -> Dict[str, float]:
    distance = np.linalg.norm(states[:, :2] - goal[None, :2], axis=1)
    metric = {
        "initial_pos_dist": float(distance[0]),
        "cost_k5": float(distance[min(5, len(distance) - 1)]),
        "cost_k10": float(distance[min(10, len(distance) - 1)]),
        "cost_k25": float(distance[min(25, len(distance) - 1)]),
        "auc_k5": float(np.mean(distance[: min(6, len(distance))])),
        "auc_k10": float(np.mean(distance[: min(11, len(distance))])),
        "auc_k25": float(np.mean(distance[: min(26, len(distance))])),
        "success_k25": float(distance[min(25, len(distance) - 1)] < 0.5),
    }
    return metric


def _adjacency() -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    cells = {tuple(map(int, cell)) for cell in HARD_CELLS.tolist()}
    return {
        cell: [
            neighbour
            for neighbour in (
                (cell[0] + 1, cell[1]),
                (cell[0] - 1, cell[1]),
                (cell[0], cell[1] + 1),
                (cell[0], cell[1] - 1),
            )
            if neighbour in cells
        ]
        for cell in cells
    }


def _graph_distance(graph, source, target) -> int:
    frontier = [(source, 0)]
    seen = {source}
    while frontier:
        node, distance = frontier.pop(0)
        if node == target:
            return distance
        for neighbour in graph[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                frontier.append((neighbour, distance + 1))
    raise ValueError("disconnected free cells")


def local_waypoint_start_goal(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    graph = _adjacency()
    pairs = [
        (source, target)
        for source in sorted(graph)
        for target in sorted(graph)
        if _graph_distance(graph, source, target) == 2
    ]
    source, target = pairs[int(rng.randint(len(pairs)))]
    start_pos = np.asarray(source, dtype=np.float32) + rng.uniform(-0.1, 0.1, 2).astype(np.float32)
    goal_pos = np.asarray(target, dtype=np.float32) + rng.uniform(-0.1, 0.1, 2).astype(np.float32)
    start = np.concatenate([start_pos, np.zeros(2, dtype=np.float32)])
    goal = np.concatenate([goal_pos, np.zeros(2, dtype=np.float32)])
    return start, goal


def read_contract(path: Path) -> Dict:
    actual = sha256(path)
    if actual != CONTRACT_SHA256:
        raise RuntimeError(f"frozen contract hash mismatch: {actual}")
    payload = json.loads(path.read_text())
    if payload["status"] != "FROZEN_BEFORE_DEVELOPMENT_RESULTS":
        raise RuntimeError("contract is not frozen")
    return payload


def candidate_config(contract: Dict, candidate_id: str) -> Dict:
    for candidate in contract["candidates"]:
        if candidate["id"] == candidate_id:
            return candidate
    raise ValueError(candidate_id)


def run_candidate(args) -> None:
    contract_path = Path(args.contract)
    contract = read_contract(contract_path)
    candidate = candidate_config(contract, args.candidate)
    output_dir = Path(args.output_dir) / args.candidate
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    _, _, _, stats = load_data_arrays(Path(args.data_dir))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wm, checkpoint = load_lag_world_model(Path(args.checkpoint_dir), stats, device)
    preprocessor = make_preprocessor(stats)
    env = create_env()
    started = time.time()
    cem = contract["cem"]
    for factor_index, factor in enumerate(DEV_FACTORS):
        for local_index in range(int(candidate["scenarios_per_factor"])):
            scenario_seed = (
                int(candidate["scenario_seed_base"])
                + int(candidate["scenario_seed_stride"]) * factor_index
                + local_index
            )
            if args.candidate == "A_hard_goal":
                init_state, goal_state = hard_start_goal(scenario_seed)
            else:
                init_state, goal_state = local_waypoint_start_goal(scenario_seed)
            obs_0, _ = env.prepare(scenario_seed, init_state)
            obs_g, _ = env.prepare(scenario_seed, goal_state)
            plan_seed = args.cem_seed + 1000 * factor_index + local_index
            for policy in ("population_prior", "true_context"):
                context = POPULATION_PRIOR if policy == "population_prior" else factor
                commands, trace = plan_cem(
                    wm,
                    preprocessor,
                    obs_0,
                    obs_g,
                    context,
                    plan_seed,
                    int(cem["num_samples"]),
                    int(cem["topk"]),
                    int(cem["opt_steps"]),
                    int(cem["model_horizon"]),
                )
                _, states, executed = execute_lag(
                    env, scenario_seed, init_state, commands, factor
                )
                append_jsonl(
                    raw_path,
                    {
                        "contract_sha256": CONTRACT_SHA256,
                        "candidate": args.candidate,
                        "factor_index": factor_index,
                        "factor_audit": factor,
                        "factor_hash": hashlib.sha256(str(factor).encode()).hexdigest(),
                        "local_index": local_index,
                        "scenario_seed": scenario_seed,
                        "plan_seed": plan_seed,
                        "policy": policy,
                        "context": context,
                        "init_state": init_state,
                        "goal_state": goal_state,
                        "commands": commands,
                        "executed_actions": executed,
                        "states": states,
                        "metrics": lag_metrics(states, goal_state),
                        "plan": trace,
                    },
                )
    manifest = {
        "candidate": args.candidate,
        "contract_sha256": CONTRACT_SHA256,
        "code_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(checkpoint),
        "git_revision": git_revision(),
        "elapsed_after_setup_s": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    dump_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def audit_candidate(args) -> Dict:
    contract = read_contract(Path(args.contract))
    candidate = candidate_config(contract, args.candidate)
    output_dir = Path(args.output_dir) / args.candidate
    raw_path = output_dir / "raw.jsonl"
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    pairs = {}
    for record in records:
        identity = (record["factor_index"], record["local_index"])
        pairs.setdefault(identity, {})[record["policy"]] = record
    expected = len(DEV_FACTORS) * int(candidate["scenarios_per_factor"])
    identity_ok = len(pairs) == expected and all(
        set(pair) == {"population_prior", "true_context"} for pair in pairs.values()
    )
    paired_nuisance = all(
        pair["population_prior"]["scenario_seed"] == pair["true_context"]["scenario_seed"]
        and pair["population_prior"]["plan_seed"] == pair["true_context"]["plan_seed"]
        and pair["population_prior"]["init_state"] == pair["true_context"]["init_state"]
        and pair["population_prior"]["goal_state"] == pair["true_context"]["goal_state"]
        for pair in pairs.values()
    )
    metrics = ("auc_k5", "auc_k10", "auc_k25", "cost_k5", "cost_k10", "cost_k25")
    results = {}
    for metric in metrics:
        population = np.asarray(
            [pair["population_prior"]["metrics"][metric] for pair in pairs.values()]
        )
        oracle = np.asarray([pair["true_context"]["metrics"][metric] for pair in pairs.values()])
        effect = population - oracle
        results[metric] = {
            "population_mean": float(population.mean()),
            "true_context_mean": float(oracle.mean()),
            "pop_minus_true_mean": float(effect.mean()),
            "ci95": bootstrap_ci(
                effect,
                seed=int(contract["bootstrap"]["seed"]),
                draws=int(contract["bootstrap"]["draws"]),
            ),
            "relative_improvement": float(effect.mean() / population.mean()),
            "direction_fraction": float(np.mean(effect > 0)),
        }
    action_diff = np.asarray(
        [
            np.mean(
                np.abs(
                    np.asarray(pair["population_prior"]["commands"])
                    - np.asarray(pair["true_context"]["commands"])
                )
            )
            for pair in pairs.values()
        ]
    )
    primary = candidate["primary_metric"]
    primary_result = results[primary]
    gates = {
        "identity_complete": bool(identity_ok),
        "paired_nuisance_exact": bool(paired_nuisance),
        "action_plan_change_fraction": float(np.mean(action_diff > 1e-6))
        >= float(contract["go_gates"]["action_plan_change_fraction"]),
        "paired_ci_lower_positive": primary_result["ci95"][0] > 0,
        "relative_improvement": primary_result["relative_improvement"]
        >= float(contract["go_gates"]["relative_improvement_min"]),
        "direction_fraction": primary_result["direction_fraction"]
        >= float(contract["go_gates"]["direction_fraction_min"]),
    }
    decision = "GO" if all(gates.values()) else "NO_GO"
    payload = {
        "candidate": args.candidate,
        "decision": decision,
        "primary_metric": primary,
        "n_pairs": len(pairs),
        "expected_pairs": expected,
        "metrics": results,
        "mean_abs_command_difference": float(action_diff.mean()),
        "gates": gates,
        "raw_sha256": sha256(raw_path),
        "contract_sha256": CONTRACT_SHA256,
    }
    dump_json(output_dir / "audit_summary.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def finalize_stage0(args) -> Dict:
    contract_path = Path(args.contract)
    read_contract(contract_path)
    output_dir = Path(args.output_dir)
    summaries = {}
    for candidate in ("A_hard_goal", "B_local_waypoint"):
        path = output_dir / candidate / "audit_summary.json"
        if path.exists():
            summaries[candidate] = json.loads(path.read_text())
    if "A_hard_goal" not in summaries:
        raise RuntimeError("candidate A audit is required")
    if summaries["A_hard_goal"]["decision"] == "GO":
        decision = "GO_FREEZE_POINTMAZE_LAG_HISTORY_CONTRACT"
    elif "B_local_waypoint" not in summaries:
        raise RuntimeError("candidate B audit is required after candidate A NO-GO")
    elif summaries["B_local_waypoint"]["decision"] == "GO":
        decision = "GO_FREEZE_POINTMAZE_LAG_HISTORY_CONTRACT"
    else:
        decision = "NO_GO_POINTMAZE_ACTUATOR_LAG_TASK"
    payload = {
        "decision": decision,
        "contract_id": "persistent-context-v2-pointmaze-actuator-lag-stage0-v1",
        "contract_sha256": CONTRACT_SHA256,
        "candidate_decisions": {
            key: {
                "decision": value["decision"],
                "primary_metric": value["primary_metric"],
                "n_pairs": value["n_pairs"],
                "primary_result": value["metrics"][value["primary_metric"]],
                "raw_sha256": value["raw_sha256"],
                "gates": value["gates"],
            }
            for key, value in summaries.items()
        },
        "formal_history_contract_authorized": decision.startswith("GO_"),
        "history_rls_executed": False,
        "stop_reason": (
            None
            if decision.startswith("GO_")
            else "Both preregistered true-factor behavior upper-bound candidates failed."
        ),
    }
    dump_json(output_dir / "stage0_decision.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "audit", "finalize"])
    parser.add_argument("--candidate", choices=["A_hard_goal", "B_local_waypoint"])
    parser.add_argument(
        "--contract", default="docs/research/persistent_context_v2_pointmaze_lag_stage0_design.json"
    )
    parser.add_argument("--data-dir", default="data/point_maze_medium")
    parser.add_argument(
        "--checkpoint-dir",
        default="/home/zhaoqing/adajepa/checkpoints/mediummaze_dynamics_shift",
    )
    parser.add_argument(
        "--output-dir", default="persistent_context_v2_outputs/pointmaze_lag_stage0_v1"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cem-seed", type=int, default=310000)
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "run":
        if args.candidate is None:
            raise ValueError("run requires --candidate")
        run_candidate(args)
    elif args.mode == "audit":
        if args.candidate is None:
            raise ValueError("audit requires --candidate")
        audit_candidate(args)
    else:
        finalize_stage0(args)


if __name__ == "__main__":
    main()
