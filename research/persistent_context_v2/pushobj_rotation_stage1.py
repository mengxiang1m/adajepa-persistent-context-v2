"""Formal PushObj rotation history-value experiment.

The estimator consumes only commanded actions and observed agent proprioception.
Factor labels, effective actions, and contact labels are never passed to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from research.persistent_context_v2.pushobj_rotation_stage0 import (
    POPULATION_PRIOR_DEG,
    append_jsonl,
    dump_json,
    execute,
    git_revision,
    identity_audit,
    load_world_model,
    make_env,
    make_preprocessor,
    plan_cem,
    pose_metrics,
    prepare_target,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-pushobj-rotation-history-stage1-v1"
EXPECTED_DESIGN_SHA256 = "9950e4fb486178115907106451e1b81923a9c5068d625b4377f14820b65b7f3e"
FORMAL_FACTORS_DEG = (-25.0, -10.0, 10.0, 25.0)
POLICIES = (
    "population_prior",
    "current_only",
    "correct_history",
    "shuffled_history",
    "wrong_sequence_history",
    "true_factor_oracle",
)
CONDITIONS = ("persistent", "no_persistence")
N_SEQUENCES = 32
N_EPISODES = 4
WINDOW = 25
SELECTION_SEED = 610000
BOOTSTRAP_SEED = 7601
BOOTSTRAP_RESAMPLES = 20_000


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def wrapped_degrees_error(estimate: float, target: float) -> float:
    return float((float(estimate) - float(target) + 180.0) % 360.0 - 180.0)


def pd_coefficients(k_p: float = 100.0, k_v: float = 20.0, dt: float = 0.01, steps: int = 10):
    """Return p_final coefficients for initial position, velocity, and target."""

    def simulate(position: float, velocity: float, target: float) -> Tuple[float, float]:
        for _ in range(int(steps)):
            velocity += (k_p * (target - position) - k_v * velocity) * dt
            position += velocity * dt
        return position, velocity

    target_position, _ = simulate(0.0, 0.0, 1.0)
    initial_position, _ = simulate(1.0, 0.0, 0.0)
    initial_velocity, _ = simulate(0.0, 1.0, 0.0)
    return initial_position, initial_velocity, target_position


@dataclass
class RotationMLE:
    norm_residual_max: float = 0.002
    command_norm_min: float = 1e-4
    angle_clip_degrees: float = 35.0
    c_sum: float = 0.0
    s_sum: float = 0.0
    transition_count: int = 0
    accepted_count: int = 0

    def update(self, commands: np.ndarray, states: np.ndarray) -> None:
        """Update from commands and observed [agent pos, ..., agent velocity]."""

        commands = np.asarray(commands, dtype=np.float64)
        states = np.asarray(states, dtype=np.float64)
        if states.shape[0] != commands.shape[0] + 1 or states.shape[1] < 7:
            raise ValueError("commands/states do not form PushObj proprio transitions")
        p_coef, v_coef, target_coef = pd_coefficients()
        for index, command in enumerate(commands):
            self.transition_count += 1
            p0 = states[index, :2]
            p1 = states[index + 1, :2]
            v0 = states[index, 5:7]
            target_position = (p1 - p_coef * p0 - v_coef * v0) / target_coef
            inferred_effective = (target_position - p0) / 100.0
            command_norm = float(np.linalg.norm(command))
            residual = abs(float(np.linalg.norm(inferred_effective)) - command_norm)
            if command_norm < self.command_norm_min or residual > self.norm_residual_max:
                continue
            self.c_sum += float(np.dot(command, inferred_effective))
            self.s_sum += float(
                command[0] * inferred_effective[1] - command[1] * inferred_effective[0]
            )
            self.accepted_count += 1

    @property
    def estimate_degrees(self) -> float:
        if self.accepted_count == 0 or (self.c_sum == 0.0 and self.s_sum == 0.0):
            return float(POPULATION_PRIOR_DEG)
        degrees = math.degrees(math.atan2(self.s_sum, self.c_sum))
        return float(np.clip(degrees, -self.angle_clip_degrees, self.angle_clip_degrees))

    def as_dict(self) -> Dict[str, float]:
        return {
            "c_sum": self.c_sum,
            "s_sum": self.s_sum,
            "transition_count": self.transition_count,
            "accepted_count": self.accepted_count,
            "estimate_degrees": self.estimate_degrees,
        }


def formal_segment_indices() -> np.ndarray:
    rng = np.random.default_rng(SELECTION_SEED)
    return rng.permutation(np.arange(500, 1000, dtype=np.int64))[: N_SEQUENCES * N_EPISODES]


def factor_for(condition: str, sequence_id: int, episode_index: int) -> float:
    base = int(sequence_id) % len(FORMAL_FACTORS_DEG)
    if condition == "persistent":
        return FORMAL_FACTORS_DEG[base]
    if condition == "no_persistence":
        return FORMAL_FACTORS_DEG[(base + int(episode_index)) % len(FORMAL_FACTORS_DEG)]
    raise ValueError(condition)


def scenario(condition: str, sequence_id: int, episode_index: int) -> Dict:
    indices = formal_segment_indices().reshape(N_SEQUENCES, N_EPISODES)
    return {
        "condition": condition,
        "sequence_id": int(sequence_id),
        "episode": int(episode_index + 1),
        "episode_index": int(episode_index),
        "segment_index": int(indices[sequence_id, episode_index]),
        "factor_deg": factor_for(condition, sequence_id, episode_index),
        "env_seed": int(710_000 + sequence_id * 100 + episode_index),
        "cem_seed": int(720_000 + sequence_id * 100 + episode_index),
    }


def donor_sequence(policy: str, sequence_id: int, history_episode_index: int, n_sequences: int) -> int:
    if policy == "correct_history":
        return int(sequence_id)
    if policy == "wrong_sequence_history":
        return int((sequence_id + 1) % n_sequences)
    if policy == "shuffled_history":
        return int((sequence_id + 1 + history_episode_index) % n_sequences)
    raise ValueError(policy)


def build_estimator(
    evidence: Dict[Tuple[int, int], Dict],
    policy: str,
    sequence_id: int,
    episode_index: int,
    n_sequences: int,
) -> Tuple[RotationMLE, List[Dict]]:
    estimator = RotationMLE()
    donors = []
    for history_episode_index in range(episode_index):
        donor = donor_sequence(policy, sequence_id, history_episode_index, n_sequences)
        row = evidence[(donor, history_episode_index)]
        # Deliberately pass only commands and observed states to the estimator.
        estimator.update(np.asarray(row["commands"]), np.asarray(row["states"]))
        donors.append(
            {
                "history_episode": history_episode_index + 1,
                "donor_sequence_id": donor,
                "evidence_sha256": row["evidence_sha256"],
            }
        )
    return estimator, donors


def bootstrap_ci(values: np.ndarray, stream: int, resamples: int = BOOTSTRAP_RESAMPLES):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(stream))
    indexes = rng.integers(0, len(values), size=(int(resamples), len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def history_policy_payload(
    evidence: Dict[Tuple[int, int], Dict],
    policy: str,
    sequence_id: int,
    episode_index: int,
    factor_deg: float,
    n_sequences: int,
) -> Dict:
    if policy in ("population_prior", "current_only"):
        return {"context_degrees": 0.0, "estimator": None, "donors": []}
    if policy == "true_factor_oracle":
        return {"context_degrees": float(factor_deg), "estimator": None, "donors": []}
    estimator, donors = build_estimator(
        evidence, policy, sequence_id, episode_index, n_sequences
    )
    return {
        "context_degrees": estimator.estimate_degrees,
        "estimator": estimator.as_dict(),
        "donors": donors,
    }


def completed_keys(path: Path, record_type: str) -> set:
    rows = read_jsonl(path)
    return {
        (int(row["sequence_id"]), int(row["episode"]))
        for row in rows
        if row.get("record_type") == record_type
    }


def evidence_lookup(path: Path) -> Dict[Tuple[int, int], Dict]:
    result = {}
    for row in read_jsonl(path):
        if row.get("record_type") == "evidence_episode":
            result[(int(row["sequence_id"]), int(row["episode_index"]))] = row
    return result


def generate_evidence(args, condition, wrapper, preprocessor, segments, env, sequence_ids):
    raw_path = args.output_dir / f"{condition}_raw.jsonl"
    completed = completed_keys(raw_path, "evidence_episode")
    for sequence_id in sequence_ids:
        for episode_index in range(args.episodes):
            meta = scenario(condition, sequence_id, episode_index)
            key = (sequence_id, episode_index + 1)
            if key in completed:
                continue
            started = time.perf_counter()
            segment = segments[meta["segment_index"]]
            initial_state, goal_obs, nominal_states, nominal_actions = prepare_target(
                env, segment, meta["env_seed"]
            )
            start_obs, _ = env.prepare(meta["env_seed"], initial_state)
            commands, planner = plan_cem(
                wrapper, preprocessor, start_obs, goal_obs, 0.0, meta["cem_seed"]
            )
            states, effective, contacts, coverages = execute(
                env, initial_state, meta["env_seed"], commands, meta["factor_deg"]
            )
            row = {
                "record_type": "evidence_episode",
                "contract_id": CONTRACT_ID,
                **meta,
                "initial_state": initial_state,
                "goal_state": nominal_states[-1],
                "nominal_actions": nominal_actions,
                "commands": commands,
                "states": states,
                "effective_actions": effective,
                "contacts": contacts,
                "coverages": coverages,
                "planner": planner,
                "evidence_sha256": array_sha256(commands) + ":" + array_sha256(states),
                "elapsed_s": time.perf_counter() - started,
                "resource": resource_snapshot(next(wrapper.parameters()).device),
            }
            append_jsonl(raw_path, row)
            print(
                f"EVIDENCE {condition} s={sequence_id} e={episode_index+1} "
                f"factor={meta['factor_deg']:+g} elapsed={row['elapsed_s']:.2f}s",
                flush=True,
            )
    return evidence_lookup(raw_path)


def run_evaluations(args, condition, wrapper, preprocessor, segments, env, sequence_ids, evidence):
    raw_path = args.output_dir / f"{condition}_raw.jsonl"
    completed = completed_keys(raw_path, "evaluation_episode")
    for sequence_id in sequence_ids:
        for episode_index in range(args.episodes):
            meta = scenario(condition, sequence_id, episode_index)
            key = (sequence_id, episode_index + 1)
            if key in completed:
                continue
            started = time.perf_counter()
            segment = segments[meta["segment_index"]]
            initial_state, goal_obs, nominal_states, _ = prepare_target(
                env, segment, meta["env_seed"]
            )
            start_obs, _ = env.prepare(meta["env_seed"], initial_state)
            policy_context = {
                policy: history_policy_payload(
                    evidence,
                    policy,
                    sequence_id,
                    episode_index,
                    meta["factor_deg"],
                    args.sequences,
                )
                for policy in POLICIES
            }
            planned = {}
            policies = {}
            for policy in POLICIES:
                context = float(policy_context[policy]["context_degrees"])
                context_key = float(round(context, 10))
                if context_key not in planned:
                    commands, planner = plan_cem(
                        wrapper,
                        preprocessor,
                        start_obs,
                        goal_obs,
                        context,
                        meta["cem_seed"],
                    )
                    states, effective, contacts, coverages = execute(
                        env, initial_state, meta["env_seed"], commands, meta["factor_deg"]
                    )
                    planned[context_key] = {
                        "commands": commands,
                        "states": states,
                        "effective_actions": effective,
                        "contacts": contacts,
                        "coverages": coverages,
                        "metrics": pose_metrics(states, nominal_states[-1], WINDOW),
                        "planner": planner,
                        "command_sha256": array_sha256(commands),
                        "state_sha256": array_sha256(states),
                    }
                policies[policy] = {**policy_context[policy], **planned[context_key]}
            row = {
                "record_type": "evaluation_episode",
                "contract_id": CONTRACT_ID,
                **meta,
                "initial_state": initial_state,
                "goal_state": nominal_states[-1],
                "policies": policies,
                "elapsed_s": time.perf_counter() - started,
                "resource": resource_snapshot(next(wrapper.parameters()).device),
            }
            append_jsonl(raw_path, row)
            print(
                f"EVAL {condition} s={sequence_id} e={episode_index+1} "
                f"theta={meta['factor_deg']:+g} hist={policies['correct_history']['context_degrees']:+.3f} "
                f"current={policies['current_only']['metrics']['pose_auc25']:.4f} "
                f"history={policies['correct_history']['metrics']['pose_auc25']:.4f} "
                f"elapsed={row['elapsed_s']:.2f}s",
                flush=True,
            )


def load_evaluations(path: Path) -> List[Dict]:
    return [row for row in read_jsonl(path) if row.get("record_type") == "evaluation_episode"]


def sequence_policy_values(rows: Sequence[Dict], n_sequences: int, episodes: int) -> Dict[str, np.ndarray]:
    by_key = {(int(row["sequence_id"]), int(row["episode_index"])): row for row in rows}
    result = {policy: np.empty(n_sequences, dtype=np.float64) for policy in POLICIES}
    for sequence_id in range(n_sequences):
        later = [by_key[(sequence_id, ep)] for ep in range(1, episodes)]
        for policy in POLICIES:
            result[policy][sequence_id] = np.mean(
                [row["policies"][policy]["metrics"]["pose_auc25"] for row in later]
            )
    return result


def effect_payload(current: np.ndarray, treatment: np.ndarray, stream: int) -> Dict:
    delta = np.asarray(current) - np.asarray(treatment)
    tolerance = 1e-12
    return {
        "current_mean": float(np.mean(current)),
        "treatment_mean": float(np.mean(treatment)),
        "mean_delta": float(np.mean(delta)),
        "relative_improvement": float(np.mean(delta) / np.mean(current)),
        "bootstrap_ci95_delta": bootstrap_ci(delta, stream),
        "positive_fraction": float(np.mean(delta > tolerance)),
        "tie_fraction": float(np.mean(np.abs(delta) <= tolerance)),
        "negative_fraction": float(np.mean(delta < -tolerance)),
        "sequence_deltas": delta.tolist(),
    }


def structural_checks(condition: str, rows: List[Dict], n_sequences: int, episodes: int) -> Dict:
    expected = n_sequences * episodes
    keys = [(int(row["sequence_id"]), int(row["episode_index"])) for row in rows]
    episode_one_identity = True
    current_population_identity = True
    for row in rows:
        policies = row["policies"]
        current_population_identity &= (
            policies["current_only"]["command_sha256"] == policies["population_prior"]["command_sha256"]
            and policies["current_only"]["state_sha256"] == policies["population_prior"]["state_sha256"]
        )
        if int(row["episode_index"]) == 0:
            hashes = {
                (policies[p]["command_sha256"], policies[p]["state_sha256"])
                for p in POLICIES
                if p != "true_factor_oracle"
            }
            episode_one_identity &= len(hashes) == 1
    factors_ok = True
    for sequence_id in range(n_sequences):
        factors = [
            float(row["factor_deg"])
            for row in rows
            if int(row["sequence_id"]) == sequence_id
        ]
        factors_ok &= len(set(factors)) == (1 if condition == "persistent" else episodes)
    return {
        "complete": len(rows) == expected and len(set(keys)) == expected,
        "episode_one_identity": bool(episode_one_identity),
        "current_population_identity": bool(current_population_identity),
        "factor_lifetime": bool(factors_ok),
    }


def summarize(output_dir: Path, n_sequences: int = N_SEQUENCES, episodes: int = N_EPISODES) -> Dict:
    condition_rows = {
        condition: load_evaluations(output_dir / f"{condition}_raw.jsonl")
        for condition in CONDITIONS
    }
    values = {
        condition: sequence_policy_values(rows, n_sequences, episodes)
        for condition, rows in condition_rows.items()
    }
    persistent_effect = effect_payload(
        values["persistent"]["current_only"], values["persistent"]["correct_history"], 100
    )
    no_persistence_effect = effect_payload(
        values["no_persistence"]["current_only"],
        values["no_persistence"]["correct_history"],
        200,
    )
    persistent_delta = np.asarray(persistent_effect["sequence_deltas"])
    no_persistence_delta = np.asarray(no_persistence_effect["sequence_deltas"])
    did = persistent_delta - no_persistence_delta
    true_effect = effect_payload(
        values["persistent"]["current_only"],
        values["persistent"]["true_factor_oracle"],
        300,
    )
    summary = {
        "contract_id": CONTRACT_ID,
        "primary_metric": "later_E2_E4_mean_pose_auc25",
        "n_sequences_per_condition": n_sequences,
        "episodes_per_sequence": episodes,
        "persistent_correct_history": persistent_effect,
        "no_persistence_correct_history": no_persistence_effect,
        "persistent_true_factor": true_effect,
        "persistent_shuffled_history": effect_payload(
            values["persistent"]["current_only"],
            values["persistent"]["shuffled_history"],
            400,
        ),
        "persistent_wrong_sequence_history": effect_payload(
            values["persistent"]["current_only"],
            values["persistent"]["wrong_sequence_history"],
            500,
        ),
        "did": {
            "mean": float(np.mean(did)),
            "bootstrap_ci95": bootstrap_ci(did, 600),
            "positive_fraction": float(np.mean(did > 1e-12)),
            "sequence_values": did.tolist(),
        },
        "true_gap_recovery": float(
            np.mean(persistent_delta)
            / np.mean(np.asarray(true_effect["sequence_deltas"], dtype=np.float64))
        ),
        "policy_means": {
            condition: {policy: float(np.mean(array)) for policy, array in mapping.items()}
            for condition, mapping in values.items()
        },
        "structural_checks": {
            condition: structural_checks(condition, condition_rows[condition], n_sequences, episodes)
            for condition in CONDITIONS
        },
        "by_persistent_factor": {},
        "by_episode": {},
        "estimator": {},
    }
    for factor in FORMAL_FACTORS_DEG:
        ids = [sequence_id for sequence_id in range(n_sequences) if factor_for("persistent", sequence_id, 0) == factor]
        current = values["persistent"]["current_only"][ids]
        history = values["persistent"]["correct_history"][ids]
        summary["by_persistent_factor"][str(factor)] = effect_payload(current, history, 700 + int(factor + 30))
    for episode_index in range(1, episodes):
        payload = {}
        for condition in CONDITIONS:
            episode_rows = [row for row in condition_rows[condition] if int(row["episode_index"]) == episode_index]
            current = np.asarray([row["policies"]["current_only"]["metrics"]["pose_auc25"] for row in episode_rows])
            history = np.asarray([row["policies"]["correct_history"]["metrics"]["pose_auc25"] for row in episode_rows])
            payload[condition] = effect_payload(current, history, 800 + episode_index * 10 + (condition == "no_persistence"))
        summary["by_episode"][str(episode_index + 1)] = payload
    for condition in CONDITIONS:
        later = [row for row in condition_rows[condition] if int(row["episode_index"]) > 0]
        errors = [
            abs(wrapped_degrees_error(row["policies"]["correct_history"]["context_degrees"], row["factor_deg"]))
            for row in later
        ]
        accepted = [row["policies"]["correct_history"]["estimator"]["accepted_count"] for row in later]
        summary["estimator"][condition] = {
            "angle_mae_degrees": float(np.mean(errors)),
            "angle_median_absolute_error_degrees": float(np.median(errors)),
            "zero_accepted_fraction": float(np.mean(np.asarray(accepted) == 0)),
            "mean_accepted_transitions": float(np.mean(accepted)),
        }
    summary["valid"] = all(
        all(checks.values()) for checks in summary["structural_checks"].values()
    )
    return summary


def make_manifest(args, device: torch.device) -> Dict:
    design_hash = sha256(args.design)
    if design_hash != EXPECTED_DESIGN_SHA256:
        raise RuntimeError(f"design hash mismatch: expected {EXPECTED_DESIGN_SHA256}, got {design_hash}")
    return {
        "contract_id": CONTRACT_ID,
        "mode": "smoke" if args.smoke else "formal",
        "git_revision": git_revision(),
        "design_path": str(args.design),
        "design_sha256": design_hash,
        "contract_path": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "command": " ".join(__import__("sys").argv),
        "sequences": args.sequences,
        "episodes": args.episodes,
        "started_unix": time.time(),
        "resource_start": resource_snapshot(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--condition", choices=("both",) + CONDITIONS, default="both")
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
        default=Path("docs/research/persistent_context_v2_pushobj_rotation_stage1_design.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/research/persistent_context_v2_pushobj_rotation_stage1_contract_zh.md"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repro_outputs/persistent_context_v2_pushobj_rotation_stage1"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.sequences = 4 if args.smoke else N_SEQUENCES
    args.episodes = 2 if args.smoke else N_EPISODES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_ids = list(range(args.sequences))

    if args.mode == "inspect":
        payload = {
            "contract_id": CONTRACT_ID,
            "design_sha256": sha256(args.design),
            "contract_sha256": sha256(args.contract),
            "segment_indices": formal_segment_indices().reshape(N_SEQUENCES, N_EPISODES).tolist(),
            "factor_schedules": {
                condition: [
                    [factor_for(condition, sequence_id, episode) for episode in range(N_EPISODES)]
                    for sequence_id in range(N_SEQUENCES)
                ]
                for condition in CONDITIONS
            },
        }
        dump_json(args.output_dir / "selection_audit.json", payload)
        print(json.dumps(payload, indent=2))
        return

    if args.mode == "summarize":
        result = summarize(args.output_dir, args.sequences, args.episodes)
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
        return

    seed_all(9101)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but unavailable")
    device = torch.device(args.device)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["design_sha256"] != EXPECTED_DESIGN_SHA256:
            raise RuntimeError("existing manifest has a different design hash")
    else:
        manifest = make_manifest(args, device)
        dump_json(manifest_path, manifest)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    if len(segments) < 1000:
        raise RuntimeError("formal PushObj pool is incomplete")
    base, wrapper, _ = load_world_model(args.checkpoint, device)
    preprocessor = make_preprocessor()
    env = make_env()
    first_meta = scenario("persistent", sequence_ids[0], 0)
    initial_state, _, _, _ = prepare_target(env, segments[first_meta["segment_index"]], first_meta["env_seed"])
    start_obs, _ = env.prepare(first_meta["env_seed"], initial_state)
    identity = identity_audit(base, wrapper, preprocessor, start_obs)
    manifest["identity_audit"] = identity
    dump_json(manifest_path, manifest)
    conditions = CONDITIONS if args.condition == "both" else (args.condition,)
    for condition in conditions:
        evidence = generate_evidence(
            args, condition, wrapper, preprocessor, segments, env, sequence_ids
        )
        run_evaluations(
            args, condition, wrapper, preprocessor, segments, env, sequence_ids, evidence
        )
    if set(conditions) == set(CONDITIONS):
        result = summarize(args.output_dir, args.sequences, args.episodes)
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2), flush=True)
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(device)
    dump_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
