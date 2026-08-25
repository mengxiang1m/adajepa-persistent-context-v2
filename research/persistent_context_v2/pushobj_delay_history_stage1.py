"""Development-only non-privileged history estimator for discrete PushObj delay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.matrix_task_interaction_d0 import model_sha256
from research.persistent_context_v2.pushobj_delay_stage0 import (
    POPULATION_PRIOR,
    apply_discrete_delay,
    execute_delay,
    load_delay_world_model,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    deadline_success,
    nominal_block_displacement_at_10,
    plan_waypoint_cem,
    prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl,
    dump_json,
    git_revision,
    identity_audit,
    make_env,
    make_preprocessor,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)
from research.persistent_context_v2.pushobj_rotation_stage1 import pd_coefficients


CONTRACT_ID = "persistent-context-v2-pushobj-discrete-delay-history-stage1-dev-v1"
EXPECTED_DESIGN_SHA256 = "3f700ef7eb296ca8f54ab9a18e15959c32338e08a01b4fcda4276822fe6dfc66"
CANDIDATES = (0, 1, 2, 3, 4)
FACTORS = (0, 1, 3, 4)
CONDITIONS = ("persistent", "no_persistence")
POLICIES = (
    "population_prior",
    "current_only",
    "correct_history_map",
    "correct_history_high_delay_gate",
    "shuffled_history",
    "wrong_sequence_history",
    "true_factor_oracle",
)
SEGMENTS = ((108, 304), (97, 16), (663, 287), (366, 150))
NOISE_STD = 0.1
GATE_PROBABILITY = 0.95
SHUFFLE_SEED = 1_080_200


def array_sha256(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def infer_effective_actions(commands, states) -> np.ndarray:
    """Invert known agent PD dynamics using only command-aligned proprio states."""
    commands = np.asarray(commands, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[1] != 2:
        raise ValueError("commands must have shape [time,2]")
    if states.ndim != 2 or states.shape[0] != len(commands) + 1 or states.shape[1] < 7:
        raise ValueError("states must contain command-aligned PushObj proprio transitions")
    p_coef, v_coef, target_coef = pd_coefficients()
    inferred = []
    for index in range(len(commands)):
        p0, p1, v0 = states[index, :2], states[index + 1, :2], states[index, 5:7]
        target = (p1 - p_coef * p0 - v_coef * v0) / target_coef
        inferred.append((target - p0) / 100.0)
    result = np.asarray(inferred, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("non-finite inferred effective action")
    return result


def episode_log_likelihood(commands, states, noise_std: float = NOISE_STD) -> np.ndarray:
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    commands = np.asarray(commands, dtype=np.float64)
    inferred = infer_effective_actions(commands, states)
    return np.asarray(
        [
            -float(np.sum((inferred - apply_discrete_delay(commands, delay)) ** 2))
            / (2.0 * noise_std**2)
            for delay in CANDIDATES
        ],
        dtype=np.float64,
    )


def normalized_posterior(log_probability) -> np.ndarray:
    values = np.asarray(log_probability, dtype=np.float64)
    shifted = values - float(np.max(values))
    probability = np.exp(shifted)
    return probability / probability.sum()


@dataclass
class DiscreteDelayPosterior:
    noise_std: float = NOISE_STD
    log_probability: np.ndarray = field(default_factory=lambda: np.full(len(CANDIDATES), -math.log(len(CANDIDATES))))
    episode_count: int = 0
    evidence_count: int = 0
    change_detected: bool = False
    latest_episode_map: int | None = None

    def update(self, commands, states) -> None:
        likelihood = episode_log_likelihood(commands, states, self.noise_std)
        latest = normalized_posterior(likelihood)
        latest_map = CANDIDATES[int(np.argmax(latest))]
        previous_map = self.map_delay if self.episode_count else None
        self.change_detected = bool(
            previous_map is not None
            and latest_map != previous_map
            and float(latest.max()) >= GATE_PROBABILITY
        )
        self.log_probability = self.log_probability + likelihood
        self.episode_count += 1
        self.evidence_count += int(np.asarray(commands).shape[0])
        self.latest_episode_map = int(latest_map)

    @property
    def posterior(self) -> np.ndarray:
        return normalized_posterior(self.log_probability)

    @property
    def map_delay(self) -> int:
        return int(CANDIDATES[int(np.argmax(self.posterior))])

    @property
    def entropy(self) -> float:
        probability = self.posterior
        return float(-np.sum(probability * np.log(np.maximum(probability, 1e-300))))

    def as_dict(self) -> dict:
        probability = self.posterior
        return {
            "candidate_delay_steps": list(CANDIDATES),
            "posterior_probability": probability,
            "map_delay_steps": self.map_delay,
            "entropy": self.entropy,
            "probability_high_delay": float(probability[3] + probability[4]),
            "episode_count": self.episode_count,
            "evidence_count": self.evidence_count,
            "latest_episode_map": self.latest_episode_map,
            "change_detected": self.change_detected,
            "noise_std": self.noise_std,
        }


def factor_for(condition: str, sequence_id: int, episode_index: int) -> int:
    base = int(sequence_id) % len(FACTORS)
    if condition == "persistent" or int(episode_index) == 0:
        return int(FACTORS[base])
    if condition == "no_persistence":
        return int(FACTORS[(base + 1) % len(FACTORS)])
    raise ValueError(condition)


def scenario(condition: str, sequence_id: int, episode_index: int) -> dict:
    return {
        "condition": condition,
        "sequence_id": int(sequence_id),
        "episode_index": int(episode_index),
        "episode": int(episode_index) + 1,
        "segment_index": int(SEGMENTS[sequence_id][episode_index]),
        "factor_steps": factor_for(condition, sequence_id, episode_index),
        "env_seed": 1_081_000 + int(sequence_id) * 100 + int(episode_index),
        "cem_seed": 1_082_000 + int(sequence_id) * 100 + int(episode_index),
    }


def shuffled_commands(commands, sequence_id: int) -> tuple[np.ndarray, list[int]]:
    commands = np.asarray(commands, dtype=np.float64)
    order = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, int(sequence_id)])).permutation(len(commands))
    if np.array_equal(order, np.arange(len(commands))):
        order = np.roll(order, 1)
    return commands[order], order.tolist()


def history_payload(evidence: dict[int, dict], policy: str, sequence_id: int) -> dict:
    if policy in ("population_prior", "current_only"):
        return {"context_delay_steps": POPULATION_PRIOR, "estimator": None, "donors": [], "current_episode_evidence_count": 0}
    if policy == "true_factor_oracle":
        raise ValueError("oracle context must be provided by the evaluation scenario")
    own = evidence[int(sequence_id)]
    donor = int(sequence_id)
    commands = np.asarray(own["commands"], dtype=np.float64)
    states = np.asarray(own["states"], dtype=np.float64)
    shuffle_order = None
    if policy == "wrong_sequence_history":
        donor = (int(sequence_id) + 1) % len(SEGMENTS)
        row = evidence[donor]
        commands, states = np.asarray(row["commands"], dtype=np.float64), np.asarray(row["states"], dtype=np.float64)
    elif policy == "shuffled_history":
        commands, shuffle_order = shuffled_commands(commands, sequence_id)
    elif policy not in ("correct_history_map", "correct_history_high_delay_gate"):
        raise ValueError(policy)
    estimator = DiscreteDelayPosterior()
    estimator.update(commands, states)
    estimate = estimator.as_dict()
    context = int(estimate["map_delay_steps"])
    if policy == "correct_history_high_delay_gate" and estimate["probability_high_delay"] < GATE_PROBABILITY:
        context = POPULATION_PRIOR
    return {
        "context_delay_steps": context,
        "estimator": estimate,
        "donors": [{"sequence_id": donor, "evidence_sha256": evidence[donor]["evidence_sha256"]}],
        "shuffle_order": shuffle_order,
        "current_episode_evidence_count": 0,
    }


def collect_e1(args, wrapper, preprocessor, segments, env) -> dict[int, dict]:
    raw_path = args.output_dir / "e1_evidence_raw.jsonl"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    evidence = {}
    for sequence_id in range(len(SEGMENTS)):
        meta = scenario("persistent", sequence_id, 0)
        initial, goal_obs, nominal_states, nominal_actions = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
        start_obs, _ = env.prepare(meta["env_seed"], initial)
        commands, planner = plan_waypoint_cem(wrapper, preprocessor, start_obs, goal_obs, POPULATION_PRIOR, meta["cem_seed"])
        states, effective, contacts, coverages = execute_delay(env, initial, meta["env_seed"], commands, meta["factor_steps"])
        estimator = DiscreteDelayPosterior()
        estimator.update(commands, states)
        row = {
            "record_type": "delay_e1_evidence",
            "contract_id": CONTRACT_ID,
            **meta,
            "initial_state": initial,
            "goal_state": nominal_states[-1],
            "nominal_actions": nominal_actions,
            "commands": commands,
            "states": states,
            "effective_actions": effective,
            "contacts": contacts,
            "coverages": coverages,
            "planner": planner,
            "estimator": estimator.as_dict(),
            "command_sha256": array_sha256(commands),
            "state_sha256": array_sha256(states),
            "effective_action_sha256": array_sha256(effective),
        }
        row["evidence_sha256"] = row["command_sha256"] + ":" + row["state_sha256"]
        append_jsonl(raw_path, row)
        evidence[sequence_id] = row
        print(f"E1 s={sequence_id} true={meta['factor_steps']} map={row['estimator']['map_delay_steps']}", flush=True)
    return evidence


def evaluate_e2(args, condition: str, wrapper, preprocessor, segments, env, evidence: dict[int, dict]) -> None:
    raw_path = args.output_dir / "e2_evaluation_raw.jsonl"
    for sequence_id in range(len(SEGMENTS)):
        meta = scenario(condition, sequence_id, 1)
        initial, goal_obs, nominal_states, _ = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
        start_obs, _ = env.prepare(meta["env_seed"], initial)
        payloads = {policy: history_payload(evidence, policy, sequence_id) for policy in POLICIES if policy != "true_factor_oracle"}
        payloads["true_factor_oracle"] = {
            "context_delay_steps": int(meta["factor_steps"]), "estimator": None,
            "donors": [], "current_episode_evidence_count": 0,
        }
        cache, policies = {}, {}
        for policy in POLICIES:
            payload = payloads[policy]
            context = int(payload["context_delay_steps"])
            if context not in cache:
                commands, planner = plan_waypoint_cem(wrapper, preprocessor, start_obs, goal_obs, context, meta["cem_seed"])
                states, effective, contacts, coverages = execute_delay(env, initial, meta["env_seed"], commands, meta["factor_steps"])
                cache[context] = {
                    "commands": commands,
                    "states": states,
                    "effective_actions": effective,
                    "contacts": contacts,
                    "coverages": coverages,
                    "metrics": pose_metrics(states, nominal_states[-1], 10),
                    "deadline_success": deadline_success(states, nominal_states[-1]),
                    "command_sha256": array_sha256(commands),
                    "state_sha256": array_sha256(states),
                    "effective_action_sha256": array_sha256(effective),
                    "planner": planner,
                }
            policies[policy] = {**payload, **cache[context]}
        append_jsonl(raw_path, {
            "record_type": "delay_e2_evaluation",
            "contract_id": CONTRACT_ID,
            **meta,
            "initial_state": initial,
            "goal_state": nominal_states[-1],
            "policies": policies,
            "resource": resource_snapshot(next(wrapper.parameters()).device),
        })
        print(
            f"E2 {condition} s={sequence_id} true={meta['factor_steps']} "
            f"map={policies['correct_history_map']['context_delay_steps']} ",
            flush=True,
        )


def effect(rows: list[dict], treatment: str) -> dict:
    current = np.asarray([row["policies"]["current_only"]["metrics"]["pose_auc10"] for row in rows], dtype=np.float64)
    target = np.asarray([row["policies"][treatment]["metrics"]["pose_auc10"] for row in rows], dtype=np.float64)
    delta = current - target
    return {
        "current_mean": float(current.mean()),
        "treatment_mean": float(target.mean()),
        "mean_delta": float(delta.mean()),
        "relative_improvement": float(delta.mean() / current.mean()),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)),
        "sequence_deltas": delta.tolist(),
    }


def summarize(output_dir: Path) -> dict:
    evidence = read_jsonl(output_dir / "e1_evidence_raw.jsonl")
    evaluations = read_jsonl(output_dir / "e2_evaluation_raw.jsonl")
    by_condition = {condition: [row for row in evaluations if row["condition"] == condition] for condition in CONDITIONS}
    effects = {condition: {policy: effect(by_condition[condition], policy) for policy in POLICIES if policy != "current_only"} for condition in CONDITIONS}
    persistent = np.asarray(effects["persistent"]["correct_history_map"]["sequence_deltas"])
    no_persistence = np.asarray(effects["no_persistence"]["correct_history_map"]["sequence_deltas"])
    paired = {(row["condition"], row["sequence_id"]): row for row in evaluations}
    checks = {
        "e1_complete": len(evidence) == len(SEGMENTS),
        "e2_complete": all(len(by_condition[condition]) == len(SEGMENTS) for condition in CONDITIONS),
        "e1_map_correct": all(int(row["estimator"]["map_delay_steps"]) == int(row["factor_steps"]) for row in evidence),
        "e2_zero_current_evidence": all(policy["current_episode_evidence_count"] == 0 for row in evaluations for policy in row["policies"].values()),
        "population_current_identity": all(row["policies"]["population_prior"]["state_sha256"] == row["policies"]["current_only"]["state_sha256"] for row in evaluations),
        "paired_e2_scene": all(
            paired[("persistent", sequence_id)]["segment_index"] == paired[("no_persistence", sequence_id)]["segment_index"]
            and np.array_equal(paired[("persistent", sequence_id)]["initial_state"], paired[("no_persistence", sequence_id)]["initial_state"])
            and np.array_equal(paired[("persistent", sequence_id)]["goal_state"], paired[("no_persistence", sequence_id)]["goal_state"])
            for sequence_id in range(len(SEGMENTS))
        ),
        "all_finite": all(np.isfinite(row["policies"][policy]["metrics"]["pose_auc10"]) for row in evaluations for policy in POLICIES),
    }
    return {
        "contract_id": CONTRACT_ID,
        "evidence_level": "development_smoke_not_formal",
        "n_sequences": len(SEGMENTS),
        "estimator_map_accuracy_e1": float(np.mean([int(row["estimator"]["map_delay_steps"]) == int(row["factor_steps"]) for row in evidence])),
        "estimator_mean_entropy_e1": float(np.mean([row["estimator"]["entropy"] for row in evidence])),
        "effects": effects,
        "did_correct_history_map": {
            "mean": float(np.mean(persistent - no_persistence)),
            "sequence_deltas": (persistent - no_persistence).tolist(),
        },
        "policy_means": {condition: {policy: float(np.mean([row["policies"][policy]["metrics"]["pose_auc10"] for row in by_condition[condition]])) for policy in POLICIES} for condition in CONDITIONS},
        "structural_checks": checks,
        "valid": all(checks.values()),
    }


def inspect(args, segments) -> None:
    rows = []
    for sequence_id in range(len(SEGMENTS)):
        for episode_index in range(2):
            meta = scenario("persistent", sequence_id, episode_index)
            meta["nominal_block_displacement_at_10"] = nominal_block_displacement_at_10(segments[meta["segment_index"]])
            rows.append(meta)
    payload = {"contract_id": CONTRACT_ID, "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract), "smoke_scenarios": rows}
    dump_json(args.output_dir / "selection_audit.json", payload)
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_delay_history_stage1_dev_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_delay_history_stage1_dev_contract_zh.md"))
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_delay_history_stage1_dev_smoke_v1"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    if args.mode == "inspect":
        inspect(args, segments)
        return
    if args.mode == "summarize":
        result = summarize(args.output_dir)
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
        return
    if not args.smoke:
        raise RuntimeError("development runner only authorizes --smoke")
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"non-empty smoke output: {args.output_dir}")
    if sha256(args.design) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen design hash mismatch")
    if args.source_snapshot is None:
        raise ValueError("--source-snapshot is required")
    seed_all(1_083_001)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index or 0)
    base, wrapper, _ = load_delay_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    first = scenario("persistent", 0, 0)
    initial, _, _, _ = prepare_waypoint(env, segments[first["segment_index"]], first["env_seed"])
    start_obs, _ = env.prepare(first["env_seed"], initial)
    identity = identity_audit(base, wrapper, preprocessor, start_obs)
    model_before = model_sha256(wrapper)
    started = time.time()
    manifest = {
        "contract_id": CONTRACT_ID,
        "evidence_level": "development_smoke_not_formal",
        "git_revision": git_revision(),
        "design_sha256": sha256(args.design),
        "contract_sha256": sha256(args.contract),
        "source_snapshot_sha256": sha256(args.source_snapshot),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_sha256": sha256(args.data),
        "command": " ".join(__import__("sys").argv),
        "started_unix": started,
        "resource_start": resource_snapshot(device),
        "identity_audit": identity,
        "model_state_sha256_before": model_before,
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    evidence = collect_e1(args, wrapper, preprocessor, segments, env)
    for condition in CONDITIONS:
        evaluate_e2(args, condition, wrapper, preprocessor, segments, env, evidence)
    model_after = model_sha256(wrapper)
    manifest.update({
        "finished_unix": time.time(),
        "wall_time_s": time.time() - started,
        "resource_end": resource_snapshot(device),
        "model_state_sha256_after": model_after,
        "model_state_unchanged": model_before == model_after,
        "e1_raw_sha256": sha256(args.output_dir / "e1_evidence_raw.jsonl"),
        "e2_raw_sha256": sha256(args.output_dir / "e2_evaluation_raw.jsonl"),
    })
    if device.type == "cuda":
        manifest["resource_end"]["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device.index or 0))
        manifest["resource_end"]["cuda_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device.index or 0))
    dump_json(args.output_dir / "manifest.json", manifest)
    result = summarize(args.output_dir)
    result["model_state_unchanged"] = manifest["model_state_unchanged"]
    result["valid"] = bool(result["valid"] and manifest["model_state_unchanged"] and max(identity.values(), default=math.inf) <= 1e-6)
    dump_json(args.output_dir / "runner_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
