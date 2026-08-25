"""Frozen formal evaluation of non-privileged discrete-delay history on PushObj."""

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

from research.persistent_context_v2.matrix_task_interaction_d0 import model_sha256
from research.persistent_context_v2.pushobj_delay_history_stage1 import DiscreteDelayPosterior
from research.persistent_context_v2.pushobj_delay_stage0 import POPULATION_PRIOR, execute_delay, load_delay_world_model
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


CONTRACT_ID = "persistent-context-v2-pushobj-discrete-delay-history-stage1-formal-v1"
EXPECTED_DESIGN_SHA256 = "5a3aa28becb91da47261ec19293d31286f163e8444380acec5e380a8a8865f7e"
EXPECTED_SELECTION_SHA256 = "78755d050e0e09b67fb07f013db074ba63732688a1c805b0acf56fbf4dc76eb7"
EXPECTED_POOL_AUDIT_SHA256 = "557d95f910668b1217da04b74a6b7821f4a6eab0fe9dccd6518940d40f7e8fe6"
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


def array_sha256(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def segment_hash(segment: dict) -> str:
    digest = hashlib.sha256()
    for key in ("states", "actions"):
        value = np.asarray(segment[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def split_pairs(selection: dict, split: str) -> list[list[int]]:
    key = f"{split}_segment_indices_by_sequence_e1_e2"
    pairs = [[int(value) for value in pair] for pair in selection[key]]
    expected = 4 if split == "smoke" else 32
    if len(pairs) != expected or any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"invalid {split} selection")
    if len({value for pair in pairs for value in pair}) != expected * 2:
        raise ValueError(f"duplicate {split} segment")
    return pairs


def factor_for(condition: str, sequence_id: int, episode_index: int) -> int:
    base = int(sequence_id) % len(FACTORS)
    if condition == "persistent" or int(episode_index) == 0:
        return int(FACTORS[base])
    if condition == "no_persistence":
        return int(FACTORS[(base + 1) % len(FACTORS)])
    raise ValueError(condition)


def scenario(pairs: list[list[int]], condition: str, sequence_id: int, episode_index: int) -> dict:
    return {
        "condition": condition,
        "sequence_id": int(sequence_id),
        "episode_index": int(episode_index),
        "episode": int(episode_index) + 1,
        "segment_index": int(pairs[sequence_id][episode_index]),
        "factor_steps": factor_for(condition, sequence_id, episode_index),
        "env_seed": 1_091_000 + int(sequence_id) * 100 + int(episode_index),
        "cem_seed": 1_092_000 + int(sequence_id) * 100 + int(episode_index),
    }


def shuffle(commands, sequence_id: int, seed: int) -> tuple[np.ndarray, list[int]]:
    commands = np.asarray(commands, dtype=np.float64)
    order = np.random.default_rng(np.random.SeedSequence([int(seed), int(sequence_id)])).permutation(len(commands))
    if np.array_equal(order, np.arange(len(commands))):
        order = np.roll(order, 1)
    return commands[order], order.tolist()


def estimate(commands, states) -> dict:
    estimator = DiscreteDelayPosterior()
    estimator.update(commands, states)
    return estimator.as_dict()


def history_payload(evidence: dict[int, dict], policy: str, sequence_id: int, design: dict) -> dict:
    if policy in ("population_prior", "current_only"):
        return {"context_delay_steps": POPULATION_PRIOR, "estimator": None, "donors": [], "shuffle_order": None, "current_episode_evidence_count": 0}
    if policy == "true_factor_oracle":
        raise ValueError("oracle is scenario-local")
    n_sequences = len(evidence)
    own = evidence[int(sequence_id)]
    donor = int(sequence_id)
    commands, states = np.asarray(own["commands"], float), np.asarray(own["states"], float)
    order = None
    if policy == "wrong_sequence_history":
        donor = (int(sequence_id) + int(design["wrong_donor_sequence_offset"])) % n_sequences
        commands, states = np.asarray(evidence[donor]["commands"], float), np.asarray(evidence[donor]["states"], float)
    elif policy == "shuffled_history":
        commands, order = shuffle(commands, sequence_id, int(design["shuffled_command_seed"]))
    elif policy not in ("correct_history_map", "correct_history_high_delay_gate"):
        raise ValueError(policy)
    posterior = estimate(commands, states)
    context = int(posterior["map_delay_steps"])
    if policy == "correct_history_high_delay_gate" and posterior["probability_high_delay"] < float(design["estimator"]["high_delay_gate_probability"]):
        context = POPULATION_PRIOR
    return {
        "context_delay_steps": context,
        "estimator": posterior,
        "donors": [{
            "sequence_id": donor,
            "factor_steps": int(evidence[donor]["factor_steps"]),
            "evidence_sha256": evidence[donor]["evidence_sha256"],
        }],
        "shuffle_order": order,
        "current_episode_evidence_count": 0,
    }


def collect_e1(args, pairs, wrapper, preprocessor, segments, env) -> dict[int, dict]:
    path = args.output_dir / "e1_evidence_raw.jsonl"
    evidence = {}
    for sequence_id in range(len(pairs)):
        meta = scenario(pairs, "persistent", sequence_id, 0)
        initial, goal_obs, nominal_states, nominal_actions = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
        start_obs, _ = env.prepare(meta["env_seed"], initial)
        commands, planner = plan_waypoint_cem(wrapper, preprocessor, start_obs, goal_obs, POPULATION_PRIOR, meta["cem_seed"])
        states, effective, contacts, coverages = execute_delay(env, initial, meta["env_seed"], commands, meta["factor_steps"])
        posterior = estimate(commands, states)
        row = {
            "record_type": "delay_formal_e1_evidence", "contract_id": CONTRACT_ID, **meta,
            "segment_sha256": segment_hash(segments[meta["segment_index"]]),
            "initial_state": initial, "goal_state": nominal_states[-1], "nominal_actions": nominal_actions,
            "commands": commands, "states": states, "effective_actions": effective,
            "contacts": contacts, "coverages": coverages, "planner": planner, "estimator": posterior,
            "command_sha256": array_sha256(commands), "state_sha256": array_sha256(states),
            "effective_action_sha256": array_sha256(effective),
        }
        row["evidence_sha256"] = row["command_sha256"] + ":" + row["state_sha256"]
        append_jsonl(path, row)
        evidence[sequence_id] = row
        print(f"E1 {args.split} {sequence_id + 1}/{len(pairs)} true={meta['factor_steps']} map={posterior['map_delay_steps']}", flush=True)
    return evidence


def evaluate_e2(args, pairs, condition, wrapper, preprocessor, segments, env, evidence, design) -> None:
    path = args.output_dir / "e2_evaluation_raw.jsonl"
    for sequence_id in range(len(pairs)):
        meta = scenario(pairs, condition, sequence_id, 1)
        initial, goal_obs, nominal_states, _ = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
        start_obs, _ = env.prepare(meta["env_seed"], initial)
        payloads = {policy: history_payload(evidence, policy, sequence_id, design) for policy in POLICIES if policy != "true_factor_oracle"}
        payloads["true_factor_oracle"] = {
            "context_delay_steps": int(meta["factor_steps"]), "estimator": None,
            "donors": [], "shuffle_order": None, "current_episode_evidence_count": 0,
        }
        # Every decision is fixed before any E2 branch is executed.
        decisions = {policy: int(payloads[policy]["context_delay_steps"]) for policy in POLICIES}
        cache, policies = {}, {}
        for policy in POLICIES:
            context = decisions[policy]
            if context not in cache:
                commands, planner = plan_waypoint_cem(wrapper, preprocessor, start_obs, goal_obs, context, meta["cem_seed"])
                states, effective, contacts, coverages = execute_delay(env, initial, meta["env_seed"], commands, meta["factor_steps"])
                cache[context] = {
                    "commands": commands, "states": states, "effective_actions": effective,
                    "contacts": contacts, "coverages": coverages,
                    "metrics": pose_metrics(states, nominal_states[-1], 10),
                    "deadline_success": deadline_success(states, nominal_states[-1]),
                    "command_sha256": array_sha256(commands), "state_sha256": array_sha256(states),
                    "effective_action_sha256": array_sha256(effective), "planner": planner,
                }
            policies[policy] = {**payloads[policy], **cache[context]}
        append_jsonl(path, {
            "record_type": "delay_formal_e2_evaluation", "contract_id": CONTRACT_ID, **meta,
            "segment_sha256": segment_hash(segments[meta["segment_index"]]),
            "initial_state": initial, "goal_state": nominal_states[-1],
            "decision_pre_e2_execution": True, "decision_contexts": decisions,
            "policy_evaluation_order": list(POLICIES), "policies": policies,
            "resource": resource_snapshot(next(wrapper.parameters()).device),
        })
        print(f"E2 {args.split} {condition} {sequence_id + 1}/{len(pairs)} true={meta['factor_steps']} old_map={decisions['correct_history_map']}", flush=True)


def bootstrap(values, design: dict, stream: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(design["bootstrap_seed"]) + int(stream))
    indexes = rng.integers(0, len(values), size=(int(design["bootstrap_resamples"]), len(values)))
    means = values[indexes].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def contrast(values, design: dict, stream: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "bootstrap_ci95": bootstrap(values, design, stream),
        "positive_fraction": float(np.mean(values > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(values) <= 1e-12)),
        "negative_fraction": float(np.mean(values < -1e-12)), "sequence_deltas": values.tolist(),
    }


def effect(rows: list[dict], treatment: str, design: dict, stream: int) -> dict:
    current = np.asarray([row["policies"]["current_only"]["metrics"]["pose_auc10"] for row in rows])
    target = np.asarray([row["policies"][treatment]["metrics"]["pose_auc10"] for row in rows])
    result = contrast(current - target, design, stream)
    result.update({
        "current_mean": float(current.mean()), "treatment_mean": float(target.mean()),
        "relative_improvement": float(result["mean"] / current.mean()),
    })
    return result


def summarize(output_dir: Path, design: dict, expected_n: int) -> dict:
    evidence = read_jsonl(output_dir / "e1_evidence_raw.jsonl")
    evaluations = read_jsonl(output_dir / "e2_evaluation_raw.jsonl")
    rows = {condition: sorted([row for row in evaluations if row["condition"] == condition], key=lambda row: row["sequence_id"]) for condition in CONDITIONS}
    effects = {
        condition: {policy: effect(rows[condition], policy, design, 1000 * condition_index + policy_index) for policy_index, policy in enumerate(POLICIES) if policy != "current_only"}
        for condition_index, condition in enumerate(CONDITIONS)
    }
    persistent = np.asarray(effects["persistent"]["correct_history_map"]["sequence_deltas"])
    no_persistence = np.asarray(effects["no_persistence"]["correct_history_map"]["sequence_deltas"])
    did = contrast(persistent - no_persistence, design, 9000)
    paired = {(row["condition"], int(row["sequence_id"])): row for row in evaluations}
    checks = {
        "e1_complete": len(evidence) == expected_n,
        "e2_complete": all(len(rows[condition]) == expected_n for condition in CONDITIONS),
        "unique_e1_e2_segments": len({row["segment_index"] for row in evidence + rows["persistent"]}) == 2 * expected_n,
        "e1_map_correct": all(int(row["estimator"]["map_delay_steps"]) == int(row["factor_steps"]) for row in evidence),
        "decision_pre_execution": all(row.get("decision_pre_e2_execution") is True for row in evaluations),
        "e2_zero_current_evidence": all(value["current_episode_evidence_count"] == 0 for row in evaluations for value in row["policies"].values()),
        "population_current_identity": all(row["policies"]["population_prior"]["state_sha256"] == row["policies"]["current_only"]["state_sha256"] for row in evaluations),
        "paired_e2_scene": all(
            paired[("persistent", sequence_id)]["segment_index"] == paired[("no_persistence", sequence_id)]["segment_index"]
            and np.array_equal(paired[("persistent", sequence_id)]["initial_state"], paired[("no_persistence", sequence_id)]["initial_state"])
            and np.array_equal(paired[("persistent", sequence_id)]["goal_state"], paired[("no_persistence", sequence_id)]["goal_state"])
            for sequence_id in range(expected_n)
        ),
        "wrong_donor_factor_no_collision": all(
            row["policies"]["wrong_sequence_history"]["donors"][0]["factor_steps"] != row["factor_steps"]
            for row in evaluations
        ),
        "all_finite": all(np.isfinite(row["policies"][policy]["metrics"]["pose_auc10"]) for row in evaluations for policy in POLICIES),
    }
    by_factor = {}
    for condition in CONDITIONS:
        by_factor[condition] = {}
        for factor in FACTORS:
            subset = [row for row in rows[condition] if int(row["factor_steps"]) == factor]
            by_factor[condition][str(factor)] = effect(subset, "correct_history_map", design, 10_000 + 100 * CONDITIONS.index(condition) + factor)
    context_match = {
        condition: {
            policy: float(np.mean([int(row["policies"][policy]["context_delay_steps"]) == int(row["factor_steps"]) for row in rows[condition]]))
            for policy in ("correct_history_map", "shuffled_history", "wrong_sequence_history")
        } for condition in CONDITIONS
    }
    return {
        "contract_id": CONTRACT_ID, "evidence_level": "formal" if expected_n == 32 else "formal_pipeline_smoke",
        "n_sequences": expected_n, "primary_metric": "E2_pose_auc10_to_waypoint",
        "estimator_e1_map_accuracy": float(np.mean([int(row["estimator"]["map_delay_steps"]) == int(row["factor_steps"]) for row in evidence])),
        "estimator_e1_mean_entropy": float(np.mean([row["estimator"]["entropy"] for row in evidence])),
        "effects": effects, "did_correct_history_map": did, "by_current_factor": by_factor,
        "context_current_factor_match": context_match,
        "persistent_oracle_gap_recovery": float(effects["persistent"]["correct_history_map"]["mean"] / effects["persistent"]["true_factor_oracle"]["mean"]) if effects["persistent"]["true_factor_oracle"]["mean"] != 0 else None,
        "deadline_success": {condition: {policy: float(np.mean([row["policies"][policy]["deadline_success"] for row in rows[condition]])) for policy in POLICIES} for condition in CONDITIONS},
        "policy_means": {condition: {policy: float(np.mean([row["policies"][policy]["metrics"]["pose_auc10"] for row in rows[condition]])) for policy in POLICIES} for condition in CONDITIONS},
        "structural_checks": checks, "valid": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument("--split", choices=("smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_delay_history_stage1_formal_design.json"))
    parser.add_argument("--selection", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_delay_history_stage1_formal_selection.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_delay_history_stage1_formal_contract_zh.md"))
    parser.add_argument("--pool-audit", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_delay_history_stage1_formal_pool_audit_20260824.json"))
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    pairs = split_pairs(selection, args.split)
    expected_n = len(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    if args.mode == "inspect":
        pool = json.loads(args.pool_audit.read_text(encoding="utf-8"))
        group = {int(row["segment_index"]): row for row in pool["groups"][args.split]}
        payload = {
            "contract_id": CONTRACT_ID, "split": args.split,
            "design_sha256": sha256(args.design), "selection_sha256": sha256(args.selection),
            "contract_sha256": sha256(args.contract), "pool_audit_sha256": sha256(args.pool_audit),
            "scenarios": [{**scenario(pairs, "persistent", sequence_id, episode_index),
                           "segment_sha256": segment_hash(segments[pairs[sequence_id][episode_index]]),
                           "pool_segment_sha256": group[pairs[sequence_id][episode_index]]["segment_sha256"],
                           "nominal_block_displacement_at_10": nominal_block_displacement_at_10(segments[pairs[sequence_id][episode_index]])}
                          for sequence_id in range(expected_n) for episode_index in range(2)],
        }
        dump_json(args.output_dir / "selection_audit.json", payload)
        print(json.dumps({"split": args.split, "scenarios": len(payload["scenarios"]), "hashes_match": all(row["segment_sha256"] == row["pool_segment_sha256"] for row in payload["scenarios"])}, indent=2))
        return
    if args.mode == "summarize":
        result = summarize(args.output_dir, design, expected_n)
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
        return
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"non-empty output: {args.output_dir}")
    for path, expected in ((args.design, EXPECTED_DESIGN_SHA256), (args.selection, EXPECTED_SELECTION_SHA256), (args.pool_audit, EXPECTED_POOL_AUDIT_SHA256)):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    if args.source_snapshot is None:
        raise ValueError("--source-snapshot is required")
    seed_all(1_093_001)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device.index or 0)
    base, wrapper, _ = load_delay_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    first = scenario(pairs, "persistent", 0, 0)
    initial, _, _, _ = prepare_waypoint(env, segments[first["segment_index"]], first["env_seed"])
    start_obs, _ = env.prepare(first["env_seed"], initial)
    identity = identity_audit(base, wrapper, preprocessor, start_obs)
    model_before, started = model_sha256(wrapper), time.time()
    manifest = {
        "contract_id": CONTRACT_ID, "split": args.split,
        "evidence_level": "formal" if args.split == "formal" else "formal_pipeline_smoke",
        "git_revision": git_revision(), "design_sha256": sha256(args.design),
        "selection_sha256": sha256(args.selection), "contract_sha256": sha256(args.contract),
        "pool_audit_sha256": sha256(args.pool_audit), "source_snapshot_sha256": sha256(args.source_snapshot),
        "checkpoint_sha256": sha256(args.checkpoint), "data_sha256": sha256(args.data),
        "command": " ".join(__import__("sys").argv), "started_unix": started,
        "resource_start": resource_snapshot(device), "identity_audit": identity,
        "model_state_sha256_before": model_before, "sequence_count": expected_n,
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    evidence = collect_e1(args, pairs, wrapper, preprocessor, segments, env)
    for condition in CONDITIONS:
        evaluate_e2(args, pairs, condition, wrapper, preprocessor, segments, env, evidence, design)
    model_after = model_sha256(wrapper)
    manifest.update({
        "finished_unix": time.time(), "wall_time_s": time.time() - started,
        "resource_end": resource_snapshot(device), "model_state_sha256_after": model_after,
        "model_state_unchanged": model_before == model_after,
        "e1_raw_sha256": sha256(args.output_dir / "e1_evidence_raw.jsonl"),
        "e2_raw_sha256": sha256(args.output_dir / "e2_evaluation_raw.jsonl"),
    })
    if device.type == "cuda":
        manifest["resource_end"]["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device.index or 0))
        manifest["resource_end"]["cuda_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device.index or 0))
    dump_json(args.output_dir / "manifest.json", manifest)
    result = summarize(args.output_dir, design, expected_n)
    result["model_state_unchanged"] = manifest["model_state_unchanged"]
    result["valid"] = bool(result["valid"] and manifest["model_state_unchanged"] and max(identity.values(), default=math.inf) <= 1e-6)
    dump_json(args.output_dir / "runner_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
