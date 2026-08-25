"""Cross-episode rotation estimation on the 10-action PushObj waypoint task."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    append_jsonl,
    dump_json,
    execute,
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
from research.persistent_context_v2.pushobj_rotation_stage1 import (
    RotationMLE,
    wrapped_degrees_error,
)


CONTRACT_ID = "persistent-context-v2-pushobj-rotation-early-waypoint-history-stage1-v1"
EXPECTED_DESIGN_SHA256 = "67ce433c5fb76e0bc26d625fd07ea8589e06f07f8770def8bb386c63b8627f61"
FACTORS_DEG = (-25.0, -10.0, 10.0, 25.0)
POLICIES = (
    "population_prior", "current_only", "correct_history",
    "shuffled_history", "wrong_sequence_history", "true_factor_oracle",
)
CONDITIONS = ("persistent", "no_persistence")
N_SEQUENCES = 32
N_EPISODES = 4
SELECTION_SEED = 850000
FACTOR_SCHEDULE_SEED = 850100
DONOR_SEED = 850200
BOOTSTRAP_SEED = 850301
BOOTSTRAP_RESAMPLES = 20_000


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def formal_segment_indices(segments: Sequence[Dict]) -> np.ndarray:
    pool = np.asarray(
        [
            index for index in range(500, min(1000, len(segments)))
            if nominal_block_displacement_at_10(segments[index]) >= 10.0
        ],
        dtype=np.int64,
    )
    if len(pool) < N_SEQUENCES * N_EPISODES:
        raise RuntimeError("formal waypoint pool is too small")
    return np.random.default_rng(SELECTION_SEED).permutation(pool)[: N_SEQUENCES * N_EPISODES]


def factor_schedules(n_sequences: int = N_SEQUENCES, episodes: int = N_EPISODES):
    if n_sequences % 4:
        raise ValueError("sequence count must be divisible by four")
    base = np.arange(n_sequences, dtype=np.int64) % 4
    persistent = np.repeat(base[:, None], episodes, axis=1)
    changing = np.empty_like(persistent)
    changing[:, 0] = base
    rng = np.random.default_rng(FACTOR_SCHEDULE_SEED)
    balanced = np.repeat(np.arange(4, dtype=np.int64), n_sequences // 4)
    for episode in range(1, episodes):
        for _ in range(100_000):
            candidate = rng.permutation(balanced)
            if np.all(candidate != changing[:, episode - 1]):
                changing[:, episode] = candidate
                break
        else:  # pragma: no cover
            raise RuntimeError("could not generate balanced changing schedule")
    return persistent, changing


def _cross_factor_map(rng, n_sequences: int, factor_shift: int) -> np.ndarray:
    mapping = np.empty(n_sequences, dtype=np.int64)
    for target_factor in range(4):
        targets = np.flatnonzero(np.arange(n_sequences) % 4 == target_factor)
        donor_factor = (target_factor + factor_shift) % 4
        donors = np.flatnonzero(np.arange(n_sequences) % 4 == donor_factor)
        mapping[targets] = rng.permutation(donors)
    return mapping


def donor_maps(n_sequences: int = N_SEQUENCES):
    if n_sequences % 4:
        raise ValueError("sequence count must be divisible by four")
    rng = np.random.default_rng(DONOR_SEED)
    wrong_shift = int(rng.integers(1, 4))
    wrong = _cross_factor_map(rng, n_sequences, wrong_shift)
    shifts = rng.permutation(np.asarray([1, 2, 3], dtype=np.int64))
    shuffled = np.stack(
        [_cross_factor_map(rng, n_sequences, int(shift)) for shift in shifts], axis=1
    )
    return wrong, shuffled


def factor_for(condition: str, sequence_id: int, episode_index: int, n_sequences=N_SEQUENCES, episodes=N_EPISODES):
    persistent, changing = factor_schedules(n_sequences, episodes)
    ids = persistent if condition == "persistent" else changing
    return FACTORS_DEG[int(ids[sequence_id, episode_index])]


def scenario(segments, condition, sequence_id, episode_index, n_sequences=N_SEQUENCES, episodes=N_EPISODES):
    indices = formal_segment_indices(segments).reshape(N_SEQUENCES, N_EPISODES)
    return {
        "condition": condition,
        "sequence_id": int(sequence_id),
        "episode": int(episode_index + 1),
        "episode_index": int(episode_index),
        "segment_index": int(indices[sequence_id, episode_index]),
        "factor_deg": factor_for(condition, sequence_id, episode_index, n_sequences, episodes),
        "env_seed": int(860_000 + sequence_id * 100 + episode_index),
        "cem_seed": int(870_000 + sequence_id * 100 + episode_index),
    }


def donor_for(policy, sequence_id, history_episode_index, n_sequences=N_SEQUENCES):
    if policy == "correct_history":
        return int(sequence_id)
    wrong, shuffled = donor_maps(n_sequences)
    if policy == "wrong_sequence_history":
        return int(wrong[sequence_id])
    if policy == "shuffled_history":
        return int(shuffled[sequence_id, history_episode_index])
    raise ValueError(policy)


def build_estimator(evidence, policy, sequence_id, episode_index, n_sequences):
    estimator = RotationMLE()
    donors = []
    for history_episode_index in range(episode_index):
        donor = donor_for(policy, sequence_id, history_episode_index, n_sequences)
        row = evidence[(donor, history_episode_index)]
        estimator.update(np.asarray(row["commands"]), np.asarray(row["states"]))
        donors.append(
            {
                "history_episode": history_episode_index + 1,
                "donor_sequence_id": donor,
                "donor_factor_deg": float(row["factor_deg"]),
                "evidence_sha256": row["evidence_sha256"],
            }
        )
    return estimator, donors


def policy_context(evidence, policy, sequence_id, episode_index, factor_deg, n_sequences):
    if policy in ("population_prior", "current_only"):
        return {"context_degrees": 0.0, "estimator": None, "donors": []}
    if policy == "true_factor_oracle":
        return {"context_degrees": float(factor_deg), "estimator": None, "donors": []}
    estimator, donors = build_estimator(evidence, policy, sequence_id, episode_index, n_sequences)
    return {"context_degrees": estimator.estimate_degrees, "estimator": estimator.as_dict(), "donors": donors}


def completed_keys(path, record_type):
    return {
        (int(row["sequence_id"]), int(row["episode"]))
        for row in read_jsonl(path) if row.get("record_type") == record_type
    }


def evidence_lookup(path):
    return {
        (int(row["sequence_id"]), int(row["episode_index"])): row
        for row in read_jsonl(path) if row.get("record_type") == "evidence_episode"
    }


def generate_evidence(args, condition, wrapper, preprocessor, segments, env):
    path = args.output_dir / f"{condition}_raw.jsonl"
    completed = completed_keys(path, "evidence_episode")
    for sequence_id in range(args.sequences):
        for episode_index in range(args.episodes):
            meta = scenario(segments, condition, sequence_id, episode_index, args.sequences, args.episodes)
            if (sequence_id, episode_index + 1) in completed:
                continue
            started = time.perf_counter()
            initial_state, goal_obs, nominal_states, nominal_actions = prepare_waypoint(
                env, segments[meta["segment_index"]], meta["env_seed"]
            )
            start_obs, _ = env.prepare(meta["env_seed"], initial_state)
            commands, planner = plan_waypoint_cem(wrapper, preprocessor, start_obs, goal_obs, 0.0, meta["cem_seed"])
            states, effective, contacts, coverages = execute(
                env, initial_state, meta["env_seed"], commands, meta["factor_deg"]
            )
            row = {
                "record_type": "evidence_episode", "contract_id": CONTRACT_ID, **meta,
                "initial_state": initial_state, "goal_state": nominal_states[-1],
                "nominal_actions": nominal_actions, "commands": commands, "states": states,
                "effective_actions": effective, "contacts": contacts, "coverages": coverages,
                "planner": planner,
                "evidence_sha256": array_sha256(commands) + ":" + array_sha256(states),
                "elapsed_s": time.perf_counter() - started,
                "resource": resource_snapshot(next(wrapper.parameters()).device),
            }
            append_jsonl(path, row)
            print(f"EVIDENCE {condition} s={sequence_id} e={episode_index+1} factor={meta['factor_deg']:+g}", flush=True)
    return evidence_lookup(path)


def run_evaluations(args, condition, wrapper, preprocessor, segments, env, evidence):
    path = args.output_dir / f"{condition}_raw.jsonl"
    completed = completed_keys(path, "evaluation_episode")
    for sequence_id in range(args.sequences):
        for episode_index in range(args.episodes):
            meta = scenario(segments, condition, sequence_id, episode_index, args.sequences, args.episodes)
            if (sequence_id, episode_index + 1) in completed:
                continue
            started = time.perf_counter()
            initial_state, goal_obs, nominal_states, _ = prepare_waypoint(
                env, segments[meta["segment_index"]], meta["env_seed"]
            )
            start_obs, _ = env.prepare(meta["env_seed"], initial_state)
            contexts = {
                policy: policy_context(evidence, policy, sequence_id, episode_index, meta["factor_deg"], args.sequences)
                for policy in POLICIES
            }
            cache, policies = {}, {}
            for policy in POLICIES:
                context = float(contexts[policy]["context_degrees"])
                key = float(round(context, 10))
                if key not in cache:
                    commands, planner = plan_waypoint_cem(
                        wrapper, preprocessor, start_obs, goal_obs, context, meta["cem_seed"]
                    )
                    states, effective, contacts, coverages = execute(
                        env, initial_state, meta["env_seed"], commands, meta["factor_deg"]
                    )
                    cache[key] = {
                        "commands": commands, "states": states, "effective_actions": effective,
                        "contacts": contacts, "coverages": coverages,
                        "metrics": pose_metrics(states, nominal_states[-1], WINDOW),
                        "deadline_success": deadline_success(states, nominal_states[-1]),
                        "planner": planner, "command_sha256": array_sha256(commands),
                        "state_sha256": array_sha256(states),
                    }
                policies[policy] = {**contexts[policy], **cache[key]}
            row = {
                "record_type": "evaluation_episode", "contract_id": CONTRACT_ID, **meta,
                "initial_state": initial_state, "goal_state": nominal_states[-1],
                "policies": policies, "elapsed_s": time.perf_counter() - started,
                "resource": resource_snapshot(next(wrapper.parameters()).device),
            }
            append_jsonl(path, row)
            print(
                f"EVAL {condition} s={sequence_id} e={episode_index+1} theta={meta['factor_deg']:+g} "
                f"hist={policies['correct_history']['context_degrees']:+.3f} "
                f"current={policies['current_only']['metrics']['pose_auc10']:.4f} "
                f"history={policies['correct_history']['metrics']['pose_auc10']:.4f}", flush=True,
            )


def bootstrap_ci(values, stream):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(stream))
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def effect(current, treatment, stream):
    delta = np.asarray(current) - np.asarray(treatment)
    return {
        "current_mean": float(np.mean(current)), "treatment_mean": float(np.mean(treatment)),
        "mean_delta": float(delta.mean()), "relative_improvement": float(delta.mean() / np.mean(current)),
        "bootstrap_ci95_delta": bootstrap_ci(delta, stream),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)), "sequence_deltas": delta.tolist(),
    }


def summarize(output_dir, n_sequences=N_SEQUENCES, episodes=N_EPISODES):
    rows = {
        c: [r for r in read_jsonl(output_dir / f"{c}_raw.jsonl") if r.get("record_type") == "evaluation_episode"]
        for c in CONDITIONS
    }
    values, successes = {}, {}
    for condition in CONDITIONS:
        lookup = {(int(r["sequence_id"]), int(r["episode_index"])): r for r in rows[condition]}
        values[condition], successes[condition] = {}, {}
        for policy in POLICIES:
            values[condition][policy] = np.asarray([
                np.mean([lookup[(s, e)]["policies"][policy]["metrics"]["pose_auc10"] for e in range(1, episodes)])
                for s in range(n_sequences)
            ])
            successes[condition][policy] = np.asarray([
                np.mean([lookup[(s, e)]["policies"][policy]["deadline_success"] for e in range(1, episodes)])
                for s in range(n_sequences)
            ])
    p = effect(values["persistent"]["current_only"], values["persistent"]["correct_history"], 100)
    n = effect(values["no_persistence"]["current_only"], values["no_persistence"]["correct_history"], 200)
    true = effect(values["persistent"]["current_only"], values["persistent"]["true_factor_oracle"], 300)
    did = np.asarray(p["sequence_deltas"]) - np.asarray(n["sequence_deltas"])
    result = {
        "contract_id": CONTRACT_ID, "primary_metric": "later_E2_E4_mean_pose_auc10_to_waypoint",
        "n_sequences_per_condition": n_sequences, "episodes_per_sequence": episodes,
        "persistent_correct_history": p, "no_persistence_correct_history": n,
        "persistent_true_factor": true,
        "persistent_shuffled_history": effect(values["persistent"]["current_only"], values["persistent"]["shuffled_history"], 400),
        "persistent_wrong_sequence_history": effect(values["persistent"]["current_only"], values["persistent"]["wrong_sequence_history"], 500),
        "did": {"mean": float(did.mean()), "bootstrap_ci95": bootstrap_ci(did, 600), "positive_fraction": float(np.mean(did > 1e-12))},
        "true_gap_recovery": float(p["mean_delta"] / true["mean_delta"]),
        "deadline_success": {
            condition: {policy: float(successes[condition][policy].mean()) for policy in POLICIES}
            for condition in CONDITIONS
        },
        "policy_means": {condition: {policy: float(v.mean()) for policy, v in values[condition].items()} for condition in CONDITIONS},
        "by_factor": {}, "by_episode": {}, "estimator": {}, "donor_factor_match": {},
    }
    for f in FACTORS_DEG:
        ids = [s for s in range(n_sequences) if factor_for("persistent", s, 0, n_sequences, episodes) == f]
        result["by_factor"][str(f)] = effect(values["persistent"]["current_only"][ids], values["persistent"]["correct_history"][ids], 700 + int(f + 30))
    for e in range(1, episodes):
        result["by_episode"][str(e + 1)] = {}
        for condition in CONDITIONS:
            erows = [r for r in rows[condition] if int(r["episode_index"]) == e]
            current = [r["policies"]["current_only"]["metrics"]["pose_auc10"] for r in erows]
            history = [r["policies"]["correct_history"]["metrics"]["pose_auc10"] for r in erows]
            result["by_episode"][str(e + 1)][condition] = effect(current, history, 800 + e)
    for condition in CONDITIONS:
        later = [r for r in rows[condition] if int(r["episode_index"]) > 0]
        for policy in ("correct_history", "shuffled_history", "wrong_sequence_history"):
            errors = [abs(wrapped_degrees_error(r["policies"][policy]["context_degrees"], r["factor_deg"])) for r in later]
            matches = [float(d["donor_factor_deg"]) == float(r["factor_deg"]) for r in later for d in r["policies"][policy]["donors"]]
            result["estimator"].setdefault(condition, {})[policy] = {
                "angle_mae_degrees": float(np.mean(errors)), "angle_median_absolute_error_degrees": float(np.median(errors)),
            }
            result["donor_factor_match"].setdefault(condition, {})[policy] = float(np.mean(matches))
    checks = {}
    for condition in CONDITIONS:
        complete = len(rows[condition]) == n_sequences * episodes
        identity = all(
            r["policies"]["current_only"]["state_sha256"] == r["policies"]["population_prior"]["state_sha256"]
            for r in rows[condition]
        )
        e1 = all(len({(r["policies"][p]["command_sha256"], r["policies"][p]["state_sha256"]) for p in POLICIES if p != "true_factor_oracle"}) == 1 for r in rows[condition] if int(r["episode_index"]) == 0)
        checks[condition] = {"complete": complete, "current_population_identity": identity, "episode_one_identity": e1}
    result["structural_checks"] = checks
    result["valid"] = all(all(x.values()) for x in checks.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_rotation_early_waypoint_stage1_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_rotation_early_waypoint_stage1_contract_zh.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage1"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.sequences = 4 if args.smoke else 32; args.episodes = 2 if args.smoke else 4
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle: segments = pickle.load(handle)["segments"]
    if args.mode == "inspect":
        persistent, changing = factor_schedules(); wrong, shuffled = donor_maps()
        payload = {"contract_id": CONTRACT_ID, "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract),
                   "segments": formal_segment_indices(segments).reshape(32, 4).tolist(), "persistent_factor_ids": persistent.tolist(),
                   "no_persistence_factor_ids": changing.tolist(), "wrong_donors": wrong.tolist(), "shuffled_donors": shuffled.tolist()}
        dump_json(args.output_dir / "selection_audit.json", payload); print(json.dumps(payload, indent=2)); return
    if args.mode == "summarize":
        result = summarize(args.output_dir, args.sequences, args.episodes); dump_json(args.output_dir / "runner_summary.json", result); print(json.dumps(result, indent=2)); return
    if sha256(args.design) != EXPECTED_DESIGN_SHA256: raise RuntimeError("frozen design hash mismatch")
    seed_all(9301); device = torch.device(args.device); manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {"contract_id": CONTRACT_ID, "mode": "smoke" if args.smoke else "formal", "git_revision": git_revision(),
                    "design_path": str(args.design), "design_sha256": sha256(args.design), "contract_path": str(args.contract),
                    "contract_sha256": sha256(args.contract), "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                    "data": str(args.data), "data_sha256": sha256(args.data), "command": " ".join(__import__("sys").argv),
                    "sequences": args.sequences, "episodes": args.episodes, "started_unix": time.time(), "resource_start": resource_snapshot(device)}
        dump_json(manifest_path, manifest)
    else: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base, wrapper, _ = load_world_model(args.checkpoint, device); preprocessor = make_preprocessor(); env = make_env()
    first = scenario(segments, "persistent", 0, 0, args.sequences, args.episodes)
    initial, _, _, _ = prepare_waypoint(env, segments[first["segment_index"]], first["env_seed"]); start_obs, _ = env.prepare(first["env_seed"], initial)
    manifest["identity_audit"] = identity_audit(base, wrapper, preprocessor, start_obs); dump_json(manifest_path, manifest)
    for condition in CONDITIONS:
        evidence = generate_evidence(args, condition, wrapper, preprocessor, segments, env)
        run_evaluations(args, condition, wrapper, preprocessor, segments, env, evidence)
    result = summarize(args.output_dir, args.sequences, args.episodes); dump_json(args.output_dir / "runner_summary.json", result); print(json.dumps(result, indent=2))
    manifest["finished_unix"] = time.time(); manifest["resource_end"] = resource_snapshot(device); dump_json(manifest_path, manifest)


if __name__ == "__main__": main()
