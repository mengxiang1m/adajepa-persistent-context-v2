"""Prospective split data collection for the matrix learned surrogate gate."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.pushobj_matrix_stage0 import (
    POPULATION_PRIOR_MATRIX,
    array_sha256,
    execute_matrix,
    factor_matrix,
    identity_audit,
    load_matrix_world_model,
    plan_matrix_waypoint,
)
from research.persistent_context_v2.pushobj_matrix_stage1 import (
    BayesianMatrixContext,
    infer_matrix_observations,
    observations_sha256,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    WINDOW,
    deadline_success,
    nominal_block_displacement_at_10,
    prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl,
    dump_json,
    git_revision,
    make_env,
    make_preprocessor,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-matrix-learned-surrogate-gate-v1"
EXPECTED_DESIGN_SHA256 = "607d5e943635e34c10883ac16b37162c212e1b0e30fd075bcb1f7e6136f3d756"
SPLITS = ("train", "dev", "formal")


def load_design(path: Path) -> dict:
    if sha256(path) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen learned-gate design hash mismatch")
    design = json.loads(path.read_text(encoding="utf-8"))
    if design["contract_id"] != CONTRACT_ID:
        raise RuntimeError("contract id mismatch")
    return design


def scenario(design: dict, split: str, sequence_id: int, episode_index: int) -> dict:
    split_design = design["splits"][split]
    factor = design["formal_factors"][int(sequence_id) % len(design["formal_factors"])]
    ordinal = int(sequence_id) * 2 + int(episode_index)
    return {
        "split": split,
        "sequence_id": int(sequence_id),
        "episode_index": int(episode_index),
        "episode": int(episode_index) + 1,
        "segment_index": int(split_design["segment_indices"][ordinal]),
        "factor_index": int(sequence_id) % len(design["formal_factors"]),
        "rotation_degrees": float(factor["rotation_degrees"]),
        "gain": float(factor["gain"]),
        "true_matrix": factor_matrix(float(factor["rotation_degrees"]), float(factor["gain"])),
        "env_seed": int(split_design["env_seed_base"]) + int(sequence_id) * 100 + int(episode_index),
        "cem_seed": int(split_design["cem_seed_base"]) + int(sequence_id) * 100 + int(episode_index),
    }


def inspect_design(design: dict, segments) -> dict:
    all_indices = []
    split_checks = {}
    for split in SPLITS:
        indices = [int(value) for value in design["splits"][split]["segment_indices"]]
        all_indices.extend(indices)
        displacements = [nominal_block_displacement_at_10(segments[index]) for index in indices]
        split_checks[split] = {
            "count": len(indices),
            "unique": len(set(indices)) == len(indices),
            "minimum_displacement": float(min(displacements)),
            "eligible": bool(min(displacements) >= float(design["segment_pool"]["minimum_nominal_block_displacement_at_10"])),
            "factor_counts": {
                str(index): int(sum(sequence_id % 8 == index for sequence_id in range(design["sequences_per_split"])))
                for index in range(8)
            },
        }
    excluded = set(int(value) for value in design["segment_pool"]["excluded_matrix_stage1_segments"])
    return {
        "contract_id": CONTRACT_ID,
        "split_checks": split_checks,
        "all_segment_count": len(all_indices),
        "all_segments_unique": len(set(all_indices)) == len(all_indices),
        "overlap_with_matrix_stage1": len(set(all_indices) & excluded),
        "valid": bool(
            len(all_indices) == 192
            and len(set(all_indices)) == 192
            and not (set(all_indices) & excluded)
            and all(row["count"] == 64 and row["unique"] and row["eligible"] for row in split_checks.values())
        ),
    }


def completed_sequence_ids(path: Path) -> set[int]:
    return {int(row["sequence_id"]) for row in read_jsonl(path) if row.get("record_type") == "paired_sequence"}


def plan_and_execute(wrapper, preprocessor, env, segments, meta: dict, context_matrix) -> dict:
    initial, goal_obs, nominal_states, _ = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
    start_obs, _ = env.prepare(meta["env_seed"], initial)
    commands, planner = plan_matrix_waypoint(
        wrapper, preprocessor, start_obs, goal_obs, np.asarray(context_matrix, dtype=np.float64), meta["cem_seed"]
    )
    states, effective, contacts, coverages = execute_matrix(
        env, initial, meta["env_seed"], commands, np.asarray(meta["true_matrix"], dtype=np.float64)
    )
    return {
        "initial_state": initial,
        "goal_state": nominal_states[-1],
        "commands": commands,
        "effective_actions": effective,
        "states": states,
        "contacts": contacts,
        "coverages": coverages,
        "metrics": pose_metrics(states, nominal_states[-1], WINDOW),
        "deadline_success": deadline_success(states, nominal_states[-1]),
        "command_sha256": array_sha256(commands),
        "state_sha256": array_sha256(states),
        "planner": planner,
    }


def collect_split(args, design: dict, segments) -> dict:
    split_dir = args.output_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    raw_path = split_dir / "raw.jsonl"
    manifest_path = split_dir / "manifest.json"
    device = torch.device(args.device)
    seed_all(int(design["splits"][args.split]["env_seed_base"]) - 1)
    manifest = {
        "contract_id": CONTRACT_ID,
        "split": args.split,
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
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("design_sha256", "checkpoint_sha256", "data_sha256", "split"):
            if previous[key] != manifest[key]:
                raise RuntimeError(f"resume manifest mismatch: {key}")
        manifest = previous
    else:
        dump_json(manifest_path, manifest)
    base, wrapper, _ = load_matrix_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    first = scenario(design, args.split, 0, 0)
    initial, _, _, _ = prepare_waypoint(env, segments[first["segment_index"]], first["env_seed"])
    start_obs, _ = env.prepare(first["env_seed"], initial)
    manifest["identity_audit"] = identity_audit(base, wrapper, preprocessor, start_obs)
    completed = completed_sequence_ids(raw_path)
    n_sequences = min(int(args.limit_sequences), int(design["sequences_per_split"]))
    for sequence_id in range(n_sequences):
        if sequence_id in completed:
            continue
        started = time.perf_counter()
        e1_meta = scenario(design, args.split, sequence_id, 0)
        evidence = plan_and_execute(wrapper, preprocessor, env, segments, e1_meta, POPULATION_PRIOR_MATRIX)
        observations, accepted_indices = infer_matrix_observations(evidence["commands"], evidence["states"])
        posterior = BayesianMatrixContext()
        posterior.update_observations(observations)
        e2_meta = scenario(design, args.split, sequence_id, 1)
        population = plan_and_execute(wrapper, preprocessor, env, segments, e2_meta, POPULATION_PRIOR_MATRIX)
        context = plan_and_execute(wrapper, preprocessor, env, segments, e2_meta, posterior.mean_matrix)
        row = {
            "record_type": "paired_sequence",
            "contract_id": CONTRACT_ID,
            "split": args.split,
            "sequence_id": sequence_id,
            "factor_index": e2_meta["factor_index"],
            "rotation_degrees": e2_meta["rotation_degrees"],
            "gain": e2_meta["gain"],
            "e1": {
                **e1_meta,
                "commands": evidence["commands"],
                "states": evidence["states"],
                "accepted_indices": accepted_indices,
                "matrix_observations": observations,
                "observation_sha256": observations_sha256(observations),
                "command_sha256": evidence["command_sha256"],
                "state_sha256": evidence["state_sha256"],
                "planner": evidence["planner"],
            },
            "posterior": posterior.as_dict(),
            "e2": {
                **e2_meta,
                "initial_state": population["initial_state"],
                "goal_state": population["goal_state"],
                "population": population,
                "context": context,
            },
            "elapsed_s": time.perf_counter() - started,
            "resource": resource_snapshot(device),
        }
        append_jsonl(raw_path, row)
        delta = population["metrics"]["pose_auc10"] - context["metrics"]["pose_auc10"]
        print(f"{args.split} sequence={sequence_id} factor={e2_meta['factor_index']} delta={delta:+.6f}", flush=True)
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "paired_sequence"]
    deltas = np.asarray([
        row["e2"]["population"]["metrics"]["pose_auc10"] - row["e2"]["context"]["metrics"]["pose_auc10"]
        for row in rows
    ], dtype=np.float64)
    summary = {
        "contract_id": CONTRACT_ID,
        "split": args.split,
        "n_sequences": len(rows),
        "population_mean": float(np.mean([row["e2"]["population"]["metrics"]["pose_auc10"] for row in rows])),
        "context_mean": float(np.mean([row["e2"]["context"]["metrics"]["pose_auc10"] for row in rows])),
        "mean_delta": float(deltas.mean()),
        "relative_improvement": float(deltas.mean() / np.mean([row["e2"]["population"]["metrics"]["pose_auc10"] for row in rows])),
        "positive_fraction": float(np.mean(deltas > 1e-12)),
        "negative_fraction": float(np.mean(deltas < -1e-12)),
        "all_observations_positive": bool(all(len(row["e1"]["matrix_observations"]) > 0 for row in rows)),
        "valid": len(rows) == n_sequences,
    }
    dump_json(split_dir / "collection_summary.json", summary)
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(device)
    manifest["raw_sha256"] = sha256(raw_path)
    dump_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run"))
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_learned_gate_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_matrix_learned_gate_contract_zh.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-sequences", type=int, default=32)
    args = parser.parse_args()
    design = load_design(args.design)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    inspection = inspect_design(design, segments)
    if not inspection["valid"]:
        raise RuntimeError(f"invalid split design: {inspection}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "selection_audit.json", inspection)
    if args.mode == "inspect":
        print(json.dumps(inspection, indent=2))
        return
    collect_split(args, design, segments)


if __name__ == "__main__":
    main()
