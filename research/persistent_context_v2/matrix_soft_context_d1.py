"""Frozen D1 exploratory dose response for partial matrix context."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.matrix_task_interaction_d0 import model_sha256, rng_state_digest
from research.persistent_context_v2.pushobj_matrix_stage0 import (
    POPULATION_PRIOR_MATRIX,
    array_sha256,
    execute_matrix,
    load_matrix_world_model,
    plan_matrix_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    WINDOW,
    deadline_success,
    prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl,
    dump_json,
    make_env,
    make_preprocessor,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    sha256,
)


D1_ID = "persistent-context-v2-matrix-soft-context-d1-exploratory-v1"


def blend_context(prior, posterior, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    prior = np.asarray(prior, dtype=np.float64)
    posterior = np.asarray(posterior, dtype=np.float64)
    if prior.shape != (2, 2) or posterior.shape != (2, 2):
        raise ValueError("prior and posterior must be 2x2")
    return (1.0 - alpha) * prior + alpha * posterior


def load_design(path: Path) -> dict:
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("design_id") != D1_ID:
        raise RuntimeError("D1 design id mismatch")
    if [float(value) for value in design.get("alphas", [])] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("D1 alpha grid mismatch")
    return design


def load_input_rows(data_dir: Path, design: dict, limit_per_split: int | None = None) -> list[dict]:
    result = []
    for split in design["splits"]:
        rows = [row for row in read_jsonl(data_dir / split / "raw.jsonl") if row.get("record_type") == "paired_sequence"]
        rows.sort(key=lambda row: int(row["sequence_id"]))
        if limit_per_split:
            rows = rows[: int(limit_per_split)]
        elif len(rows) != int(design["expected_sequences_per_split"]):
            raise RuntimeError(f"expected 32 {split} rows, got {len(rows)}")
        result.extend(rows)
    return result


def endpoint_reference(row: dict, alpha: float) -> dict | None:
    if abs(float(alpha)) <= 1e-12:
        return row["e2"]["population"]
    if abs(float(alpha) - 1.0) <= 1e-12:
        return row["e2"]["context"]
    return None


def run(args) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design = load_design(args.design)
    rows = load_input_rows(args.data_dir, design, args.limit_per_split)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index if device.index is not None else torch.cuda.current_device())
    resource_start = resource_snapshot(device)
    _, wrapper, _ = load_matrix_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    raw_path = args.output_dir / "raw.jsonl"
    completed = {
        (str(row["split"]), int(row["sequence_id"]), float(row["alpha"]))
        for row in read_jsonl(raw_path)
        if row.get("record_type") == "d1_alpha_treatment"
    }
    started = time.time()
    model_before, rng_before = model_sha256(wrapper), rng_state_digest()
    max_initial_error = max_goal_error = max_endpoint_state_error = max_endpoint_metric_error = 0.0
    endpoint_hash_mismatches = 0
    written = 0
    for row in rows:
        episode = row["e2"]
        initial, goal_obs, nominal_states, _ = prepare_waypoint(
            env, segments[int(episode["segment_index"])], int(episode["env_seed"])
        )
        initial_error = float(np.max(np.abs(np.asarray(initial) - np.asarray(episode["initial_state"]))))
        goal_error = float(np.max(np.abs(np.asarray(nominal_states[-1]) - np.asarray(episode["goal_state"]))))
        max_initial_error, max_goal_error = max(max_initial_error, initial_error), max(max_goal_error, goal_error)
        if initial_error > 1e-6 or goal_error > 1e-6:
            raise RuntimeError(f"scene replay mismatch initial={initial_error} goal={goal_error}")
        start_obs, _ = env.prepare(int(episode["env_seed"]), initial)
        posterior = np.asarray(row["posterior"]["mean_matrix"], dtype=np.float64)
        truth = np.asarray(episode["true_matrix"], dtype=np.float64)
        for alpha in [float(value) for value in design["alphas"]]:
            key = (str(row["split"]), int(row["sequence_id"]), alpha)
            if key in completed:
                continue
            context = blend_context(POPULATION_PRIOR_MATRIX, posterior, alpha)
            commands, planner = plan_matrix_waypoint(
                wrapper, preprocessor, start_obs, goal_obs, context, int(episode["cem_seed"])
            )
            states, effective, contacts, coverages = execute_matrix(
                env, initial, int(episode["env_seed"]), commands, truth
            )
            metrics = pose_metrics(states, nominal_states[-1], WINDOW)
            reference = endpoint_reference(row, alpha)
            endpoint = None
            if reference is not None:
                reference_states = np.asarray(reference["states"], dtype=np.float64)
                state_error = float(np.max(np.abs(np.asarray(states, dtype=np.float64) - reference_states)))
                metric_error = abs(float(metrics["pose_auc10"]) - float(reference["metrics"]["pose_auc10"]))
                command_match = array_sha256(commands) == str(reference["command_sha256"])
                state_match = array_sha256(states) == str(reference["state_sha256"])
                max_endpoint_state_error = max(max_endpoint_state_error, state_error)
                max_endpoint_metric_error = max(max_endpoint_metric_error, metric_error)
                endpoint_hash_mismatches += int(not (command_match and state_match))
                endpoint = {
                    "reference_policy": "population" if alpha == 0.0 else "context",
                    "command_hash_match": command_match,
                    "state_hash_match": state_match,
                    "state_max_abs": state_error,
                    "pose_auc10_abs": metric_error,
                }
            append_jsonl(
                raw_path,
                {
                    "record_type": "d1_alpha_treatment",
                    "design_id": D1_ID,
                    "split": row["split"],
                    "sequence_id": int(row["sequence_id"]),
                    "factor_index": int(row["factor_index"]),
                    "rotation_degrees": float(row["rotation_degrees"]),
                    "gain": float(row["gain"]),
                    "segment_index": int(episode["segment_index"]),
                    "env_seed": int(episode["env_seed"]),
                    "cem_seed": int(episode["cem_seed"]),
                    "alpha": alpha,
                    "prior_matrix": POPULATION_PRIOR_MATRIX,
                    "posterior_matrix": posterior,
                    "context_matrix": context,
                    "true_matrix": truth,
                    "initial_state": initial,
                    "goal_state": nominal_states[-1],
                    "commands": commands,
                    "effective_actions": effective,
                    "states": states,
                    "contacts": contacts,
                    "coverages": coverages,
                    "metrics": metrics,
                    "deadline_success": deadline_success(states, nominal_states[-1]),
                    "command_sha256": array_sha256(commands),
                    "state_sha256": array_sha256(states),
                    "planner": planner,
                    "endpoint_identity": endpoint,
                },
            )
            written += 1
        print(f"D1 {row['split']} sequence={row['sequence_id']} treatments complete", flush=True)
    model_after, rng_after = model_sha256(wrapper), rng_state_digest()
    manifest = {
        "design_id": D1_ID,
        "evidence_level": design["evidence_level"],
        "command": " ".join(__import__("sys").argv),
        "design": str(args.design),
        "design_sha256": sha256(args.design),
        "source_snapshot": str(args.source_snapshot),
        "source_snapshot_sha256": sha256(args.source_snapshot),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "input_raw_sha256": {split: sha256(args.data_dir / split / "raw.jsonl") for split in design["splits"]},
        "expected_treatments": len(rows) * len(design["alphas"]),
        "new_treatments_written": written,
        "started_unix": started,
        "finished_unix": time.time(),
        "resource_start": resource_start,
        "resource_end": resource_snapshot(device),
        "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after,
        "model_state_unchanged": model_before == model_after,
        "rng_digest_before": rng_before,
        "rng_digest_after": rng_after,
        "rng_unchanged": rng_before == rng_after,
        "initial_replay_max_abs": max_initial_error,
        "goal_replay_max_abs": max_goal_error,
        "endpoint_identity": {
            "hash_mismatches": endpoint_hash_mismatches,
            "state_max_abs": max_endpoint_state_error,
            "pose_auc10_max_abs": max_endpoint_metric_error,
        },
        "raw_sha256": sha256(raw_path),
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    return manifest


def bootstrap_ci(values: np.ndarray, seed: int, resamples: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    indexes = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def dose_summary(rows: list[dict], design: dict) -> dict:
    alphas = [float(value) for value in design["alphas"]]
    keys = sorted({(row["split"], int(row["sequence_id"])) for row in rows})
    lookup = {(row["split"], int(row["sequence_id"]), float(row["alpha"])): row for row in rows}
    if any((split, sequence, alpha) not in lookup for split, sequence in keys for alpha in alphas):
        raise RuntimeError("incomplete alpha grid")
    costs = np.asarray([[lookup[(split, sequence, alpha)]["metrics"]["pose_auc10"] for alpha in alphas] for split, sequence in keys], dtype=np.float64)
    population = costs[:, 0]
    result = {"n_sequences": len(keys), "alphas": alphas, "by_alpha": {}, "by_split": {}, "by_factor": {}}
    for index, alpha in enumerate(alphas):
        delta = population - costs[:, index]
        result["by_alpha"][str(alpha)] = {
            "mean_cost": float(costs[:, index].mean()),
            "mean_delta_vs_population": float(delta.mean()),
            "relative_improvement_vs_population": float(delta.mean() / population.mean()),
            "bootstrap_ci95_delta": bootstrap_ci(delta, int(design["bootstrap_seed"]) + index, int(design["bootstrap_resamples"])),
            "positive_fraction": float(np.mean(delta > 1e-12)),
            "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
            "harm_fraction": float(np.mean(delta < -1e-12)),
            "mean_delta_vs_full_context": float(np.mean(costs[:, -1] - costs[:, index])),
        }
    best_fixed = int(np.argmin(costs.mean(axis=0)))
    best_per_sequence = np.argmin(costs, axis=1)
    monotonic_up = np.all(np.diff(costs, axis=1) >= -1e-12, axis=1)
    monotonic_down = np.all(np.diff(costs, axis=1) <= 1e-12, axis=1)
    result["exploratory_ceilings"] = {
        "best_fixed_alpha": alphas[best_fixed],
        "best_fixed_mean_cost": float(costs[:, best_fixed].mean()),
        "per_sequence_best_mean_cost": float(np.min(costs, axis=1).mean()),
        "per_sequence_best_alpha_counts": {str(alpha): int(np.sum(best_per_sequence == index)) for index, alpha in enumerate(alphas)},
        "interior_best_fraction": float(np.mean((best_per_sequence > 0) & (best_per_sequence < len(alphas) - 1))),
        "non_monotonic_fraction": float(np.mean(~(monotonic_up | monotonic_down))),
    }
    for split in design["splits"]:
        mask = np.asarray([key[0] == split for key in keys])
        result["by_split"][split] = {
            str(alpha): {
                "mean_delta_vs_population": float(np.mean(population[mask] - costs[mask, index])),
                "harm_fraction": float(np.mean((population[mask] - costs[mask, index]) < -1e-12)),
            }
            for index, alpha in enumerate(alphas)
        }
    factor_indexes = np.asarray([int(lookup[(split, sequence, alphas[0])]["factor_index"]) for split, sequence in keys])
    for factor in sorted(set(factor_indexes.tolist())):
        mask = factor_indexes == factor
        result["by_factor"][str(factor)] = {
            str(alpha): {
                "mean_delta_vs_population": float(np.mean(population[mask] - costs[mask, index])),
                "harm_fraction": float(np.mean((population[mask] - costs[mask, index]) < -1e-12)),
            }
            for index, alpha in enumerate(alphas)
        }
    return result


def summarize(args) -> dict:
    design = load_design(args.design)
    rows = [row for row in read_jsonl(args.output_dir / "raw.jsonl") if row.get("record_type") == "d1_alpha_treatment"]
    result = dose_summary(rows, design)
    result.update({"design_id": D1_ID, "evidence_level": design["evidence_level"]})
    dump_json(args.output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "summarize"))
    parser.add_argument("--data-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_soft_context_d1_design.json"))
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args) if args.mode == "run" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
