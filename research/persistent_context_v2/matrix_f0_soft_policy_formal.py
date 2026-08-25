"""Prospective F0 soft-context policy formal experiment."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.matrix_learned_gate import feature_row
from research.persistent_context_v2.matrix_soft_context_d1 import blend_context
from research.persistent_context_v2.matrix_soft_policy_d2 import decisions, fit_model, predict
from research.persistent_context_v2.matrix_task_interaction_d0 import model_sha256, rng_state_digest
from research.persistent_context_v2.pushobj_matrix_stage0 import (
    POPULATION_PRIOR_MATRIX,
    array_sha256,
    execute_matrix,
    factor_matrix,
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
    make_env,
    make_preprocessor,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-matrix-f0-soft-policy-formal-v1"
EXPECTED_DESIGN_SHA256 = "a27cd6994b2fb2a6720ddf2949ccc4534bea11b35159da26905bd2f85b30fc5f"
EXPECTED_CONTRACT_SHA256 = "98d0645db072c26d15669b931143dc51c211395f7f3849c017f98b11f8dd7315"
EXPECTED_SELECTION_SHA256 = "3338e6eca7d63de0649cb52fba1c4c8ebc1e0904c5e79e151813dc09b5e71d4c"
SPLITS = ("train", "dev", "formal")


def load_frozen(design_path: Path, contract_path: Path, selection_path: Path) -> tuple[dict, dict]:
    if sha256(design_path) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen design hash mismatch")
    if sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract hash mismatch")
    if sha256(selection_path) != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("frozen selection hash mismatch")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if design["contract_id"] != CONTRACT_ID:
        raise RuntimeError("contract id mismatch")
    return design, selection


def inspect_selection(design: dict, selection: dict, segments, availability_audit: Path) -> dict:
    audit_hash = sha256(availability_audit)
    audit = json.loads(availability_audit.read_text(encoding="utf-8"))
    allowed = set(int(value) for value in audit["eligible_matrix_unexposed_indices"])
    used = []
    split_checks = {}
    for split in SPLITS:
        indexes = [int(value) for value in selection[split]]
        expected = 2 * int(design["splits"][split]["sequences"])
        used.extend(indexes)
        displacement = [nominal_block_displacement_at_10(segments[index]) for index in indexes]
        split_checks[split] = {
            "count": len(indexes),
            "expected": expected,
            "unique": len(set(indexes)) == len(indexes),
            "minimum_displacement": float(min(displacement)),
            "all_in_frozen_unexposed_pool": set(indexes) <= allowed,
        }
    valid = (
        audit_hash == design["availability_audit_sha256"]
        and len(used) == 384
        and len(set(used)) == 384
        and all(row["count"] == row["expected"] and row["unique"] and row["minimum_displacement"] >= 10.0 and row["all_in_frozen_unexposed_pool"] for row in split_checks.values())
    )
    return {
        "contract_id": CONTRACT_ID,
        "availability_audit_sha256": audit_hash,
        "split_checks": split_checks,
        "all_segment_count": len(used),
        "all_segments_unique": len(set(used)) == len(used),
        "reserve_count": len(selection["reserve"]),
        "valid": bool(valid),
    }


def scenario(design: dict, selection: dict, split: str, sequence_id: int, episode_index: int) -> dict:
    factor_index = int(sequence_id) % len(design["factors"])
    factor = design["factors"][factor_index]
    split_design = design["splits"][split]
    return {
        "split": split,
        "sequence_id": int(sequence_id),
        "episode_index": int(episode_index),
        "episode": int(episode_index) + 1,
        "segment_index": int(selection[split][2 * int(sequence_id) + int(episode_index)]),
        "factor_index": factor_index,
        "rotation_degrees": float(factor["rotation_degrees"]),
        "gain": float(factor["gain"]),
        "true_matrix": factor_matrix(float(factor["rotation_degrees"]), float(factor["gain"])),
        "env_seed": int(split_design["env_seed_base"]) + 100 * int(sequence_id) + int(episode_index),
        "cem_seed": int(split_design["cem_seed_base"]) + 100 * int(sequence_id) + int(episode_index),
    }


def plan_execute(wrapper, preprocessor, env, segments, meta: dict, context_matrix) -> dict:
    initial, goal_obs, nominal_states, _ = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
    start_obs, _ = env.prepare(meta["env_seed"], initial)
    commands, planner = plan_matrix_waypoint(wrapper, preprocessor, start_obs, goal_obs, context_matrix, meta["cem_seed"])
    states, effective, contacts, coverages = execute_matrix(env, initial, meta["env_seed"], commands, meta["true_matrix"])
    return {
        "initial_state": initial,
        "goal_state": nominal_states[-1],
        "context_matrix": np.asarray(context_matrix),
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


def posterior_feature(posterior: dict, design: dict) -> np.ndarray:
    return feature_row({"posterior": posterior}, design)


def load_locked_model(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract_id") != CONTRACT_ID or value.get("locked") is not True:
        raise RuntimeError("invalid locked model")
    return value


def model_prediction(model: dict, x: np.ndarray, alphas: list[float]) -> np.ndarray:
    holder = {"mean": np.asarray(model["mean"]), "scale": np.asarray(model["scale"]), "beta": np.asarray(model["beta"])}
    rows = [{"features": np.asarray(x)}]
    return predict(holder, rows, len(x), alphas)[0]


def collect(args, design: dict, selection: dict, segments) -> dict:
    split = args.split
    split_dir = args.output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    raw_path, manifest_path = split_dir / "raw.jsonl", split_dir / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"existing split manifest: {manifest_path}")
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    seed_all(int(design["splits"][split]["env_seed_base"]) - 1)
    locked_model = None
    model_audit = None
    if split == "formal":
        if args.locked_model is None or args.model_audit is None:
            raise RuntimeError("formal requires --locked-model and --model-audit")
        locked_model = load_locked_model(args.locked_model)
        model_audit = json.loads(args.model_audit.read_text(encoding="utf-8"))
        if not model_audit.get("valid") or model_audit.get("locked_model_sha256") != sha256(args.locked_model):
            raise RuntimeError("formal model audit is absent or invalid")
    started = time.time()
    _, wrapper, _ = load_matrix_world_model(args.checkpoint, device)
    model_before, rng_before = model_sha256(wrapper), rng_state_digest()
    manifest = {
        "contract_id": CONTRACT_ID,
        "split": split,
        "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract), "selection_sha256": sha256(args.selection),
        "source_snapshot_sha256": sha256(args.source_snapshot), "checkpoint_sha256": sha256(args.checkpoint), "data_sha256": sha256(args.data),
        "locked_model_sha256": sha256(args.locked_model) if locked_model else None,
        "model_audit_sha256": sha256(args.model_audit) if model_audit else None,
        "command": " ".join(__import__("sys").argv), "started_unix": started, "resource_start": resource_snapshot(device),
        "model_state_sha256_before": model_before, "rng_digest_before": rng_before,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("design_sha256", "contract_sha256", "selection_sha256", "source_snapshot_sha256", "checkpoint_sha256", "data_sha256", "locked_model_sha256", "model_state_sha256_before"):
            if previous.get(key) != manifest.get(key): raise RuntimeError(f"resume mismatch: {key}")
        manifest = previous
    else:
        dump_json(manifest_path, manifest)
    preprocessor, env = make_preprocessor(), make_env()
    completed = {int(row["sequence_id"]) for row in read_jsonl(raw_path) if row.get("record_type") == "f0_soft_policy_sequence"}
    expected = int(design["splits"][split]["sequences"])
    limit = min(expected, int(args.limit_sequences) if args.limit_sequences else expected)
    alphas = [float(value) for value in design["alphas"]]
    for sequence_id in range(limit):
        if sequence_id in completed: continue
        sequence_started = time.perf_counter()
        e1_meta = scenario(design, selection, split, sequence_id, 0)
        e1 = plan_execute(wrapper, preprocessor, env, segments, e1_meta, POPULATION_PRIOR_MATRIX)
        observations, accepted = infer_matrix_observations(e1["commands"], e1["states"])
        posterior = BayesianMatrixContext(); posterior.update_observations(observations)
        posterior_dict = posterior.as_dict()
        x = posterior_feature(posterior_dict, design)
        selected_alpha, predictions = None, None
        order = list(alphas)
        decision_unix = None
        if locked_model is not None:
            predictions = model_prediction(locked_model, x, alphas)
            selected_index = int(decisions(predictions[None, :])[0])
            selected_alpha = alphas[selected_index]
            order = [selected_alpha] + [alpha for alpha in alphas if alpha != selected_alpha]
            decision_unix = time.time()
        e2_meta = scenario(design, selection, split, sequence_id, 1)
        treatments = {}
        for execution_index, alpha in enumerate(order):
            matrix = blend_context(POPULATION_PRIOR_MATRIX, posterior.mean_matrix, alpha)
            treatment = plan_execute(wrapper, preprocessor, env, segments, e2_meta, matrix)
            treatment["execution_index"] = execution_index
            treatments[str(alpha)] = treatment
        append_jsonl(raw_path, {
            "record_type": "f0_soft_policy_sequence", "contract_id": CONTRACT_ID, "split": split,
            "sequence_id": sequence_id, "factor_index": e2_meta["factor_index"], "rotation_degrees": e2_meta["rotation_degrees"], "gain": e2_meta["gain"],
            "e1": {**e1_meta, "commands": e1["commands"], "states": e1["states"], "accepted_indices": accepted, "matrix_observations": observations, "observation_sha256": observations_sha256(observations), "command_sha256": e1["command_sha256"], "state_sha256": e1["state_sha256"]},
            "posterior": posterior_dict, "features": x, "feature_names": design["feature_basis"],
            "e2_meta": e2_meta, "selected_alpha": selected_alpha, "predicted_benefits": predictions,
            "decision_unix": decision_unix, "execution_order": order, "treatments": treatments,
            "elapsed_s": time.perf_counter() - sequence_started, "resource": resource_snapshot(device),
        })
        print(f"{split} sequence={sequence_id} selected={selected_alpha} complete", flush=True)
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "f0_soft_policy_sequence"]
    model_after, rng_after = model_sha256(wrapper), rng_state_digest()
    manifest.update({"finished_unix": time.time(), "resource_end": resource_snapshot(device), "rows": len(rows), "raw_sha256": sha256(raw_path),
                     "model_state_sha256_after": model_after, "model_state_unchanged": model_before == model_after,
                     "rng_digest_after": rng_after, "rng_unchanged": rng_before == rng_after})
    dump_json(manifest_path, manifest)
    return manifest


def row_for_fit(row: dict, alphas: list[float]) -> dict:
    return {"features": np.asarray(row["features"]), "costs": np.asarray([row["treatments"][str(alpha)]["metrics"]["pose_auc10"] for alpha in alphas])}


def fit_lock(args, design: dict) -> dict:
    train_path, dev_path = args.output_dir / "train" / "raw.jsonl", args.output_dir / "dev" / "raw.jsonl"
    train = [row_for_fit(row, design["alphas"]) for row in read_jsonl(train_path) if row.get("record_type") == "f0_soft_policy_sequence"]
    dev = [row_for_fit(row, design["alphas"]) for row in read_jsonl(dev_path) if row.get("record_type") == "f0_soft_policy_sequence"]
    if len(train) != 64 or len(dev) != 32: raise RuntimeError("fit requires complete train=64 and dev=32")
    candidates = []
    for ridge in design["ridge_alphas"]:
        model = fit_model(train, 6, float(ridge)); prediction = predict(model, dev, 6, design["alphas"]); selected = decisions(prediction)
        costs = np.stack([row["costs"] for row in dev]); outcome = costs[np.arange(len(dev)), selected]
        candidates.append({"ridge_alpha": float(ridge), "dev_mean_delta": float(np.mean(costs[:, 0] - outcome)), "selection_counts": {str(alpha): int(np.sum(selected == i)) for i, alpha in enumerate(design["alphas"])}})
    chosen = max(candidates, key=lambda row: (row["dev_mean_delta"], row["ridge_alpha"]))
    model = fit_model(train + dev, 6, chosen["ridge_alpha"])
    locked = {
        "contract_id": CONTRACT_ID, "locked": True, "selected_ridge_alpha": chosen["ridge_alpha"], "ridge_candidates": candidates,
        "feature_names": design["feature_basis"], "alphas": design["alphas"], "mean": model["mean"], "scale": model["scale"], "beta": model["beta"],
        "train_raw_sha256": sha256(train_path), "dev_raw_sha256": sha256(dev_path), "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract), "selection_sha256": sha256(args.selection), "source_snapshot_sha256": sha256(args.source_snapshot),
    }
    dump_json(args.output_dir / "locked_model.json", locked); dump_json(args.output_dir / "model_selection.json", {"chosen": chosen, "candidates": candidates})
    print(json.dumps({"locked_model_sha256": sha256(args.output_dir / "locked_model.json"), "chosen": chosen}, indent=2))
    return locked


def bootstrap_ci(values, design, stream):
    values = np.asarray(values); rng = np.random.default_rng(int(design["bootstrap_seed"]) + stream)
    indexes = rng.integers(0, len(values), size=(int(design["bootstrap_resamples"]), len(values)))
    means = values[indexes].mean(1); return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def evaluate(args, design: dict) -> dict:
    rows = [row for row in read_jsonl(args.output_dir / "formal" / "raw.jsonl") if row.get("record_type") == "f0_soft_policy_sequence"]
    if len(rows) != 96: raise RuntimeError("evaluation requires 96 formal rows")
    alphas = design["alphas"]; costs = np.asarray([[row["treatments"][str(a)]["metrics"]["pose_auc10"] for a in alphas] for row in rows])
    selected_index = np.asarray([alphas.index(float(row["selected_alpha"])) for row in rows]); learned = costs[np.arange(96), selected_index]
    population, fixed05, fixed075, full = costs[:, 0], costs[:, 2], costs[:, 3], costs[:, 4]
    def effect(treatment, stream):
        delta = population - treatment
        return {"mean_cost": float(treatment.mean()), "mean_delta_vs_population": float(delta.mean()), "relative_improvement": float(delta.mean()/population.mean()), "bootstrap_ci95": bootstrap_ci(delta, design, stream), "harm_fraction": float(np.mean(delta < -1e-12)), "positive_fraction": float(np.mean(delta > 1e-12)), "unit_deltas": delta.tolist()}
    primary = fixed075 - learned
    result = {
        "contract_id": CONTRACT_ID, "n_formal": 96, "primary": {"mean_delta_fixed075_minus_learned": float(primary.mean()), "bootstrap_ci95": bootstrap_ci(primary, design, 100), "unit_deltas": primary.tolist(), "learned_minus_fixed_harm_fraction": float(np.mean((population-learned)<-1e-12)-np.mean((population-fixed075)<-1e-12))},
        "policies": {"fixed_alpha_0.5": effect(fixed05, 1), "fixed_alpha_0.75": effect(fixed075, 2), "full_context": effect(full, 3), "F0_soft_policy": effect(learned, 4), "per_sequence_best_ceiling": effect(costs.min(1), 5)},
        "selection_counts": {str(alpha): int(np.sum(selected_index == i)) for i, alpha in enumerate(alphas)},
        "selected_branch_first_valid": bool(all(row["execution_order"][0] == row["selected_alpha"] and row["treatments"][str(row["selected_alpha"])]["execution_index"] == 0 for row in rows)),
    }
    dump_json(args.output_dir / "formal_summary.json", result); print(json.dumps(result, indent=2)); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("inspect","collect","fit","evaluate")); parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_f0_soft_policy_design.json")); parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_matrix_f0_soft_policy_contract_zh.md")); parser.add_argument("--selection", type=Path, default=Path("docs/research/persistent_context_v2_matrix_f0_soft_policy_selection.json"))
    parser.add_argument("--availability-audit", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_segment_availability_20260823.json")); parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth")); parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl")); parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_f0_soft_policy_formal_v1")); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--limit-sequences", type=int); parser.add_argument("--locked-model", type=Path); parser.add_argument("--model-audit", type=Path); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); design, selection = load_frozen(args.design, args.contract, args.selection)
    with args.data.open("rb") as handle: segments = pickle.load(handle)["segments"]
    inspection = inspect_selection(design, selection, segments, args.availability_audit); args.output_dir.mkdir(parents=True, exist_ok=True); dump_json(args.output_dir / "selection_audit.json", inspection)
    if not inspection["valid"]: raise RuntimeError("selection audit invalid")
    if args.mode == "inspect": print(json.dumps(inspection, indent=2))
    elif args.mode == "collect": collect(args, design, selection, segments)
    elif args.mode == "fit": fit_lock(args, design)
    else: evaluate(args, design)


if __name__ == "__main__": main()
