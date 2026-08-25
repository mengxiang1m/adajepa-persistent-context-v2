"""Prospective cross-shape persistent matrix-context formal experiment."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.matrix_soft_context_d1 import blend_context
from research.persistent_context_v2.matrix_task_interaction_d0 import model_sha256, rng_state_digest
from research.persistent_context_v2.pushobj_matrix_stage0 import (
    POPULATION_PRIOR_MATRIX, array_sha256, execute_matrix, factor_matrix,
    load_matrix_world_model, plan_matrix_waypoint,
)
from research.persistent_context_v2.pushobj_matrix_stage1 import (
    BayesianMatrixContext, infer_matrix_observations, observations_sha256,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    WINDOW, deadline_success, nominal_block_displacement_at_10, prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl, dump_json, make_env, make_preprocessor, pose_metrics,
    read_jsonl, resource_snapshot, seed_all, sha256,
)


CONTRACT_ID = "persistent-context-v2-cross-shape-matrix-history-formal-v1"
EXPECTED_DESIGN_SHA256 = "616aaf61054ed77e04c08986cf504486b0363ebdef687f24f7529523a46b05b5"
EXPECTED_CONTRACT_SHA256 = "f657aa557c9a460938ead7583b729cc551ceea27646a0596ebf3e335d7a793f8"
EXPECTED_SELECTION_SHA256 = "f7c3b0e21389ede3010a5c9ea05a246f7d865c401f14d5da0c260f5dcd7a7b80"
SPLITS = ("smoke", "formal")


def segment_hash(segment: dict) -> str:
    import hashlib
    digest = hashlib.sha256()
    for key in ("states", "actions"):
        value = np.asarray(segment[key])
        digest.update(key.encode("ascii")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def load_frozen(design_path: Path, contract_path: Path, selection_path: Path) -> tuple[dict, dict]:
    for path, expected, name in ((design_path, EXPECTED_DESIGN_SHA256, "design"),
                                 (contract_path, EXPECTED_CONTRACT_SHA256, "contract"),
                                 (selection_path, EXPECTED_SELECTION_SHA256, "selection")):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen {name} hash mismatch")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if design.get("contract_id") != CONTRACT_ID or selection.get("contract_id") != CONTRACT_ID:
        raise RuntimeError("contract id mismatch")
    return design, selection


def load_pools(design: dict) -> dict[str, list[dict]]:
    pools = {}
    root = Path(design["data_root"])
    for shape in design["shapes"]:
        path = root / f"val_{shape}" / "plan_targets.pkl"
        if sha256(path) != design["pool_sha256"][shape]:
            raise RuntimeError(f"pool hash mismatch: {shape}")
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if str(data.get("shape")) != shape or int(data.get("seed")) != 42:
            raise RuntimeError(f"pool metadata mismatch: {shape}")
        pools[shape] = data["segments"]
    return pools


def inspect_selection(design: dict, selection: dict, pools: dict, audit_path: Path) -> dict:
    failures, hashes, provenance = [], [], []
    from collections import Counter
    global_hash_counts = Counter(segment_hash(segment) for segments in pools.values() for segment in segments)
    if sha256(audit_path) != design["author_pool_audit_sha256"]:
        failures.append("author pool audit hash mismatch")
    checks = {}
    for split, expected in (("smoke", 6), ("formal", 96)):
        rows = selection[split]
        if len(rows) != expected:
            failures.append(f"{split} row count")
        pair_counts = {str(i): 0 for i in range(6)}; factor_counts = {str(i): 0 for i in range(8)}
        combo_counts = {}
        minimum = float("inf")
        for sequence_id, row in enumerate(rows):
            if int(row["sequence_id"]) != sequence_id:
                failures.append(f"{split} sequence order")
            pair_index, factor_index = int(row["shape_pair_index"]), int(row["factor_index"])
            pair_counts[str(pair_index)] += 1; factor_counts[str(factor_index)] += 1
            combo_counts[f"{pair_index}:{factor_index}"] = combo_counts.get(f"{pair_index}:{factor_index}", 0) + 1
            if [row["e1"]["shape"], row["e2"]["shape"]] != design["shape_pairs"][pair_index]:
                failures.append(f"{split} shape pair")
            if int(row["no_persistence_factor_index"]) != (factor_index + 1) % 8:
                failures.append(f"{split} factor derangement")
            for episode in ("e1", "e2"):
                item = row[episode]; segment = pools[item["shape"]][int(item["segment_index"])]
                digest = segment_hash(segment); key = f"{int(segment['ep_idx'])}:{int(segment['offset'])}"
                hashes.append(digest); provenance.append(key)
                minimum = min(minimum, nominal_block_displacement_at_10(segment))
                if digest != item["segment_sha256"] or key != item["provenance_key"] or global_hash_counts[digest] != 1:
                    failures.append(f"{split} segment identity")
        checks[split] = {"rows": len(rows), "pair_counts": pair_counts, "factor_counts": factor_counts,
                         "combo_counts": combo_counts, "minimum_displacement": float(minimum)}
    for shape, rows in selection["reserve"].items():
        if len(rows) != int(design["reserve_segments_per_shape"]): failures.append("reserve count")
        for item in rows:
            segment = pools[shape][int(item["segment_index"])]
            digest = segment_hash(segment); key = f"{int(segment['ep_idx'])}:{int(segment['offset'])}"
            hashes.append(digest); provenance.append(key)
            if digest != item["segment_sha256"] or key != item["provenance_key"] or global_hash_counts[digest] != 1:
                failures.append("reserve segment identity")
            if nominal_block_displacement_at_10(segment) < 10.0: failures.append("reserve eligibility")
    formal = checks["formal"]
    if set(formal["pair_counts"].values()) != {16} or set(formal["factor_counts"].values()) != {12} or set(formal["combo_counts"].values()) != {2}:
        failures.append("formal balance")
    if len(hashes) != 228 or len(set(hashes)) != 228 or len(set(provenance)) != 228:
        failures.append("global uniqueness")
    if min(row["minimum_displacement"] for row in checks.values()) < 10.0:
        failures.append("eligibility")
    return {"contract_id": CONTRACT_ID, "valid": not failures, "failures": failures,
            "checks": checks, "selected_segment_count": len(hashes),
            "unique_segment_hash_count": len(set(hashes)), "unique_provenance_count": len(set(provenance)),
            "author_pool_audit_sha256": sha256(audit_path)}
def feature(posterior: dict, design: dict) -> np.ndarray:
    z = np.asarray(posterior["mean_z"], dtype=np.float64)
    gain = float(np.linalg.norm(z)); rotation = math.degrees(math.atan2(z[1], z[0]))
    g = (gain - float(design["gain_center"])) / float(design["gain_scale"])
    r = rotation / float(design["rotation_scale_degrees"])
    return np.asarray([1.0, g, r, g * g, g * r, r * r])


def predict_f0(model: dict, x: np.ndarray) -> np.ndarray:
    alphas = np.asarray(model["alphas"], dtype=np.float64)
    normalized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    beta = np.asarray(model["beta"])
    return np.asarray([np.concatenate([a * normalized, a * (1.0 - a) * normalized]) @ beta for a in alphas])


def execute_scene(wrapper, preprocessor, env, segment: dict, env_seed: int, cem_seed: int, true_matrix, context_matrix) -> dict:
    started_unix = time.time()
    initial, goal_obs, nominal, _ = prepare_waypoint(env, segment, env_seed)
    start_obs, _ = env.prepare(env_seed, initial)
    commands, planner = plan_matrix_waypoint(wrapper, preprocessor, start_obs, goal_obs, context_matrix, cem_seed)
    states, effective, contacts, coverages = execute_matrix(env, initial, env_seed, commands, true_matrix)
    return {"execution_started_unix": started_unix, "initial_state": initial, "goal_state": nominal[-1], "context_matrix": np.asarray(context_matrix),
            "commands": commands, "effective_actions": effective, "states": states, "contacts": contacts,
            "coverages": coverages, "metrics": pose_metrics(states, nominal[-1], WINDOW),
            "deadline_success": deadline_success(states, nominal[-1]), "command_sha256": array_sha256(commands),
            "state_sha256": array_sha256(states), "planner": planner}


def evidence_pair(wrapper, preprocessor, env, segment: dict, env_seed: int, cem_seed: int, correct_matrix, no_persistence_matrix) -> dict:
    initial, goal_obs, nominal, _ = prepare_waypoint(env, segment, env_seed)
    start_obs, _ = env.prepare(env_seed, initial)
    commands, planner = plan_matrix_waypoint(wrapper, preprocessor, start_obs, goal_obs, POPULATION_PRIOR_MATRIX, cem_seed)
    result = {"initial_state": initial, "goal_state": nominal[-1], "commands": commands,
              "command_sha256": array_sha256(commands), "planner": planner}
    for name, matrix in (("correct", correct_matrix), ("no_persistence", no_persistence_matrix)):
        states, effective, contacts, coverages = execute_matrix(env, initial, env_seed, commands, matrix)
        observations, accepted = infer_matrix_observations(commands, states)
        posterior = BayesianMatrixContext(); posterior.update_observations(observations)
        result[name] = {"true_matrix": np.asarray(matrix), "states": states, "effective_actions": effective,
                        "contacts": contacts, "coverages": coverages, "state_sha256": array_sha256(states),
                        "matrix_observations": observations, "accepted_indices": accepted,
                        "observation_sha256": observations_sha256(observations), "posterior": posterior.as_dict()}
    return result


def collect(args, design: dict, selection: dict, pools: dict) -> dict:
    split, rows = args.split, selection[args.split]
    output = args.output_dir / split; output.mkdir(parents=True, exist_ok=True)
    raw_path, manifest_path = output / "raw.jsonl", output / "manifest.json"
    if manifest_path.exists() and not args.resume: raise FileExistsError(f"existing manifest: {manifest_path}")
    for path, expected, name in ((args.external_f0_model, design["external_f0_model_sha256"], "external model"),
                                 (args.external_f0_model_audit, design["external_f0_model_audit_sha256"], "external audit")):
        if sha256(path) != expected: raise RuntimeError(f"{name} hash mismatch")
    model = json.loads(args.external_f0_model.read_text(encoding="utf-8"))
    model_audit = json.loads(args.external_f0_model_audit.read_text(encoding="utf-8"))
    if not model.get("locked") or not model_audit.get("valid") or model_audit.get("locked_model_sha256") != sha256(args.external_f0_model):
        raise RuntimeError("external F0 model is not validly locked")
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    seed_all(int(design[split]["env_seed_base"]) - 1)
    _, wrapper, _ = load_matrix_world_model(args.checkpoint, device)
    model_before, rng_before = model_sha256(wrapper), rng_state_digest()
    manifest = {"contract_id": CONTRACT_ID, "split": split, "design_sha256": sha256(args.design),
                "contract_sha256": sha256(args.contract), "selection_sha256": sha256(args.selection),
                "source_snapshot_sha256": sha256(args.source_snapshot), "checkpoint_sha256": sha256(args.checkpoint),
                "pool_sha256": design["pool_sha256"], "external_f0_model_sha256": sha256(args.external_f0_model),
                "external_f0_model_audit_sha256": sha256(args.external_f0_model_audit),
                "command": " ".join(__import__("sys").argv), "started_unix": time.time(),
                "resource_start": resource_snapshot(device), "model_state_sha256_before": model_before,
                "rng_digest_before": rng_before}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("design_sha256", "contract_sha256", "selection_sha256", "source_snapshot_sha256", "checkpoint_sha256", "external_f0_model_sha256"):
            if previous.get(key) != manifest.get(key): raise RuntimeError(f"resume mismatch: {key}")
        manifest = previous
    else: dump_json(manifest_path, manifest)
    preprocessor, env = make_preprocessor(), make_env()
    completed = {int(row["sequence_id"]) for row in read_jsonl(raw_path) if row.get("record_type") == "cross_shape_sequence"}
    limit = min(len(rows), int(args.limit_sequences) if args.limit_sequences else len(rows))
    for sequence_id in range(limit):
        if sequence_id in completed: continue
        started = time.perf_counter(); selected = rows[sequence_id]
        factor_index = int(selected["factor_index"]); no_index = int(selected["no_persistence_factor_index"])
        factor_row, no_row = design["factors"][factor_index], design["factors"][no_index]
        correct_matrix = factor_matrix(factor_row["rotation_degrees"], factor_row["gain"])
        no_matrix = factor_matrix(no_row["rotation_degrees"], no_row["gain"])
        env_seed = int(design[split]["env_seed_base"]) + 100 * sequence_id
        cem_seed = int(design[split]["cem_seed_base"]) + 100 * sequence_id
        e1_segment = pools[selected["e1"]["shape"]][int(selected["e1"]["segment_index"])]
        evidence = evidence_pair(wrapper, preprocessor, env, e1_segment, env_seed, cem_seed, correct_matrix, no_matrix)
        correct_posterior = evidence["correct"]["posterior"]; no_posterior = evidence["no_persistence"]["posterior"]
        x = feature(correct_posterior, design); predictions = predict_f0(model, x)
        model_alphas = [float(value) for value in model["alphas"]]
        selected_alpha = model_alphas[int(np.argmax(predictions))]
        decision_unix = time.time()
        correct_order = [selected_alpha] + [float(a) for a in design["correct_context_alphas"] if float(a) != selected_alpha]
        e2_segment = pools[selected["e2"]["shape"]][int(selected["e2"]["segment_index"])]
        correct_treatments = {}
        for execution_index, alpha in enumerate(correct_order):
            context = blend_context(POPULATION_PRIOR_MATRIX, np.asarray(correct_posterior["mean_matrix"]), alpha)
            treatment = execute_scene(wrapper, preprocessor, env, e2_segment, env_seed + 1, cem_seed + 1, correct_matrix, context)
            treatment["execution_index"] = execution_index; correct_treatments[str(alpha)] = treatment
        no_context = blend_context(POPULATION_PRIOR_MATRIX, np.asarray(no_posterior["mean_matrix"]), float(design["fixed_alpha"]))
        no_treatment = execute_scene(wrapper, preprocessor, env, e2_segment, env_seed + 1, cem_seed + 1, correct_matrix, no_context)
        no_treatment["execution_index"] = len(correct_order)
        append_jsonl(raw_path, {"record_type": "cross_shape_sequence", "contract_id": CONTRACT_ID, "split": split,
            "sequence_id": sequence_id, "shape_pair_index": int(selected["shape_pair_index"]),
            "factor_index": factor_index, "no_persistence_factor_index": no_index,
            "correct_factor": factor_row, "no_persistence_e1_factor": no_row,
            "selection": selected, "env_seed": env_seed, "cem_seed": cem_seed,
            "e1": evidence, "features": x, "feature_names": design["feature_basis"],
            "external_f0_predicted_benefits": predictions, "external_f0_alphas": model_alphas,
            "external_f0_selected_alpha": selected_alpha, "decision_unix": decision_unix,
            "correct_execution_order": correct_order, "correct_treatments": correct_treatments,
            "no_persistence_treatment": no_treatment, "elapsed_s": time.perf_counter() - started,
            "resource": resource_snapshot(device)})
        print(f"{split} sequence={sequence_id} pair={selected['e1']['shape']}->{selected['e2']['shape']} alpha={selected_alpha} complete", flush=True)
    output_rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "cross_shape_sequence"]
    model_after, rng_after = model_sha256(wrapper), rng_state_digest()
    manifest.update({"finished_unix": time.time(), "resource_end": resource_snapshot(device), "rows": len(output_rows),
                     "raw_sha256": sha256(raw_path), "model_state_sha256_after": model_after,
                     "model_state_unchanged": model_before == model_after, "rng_digest_after": rng_after,
                     "rng_unchanged": rng_before == rng_after})
    dump_json(manifest_path, manifest); return manifest


def bootstrap_ci(values: np.ndarray, design: dict, stream: int) -> list[float]:
    rng = np.random.default_rng(int(design["bootstrap_seed"]) + stream)
    indexes = rng.integers(0, len(values), size=(int(design["bootstrap_resamples"]), len(values)))
    return [float(x) for x in np.quantile(values[indexes].mean(1), [.025, .975])]


def contrast(values: np.ndarray, design: dict, stream: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "bootstrap_ci95": bootstrap_ci(values, design, stream),
            "positive_fraction": float(np.mean(values > 1e-12)), "tie_fraction": float(np.mean(np.abs(values) <= 1e-12)),
            "negative_fraction": float(np.mean(values < -1e-12)), "unit_deltas": values.tolist()}


def evaluate(args, design: dict) -> dict:
    rows = [row for row in read_jsonl(args.output_dir / "formal/raw.jsonl") if row.get("record_type") == "cross_shape_sequence"]
    if len(rows) != 96: raise RuntimeError("evaluation requires 96 formal rows")
    def correct(row, alpha): return float(row["correct_treatments"][str(float(alpha))]["metrics"]["pose_auc10"])
    population = np.asarray([correct(row, 0) for row in rows]); fixed05 = np.asarray([correct(row, .5) for row in rows])
    fixed075 = np.asarray([correct(row, .75) for row in rows]); full = np.asarray([correct(row, 1) for row in rows])
    no_persistence = np.asarray([row["no_persistence_treatment"]["metrics"]["pose_auc10"] for row in rows])
    f0 = np.asarray([correct(row, row["external_f0_selected_alpha"]) for row in rows])
    best = np.minimum.reduce([population, fixed05, fixed075, full])
    primary, persistence = population - fixed075, no_persistence - fixed075
    def effect(cost, stream):
        delta = population - cost
        value = contrast(delta, design, stream); value.update({"mean_cost": float(cost.mean()),
            "relative_improvement": float(delta.mean() / population.mean()), "harm_fraction": float(np.mean(delta < -1e-12))})
        return value
    by_pair, by_factor = {}, {}
    for field, target, count in (("shape_pair_index", by_pair, 6), ("factor_index", by_factor, 8)):
        ids = np.asarray([int(row[field]) for row in rows])
        for index in range(count):
            keep = ids == index; target[str(index)] = {"n": int(keep.sum()), "primary_mean": float(primary[keep].mean()),
                                                      "persistence_mean": float(persistence[keep].mean())}
    result = {"contract_id": CONTRACT_ID, "n_formal": 96,
              "primary": contrast(primary, design, 100), "persistence_specific": contrast(persistence, design, 101),
              "policies": {"correct_fixed_alpha_0.5": effect(fixed05, 1), "correct_fixed_alpha_0.75": effect(fixed075, 2),
                           "correct_full_context": effect(full, 3), "external_T_F0": effect(f0, 4),
                           "no_persistence_alpha_0.75": effect(no_persistence, 5), "per_sequence_best_fixed_grid_ceiling": effect(best, 6)},
              "external_F0_vs_fixed_0.75": contrast(fixed075 - f0, design, 102),
              "external_f0_selection_counts": {str(alpha): int(sum(float(row["external_f0_selected_alpha"]) == alpha for row in rows))
                                                 for alpha in [float(a) for a in rows[0]["external_f0_alphas"]]},
              "by_shape_pair": by_pair, "by_factor": by_factor,
              "selected_branch_first_valid": bool(all(float(row["correct_execution_order"][0]) == float(row["external_f0_selected_alpha"])
                                                       and int(row["correct_treatments"][str(float(row["external_f0_selected_alpha"]))]["execution_index"]) == 0 for row in rows))}
    dump_json(args.output_dir / "formal_summary.json", result); print(json.dumps(result, indent=2)); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("inspect", "collect", "evaluate"))
    parser.add_argument("--split", choices=SPLITS, default="smoke")
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_cross_shape_matrix_history_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_cross_shape_matrix_history_contract_zh.md"))
    parser.add_argument("--selection", type=Path, default=Path("docs/research/persistent_context_v2_cross_shape_matrix_history_selection.json"))
    parser.add_argument("--author-pool-audit", type=Path, default=Path("repro_outputs/persistent_context_v2_cross_shape_author_pool_audit_20260823.json"))
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--external-f0-model", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_f0_soft_policy_formal_v1/locked_model.json"))
    parser.add_argument("--external-f0-model-audit", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_f0_soft_policy_formal_v1/model_audit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_cross_shape_matrix_history_formal_v1"))
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--limit-sequences", type=int); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); design, selection = load_frozen(args.design, args.contract, args.selection); pools = load_pools(design)
    inspection = inspect_selection(design, selection, pools, args.author_pool_audit); args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "selection_audit.json", inspection)
    if not inspection["valid"]: raise RuntimeError("selection audit invalid")
    if args.mode == "inspect": print(json.dumps(inspection, indent=2))
    elif args.mode == "collect": collect(args, design, selection, pools)
    else: evaluate(args, design)


if __name__ == "__main__": main()
