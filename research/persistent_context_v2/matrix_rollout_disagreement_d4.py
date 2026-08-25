"""Prospective D4 development study of pre-outcome rollout disagreement.

The development outcomes are new, but this is a feasibility study rather than
a formal gate evaluation.  The frozen primary score is computed before either
E2 policy is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.matrix_soft_context_d1 import blend_context
from research.persistent_context_v2.matrix_task_interaction_d0 import model_interaction_features, model_sha256
from research.persistent_context_v2.pushobj_matrix_stage0 import (
    ACTION_DIM,
    FRAMESKIP,
    MODEL_HORIZON,
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
    make_env,
    make_preprocessor,
    obs_batch,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-matrix-rollout-disagreement-d4-development-v1"
EXPECTED_DESIGN_SHA256 = "40f36841e9665d80d4ed2ae9a7767d21eb611d8415297a54e96c5caac46823c2"
EXPECTED_CONTRACT_SHA256 = "930ead1aac500e21a02ce32cf8e862616cefa31f3c0cf7cb6e13d9860b8b0c1e"
# Filled only after the outcome-blind selection is generated and audited.
EXPECTED_SELECTION_SHA256 = "TO_BE_FROZEN"
SPLITS = ("smoke", "development")


def segment_hash(segment: dict) -> str:
    digest = hashlib.sha256()
    for key in ("states", "actions"):
        value = np.asarray(segment[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def full_rng_digest() -> str:
    cuda = [value.cpu().numpy() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    payload = pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy(), cuda))
    return hashlib.sha256(payload).hexdigest()


def load_frozen(design_path: Path, contract_path: Path, selection_path: Path) -> tuple[dict, dict]:
    expected = (
        (design_path, EXPECTED_DESIGN_SHA256, "design"),
        (contract_path, EXPECTED_CONTRACT_SHA256, "contract"),
        (selection_path, EXPECTED_SELECTION_SHA256, "selection"),
    )
    for path, digest, label in expected:
        if digest == "TO_BE_FROZEN" or sha256(path) != digest:
            raise RuntimeError(f"frozen {label} hash mismatch")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if design.get("contract_id") != CONTRACT_ID or selection.get("contract_id") != CONTRACT_ID:
        raise RuntimeError("contract id mismatch")
    return design, selection


def load_pools(design: dict) -> dict[str, list[dict]]:
    audit_path = Path(design["author_pool_audit"])
    if sha256(audit_path) != design["author_pool_audit_sha256"]:
        raise RuntimeError("author pool audit hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    pools = {}
    for shape in design["shapes"]:
        path = Path(audit["pools"][shape]["path"])
        if sha256(path) != design["pool_sha256"][shape]:
            raise RuntimeError(f"pool hash mismatch: {shape}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if str(payload.get("shape")) != shape or int(payload.get("seed")) != 42:
            raise RuntimeError(f"pool metadata mismatch: {shape}")
        pools[shape] = payload["segments"]
    return pools


def inspect_selection(design: dict, selection: dict, pools: dict[str, list[dict]]) -> dict:
    from collections import Counter

    old_path = Path(design["excluded_selection"])
    if sha256(old_path) != design["excluded_selection_sha256"]:
        raise RuntimeError("excluded selection hash mismatch")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_items = [row[episode] for split in ("smoke", "formal") for row in old[split] for episode in ("e1", "e2")]
    old_items += [item for rows in old["reserve"].values() for item in rows]
    old_hashes = {str(item["segment_sha256"]) for item in old_items}
    old_provenance = {str(item["provenance_key"]) for item in old_items}
    global_counts = Counter(segment_hash(segment) for values in pools.values() for segment in values)
    failures, selected_hashes, selected_provenance = [], [], []
    checks = {}
    for split, expected in (("smoke", 6), ("development", 96)):
        rows = selection.get(split, [])
        pair_counts = Counter()
        factor_counts = Counter()
        combo_counts = Counter()
        minimum = math.inf
        if len(rows) != expected:
            failures.append(f"{split} row count")
        for ordinal, row in enumerate(rows):
            if int(row["sequence_id"]) != ordinal:
                failures.append(f"{split} sequence order")
            pair, factor = int(row["shape_pair_index"]), int(row["factor_index"])
            pair_counts[pair] += 1
            factor_counts[factor] += 1
            combo_counts[(pair, factor)] += 1
            if [row["e1"]["shape"], row["e2"]["shape"]] != design["shape_pairs"][pair]:
                failures.append(f"{split} shape pair")
            for episode in ("e1", "e2"):
                item = row[episode]
                segment = pools[item["shape"]][int(item["segment_index"])]
                digest = segment_hash(segment)
                provenance = f"{int(segment['ep_idx'])}:{int(segment['offset'])}"
                selected_hashes.append(digest)
                selected_provenance.append(provenance)
                minimum = min(minimum, nominal_block_displacement_at_10(segment))
                if digest != item["segment_sha256"] or provenance != item["provenance_key"]:
                    failures.append(f"{split} segment identity")
                if digest in old_hashes or provenance in old_provenance or global_counts[digest] != 1:
                    failures.append(f"{split} exclusion or duplication")
        checks[split] = {
            "rows": len(rows),
            "pair_counts": dict(sorted(pair_counts.items())),
            "factor_counts": dict(sorted(factor_counts.items())),
            "combo_counts": {f"{a}:{b}": count for (a, b), count in sorted(combo_counts.items())},
            "minimum_displacement": float(minimum),
        }
    if len(selected_hashes) != 204 or len(set(selected_hashes)) != 204 or len(set(selected_provenance)) != 204:
        failures.append("selected segment global uniqueness")
    dev = checks["development"]
    if set(dev["pair_counts"].values()) != {16} or set(dev["factor_counts"].values()) != {12} or set(dev["combo_counts"].values()) != {2}:
        failures.append("development balance")
    if min(item["minimum_displacement"] for item in checks.values()) < 10.0:
        failures.append("waypoint eligibility")
    return {
        "contract_id": CONTRACT_ID,
        "valid": not failures,
        "failures": failures,
        "checks": checks,
        "selected_segment_count": len(selected_hashes),
        "unique_segment_hash_count": len(set(selected_hashes)),
        "unique_provenance_count": len(set(selected_provenance)),
    }


def _cpu_dict(values: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {str(key): value.detach().cpu().contiguous().numpy() for key, value in sorted(values.items())}


def flatten_latent(values: dict[str, np.ndarray]) -> np.ndarray:
    """Return model-step by flattened-latent array in deterministic key order."""
    pieces, steps = [], None
    for key in sorted(values):
        value = np.asarray(values[key])
        if value.ndim < 2 or value.shape[0] != 1:
            raise ValueError(f"unexpected latent shape for {key}: {value.shape}")
        value = value[0]
        current_steps = int(value.shape[0]) if value.ndim > 1 else 1
        flat = value.reshape(current_steps, -1)
        if steps is None:
            steps = current_steps
        elif current_steps != steps:
            raise ValueError("latent keys disagree on model-step count")
        pieces.append(flat.astype(np.float64, copy=False))
    if not pieces:
        raise ValueError("empty latent dictionary")
    return np.concatenate(pieces, axis=1)


def rms_pair(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> tuple[float, float]:
    delta = flatten_latent(first) - flatten_latent(second)
    return float(np.sqrt(np.mean(delta**2))), float(np.sqrt(np.mean(delta[-1] ** 2)))


def rollout_features(traces: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    pp, pc = traces["prior_population"], traces["prior_context"]
    cp, cc = traces["context_population"], traces["context_context"]
    return np.asarray((*rms_pair(pp, cp), *rms_pair(pc, cc), *rms_pair(pp, pc), *rms_pair(cp, cc)), dtype=np.float64)


def readonly_rollout(wm, preprocessor, obs_0, obs_g, commands, context_matrix) -> tuple[dict, dict, float]:
    from planning.objectives import create_objective_fn
    from utils import move_to_device

    device = next(wm.parameters()).device
    wm.set_context(np.asarray(context_matrix, dtype=np.float64))
    transformed_0 = move_to_device(preprocessor.transform_obs(obs_batch(obs_0)), device)
    transformed_g = move_to_device(preprocessor.transform_obs(obs_batch(obs_g)), device)
    command_tensor = torch.as_tensor(np.asarray(commands), dtype=torch.float32)
    normalized = preprocessor.normalize_actions(command_tensor)
    actions = normalized.reshape(1, MODEL_HORIZON, FRAMESKIP * ACTION_DIM).to(device)
    objective = create_objective_fn(alpha=1, base=2, mode="staged")
    with torch.inference_mode():
        goal = wm.encode_obs(transformed_g)
        prediction, _ = wm.rollout(transformed_0, actions)
        loss = objective(prediction, goal, step=0)
    return _cpu_dict(prediction), _cpu_dict(goal), float(loss.reshape(-1)[0].item())


def preoutcome_traces(wm, preprocessor, obs_0, obs_g, population_commands, context_commands, context_matrix):
    rng_before = full_rng_digest()
    traces, scores, goal = {}, {}, None
    for model_name, matrix in (("prior", POPULATION_PRIOR_MATRIX), ("context", context_matrix)):
        for plan_name, commands in (("population", population_commands), ("context", context_commands)):
            prediction, current_goal, score = readonly_rollout(wm, preprocessor, obs_0, obs_g, commands, matrix)
            traces[f"{model_name}_{plan_name}"] = prediction
            action_name = "prior" if plan_name == "population" else "context"
            scores[f"J_{model_name}_a_{action_name}"] = score
            if goal is None:
                goal = current_goal
            elif any(not np.array_equal(goal[key], current_goal[key]) for key in goal):
                raise RuntimeError("goal latent changed across read-only probes")
    wm.set_context(POPULATION_PRIOR_MATRIX)
    rng_after = full_rng_digest()
    if rng_before != rng_after:
        raise RuntimeError("read-only rollout probe changed global RNG")
    latent = rollout_features(traces)
    objective = model_interaction_features(scores)
    return traces, goal, scores, latent, objective, rng_before, rng_after


def save_trace_npz(path: Path, traces: dict, goal: dict) -> dict:
    arrays = {}
    for trace_name, values in sorted(traces.items()):
        for key, value in sorted(values.items()):
            arrays[f"{trace_name}__{key}"] = np.asarray(value)
    for key, value in sorted(goal.items()):
        arrays[f"goal_latent__{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha256(value)} for key, value in arrays.items()},
    }


def evidence_episode(wm, preprocessor, env, segment, env_seed, cem_seed, true_matrix) -> dict:
    initial, goal_obs, nominal, _ = prepare_waypoint(env, segment, env_seed)
    start_obs, _ = env.prepare(env_seed, initial)
    commands, planner = plan_matrix_waypoint(wm, preprocessor, start_obs, goal_obs, POPULATION_PRIOR_MATRIX, cem_seed)
    states, effective, contacts, coverages = execute_matrix(env, initial, env_seed, commands, true_matrix)
    observations, accepted = infer_matrix_observations(commands, states)
    posterior = BayesianMatrixContext()
    posterior.update_observations(observations)
    return {
        "initial_state": initial,
        "goal_state": nominal[-1],
        "commands": commands,
        "command_sha256": array_sha256(commands),
        "states": states,
        "state_sha256": array_sha256(states),
        "effective_actions": effective,
        "contacts": contacts,
        "coverages": coverages,
        "matrix_observations": observations,
        "accepted_indices": accepted,
        "observation_sha256": observations_sha256(observations),
        "posterior": posterior.as_dict(),
        "planner": planner,
    }


def execute_preplanned(env, initial, goal, env_seed, commands, true_matrix, context_matrix, execution_index) -> dict:
    started = time.time()
    states, effective, contacts, coverages = execute_matrix(env, initial, env_seed, commands, true_matrix)
    return {
        "execution_index": int(execution_index),
        "execution_started_unix": started,
        "context_matrix": np.asarray(context_matrix),
        "commands": np.asarray(commands),
        "command_sha256": array_sha256(commands),
        "states": states,
        "state_sha256": array_sha256(states),
        "effective_actions": effective,
        "contacts": contacts,
        "coverages": coverages,
        "metrics": pose_metrics(states, goal, WINDOW),
        "deadline_success": deadline_success(states, goal),
    }


def collect(args, design: dict, selection: dict, pools: dict[str, list[dict]]) -> dict:
    split, selected_rows = args.split, selection[args.split]
    output = args.output_dir / split
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_path, manifest_path = output / "raw.jsonl", output / "manifest.json"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index if device.index is not None else torch.cuda.current_device())
    seed_all(int(design[split]["env_seed_base"]) - 1)
    base, wm, _ = load_matrix_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    model_before = model_sha256(wm)
    manifest = {
        "contract_id": CONTRACT_ID,
        "evidence_level": design["evidence_level"],
        "split": split,
        "design_sha256": sha256(args.design),
        "contract_sha256": sha256(args.contract),
        "selection_sha256": sha256(args.selection),
        "source_snapshot_sha256": sha256(args.source_snapshot),
        "checkpoint_sha256": sha256(args.checkpoint),
        "pool_sha256": design["pool_sha256"],
        "command": " ".join(__import__("sys").argv),
        "started_unix": time.time(),
        "resource_start": resource_snapshot(device),
        "model_state_sha256_before": model_before,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("design_sha256", "contract_sha256", "selection_sha256", "source_snapshot_sha256", "checkpoint_sha256"):
            if previous.get(key) != manifest[key]:
                raise RuntimeError(f"resume mismatch: {key}")
        manifest = previous
    else:
        dump_json(manifest_path, manifest)
    completed = {int(row["sequence_id"]) for row in read_jsonl(raw_path) if row.get("record_type") == "d4_sequence"}
    limit = min(len(selected_rows), int(args.limit_sequences) if args.limit_sequences else len(selected_rows))
    for sequence_id in range(limit):
        if sequence_id in completed:
            continue
        selected = selected_rows[sequence_id]
        factor_index = int(selected["factor_index"])
        factor = design["factors"][factor_index]
        true_matrix = factor_matrix(factor["rotation_degrees"], factor["gain"])
        env_seed = int(design[split]["env_seed_base"]) + 100 * sequence_id
        cem_seed = int(design[split]["cem_seed_base"]) + 100 * sequence_id
        e1_segment = pools[selected["e1"]["shape"]][int(selected["e1"]["segment_index"])]
        e1 = evidence_episode(wm, preprocessor, env, e1_segment, env_seed, cem_seed, true_matrix)
        posterior_matrix = np.asarray(e1["posterior"]["mean_matrix"], dtype=np.float64)
        context_matrix = blend_context(POPULATION_PRIOR_MATRIX, posterior_matrix, float(design["fixed_alpha"]))

        e2_segment = pools[selected["e2"]["shape"]][int(selected["e2"]["segment_index"])]
        initial, goal_obs, nominal, _ = prepare_waypoint(env, e2_segment, env_seed + 1)
        start_obs, _ = env.prepare(env_seed + 1, initial)
        population_commands, population_planner = plan_matrix_waypoint(
            wm, preprocessor, start_obs, goal_obs, POPULATION_PRIOR_MATRIX, cem_seed + 1
        )
        context_commands, context_planner = plan_matrix_waypoint(
            wm, preprocessor, start_obs, goal_obs, context_matrix, cem_seed + 1
        )
        traces, goal_latent, scores, latent_features, objective_features, rng_before, rng_after = preoutcome_traces(
            wm, preprocessor, start_obs, goal_obs, population_commands, context_commands, context_matrix
        )
        trace_path = output / "traces" / f"sequence_{sequence_id:03d}.npz"
        trace_manifest = save_trace_npz(trace_path, traces, goal_latent)
        repeat_exact = None
        if split == "smoke":
            repeated = preoutcome_traces(wm, preprocessor, start_obs, goal_obs, population_commands, context_commands, context_matrix)
            repeat_exact = bool(
                all(np.array_equal(traces[name][key], repeated[0][name][key]) for name in traces for key in traces[name])
                and all(np.array_equal(goal_latent[key], repeated[1][key]) for key in goal_latent)
            )
            if not repeat_exact:
                raise RuntimeError("smoke repeated rollout trace mismatch")
        trace_completed_unix = time.time()
        order = ["population", "context"] if sequence_id % 2 == 0 else ["context", "population"]
        policies = {}
        specifications = {
            "population": (population_commands, POPULATION_PRIOR_MATRIX),
            "context": (context_commands, context_matrix),
        }
        for execution_index, name in enumerate(order):
            commands, matrix = specifications[name]
            policies[name] = execute_preplanned(
                env, initial, nominal[-1], env_seed + 1, commands, true_matrix, matrix, execution_index
            )
            if policies[name]["execution_started_unix"] < trace_completed_unix:
                raise RuntimeError("E2 outcome started before trace completion")
        identity = identity_audit(base, wm, preprocessor, start_obs) if split == "smoke" else None
        append_jsonl(raw_path, {
            "record_type": "d4_sequence",
            "contract_id": CONTRACT_ID,
            "split": split,
            "sequence_id": sequence_id,
            "shape_pair_index": int(selected["shape_pair_index"]),
            "factor_index": factor_index,
            "replicate": int(selected["replicate"]),
            "selection": selected,
            "factor": factor,
            "true_matrix": true_matrix,
            "env_seed": env_seed,
            "cem_seed": cem_seed,
            "e1": e1,
            "e2": {
                "initial_state": initial,
                "goal_state": nominal[-1],
                "population_planner": population_planner,
                "context_planner": context_planner,
                "execution_order": order,
                "trace_completed_unix": trace_completed_unix,
                "trace": trace_manifest,
                "trace_repeat_exact": repeat_exact,
                "model_scores": scores,
                "latent_feature_names": design["latent_feature_names"],
                "latent_features": latent_features,
                "objective_feature_names": design["objective_feature_names"],
                "objective_features": objective_features,
                "probe_rng_digest_before": rng_before,
                "probe_rng_digest_after": rng_after,
                "policies": policies,
            },
            "population_identity": identity,
            "resource": resource_snapshot(device),
        })
        print(f"D4 {split} sequence={sequence_id} {selected['e1']['shape']}->{selected['e2']['shape']} complete", flush=True)
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "d4_sequence"]
    model_after = model_sha256(wm)
    resource_end = resource_snapshot(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        resource_end["cuda_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(index))
        resource_end["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(index))
    manifest.update({
        "finished_unix": time.time(),
        "rows": len(rows),
        "raw_sha256": sha256(raw_path),
        "model_state_sha256_after": model_after,
        "model_state_unchanged": model_before == model_after,
        "resource_end": resource_end,
    })
    if not manifest["model_state_unchanged"]:
        raise RuntimeError("collection changed model state")
    dump_json(manifest_path, manifest)
    return manifest


def correlation(first, second) -> float | None:
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def rankdata(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def roc_auc(labels, scores) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def bootstrap_auc(labels, scores, design: dict) -> tuple[list[float] | None, int]:
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(int(design["bootstrap_seed"]))
    values = []
    for _ in range(int(design["bootstrap_resamples"])):
        indexes = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indexes])) == 2:
            values.append(roc_auc(labels[indexes], scores[indexes]))
    if not values:
        return None, 0
    return [float(value) for value in np.quantile(values, (0.025, 0.975))], len(values)


def quartile_summary(scores, benefit, tolerance: float) -> list[dict]:
    order = np.argsort(np.asarray(scores), kind="mergesort")
    groups = np.array_split(order, 4)
    result = []
    for index, group in enumerate(groups):
        group_benefit = np.asarray(benefit)[group]
        result.append({
            "quartile": index + 1,
            "n": len(group),
            "score_min": float(np.asarray(scores)[group].min()),
            "score_max": float(np.asarray(scores)[group].max()),
            "mean_benefit": float(group_benefit.mean()),
            "harm_fraction": float(np.mean(group_benefit < -tolerance)),
            "sequence_indices": group.tolist(),
        })
    return result


def fit_ridge(rows, indexes, alpha):
    x = np.asarray([rows[index]["x"] for index in indexes], dtype=np.float64)
    y = np.asarray([rows[index]["benefit"] for index in indexes], dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    z = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(z.shape[1]); penalty[0, 0] = 0.0
    beta = np.linalg.pinv(z.T @ z + float(alpha) * penalty) @ z.T @ y
    return mean, scale, beta


def predict_ridge(model, rows, indexes):
    mean, scale, beta = model
    x = np.asarray([rows[index]["x"] for index in indexes], dtype=np.float64)
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    return z @ beta


def nested_leave_pair_out(rows, ridge_alphas):
    predictions = np.zeros(len(rows), dtype=np.float64)
    chosen = {}
    groups = sorted({int(row["shape_pair_index"]) for row in rows})
    for outer in groups:
        train = [index for index, row in enumerate(rows) if int(row["shape_pair_index"]) != outer]
        test = [index for index, row in enumerate(rows) if int(row["shape_pair_index"]) == outer]
        candidates = []
        for alpha in ridge_alphas:
            squared = []
            for inner in sorted({int(rows[index]["shape_pair_index"]) for index in train}):
                inner_train = [index for index in train if int(rows[index]["shape_pair_index"]) != inner]
                inner_test = [index for index in train if int(rows[index]["shape_pair_index"]) == inner]
                prediction = predict_ridge(fit_ridge(rows, inner_train, alpha), rows, inner_test)
                truth = np.asarray([rows[index]["benefit"] for index in inner_test])
                squared.extend((prediction - truth) ** 2)
            candidates.append((float(np.mean(squared)), float(alpha)))
        alpha = min(candidates, key=lambda item: (item[0], item[1]))[1]
        chosen[str(outer)] = alpha
        predictions[test] = predict_ridge(fit_ridge(rows, train, alpha), rows, test)
    return predictions, chosen


def analyze(args, design: dict) -> dict:
    raw_path = args.output_dir / "development" / "raw.jsonl"
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "d4_sequence"]
    if len(rows) != int(design["development"]["sequences"]):
        raise RuntimeError("D4 analysis requires all 96 development sequences")
    rows.sort(key=lambda row: int(row["sequence_id"]))
    population = np.asarray([row["e2"]["policies"]["population"]["metrics"]["pose_auc10"] for row in rows])
    context = np.asarray([row["e2"]["policies"]["context"]["metrics"]["pose_auc10"] for row in rows])
    benefit = population - context
    tolerance = float(design["harm_tolerance"])
    harm = benefit < -tolerance
    latent = np.asarray([row["e2"]["latent_features"] for row in rows], dtype=np.float64)
    objective = np.asarray([row["e2"]["objective_features"] for row in rows], dtype=np.float64)
    primary_index = design["latent_feature_names"].index(design["primary_risk_score"])
    primary = latent[:, primary_index]
    auc = roc_auc(harm, primary)
    auc_ci, valid_bootstraps = bootstrap_auc(harm, primary, design)
    secondary = {}
    risk = -benefit
    for index, name in enumerate(design["latent_feature_names"]):
        secondary[name] = {
            "roc_auc_harm": roc_auc(harm, latent[:, index]),
            "pearson_with_negative_benefit": correlation(latent[:, index], risk),
            "spearman_with_negative_benefit": correlation(rankdata(latent[:, index]), rankdata(risk)),
            "unit_values": latent[:, index].tolist(),
        }
    for index, name in enumerate(design["objective_feature_names"]):
        secondary[name] = {
            "roc_auc_harm": roc_auc(harm, objective[:, index]),
            "pearson_with_negative_benefit": correlation(objective[:, index], risk),
            "spearman_with_negative_benefit": correlation(rankdata(objective[:, index]), rankdata(risk)),
            "unit_values": objective[:, index].tolist(),
        }
    mechanism_rows = [{"x": np.concatenate((latent[index], objective[index])), "benefit": benefit[index],
                       "shape_pair_index": int(rows[index]["shape_pair_index"])} for index in range(len(rows))]
    ridge_prediction, chosen = nested_leave_pair_out(mechanism_rows, design["ridge_alphas"])
    shadow = np.where(ridge_prediction > 0.0, benefit, 0.0)
    result = {
        "contract_id": CONTRACT_ID,
        "evidence_level": design["evidence_level"],
        "n_sequences": len(rows),
        "population_mean_pose_auc10": float(population.mean()),
        "context_mean_pose_auc10": float(context.mean()),
        "mean_benefit": float(benefit.mean()),
        "harm_count": int(harm.sum()),
        "harm_fraction": float(harm.mean()),
        "positive_count": int(np.sum(benefit > tolerance)),
        "tie_count": int(np.sum(np.abs(benefit) <= tolerance)),
        "negative_count": int(np.sum(benefit < -tolerance)),
        "unit_benefit": benefit.tolist(),
        "unit_harm": harm.astype(int).tolist(),
        "primary": {
            "score_name": design["primary_risk_score"],
            "direction": design["primary_direction"],
            "roc_auc_harm": auc,
            "bootstrap_ci95_auc": auc_ci,
            "valid_bootstrap_resamples": valid_bootstraps,
            "pearson_with_negative_benefit": correlation(primary, risk),
            "spearman_with_negative_benefit": correlation(rankdata(primary), rankdata(risk)),
            "quartiles": quartile_summary(primary, benefit, tolerance),
            "unit_scores": primary.tolist(),
        },
        "secondary_scores": secondary,
        "ridge_leave_one_shape_pair_out": {
            "feature_names": design["latent_feature_names"] + design["objective_feature_names"],
            "chosen_ridge_by_outer_pair": chosen,
            "predictions": ridge_prediction.tolist(),
            "prediction_correlation": correlation(ridge_prediction, benefit),
            "shadow_selection_rate": float(np.mean(ridge_prediction > 0.0)),
            "shadow_mean_benefit": float(shadow.mean()),
            "shadow_harm_fraction": float(np.mean(shadow < -tolerance)),
            "shadow_unit_benefit": shadow.tolist(),
        },
        "raw_sha256": sha256(raw_path),
    }
    dump_json(args.output_dir / "development" / "analysis.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "collect", "analyze"))
    parser.add_argument("--split", choices=SPLITS, default="smoke")
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_contract_zh.md"))
    parser.add_argument("--selection", type=Path, default=Path("docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_selection.json"))
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_rollout_disagreement_d4_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-sequences", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    design, selection = load_frozen(args.design, args.contract, args.selection)
    pools = load_pools(design)
    inspection = inspect_selection(design, selection, pools)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "selection_audit.json", inspection)
    if not inspection["valid"]:
        raise RuntimeError("selection audit invalid")
    if args.mode == "inspect":
        print(json.dumps(inspection, indent=2))
    elif args.mode == "collect":
        collect(args, design, selection, pools)
    else:
        analyze(args, design)


if __name__ == "__main__":
    main()
