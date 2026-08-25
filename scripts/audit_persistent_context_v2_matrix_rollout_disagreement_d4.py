#!/usr/bin/env python3
"""Independent raw-artifact audit for the D4 development study.

This file intentionally does not import the D4 collector or analyzer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.matrix_soft_context_d1 import blend_context
from research.persistent_context_v2.pushobj_matrix_stage0 import (
    ACTION_DIM,
    FRAMESKIP,
    MODEL_HORIZON,
    POPULATION_PRIOR_MATRIX,
    apply_action_matrix,
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
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import WINDOW, prepare_waypoint
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    dump_json,
    make_env,
    make_preprocessor,
    obs_batch,
    pose_metrics,
    read_jsonl,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-matrix-rollout-disagreement-d4-development-v1"
EXPECTED_DESIGN_SHA256 = "40f36841e9665d80d4ed2ae9a7767d21eb611d8415297a54e96c5caac46823c2"
EXPECTED_CONTRACT_SHA256 = "930ead1aac500e21a02ce32cf8e862616cefa31f3c0cf7cb6e13d9860b8b0c1e"
EXPECTED_SELECTION_SHA256 = "208cd91e4a68af891beddf8465f73ceb048cfca60eb1c9244b06228659da4005"


def full_rng_digest() -> str:
    cuda = [value.cpu().numpy() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy(), cuda))).hexdigest()


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_dict(values):
    return {str(key): value.detach().cpu().contiguous().numpy() for key, value in sorted(values.items())}


def independent_rollout(wm, preprocessor, obs_0, obs_g, commands, context_matrix):
    from planning.objectives import create_objective_fn
    from utils import move_to_device

    device = next(wm.parameters()).device
    wm.set_context(np.asarray(context_matrix, dtype=np.float64))
    transformed_0 = move_to_device(preprocessor.transform_obs(obs_batch(obs_0)), device)
    transformed_g = move_to_device(preprocessor.transform_obs(obs_batch(obs_g)), device)
    commands = torch.as_tensor(np.asarray(commands), dtype=torch.float32)
    normalized = preprocessor.normalize_actions(commands).reshape(1, MODEL_HORIZON, FRAMESKIP * ACTION_DIM).to(device)
    objective = create_objective_fn(alpha=1, base=2, mode="staged")
    with torch.inference_mode():
        goal = wm.encode_obs(transformed_g)
        prediction, _ = wm.rollout(transformed_0, normalized)
        score = objective(prediction, goal, step=0)
    return cpu_dict(prediction), cpu_dict(goal), float(score.reshape(-1)[0].item())


def flatten(values):
    pieces, steps = [], None
    for key in sorted(values):
        value = np.asarray(values[key])
        if value.ndim < 2 or value.shape[0] != 1:
            raise ValueError(f"unexpected latent shape: {key} {value.shape}")
        value = value[0]
        current = int(value.shape[0]) if value.ndim > 1 else 1
        if steps is None:
            steps = current
        elif current != steps:
            raise ValueError("model-step mismatch")
        pieces.append(value.reshape(current, -1).astype(np.float64, copy=False))
    return np.concatenate(pieces, axis=1)


def rms(first, second):
    delta = flatten(first) - flatten(second)
    return float(np.sqrt(np.mean(delta**2))), float(np.sqrt(np.mean(delta[-1] ** 2)))


def features(traces, scores):
    pp, pc = traces["prior_population"], traces["prior_context"]
    cp, cc = traces["context_population"], traces["context_context"]
    latent = np.asarray((*rms(pp, cp), *rms(pc, cc), *rms(pp, pc), *rms(cp, cc)))
    objective = np.asarray([
        scores["J_prior_a_context"] - scores["J_prior_a_prior"],
        scores["J_context_a_prior"] - scores["J_context_a_context"],
        scores["J_context_a_prior"] - scores["J_prior_a_prior"],
        scores["J_context_a_context"] - scores["J_prior_a_context"],
    ])
    return latent, objective


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def correlation(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return None if np.std(first) <= 1e-12 or np.std(second) <= 1e-12 else float(np.corrcoef(first, second)[0, 1])


def bootstrap_auc(labels, scores, design):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    rng = np.random.default_rng(int(design["bootstrap_seed"]))
    values = []
    for _ in range(int(design["bootstrap_resamples"])):
        indexes = rng.integers(0, len(labels), len(labels))
        current = auc(labels[indexes], scores[indexes])
        if current is not None:
            values.append(current)
    return (None, 0) if not values else ([float(x) for x in np.quantile(values, (.025, .975))], len(values))


def fit_ridge(rows, indexes, alpha):
    x = np.asarray([rows[index]["x"] for index in indexes])
    y = np.asarray([rows[index]["benefit"] for index in indexes])
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.0
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.eye(z.shape[1]); penalty[0, 0] = 0
    beta = np.linalg.pinv(z.T @ z + float(alpha) * penalty) @ z.T @ y
    return mean, scale, beta


def predict(model, rows, indexes):
    mean, scale, beta = model
    x = np.asarray([rows[index]["x"] for index in indexes])
    return np.column_stack((np.ones(len(x)), (x - mean) / scale)) @ beta


def nested(rows, alphas):
    output = np.zeros(len(rows)); chosen = {}
    groups = sorted({int(row["shape_pair_index"]) for row in rows})
    for outer in groups:
        train = [i for i, row in enumerate(rows) if int(row["shape_pair_index"]) != outer]
        test = [i for i, row in enumerate(rows) if int(row["shape_pair_index"]) == outer]
        candidates = []
        for alpha in alphas:
            errors = []
            for inner in sorted({int(rows[i]["shape_pair_index"]) for i in train}):
                fit = [i for i in train if int(rows[i]["shape_pair_index"]) != inner]
                held = [i for i in train if int(rows[i]["shape_pair_index"]) == inner]
                errors.extend((predict(fit_ridge(rows, fit, alpha), rows, held) - np.asarray([rows[i]["benefit"] for i in held])) ** 2)
            candidates.append((float(np.mean(errors)), float(alpha)))
        selected = min(candidates, key=lambda value: (value[0], value[1]))[1]
        chosen[str(outer)] = selected
        output[test] = predict(fit_ridge(rows, train, selected), rows, test)
    return output, chosen


def compare_array(actual, expected, tolerance, label, failures):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        failures.append(f"{label}: shape mismatch")
        return
    maximum = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0
    if maximum > tolerance:
        failures.append(f"{label}: max abs {maximum}")


def audit_raw(args, design, selection):
    split_dir = args.output_dir / args.split
    raw_path, manifest_path = split_dir / "raw.jsonl", split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "d4_sequence"]
    expected_rows = selection[args.split]
    failures = []
    if len(rows) != len(expected_rows) or [int(row["sequence_id"]) for row in rows] != list(range(len(rows))):
        failures.append("row count or sequence order")
    if manifest.get("raw_sha256") != sha256(raw_path) or manifest.get("rows") != len(rows):
        failures.append("manifest raw identity")
    audit = json.loads(Path(design["author_pool_audit"]).read_text(encoding="utf-8"))
    pools = {}
    for shape in design["shapes"]:
        path = Path(audit["pools"][shape]["path"])
        if sha256(path) != design["pool_sha256"][shape]:
            failures.append(f"pool hash {shape}")
        with path.open("rb") as handle:
            pools[shape] = pickle.load(handle)["segments"]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _, wm, _ = load_matrix_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    state_before, rng_before = model_sha256(wm), full_rng_digest()
    max_trace_error = max_state_error = max_metric_error = 0.0
    for row, expected_selection in zip(rows, expected_rows):
        sequence_id = int(row["sequence_id"])
        if row["selection"] != expected_selection:
            failures.append(f"sequence {sequence_id}: frozen selection mismatch")
        factor = design["factors"][int(expected_selection["factor_index"])]
        true_matrix = factor_matrix(factor["rotation_degrees"], factor["gain"])
        compare_array(row["true_matrix"], true_matrix, 1e-7, f"sequence {sequence_id} true matrix", failures)
        env_seed = int(design[args.split]["env_seed_base"]) + 100 * sequence_id
        cem_seed = int(design[args.split]["cem_seed_base"]) + 100 * sequence_id
        if int(row["env_seed"]) != env_seed or int(row["cem_seed"]) != cem_seed:
            failures.append(f"sequence {sequence_id}: seed mismatch")

        e1_segment = pools[expected_selection["e1"]["shape"]][int(expected_selection["e1"]["segment_index"])]
        e1_initial, e1_goal_obs, _, _ = prepare_waypoint(env, e1_segment, env_seed)
        e1_start, _ = env.prepare(env_seed, e1_initial)
        e1_commands, _ = plan_matrix_waypoint(wm, preprocessor, e1_start, e1_goal_obs, POPULATION_PRIOR_MATRIX, cem_seed)
        compare_array(e1_commands, row["e1"]["commands"], 0.0, f"sequence {sequence_id} E1 plan", failures)
        e1_states, _, _, _ = execute_matrix(env, e1_initial, env_seed, e1_commands, true_matrix)
        compare_array(e1_states, row["e1"]["states"], 1e-6, f"sequence {sequence_id} E1 replay", failures)
        observations, accepted = infer_matrix_observations(e1_commands, e1_states)
        posterior = BayesianMatrixContext(); posterior.update_observations(observations)
        if accepted != row["e1"]["accepted_indices"] or observations_sha256(observations) != row["e1"]["observation_sha256"]:
            failures.append(f"sequence {sequence_id}: E1 estimator inputs")
        compare_array(posterior.mean_matrix, row["e1"]["posterior"]["mean_matrix"], 1e-10, f"sequence {sequence_id} posterior", failures)
        context_matrix = blend_context(POPULATION_PRIOR_MATRIX, posterior.mean_matrix, float(design["fixed_alpha"]))

        e2_segment = pools[expected_selection["e2"]["shape"]][int(expected_selection["e2"]["segment_index"])]
        initial, goal_obs, nominal, _ = prepare_waypoint(env, e2_segment, env_seed + 1)
        start_obs, _ = env.prepare(env_seed + 1, initial)
        compare_array(initial, row["e2"]["initial_state"], 1e-6, f"sequence {sequence_id} E2 initial", failures)
        compare_array(nominal[-1], row["e2"]["goal_state"], 1e-6, f"sequence {sequence_id} E2 goal", failures)
        commands = {}
        for name, matrix in (("population", POPULATION_PRIOR_MATRIX), ("context", context_matrix)):
            current, _ = plan_matrix_waypoint(wm, preprocessor, start_obs, goal_obs, matrix, cem_seed + 1)
            commands[name] = current
            compare_array(current, row["e2"]["policies"][name]["commands"], 0.0, f"sequence {sequence_id} {name} plan", failures)

        traces, goals, scores = {}, [], {}
        probe_rng_before = full_rng_digest()
        for model_name, matrix in (("prior", POPULATION_PRIOR_MATRIX), ("context", context_matrix)):
            for plan_name in ("population", "context"):
                prediction, goal, score = independent_rollout(wm, preprocessor, start_obs, goal_obs, commands[plan_name], matrix)
                traces[f"{model_name}_{plan_name}"] = prediction
                goals.append(goal)
                action_name = "prior" if plan_name == "population" else "context"
                scores[f"J_{model_name}_a_{action_name}"] = score
        wm.set_context(POPULATION_PRIOR_MATRIX)
        if probe_rng_before != full_rng_digest() or row["e2"]["probe_rng_digest_before"] != row["e2"]["probe_rng_digest_after"]:
            failures.append(f"sequence {sequence_id}: read-only RNG")
        latent, objective = features(traces, scores)
        compare_array(latent, row["e2"]["latent_features"], 1e-10, f"sequence {sequence_id} latent features", failures)
        compare_array(objective, row["e2"]["objective_features"], 1e-7, f"sequence {sequence_id} objective features", failures)
        for key, value in scores.items():
            max_metric_error = max(max_metric_error, abs(value - float(row["e2"]["model_scores"][key])))

        trace_path = Path(row["e2"]["trace"]["path"])
        if sha256(trace_path) != row["e2"]["trace"]["sha256"]:
            failures.append(f"sequence {sequence_id}: trace NPZ hash")
        with np.load(trace_path, allow_pickle=False) as saved:
            expected_arrays = {f"{name}__{key}": value for name, trace in traces.items() for key, value in trace.items()}
            expected_arrays.update({f"goal_latent__{key}": value for key, value in goals[0].items()})
            if set(saved.files) != set(expected_arrays):
                failures.append(f"sequence {sequence_id}: trace array keys")
            for key, value in expected_arrays.items():
                actual = saved[key]
                error = float(np.max(np.abs(actual.astype(np.float64) - value.astype(np.float64)))) if actual.size else 0.0
                max_trace_error = max(max_trace_error, error)
                compare_array(actual, value, 1e-6, f"sequence {sequence_id} trace {key}", failures)
                item = row["e2"]["trace"]["arrays"][key]
                if item["shape"] != list(actual.shape) or item["dtype"] != str(actual.dtype) or item["sha256"] != array_sha256(actual):
                    failures.append(f"sequence {sequence_id}: trace manifest {key}")

        expected_order = ["population", "context"] if sequence_id % 2 == 0 else ["context", "population"]
        if row["e2"]["execution_order"] != expected_order:
            failures.append(f"sequence {sequence_id}: execution order")
        for execution_index, name in enumerate(expected_order):
            policy = row["e2"]["policies"][name]
            matrix = POPULATION_PRIOR_MATRIX if name == "population" else context_matrix
            states, effective, _, _ = execute_matrix(env, initial, env_seed + 1, commands[name], true_matrix)
            state_error = float(np.max(np.abs(states - np.asarray(policy["states"]))))
            max_state_error = max(max_state_error, state_error)
            compare_array(effective, policy["effective_actions"], 1e-6, f"sequence {sequence_id} {name} effective", failures)
            if int(policy["execution_index"]) != execution_index or float(policy["execution_started_unix"]) < float(row["e2"]["trace_completed_unix"]):
                failures.append(f"sequence {sequence_id}: {name} execution timing/order")
            recomputed_metrics = pose_metrics(states, nominal[-1], WINDOW)
            for key, value in recomputed_metrics.items():
                error = abs(value - float(policy["metrics"][key])); max_metric_error = max(max_metric_error, error)
                if error > 1e-10:
                    failures.append(f"sequence {sequence_id}: {name} metric {key}")
        if args.split == "smoke" and row["e2"]["trace_repeat_exact"] is not True:
            failures.append(f"sequence {sequence_id}: repeat trace identity")
        print(f"D4 audit {args.split} sequence={sequence_id} complete", flush=True)
    state_after, rng_after = model_sha256(wm), full_rng_digest()
    if state_before != state_after:
        failures.append("auditor changed model state")
    # Environment replay may use global RNG; only the read-only probe has a strict RNG invariant.
    result = {
        "contract_id": CONTRACT_ID,
        "split": args.split,
        "valid": not failures,
        "failures": failures,
        "rows": len(rows),
        "raw_sha256": sha256(raw_path),
        "model_state_unchanged": state_before == state_after,
        "auditor_global_rng_before": rng_before,
        "auditor_global_rng_after": rng_after,
        "max_trace_abs_error": max_trace_error,
        "max_state_abs_error": max_state_error,
        "max_score_or_metric_abs_error": max_metric_error,
    }
    dump_json(split_dir / "independent_audit.json", result)
    return result


def audit_analysis(args, design):
    development = args.output_dir / "development"
    rows = [row for row in read_jsonl(development / "raw.jsonl") if row.get("record_type") == "d4_sequence"]
    report = json.loads((development / "analysis.json").read_text(encoding="utf-8"))
    failures = []
    population = np.asarray([row["e2"]["policies"]["population"]["metrics"]["pose_auc10"] for row in rows])
    context = np.asarray([row["e2"]["policies"]["context"]["metrics"]["pose_auc10"] for row in rows])
    benefit = population - context
    harm = benefit < -float(design["harm_tolerance"])
    latent = np.asarray([row["e2"]["latent_features"] for row in rows])
    objective = np.asarray([row["e2"]["objective_features"] for row in rows])
    primary = latent[:, design["latent_feature_names"].index(design["primary_risk_score"])]
    expected_auc = auc(harm, primary)
    expected_ci, expected_count = bootstrap_auc(harm, primary, design)
    primary_report = report["primary"]
    if expected_auc != primary_report["roc_auc_harm"] or expected_count != primary_report["valid_bootstrap_resamples"]:
        failures.append("primary AUC or bootstrap count")
    if expected_ci is None:
        if primary_report["bootstrap_ci95_auc"] is not None:
            failures.append("primary bootstrap nullability")
    elif not np.allclose(expected_ci, primary_report["bootstrap_ci95_auc"], atol=0, rtol=0):
        failures.append("primary bootstrap interval")
    if not np.allclose(benefit, report["unit_benefit"], atol=0, rtol=0):
        failures.append("unit benefit")
    if int(harm.sum()) != int(report["harm_count"]):
        failures.append("harm count")
    for index, name in enumerate(design["latent_feature_names"]):
        item = report["secondary_scores"][name]
        if item["roc_auc_harm"] != auc(harm, latent[:, index]) or not np.allclose(item["unit_values"], latent[:, index], atol=0, rtol=0):
            failures.append(f"secondary latent {name}")
    for index, name in enumerate(design["objective_feature_names"]):
        item = report["secondary_scores"][name]
        if item["roc_auc_harm"] != auc(harm, objective[:, index]) or not np.allclose(item["unit_values"], objective[:, index], atol=0, rtol=0):
            failures.append(f"secondary objective {name}")
    mechanism = [{"x": np.concatenate((latent[i], objective[i])), "benefit": benefit[i],
                  "shape_pair_index": int(rows[i]["shape_pair_index"])} for i in range(len(rows))]
    prediction, chosen = nested(mechanism, design["ridge_alphas"])
    ridge = report["ridge_leave_one_shape_pair_out"]
    if chosen != ridge["chosen_ridge_by_outer_pair"] or not np.allclose(prediction, ridge["predictions"], atol=1e-12, rtol=1e-12):
        failures.append("nested ridge")
    result = {"contract_id": CONTRACT_ID, "valid": not failures, "failures": failures,
              "analysis_sha256": sha256(development / "analysis.json"), "n_sequences": len(rows)}
    dump_json(development / "independent_analysis_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("raw", "analysis", "all"))
    parser.add_argument("--split", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_contract_zh.md"))
    parser.add_argument("--selection", type=Path, default=Path("docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_selection.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_rollout_disagreement_d4_v1"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path, expected, label in ((args.design, EXPECTED_DESIGN_SHA256, "design"),
                                  (args.contract, EXPECTED_CONTRACT_SHA256, "contract"),
                                  (args.selection, EXPECTED_SELECTION_SHA256, "selection")):
        if expected == "TO_BE_FROZEN" or sha256(path) != expected:
            raise RuntimeError(f"frozen {label} hash mismatch")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    results = {}
    if args.mode in ("raw", "all"):
        results["raw"] = audit_raw(args, design, selection)
    if args.mode in ("analysis", "all"):
        results["analysis"] = audit_analysis(args, design)
    print(json.dumps(results, indent=2))
    if any(not result["valid"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
