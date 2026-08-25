"""Development-only 100 Hz contact-response audit for hidden PushObj CoG."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.pushobj_cog_markov_contact_audit import (
    CONTACT_FEATURES,
    CONTACT_KINDS,
    ContactRecorder,
    action_variants,
    array_sha256,
    boundary_state,
)
from research.persistent_context_v2.pushobj_cog_stage0 import (
    _setup_physics,
    prepare_waypoint_physics,
    rollout_physics,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    dump_json,
    git_revision,
    make_env,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-pushobj-cog-event-response-audit-v1"
EXPECTED_DESIGN_SHA256 = "e9cf733782795997a45196f606656f1955784c2c58002f40c8707c2fa8719de6"
REPRESENTATIONS = (
    "C10_aggregate",
    "S100_state",
    "S100_state_geometry",
    "S100_state_geometry_impulse",
    "P100_true_contact",
)
PRIMARY_FIRST = "C10_aggregate"
PRIMARY_SECOND = "S100_state_geometry_impulse"
AGENT_BLOCK = CONTACT_KINDS.index("agent_block")
BLOCK_WALL = CONTACT_KINDS.index("block_wall")

GEOMETRY_FEATURES = (
    "point_count",
    "first_contact_count",
    "normal_x",
    "normal_y",
    "contact_rel_x",
    "contact_rel_y",
    "contact_radius",
    "min_distance",
)
IMPULSE_FEATURES = (
    "impulse_x",
    "impulse_y",
    "impulse_norm_sum",
    "impulse_norm_max",
    "total_ke_sum",
    "total_ke_max",
)
GEOMETRY_INDEX = np.asarray([CONTACT_FEATURES.index(name) for name in GEOMETRY_FEATURES], dtype=np.int64)
IMPULSE_INDEX = np.asarray([CONTACT_FEATURES.index(name) for name in IMPULSE_FEATURES], dtype=np.int64)


def _contact_index(name: str) -> int:
    return CONTACT_FEATURES.index(name)


def rollout_substeps(env, shape, initial_state, env_seed, commands, cog_x):
    """Run the exact controller while recording every 100 Hz physics substep."""
    from pymunk.vec2d import Vec2d

    _setup_physics(env, shape, initial_state, env_seed, cog_x)
    recorder = ContactRecorder(env)
    env.collision_handeler.post_solve = recorder
    dt = 1.0 / env.sim_hz
    substeps_per_control = env.sim_hz // env.control_hz

    boundary_states = [boundary_state(env)]
    pre_states, post_states, contacts = [], [], []
    command_rows, target_rows, control_ids, substep_ids = [], [], [], []

    for control_id, command in enumerate(np.asarray(commands)):
        action = np.asarray(command) * env.action_scale
        target = env.agent.position + action
        for substep_id in range(substeps_per_control):
            recorder.reset()
            env.n_contact_points = 0
            pre_states.append(boundary_state(env))
            acceleration = env.k_p * (target - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
            env.agent.velocity += acceleration * dt
            env.space.step(dt)
            post_states.append(boundary_state(env))
            contacts.append(recorder.as_array())
            command_rows.append(command)
            target_rows.append((float(target.x), float(target.y)))
            control_ids.append(control_id)
            substep_ids.append(substep_id)
        boundary_states.append(boundary_state(env))

    return {
        "boundary_states": np.asarray(boundary_states, dtype=np.float32),
        "pre_states": np.asarray(pre_states, dtype=np.float32),
        "post_states": np.asarray(post_states, dtype=np.float32),
        "contacts": np.asarray(contacts, dtype=np.float32),
        "commands": np.asarray(command_rows, dtype=np.float32),
        "targets": np.asarray(target_rows, dtype=np.float32),
        "control_ids": np.asarray(control_ids, dtype=np.int16),
        "substep_ids": np.asarray(substep_ids, dtype=np.int16),
    }


def aggregate_contact_substeps(contacts, control_ids, action_count=10):
    """Reconstruct the P3a-style 10 Hz contact summary from 100 Hz rows."""
    contacts = np.asarray(contacts, dtype=np.float64)
    control_ids = np.asarray(control_ids)
    result = np.zeros((action_count, len(CONTACT_KINDS), len(CONTACT_FEATURES)), dtype=np.float64)
    sum_names = (
        "callback_count", "point_count", "first_contact_count", "impulse_x", "impulse_y",
        "abs_impulse_x", "abs_impulse_y", "impulse_norm_sum", "total_ke_sum",
    )
    max_names = ("impulse_norm_max", "total_ke_max")
    for control_id in range(action_count):
        rows = contacts[control_ids == control_id]
        for kind in range(len(CONTACT_KINDS)):
            values = rows[:, kind, :]
            for name in sum_names:
                index = _contact_index(name)
                result[control_id, kind, index] = values[:, index].sum()
            for name in max_names:
                index = _contact_index(name)
                result[control_id, kind, index] = values[:, index].max(initial=0.0)

            impulse_weight = values[:, _contact_index("impulse_norm_sum")]
            impulse_total = impulse_weight.sum()
            if impulse_total > 0:
                for name in ("normal_x", "normal_y"):
                    index = _contact_index(name)
                    result[control_id, kind, index] = np.sum(values[:, index] * impulse_weight) / impulse_total

            point_weight = values[:, _contact_index("point_count")]
            point_total = point_weight.sum()
            if point_total > 0:
                for name in ("contact_rel_x", "contact_rel_y", "contact_radius"):
                    index = _contact_index(name)
                    result[control_id, kind, index] = np.sum(values[:, index] * point_weight) / point_total
                distance = values[:, _contact_index("min_distance")][point_weight > 0]
                result[control_id, kind, _contact_index("min_distance")] = distance.min()
    return result.astype(np.float32)


def eligible_event_mask(trace):
    contacts = trace["contacts"]
    agent_block_points = contacts[:, AGENT_BLOCK, _contact_index("point_count")]
    block_wall_points = contacts[:, BLOCK_WALL, _contact_index("point_count")]
    return (agent_block_points > 0) & (block_wall_points == 0)


def response_target(nominal_trace, true_trace, mask):
    nominal_delta = nominal_trace["post_states"][:, 7:10] - nominal_trace["pre_states"][:, 7:10]
    true_delta = true_trace["post_states"][:, 7:10] - true_trace["pre_states"][:, 7:10]
    return (true_delta - nominal_delta)[mask].astype(np.float32)


def representation_rows(nominal_trace, true_trace, mask):
    event_indices = np.flatnonzero(mask)
    control_ids = nominal_trace["control_ids"][event_indices]
    nominal_contact = nominal_trace["contacts"][event_indices, AGENT_BLOCK]
    true_contact = true_trace["contacts"][event_indices, AGENT_BLOCK]
    control_contact = aggregate_contact_substeps(
        nominal_trace["contacts"], nominal_trace["control_ids"],
        action_count=len(nominal_trace["boundary_states"]) - 1,
    )[control_ids, AGENT_BLOCK]

    pre = nominal_trace["pre_states"][event_indices]
    commands = nominal_trace["commands"][event_indices]
    target_delta = nominal_trace["targets"][event_indices] - pre[:, :2]
    substeps_per_control = int(nominal_trace["substep_ids"].max()) + 1
    phase = ((nominal_trace["substep_ids"][event_indices].astype(np.float32) + 0.5) / substeps_per_control)[:, None]
    state = np.concatenate([pre, commands, target_delta, phase], axis=1).astype(np.float32)

    c10 = np.concatenate([
        nominal_trace["boundary_states"][control_ids],
        commands,
        control_contact,
    ], axis=1).astype(np.float32)
    geometry = nominal_contact[:, GEOMETRY_INDEX]
    impulse = nominal_contact[:, IMPULSE_INDEX]
    true_privileged = np.concatenate([true_contact[:, GEOMETRY_INDEX], true_contact[:, IMPULSE_INDEX]], axis=1)
    return {
        "C10_aggregate": c10,
        "S100_state": state,
        "S100_state_geometry": np.concatenate([state, geometry], axis=1).astype(np.float32),
        "S100_state_geometry_impulse": np.concatenate([state, geometry, impulse], axis=1).astype(np.float32),
        "P100_true_contact": np.concatenate([state, geometry, impulse, true_privileged], axis=1).astype(np.float32),
    }


def deterministic_repeat_audit(env, segment, segment_id, design):
    env_seed = int(design["env_seed_base"]) + int(segment_id)
    shape, initial_state, commands, _ = prepare_waypoint_physics(env, segment, env_seed)
    first = rollout_substeps(env, shape, initial_state, env_seed, commands, -25.0)
    second = rollout_substeps(env, shape, initial_state, env_seed, commands, -25.0)
    keys = ("boundary_states", "pre_states", "post_states", "contacts", "commands", "targets")
    return {
        "equal": all(np.array_equal(first[key], second[key]) for key in keys),
        "hashes_first": {key: array_sha256(first[key]) for key in keys},
        "hashes_second": {key: array_sha256(second[key]) for key in keys},
    }


def generate_split(env, segments, indices, factors, design, seed, split_name):
    rng = np.random.default_rng(int(seed))
    storage = {name: [] for name in REPRESENTATIONS}
    targets, contexts, segment_ids, variant_ids = [], [], [], []
    control_ids, substep_ids, true_event_flags = [], [], []
    nominal_points, nominal_wall_points = [], []
    nominal_impulses, true_impulses = [], []
    identity_max_abs = 0.0
    event_counts = {}

    for ordinal, segment_id in enumerate(indices):
        segment = segments[int(segment_id)]
        env_seed = int(design["env_seed_base"]) + int(segment_id)
        shape, initial_state, nominal_commands, _ = prepare_waypoint_physics(env, segment, env_seed)
        segment_event_count = 0
        for variant_id, commands in enumerate(action_variants(nominal_commands, design["action_noise_sigmas"], rng)):
            nominal_trace = rollout_substeps(env, shape, initial_state, env_seed, commands, 0.0)
            reference = rollout_physics(env, shape, initial_state, env_seed, commands, 0.0)
            identity_max_abs = max(
                identity_max_abs,
                float(np.max(np.abs(nominal_trace["boundary_states"][:, :7] - reference))),
            )
            mask = eligible_event_mask(nominal_trace)
            event_indices = np.flatnonzero(mask)
            segment_event_count += int(len(event_indices))
            for factor in factors:
                true_trace = nominal_trace if float(factor) == 0.0 else rollout_substeps(
                    env, shape, initial_state, env_seed, commands, factor
                )
                rows = representation_rows(nominal_trace, true_trace, mask)
                for name in REPRESENTATIONS:
                    storage[name].append(rows[name])
                count = len(event_indices)
                targets.append(response_target(nominal_trace, true_trace, mask))
                contexts.append(np.full(count, float(factor), dtype=np.float32))
                segment_ids.append(np.full(count, int(segment_id), dtype=np.int32))
                variant_ids.append(np.full(count, int(variant_id), dtype=np.int16))
                control_ids.append(nominal_trace["control_ids"][event_indices])
                substep_ids.append(nominal_trace["substep_ids"][event_indices])
                true_contact = true_trace["contacts"][event_indices, AGENT_BLOCK]
                nominal_contact = nominal_trace["contacts"][event_indices, AGENT_BLOCK]
                true_event_flags.append(true_contact[:, _contact_index("point_count")] > 0)
                nominal_points.append(nominal_contact[:, _contact_index("point_count")])
                nominal_wall_points.append(nominal_trace["contacts"][event_indices, BLOCK_WALL, _contact_index("point_count")])
                nominal_impulses.append(nominal_contact[:, _contact_index("impulse_norm_sum")])
                true_impulses.append(true_contact[:, _contact_index("impulse_norm_sum")])
        event_counts[str(int(segment_id))] = int(segment_event_count)
        print(f"GENERATE {split_name} {ordinal + 1}/{len(indices)} segment={segment_id} events={segment_event_count}", flush=True)

    def concatenate(values, dtype=None):
        result = np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
        return result.astype(dtype) if dtype is not None else result

    payload = {f"inputs_{name}": concatenate(storage[name], np.float32) for name in REPRESENTATIONS}
    payload.update(
        targets=concatenate(targets, np.float32),
        contexts=concatenate(contexts, np.float32),
        segment_ids=concatenate(segment_ids, np.int32),
        variant_ids=concatenate(variant_ids, np.int16),
        control_ids=concatenate(control_ids, np.int16),
        substep_ids=concatenate(substep_ids, np.int16),
        true_event_flags=concatenate(true_event_flags, bool),
        nominal_point_count=concatenate(nominal_points, np.float32),
        nominal_block_wall_point_count=concatenate(nominal_wall_points, np.float32),
        nominal_impulse_norm=concatenate(nominal_impulses, np.float32),
        true_impulse_norm=concatenate(true_impulses, np.float32),
        identity_max_abs=np.asarray(identity_max_abs, dtype=np.float64),
    )
    return payload, event_counts


def standardizer(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def feature_map(values, contexts, mean, scale):
    normalized = (np.asarray(values, dtype=np.float64) - mean) / scale
    context = np.asarray(contexts, dtype=np.float64).reshape(-1, 1) / 30.0
    return np.concatenate([context, np.square(context), context * normalized, np.square(context) * normalized], axis=1)


def ridge_fit(features, targets, alpha):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.shape[1] > features.shape[0]:
        gram = features @ features.T
        gram.flat[:: len(gram) + 1] += float(alpha)
        return features.T @ np.linalg.solve(gram, targets)
    gram = features.T @ features
    gram.flat[:: len(gram) + 1] += float(alpha)
    return np.linalg.solve(gram, features.T @ targets)


def response_errors(prediction_scaled, target_scaled):
    difference = np.asarray(prediction_scaled, dtype=np.float64) - np.asarray(target_scaled, dtype=np.float64)
    return np.sqrt(np.mean(np.square(difference), axis=1))


def grouped_means(values, segment_ids):
    values = np.asarray(values, dtype=np.float64)
    segment_ids = np.asarray(segment_ids)
    unique = np.unique(segment_ids)
    return unique, np.asarray([values[segment_ids == segment].mean() for segment in unique], dtype=np.float64)


def choose_alpha(values, target_scaled, contexts, segment_ids, design):
    unique = np.unique(segment_ids)
    folds = min(int(design["group_cv_folds"]), len(unique))
    if folds < 2:
        raise ValueError("at least two train segments are required")
    fold_by_segment = {int(segment): index % folds for index, segment in enumerate(unique)}
    rows = []
    for alpha in design["ridge_alphas"]:
        fold_scores = []
        for fold in range(folds):
            valid = np.asarray([fold_by_segment[int(segment)] == fold for segment in segment_ids])
            train = ~valid
            mean, scale = standardizer(values[train])
            beta = ridge_fit(feature_map(values[train], contexts[train], mean, scale), target_scaled[train], alpha)
            prediction = feature_map(values[valid], contexts[valid], mean, scale) @ beta
            _, segment_error = grouped_means(response_errors(prediction, target_scaled[valid]), segment_ids[valid])
            fold_scores.append(float(segment_error.mean()))
        rows.append({
            "alpha": float(alpha),
            "fold_scores": fold_scores,
            "mean_segment_error": float(np.mean(fold_scores)),
            "effective_folds": folds,
        })
    selected = min(rows, key=lambda row: (row["mean_segment_error"], -row["alpha"]))
    return float(selected["alpha"]), rows


def target_scale_from_train(train, floor):
    mask = np.asarray(train["contexts"]) != 0
    scale = np.sqrt(np.mean(np.square(np.asarray(train["targets"], dtype=np.float64)[mask]), axis=0))
    return np.maximum(scale, float(floor))


def fit_frozen_models(train, design, output_dir):
    target_scale = target_scale_from_train(train, design["target_rms_floor"])
    target_scaled = np.asarray(train["targets"], dtype=np.float64) / target_scale
    arrays = {"target_scale": target_scale}
    details = {}
    for name in REPRESENTATIONS:
        values = train[f"inputs_{name}"]
        alpha, cv = choose_alpha(values, target_scaled, train["contexts"], train["segment_ids"], design)
        mean, scale = standardizer(values)
        beta = ridge_fit(feature_map(values, train["contexts"], mean, scale), target_scaled, alpha)
        zero = feature_map(values[: min(3, len(values))], np.zeros(min(3, len(values))), mean, scale) @ beta
        arrays[f"{name}_mean"] = mean
        arrays[f"{name}_scale"] = scale
        arrays[f"{name}_beta"] = beta
        arrays[f"{name}_alpha"] = np.asarray(alpha, dtype=np.float64)
        details[name] = {
            "input_dim": int(values.shape[1]),
            "mapped_dim": int(beta.shape[0]),
            "selected_alpha": alpha,
            "cv": cv,
            "zero_context_max_abs": float(np.max(np.abs(zero))) if len(zero) else 0.0,
            "mean_sha256": array_sha256(mean),
            "scale_sha256": array_sha256(scale),
            "beta_sha256": array_sha256(beta),
        }
        print(f"FIT {name} alpha={alpha}", flush=True)
    model_path = output_dir / "frozen_models.npz"
    np.savez_compressed(model_path, **arrays)
    selection = {
        "contract_id": CONTRACT_ID,
        "locked_unix": time.time(),
        "target_scale": target_scale.tolist(),
        "target_scale_sha256": array_sha256(target_scale),
        "models": details,
        "frozen_models_sha256": sha256(model_path),
    }
    dump_json(output_dir / "selection_summary.json", selection)
    return selection


def predict_frozen(evaluation, model_path):
    models = dict(np.load(model_path))
    target_scale = np.asarray(models["target_scale"], dtype=np.float64)
    predictions = {"zero_response": np.zeros_like(evaluation["targets"], dtype=np.float32)}
    for name in REPRESENTATIONS:
        prediction_scaled = feature_map(
            evaluation[f"inputs_{name}"],
            evaluation["contexts"],
            models[f"{name}_mean"],
            models[f"{name}_scale"],
        ) @ models[f"{name}_beta"]
        predictions[name] = (prediction_scaled * target_scale).astype(np.float32)
    return predictions, target_scale


def bootstrap_comparison(first_error, second_error, segment_ids, design):
    _, first = grouped_means(first_error, segment_ids)
    _, second = grouped_means(second_error, segment_ids)
    delta = first - second
    rng = np.random.default_rng(int(design["bootstrap_seed"]))
    indexes = rng.integers(0, len(delta), size=(int(design["bootstrap_resamples"]), len(delta)))
    means = delta[indexes].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "positive": int(np.sum(delta > 1e-12)),
        "tie": int(np.sum(np.abs(delta) <= 1e-12)),
        "negative": int(np.sum(delta < -1e-12)),
        "n_segments": int(len(delta)),
    }


def summarize(evaluation, predictions, target_scale, selection, design, data_manifest, smoke):
    target_scaled = np.asarray(evaluation["targets"], dtype=np.float64) / target_scale
    errors = {
        name: response_errors(np.asarray(prediction, dtype=np.float64) / target_scale, target_scaled)
        for name, prediction in predictions.items()
    }
    summary = {
        "contract_id": CONTRACT_ID,
        "nature": "smoke only" if smoke else "development event-response audit; not a formal closed-loop result",
        "primary_metric": design["primary_metric"],
        "primary_comparison": design["primary_comparison"],
        "target_scale": target_scale.tolist(),
        "models": {},
        "comparisons": {},
        "by_factor": {},
        "event_diagnostics": {},
    }
    for name, error in errors.items():
        _, segment_error = grouped_means(error, evaluation["segment_ids"])
        summary["models"][name] = {
            "mean_event_error": float(error.mean()),
            "mean_segment_error": float(segment_error.mean()),
            **selection.get("models", {}).get(name, {}),
        }
    comparisons = (
        (PRIMARY_FIRST, PRIMARY_SECOND),
        ("S100_state", "S100_state_geometry"),
        ("S100_state_geometry", "S100_state_geometry_impulse"),
        ("S100_state_geometry_impulse", "P100_true_contact"),
        ("zero_response", "S100_state_geometry_impulse"),
    )
    for first, second in comparisons:
        summary["comparisons"][f"{first}_minus_{second}"] = bootstrap_comparison(
            errors[first], errors[second], evaluation["segment_ids"], design
        )
    for factor in design["eval_factors_cog_x"][: None if not smoke else len(design["eval_factors_cog_x"])]:
        mask = evaluation["contexts"] == float(factor)
        if not mask.any():
            continue
        summary["by_factor"][str(float(factor))] = {
            name: float(grouped_means(error[mask], evaluation["segment_ids"][mask])[1].mean())
            for name, error in errors.items()
        }
    event_mismatch = ~np.asarray(evaluation["true_event_flags"], dtype=bool)
    impulse_delta = np.abs(evaluation["true_impulse_norm"] - evaluation["nominal_impulse_norm"])
    summary["event_diagnostics"] = {
        "eval_factor_event_rows": int(len(evaluation["targets"])),
        "eval_unique_segments": int(len(np.unique(evaluation["segment_ids"]))),
        "true_contact_missing_at_nominal_event_fraction": float(event_mismatch.mean()),
        "mean_abs_true_nominal_impulse_difference": float(impulse_delta.mean()),
        "train_nominal_events_by_segment": data_manifest["train_nominal_events_by_segment"],
        "eval_nominal_events_by_segment": data_manifest["eval_nominal_events_by_segment"],
    }
    required_train = len(design["train_segment_indices"]) if not smoke else data_manifest["train_segment_count"]
    required_eval = len(design["eval_segment_indices"]) if not smoke else data_manifest["eval_segment_count"]
    checks = {
        **validate_design(design),
        "full_train_segments": len(np.unique(data_manifest["train_segments"])) == required_train,
        "full_eval_segments": len(np.unique(evaluation["segment_ids"])) == required_eval,
        "all_train_segments_have_events": all(value > 0 for value in data_manifest["train_nominal_events_by_segment"].values()),
        "all_eval_segments_have_events": all(value > 0 for value in data_manifest["eval_nominal_events_by_segment"].values()),
        "rollout_identity": max(data_manifest["train_identity_max_abs"], data_manifest["eval_identity_max_abs"]) <= 1e-6,
        "deterministic_repeat": bool(data_manifest["deterministic_repeat"]["equal"]),
        "zero_context_target": bool(np.max(np.abs(np.asarray(data_manifest["zero_context_target_max_abs"]))) == 0.0),
        "zero_context_models": all(selection["models"][name]["zero_context_max_abs"] == 0.0 for name in REPRESENTATIONS),
        "model_locked_before_eval": selection["locked_unix"] <= data_manifest["eval_generated_unix"],
        "finite": bool(
            all(bool(np.isfinite(value).all()) for value in predictions.values())
            and bool(np.isfinite(evaluation["targets"]).all())
        ),
        "event_rule": bool(np.all(evaluation["nominal_point_count"] > 0) and np.all(evaluation["nominal_block_wall_point_count"] == 0)),
    }
    summary["structural_checks"] = checks
    summary["valid"] = all(checks.values())
    return summary, errors


def validate_design(design):
    train = set(map(int, design["train_segment_indices"]))
    evaluation = set(map(int, design["eval_segment_indices"]))
    formal = set(map(int, design["forbidden_formal_segment_indices"]))
    checks = {
        "contract_id": design["contract_id"] == CONTRACT_ID,
        "train_eval_disjoint": train.isdisjoint(evaluation),
        "formal_not_used": formal.isdisjoint(train | evaluation),
        "frequency": int(design["sim_hz"]) == 100 and int(design["control_hz"]) == 10,
        "representations": tuple(design["representations"]) == REPRESENTATIONS,
        "primary_comparison_frozen": design["primary_comparison"] == f"{PRIMARY_FIRST}_minus_{PRIMARY_SECOND}",
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid frozen design: {checks}")
    return checks


def load_segments(data_path):
    with data_path.open("rb") as handle:
        return pickle.load(handle)["segments"]


def run(args, design):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError("output directory is not empty; use a new append-only directory")
    if not args.source_snapshot.is_file():
        raise FileNotFoundError(f"missing source snapshot: {args.source_snapshot}")

    train_indices = design["train_segment_indices"][: args.limit_train]
    eval_indices = design["eval_segment_indices"][: args.limit_eval]
    manifest = {
        "contract_id": CONTRACT_ID,
        "command": " ".join(sys.argv),
        "git_revision": git_revision(),
        "design_sha256": sha256(args.design),
        "contract_sha256": sha256(args.contract),
        "source_snapshot": str(args.source_snapshot),
        "source_snapshot_sha256": sha256(args.source_snapshot),
        "data_sha256": sha256(args.data),
        "started_unix": time.time(),
        "resource_start": resource_snapshot(torch.device("cpu")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "allocation_rationale": "CPU-only Pymunk event logging and ridge audit; no GPU allocated",
        "smoke": bool(args.smoke),
        "dirty_status": __import__("subprocess").run(
            ["git", "status", "--short"], capture_output=True, text=True
        ).stdout.splitlines(),
    }
    dump_json(output_dir / "manifest.json", manifest)

    seed_all(int(design["model_seed"]))
    segments = load_segments(args.data)
    env = make_env()
    repeat = deterministic_repeat_audit(env, segments[int(eval_indices[0])], int(eval_indices[0]), design)

    train, train_counts = generate_split(
        env, segments, train_indices, design["train_factors_cog_x"], design, design["variant_seed"], "train"
    )
    np.savez_compressed(output_dir / "train_data.npz", **train)
    selection = fit_frozen_models(train, design, output_dir)

    eval_generated_unix = time.time()
    evaluation, eval_counts = generate_split(
        env, segments, eval_indices, design["eval_factors_cog_x"], design, design["variant_seed"] + 1, "eval"
    )
    np.savez_compressed(output_dir / "eval_data.npz", **evaluation)
    zero_context = train["contexts"] == 0
    data_manifest = {
        "train_segment_count": int(len(train_indices)),
        "eval_segment_count": int(len(eval_indices)),
        "train_segments": list(map(int, train_indices)),
        "eval_segments": list(map(int, eval_indices)),
        "train_factor_event_rows": int(len(train["targets"])),
        "eval_factor_event_rows": int(len(evaluation["targets"])),
        "train_nominal_events_by_segment": train_counts,
        "eval_nominal_events_by_segment": eval_counts,
        "train_identity_max_abs": float(train["identity_max_abs"]),
        "eval_identity_max_abs": float(evaluation["identity_max_abs"]),
        "zero_context_target_max_abs": float(np.max(np.abs(train["targets"][zero_context]))),
        "deterministic_repeat": repeat,
        "selection_summary_sha256_before_eval": sha256(output_dir / "selection_summary.json"),
        "frozen_models_sha256_before_eval": sha256(output_dir / "frozen_models.npz"),
        "eval_generated_unix": eval_generated_unix,
        "arrays": {
            "train_targets": array_sha256(train["targets"]),
            "eval_targets": array_sha256(evaluation["targets"]),
            "train_primary_input": array_sha256(train[f"inputs_{PRIMARY_SECOND}"]),
            "eval_primary_input": array_sha256(evaluation[f"inputs_{PRIMARY_SECOND}"]),
        },
    }
    dump_json(output_dir / "data_manifest.json", data_manifest)

    predictions, target_scale = predict_frozen(evaluation, output_dir / "frozen_models.npz")
    summary, errors = summarize(evaluation, predictions, target_scale, selection, design, data_manifest, args.smoke)
    dump_json(output_dir / "runner_summary.json", summary)
    np.savez_compressed(
        output_dir / "predictions_and_errors.npz",
        **{f"prediction_{name}": value for name, value in predictions.items()},
        **{f"error_{name}": value for name, value in errors.items()},
    )
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(torch.device("cpu"))
    manifest["exit_status"] = 0
    dump_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_event_response_audit_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_event_response_audit_contract_zh.md"))
    parser.add_argument("--source-snapshot", type=Path, default=Path("repro_outputs/persistent_context_v2_p3b_source_snapshot_v1/source_snapshot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_event_response_audit_v1"))
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if sha256(args.design) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen design hash mismatch")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    checks = validate_design(design)
    if args.limit_train is None:
        args.limit_train = len(design["train_segment_indices"])
    if args.limit_eval is None:
        args.limit_eval = len(design["eval_segment_indices"])
    if args.smoke and (args.limit_train >= len(design["train_segment_indices"]) or args.limit_eval >= len(design["eval_segment_indices"])):
        raise ValueError("smoke must use strict train/eval subsets")
    if not args.smoke and (
        args.limit_train != len(design["train_segment_indices"]) or args.limit_eval != len(design["eval_segment_indices"])
    ):
        raise ValueError("non-smoke run must use the full frozen design")
    if args.mode == "inspect":
        print(json.dumps({
            "contract_id": CONTRACT_ID,
            "design_sha256": sha256(args.design),
            "contract_sha256": sha256(args.contract),
            "checks": checks,
        }, indent=2))
        return
    run(args, design)


if __name__ == "__main__":
    main()
