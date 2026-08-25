"""Development-only Markov/contact representation audit for hidden PushObj CoG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.pushobj_cog_predictor import (
    ANGLE_SCALE,
    CoGFiLMResidual,
    POSITION_SCALE,
    encode_trajectory,
    load_model,
    residual_target,
    signed_angle_delta,
)
from research.persistent_context_v2.pushobj_cog_stage0 import (
    ACTION_COUNT,
    COG_Y,
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


CONTRACT_ID = "persistent-context-v2-pushobj-cog-markov-contact-audit-v1"
EXPECTED_DESIGN_SHA256 = "3c93649f79c979115f2031c84d97b4d0ab4b087c14ad10ca21eead96f61e9a6a"
CONTACT_KINDS = ("agent_block", "block_wall", "agent_wall", "other")
CONTACT_FEATURES = (
    "callback_count",
    "point_count",
    "first_contact_count",
    "impulse_x",
    "impulse_y",
    "abs_impulse_x",
    "abs_impulse_y",
    "impulse_norm_sum",
    "impulse_norm_max",
    "normal_x",
    "normal_y",
    "contact_rel_x",
    "contact_rel_y",
    "contact_radius",
    "min_distance",
    "total_ke_sum",
    "total_ke_max",
)
REPRESENTATIONS = ("R0_legacy", "R1_markov", "R2_nominal_agent_block_contact")
CORRECTIONS = ("v1_plus_C1_markov", "v1_plus_C2_markov_contact")


def array_sha256(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _xy(value):
    return np.asarray((float(value.x), float(value.y)), dtype=np.float64)


class ContactRecorder:
    """Read-only post-solve aggregation, canonicalized as impulse on the block."""

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.rows = {
            kind: {
                "callback_count": 0.0,
                "point_count": 0.0,
                "first_contact_count": 0.0,
                "impulse_x": 0.0,
                "impulse_y": 0.0,
                "abs_impulse_x": 0.0,
                "abs_impulse_y": 0.0,
                "impulse_norm_sum": 0.0,
                "impulse_norm_max": 0.0,
                "normal_weighted_x": 0.0,
                "normal_weighted_y": 0.0,
                "normal_weight": 0.0,
                "contact_rel_x_sum": 0.0,
                "contact_rel_y_sum": 0.0,
                "contact_radius_sum": 0.0,
                "contact_sample_count": 0.0,
                "min_distance": math.inf,
                "total_ke_sum": 0.0,
                "total_ke_max": 0.0,
            }
            for kind in CONTACT_KINDS
        }

    def _kind(self, arbiter):
        bodies = tuple(shape.body for shape in arbiter.shapes)
        has_agent = self.env.agent in bodies
        has_block = self.env.block in bodies
        has_static = self.env.space.static_body in bodies
        if has_agent and has_block:
            return "agent_block"
        if has_block and has_static:
            return "block_wall"
        if has_agent and has_static:
            return "agent_wall"
        return "other"

    def __call__(self, arbiter, _space, _data):
        points = arbiter.contact_point_set.points
        self.env.n_contact_points += len(points)
        kind = self._kind(arbiter)
        row = self.rows[kind]
        row["callback_count"] += 1.0
        row["point_count"] += float(len(points))
        row["first_contact_count"] += float(bool(arbiter.is_first_contact))

        impulse = _xy(arbiter.total_impulse)
        normal = _xy(arbiter.contact_point_set.normal)
        # Pymunk reports impulse/normal in shape-A orientation. Reorient to block.
        if arbiter.shapes[0].body is not self.env.block:
            impulse = -impulse
            normal = -normal
        impulse_norm = float(np.linalg.norm(impulse))
        row["impulse_x"] += float(impulse[0])
        row["impulse_y"] += float(impulse[1])
        row["abs_impulse_x"] += abs(float(impulse[0]))
        row["abs_impulse_y"] += abs(float(impulse[1]))
        row["impulse_norm_sum"] += impulse_norm
        row["impulse_norm_max"] = max(row["impulse_norm_max"], impulse_norm)
        weight = max(impulse_norm, 1e-12)
        row["normal_weighted_x"] += float(normal[0]) * weight
        row["normal_weighted_y"] += float(normal[1]) * weight
        row["normal_weight"] += weight
        row["total_ke_sum"] += float(arbiter.total_ke)
        row["total_ke_max"] = max(row["total_ke_max"], float(arbiter.total_ke))

        block_position = np.asarray(tuple(self.env.block.position), dtype=np.float64)
        for point in points:
            midpoint = 0.5 * (_xy(point.point_a) + _xy(point.point_b))
            relative = midpoint - block_position
            row["contact_rel_x_sum"] += float(relative[0])
            row["contact_rel_y_sum"] += float(relative[1])
            row["contact_radius_sum"] += float(np.linalg.norm(relative))
            row["contact_sample_count"] += 1.0
            row["min_distance"] = min(row["min_distance"], float(point.distance))

    def as_array(self):
        result = np.zeros((len(CONTACT_KINDS), len(CONTACT_FEATURES)), dtype=np.float32)
        for kind_index, kind in enumerate(CONTACT_KINDS):
            row = self.rows[kind]
            normal_weight = row["normal_weight"]
            sample_count = row["contact_sample_count"]
            values = {
                **row,
                "normal_x": row["normal_weighted_x"] / normal_weight if normal_weight else 0.0,
                "normal_y": row["normal_weighted_y"] / normal_weight if normal_weight else 0.0,
                "contact_rel_x": row["contact_rel_x_sum"] / sample_count if sample_count else 0.0,
                "contact_rel_y": row["contact_rel_y_sum"] / sample_count if sample_count else 0.0,
                "contact_radius": row["contact_radius_sum"] / sample_count if sample_count else 0.0,
                "min_distance": row["min_distance"] if sample_count else 0.0,
            }
            result[kind_index] = [float(values[name]) for name in CONTACT_FEATURES]
        return result


def boundary_state(env):
    legacy = np.asarray(env._get_obs(), dtype=np.float32)
    block_velocity = np.asarray(tuple(env.block.velocity), dtype=np.float32)
    return np.concatenate([legacy, block_velocity, np.asarray([env.block.angular_velocity], dtype=np.float32)])


def rollout_enriched(env, shape, initial_state, env_seed, commands, cog_x):
    """Exact rollout_physics integration with read-only boundary/contact capture."""
    from pymunk.vec2d import Vec2d

    _setup_physics(env, shape, initial_state, env_seed, cog_x)
    recorder = ContactRecorder(env)
    env.collision_handeler.post_solve = recorder
    states = [boundary_state(env)]
    contacts = []
    for command in np.asarray(commands):
        recorder.reset()
        env.n_contact_points = 0
        action = np.asarray(command) * env.action_scale
        target = env.agent.position + action
        dt = 1.0 / env.sim_hz
        for _ in range(env.sim_hz // env.control_hz):
            acceleration = env.k_p * (target - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
            env.agent.velocity += acceleration * dt
            env.space.step(dt)
        states.append(boundary_state(env))
        contacts.append(recorder.as_array())
    return np.asarray(states, dtype=np.float32), np.asarray(contacts, dtype=np.float32)


def action_variants(nominal, sigmas, rng):
    yield np.asarray(nominal, dtype=np.float32)
    for sigma in sigmas:
        yield (np.asarray(nominal, dtype=np.float32) + rng.normal(0.0, sigma, size=np.asarray(nominal).shape)).astype(np.float32)


def representation_inputs(commands, nominal_states, nominal_contacts):
    legacy = nominal_states[:, :7]
    r0 = encode_trajectory(commands, legacy)
    markov_extra = nominal_states[:, 7:10].reshape(-1)
    agent_block = nominal_contacts[:, CONTACT_KINDS.index("agent_block"), :].reshape(-1)
    return {
        "R0_legacy": r0.astype(np.float32),
        "R1_markov": np.concatenate([r0, markov_extra]).astype(np.float32),
        "R2_nominal_agent_block_contact": np.concatenate([r0, markov_extra, agent_block]).astype(np.float32),
    }


def generate_split(env, segments, indices, factors, design, seed, split_name):
    rng = np.random.default_rng(seed)
    storage = {name: [] for name in REPRESENTATIONS}
    targets, contexts, segment_ids, variant_ids = [], [], [], []
    true_markov, true_contacts = [], []
    identity_max_abs = 0.0
    for ordinal, segment_id in enumerate(indices):
        segment = segments[int(segment_id)]
        env_seed = int(design["env_seed_base"]) + int(segment_id)
        shape, initial_state, nominal_commands, _ = prepare_waypoint_physics(env, segment, env_seed)
        for variant_id, commands in enumerate(action_variants(nominal_commands, design["action_noise_sigmas"], rng)):
            nominal_states, nominal_contacts = rollout_enriched(env, shape, initial_state, env_seed, commands, 0.0)
            reference = rollout_physics(env, shape, initial_state, env_seed, commands, 0.0)
            identity_max_abs = max(identity_max_abs, float(np.max(np.abs(nominal_states[:, :7] - reference))))
            inputs = representation_inputs(commands, nominal_states, nominal_contacts)
            for factor in factors:
                if float(factor) == 0.0:
                    factor_states, factor_contacts = nominal_states, nominal_contacts
                else:
                    factor_states, factor_contacts = rollout_enriched(env, shape, initial_state, env_seed, commands, factor)
                for name in REPRESENTATIONS:
                    storage[name].append(inputs[name])
                targets.append(residual_target(factor_states[:, :7], nominal_states[:, :7]))
                contexts.append(float(factor))
                segment_ids.append(int(segment_id))
                variant_ids.append(int(variant_id))
                true_markov.append(factor_states)
                true_contacts.append(factor_contacts)
        print(f"GENERATE {split_name} {ordinal + 1}/{len(indices)} segment={segment_id}", flush=True)
    payload = {f"inputs_{name}": np.asarray(storage[name], dtype=np.float32) for name in REPRESENTATIONS}
    payload.update(
        targets=np.asarray(targets, dtype=np.float32),
        contexts=np.asarray(contexts, dtype=np.float32),
        segment_ids=np.asarray(segment_ids, dtype=np.int32),
        variant_ids=np.asarray(variant_ids, dtype=np.int16),
        true_markov=np.asarray(true_markov, dtype=np.float32),
        true_contacts=np.asarray(true_contacts, dtype=np.float32),
        identity_max_abs=np.asarray(identity_max_abs, dtype=np.float64),
    )
    return payload


def deterministic_repeat_audit(env, segment, segment_id, design):
    env_seed = int(design["env_seed_base"]) + int(segment_id)
    shape, initial_state, commands, _ = prepare_waypoint_physics(env, segment, env_seed)
    first_state, first_contact = rollout_enriched(env, shape, initial_state, env_seed, commands, -25.0)
    second_state, second_contact = rollout_enriched(env, shape, initial_state, env_seed, commands, -25.0)
    return {
        "state_sha256_first": array_sha256(first_state),
        "state_sha256_second": array_sha256(second_state),
        "contact_sha256_first": array_sha256(first_contact),
        "contact_sha256_second": array_sha256(second_contact),
        "state_equal": bool(np.array_equal(first_state, second_state)),
        "contact_equal": bool(np.array_equal(first_contact, second_contact)),
    }


def trajectory_errors(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, ACTION_COUNT, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, ACTION_COUNT, 3)
    position = np.linalg.norm(prediction[:, :, :2] - target[:, :, :2], axis=2)
    angle = np.abs(signed_angle_delta(prediction[:, :, 2] * ANGLE_SCALE, target[:, :, 2] * ANGLE_SCALE)) / ANGLE_SCALE
    return (position + angle).mean(axis=1)


def standardizer(x):
    mean = np.asarray(x, dtype=np.float64).mean(axis=0)
    scale = np.asarray(x, dtype=np.float64).std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def feature_map(x, contexts, mean, scale):
    z = (np.asarray(x, dtype=np.float64) - mean) / scale
    c = np.asarray(contexts, dtype=np.float64).reshape(-1, 1) / 30.0
    return np.concatenate([c, np.square(c), c * z, np.square(c) * z], axis=1)


def ridge_fit(phi, target, alpha):
    phi = np.asarray(phi, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if phi.shape[1] > phi.shape[0]:
        gram = phi @ phi.T
        gram.flat[:: len(gram) + 1] += float(alpha)
        return phi.T @ np.linalg.solve(gram, target)
    gram = phi.T @ phi
    gram.flat[:: len(gram) + 1] += float(alpha)
    return np.linalg.solve(gram, phi.T @ target)


def segment_means(values, segment_ids):
    values = np.asarray(values, dtype=np.float64)
    segment_ids = np.asarray(segment_ids)
    unique = np.unique(segment_ids)
    return unique, np.asarray([values[segment_ids == segment].mean() for segment in unique], dtype=np.float64)


def choose_alpha(x, target, contexts, segment_ids, alphas, folds, base_prediction=None):
    unique = np.unique(segment_ids)
    effective_folds = min(int(folds), len(unique))
    if effective_folds < 2:
        raise ValueError("at least two train segments are required for grouped CV")
    fold_by_segment = {int(segment): index % effective_folds for index, segment in enumerate(unique)}
    base = np.zeros_like(target) if base_prediction is None else np.asarray(base_prediction)
    rows = []
    for alpha in alphas:
        fold_scores = []
        for fold in range(effective_folds):
            valid = np.asarray([fold_by_segment[int(segment)] == fold for segment in segment_ids])
            train = ~valid
            mean, scale = standardizer(x[train])
            beta = ridge_fit(feature_map(x[train], contexts[train], mean, scale), target[train] - base[train], alpha)
            prediction = base[valid] + feature_map(x[valid], contexts[valid], mean, scale) @ beta
            errors = trajectory_errors(prediction, target[valid])
            _, grouped = segment_means(errors, segment_ids[valid])
            fold_scores.append(float(grouped.mean()))
        rows.append({"alpha": float(alpha), "effective_folds": effective_folds, "fold_scores": fold_scores, "mean_segment_pose_error": float(np.mean(fold_scores))})
    selected = min(rows, key=lambda row: (row["mean_segment_pose_error"], row["alpha"]))
    return float(selected["alpha"]), rows


def fit_and_predict(train, evaluation, name, design, x_train=None, x_eval=None, train_base=None, eval_base=None):
    if x_train is None:
        x_train = train[f"inputs_{name}"]
    if x_eval is None:
        x_eval = evaluation[f"inputs_{name}"]
    alpha, cv = choose_alpha(
        x_train,
        train["targets"],
        train["contexts"],
        train["segment_ids"],
        design["ridge_alphas"],
        design["group_cv_folds"],
        base_prediction=train_base,
    )
    train_base = np.zeros_like(train["targets"]) if train_base is None else np.asarray(train_base)
    eval_base = np.zeros_like(evaluation["targets"]) if eval_base is None else np.asarray(eval_base)
    mean, scale = standardizer(x_train)
    beta = ridge_fit(feature_map(x_train, train["contexts"], mean, scale), train["targets"] - train_base, alpha)
    prediction = eval_base + feature_map(x_eval, evaluation["contexts"], mean, scale) @ beta
    zero = feature_map(x_eval[:3], np.zeros(3), mean, scale) @ beta
    return prediction.astype(np.float32), {
        "selected_alpha": alpha,
        "cv": cv,
        "input_dim": int(x_train.shape[1]),
        "mapped_dim": int(beta.shape[0]),
        "zero_context_max_abs": float(np.max(np.abs(zero))),
        "mean_sha256": array_sha256(mean),
        "scale_sha256": array_sha256(scale),
        "beta_sha256": array_sha256(beta),
    }


def bootstrap_delta(first, second, segment_ids, seed, resamples):
    _, first_grouped = segment_means(first, segment_ids)
    _, second_grouped = segment_means(second, segment_ids)
    delta = first_grouped - second_grouped
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, len(delta), size=(int(resamples), len(delta)))
    means = delta[indexes].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "positive": int(np.sum(delta > 1e-12)),
        "tie": int(np.sum(np.abs(delta) <= 1e-12)),
        "negative": int(np.sum(delta < -1e-12)),
        "n_segments": int(len(delta)),
    }


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(first, second):
    first_rank, second_rank = rankdata(first), rankdata(second)
    if first_rank.std() == 0 or second_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def summarize_predictions(evaluation, predictions, model_details, design):
    target = evaluation["targets"]
    errors = {name: trajectory_errors(value, target) for name, value in predictions.items()}
    summary = {"models": {}, "comparisons": {}, "by_factor": {}, "posthoc_contact_strata": {}, "v1_step_correlations": {}}
    for name, value in predictions.items():
        _, grouped = segment_means(errors[name], evaluation["segment_ids"])
        summary["models"][name] = {
            "mean_trajectory_pose_error": float(errors[name].mean()),
            "mean_segment_pose_error": float(grouped.mean()),
            "normalized_residual_mse": float(np.square(np.asarray(value) - target).mean()),
            **model_details.get(name, {}),
        }
    comparisons = (
        ("R0_legacy", "R1_markov"),
        ("R1_markov", "R2_nominal_agent_block_contact"),
        ("R0_legacy", "R2_nominal_agent_block_contact"),
        ("v1_film", "R2_nominal_agent_block_contact"),
        ("v1_film", "v1_plus_C1_markov"),
        ("v1_plus_C1_markov", "v1_plus_C2_markov_contact"),
        ("v1_film", "v1_plus_C2_markov_contact"),
    )
    for first, second in comparisons:
        summary["comparisons"][f"{first}_minus_{second}"] = bootstrap_delta(
            errors[first], errors[second], evaluation["segment_ids"], design["bootstrap_seed"], design["bootstrap_resamples"]
        )
    for factor in design["eval_factors_cog_x"]:
        mask = evaluation["contexts"] == float(factor)
        summary["by_factor"][str(float(factor))] = {name: float(value[mask].mean()) for name, value in errors.items()}

    contact = evaluation["true_contacts"][:, :, CONTACT_KINDS.index("agent_block"), :]
    point_count = contact[:, :, CONTACT_FEATURES.index("point_count")].sum(axis=1)
    impulse = contact[:, :, CONTACT_FEATURES.index("impulse_norm_sum")].sum(axis=1)
    contacted = point_count > 0
    positive_impulse = impulse[contacted]
    high_threshold = float(np.quantile(positive_impulse, 0.75)) if len(positive_impulse) else math.inf
    strata = {
        "no_agent_block_contact": ~contacted,
        "agent_block_contact": contacted,
        "high_true_impulse_q75": contacted & (impulse >= high_threshold),
    }
    summary["posthoc_contact_strata"]["high_impulse_threshold"] = high_threshold
    for stratum, mask in strata.items():
        summary["posthoc_contact_strata"][stratum] = {
            "n_trajectories": int(mask.sum()),
            **{name: float(value[mask].mean()) if mask.any() else None for name, value in errors.items()},
        }

    v1_step = np.asarray(predictions["v1_film"]).reshape(-1, ACTION_COUNT, 3)
    target_step = target.reshape(-1, ACTION_COUNT, 3)
    step_error = np.linalg.norm(v1_step[:, :, :2] - target_step[:, :, :2], axis=2)
    step_error += np.abs(signed_angle_delta(v1_step[:, :, 2] * ANGLE_SCALE, target_step[:, :, 2] * ANGLE_SCALE)) / ANGLE_SCALE
    true_states = evaluation["true_markov"]
    block_speed = np.linalg.norm(true_states[:, :-1, 7:9], axis=2)
    angular_speed = np.abs(true_states[:, :-1, 9])
    step_impulse = contact[:, :, CONTACT_FEATURES.index("impulse_norm_sum")]
    summary["v1_step_correlations"] = {
        "descriptive_transition_count": int(step_error.size),
        "spearman_error_block_speed": spearman(step_error.ravel(), block_speed.ravel()),
        "spearman_error_abs_block_angular_speed": spearman(step_error.ravel(), angular_speed.ravel()),
        "spearman_error_true_contact_impulse": spearman(step_error.ravel(), step_impulse.ravel()),
        "warning": "transitions are not independent; descriptive only",
    }
    return summary, errors


def validate_design(design):
    train = set(map(int, design["train_segment_indices"]))
    evaluation = set(map(int, design["eval_segment_indices"]))
    formal = set(map(int, design["forbidden_formal_segment_indices"]))
    checks = {
        "contract_id": design["contract_id"] == CONTRACT_ID,
        "train_eval_disjoint": train.isdisjoint(evaluation),
        "formal_not_used": formal.isdisjoint(train | evaluation),
        "four_variants": int(design["variants_per_segment"]) == 4 and len(design["action_noise_sigmas"]) == 3,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid frozen design: {checks}")
    return checks


def generate_data(args, design, output_dir):
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    env = make_env()
    train_indices = design["train_segment_indices"][: args.limit_train]
    eval_indices = design["eval_segment_indices"][: args.limit_eval]
    repeat = deterministic_repeat_audit(env, segments[int(eval_indices[0])], int(eval_indices[0]), design)
    train = generate_split(env, segments, train_indices, design["train_factors_cog_x"], design, design["variant_seed"], "train")
    evaluation = generate_split(env, segments, eval_indices, design["eval_factors_cog_x"], design, design["variant_seed"] + 1, "eval")
    np.savez_compressed(output_dir / "train_data.npz", **train)
    np.savez_compressed(output_dir / "eval_data.npz", **evaluation)
    data_manifest = {
        "train_trajectories": int(len(train["targets"])),
        "eval_trajectories": int(len(evaluation["targets"])),
        "train_segments": list(map(int, train_indices)),
        "eval_segments": list(map(int, eval_indices)),
        "train_identity_max_abs": float(train["identity_max_abs"]),
        "eval_identity_max_abs": float(evaluation["identity_max_abs"]),
        "deterministic_repeat": repeat,
        "arrays": {
            "train_targets": array_sha256(train["targets"]),
            "eval_targets": array_sha256(evaluation["targets"]),
            "train_r2": array_sha256(train["inputs_R2_nominal_agent_block_contact"]),
            "eval_r2": array_sha256(evaluation["inputs_R2_nominal_agent_block_contact"]),
        },
    }
    dump_json(output_dir / "data_manifest.json", data_manifest)
    return data_manifest


def run_audit(args, design, output_dir, device):
    train = dict(np.load(output_dir / "train_data.npz"))
    evaluation = dict(np.load(output_dir / "eval_data.npz"))
    predictions, model_details = {}, {}
    model, checkpoint = load_model(args.v1_checkpoint, device)
    with torch.no_grad():
        train_x = torch.from_numpy(train["inputs_R0_legacy"]).to(device)
        train_c = torch.from_numpy(train["contexts"]).to(device)
        eval_x = torch.from_numpy(evaluation["inputs_R0_legacy"]).to(device)
        eval_c = torch.from_numpy(evaluation["contexts"]).to(device)
        v1_train = model(train_x, train_c).cpu().numpy()
        predictions["v1_film"] = model(eval_x, eval_c).cpu().numpy()
    model_details["v1_film"] = {
        "checkpoint_best_step": int(checkpoint["best_step"]),
        "checkpoint_parameter_count": int(checkpoint["parameter_count"]),
        "checkpoint_sha256": sha256(args.v1_checkpoint),
    }
    for name in REPRESENTATIONS:
        predictions[name], model_details[name] = fit_and_predict(train, evaluation, name, design)
        print(f"FIT {name} alpha={model_details[name]['selected_alpha']}", flush=True)
    correction_inputs = {
        "v1_plus_C1_markov": (
            train["inputs_R1_markov"][:, 108:],
            evaluation["inputs_R1_markov"][:, 108:],
        ),
        "v1_plus_C2_markov_contact": (
            train["inputs_R2_nominal_agent_block_contact"][:, 108:],
            evaluation["inputs_R2_nominal_agent_block_contact"][:, 108:],
        ),
    }
    for name in CORRECTIONS:
        x_train, x_eval = correction_inputs[name]
        predictions[name], model_details[name] = fit_and_predict(
            train,
            evaluation,
            name,
            design,
            x_train=x_train,
            x_eval=x_eval,
            train_base=v1_train,
            eval_base=predictions["v1_film"],
        )
        print(f"FIT {name} alpha={model_details[name]['selected_alpha']}", flush=True)
    summary, errors = summarize_predictions(evaluation, predictions, model_details, design)
    summary.update(
        contract_id=CONTRACT_ID,
        nature="development representation audit; not a formal closed-loop result",
        structural_checks={
            **validate_design(design),
            "full_frozen_train_segments": len(np.unique(train["segment_ids"])) == len(design["train_segment_indices"]),
            "full_frozen_eval_segments": len(np.unique(evaluation["segment_ids"])) == len(design["eval_segment_indices"]),
            "identity": max(float(train["identity_max_abs"]), float(evaluation["identity_max_abs"])) <= 1e-6,
            "zero_context_identity": all(model_details[name]["zero_context_max_abs"] == 0.0 for name in REPRESENTATIONS + CORRECTIONS),
        },
    )
    summary["valid"] = all(summary["structural_checks"].values())
    dump_json(output_dir / "runner_summary.json", summary)
    np.savez_compressed(output_dir / "predictions_and_errors.npz", **{f"prediction_{key}": value for key, value in predictions.items()}, **{f"error_{key}": value for key, value in errors.items()})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "generate", "audit", "run"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_markov_contact_audit_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_markov_contact_audit_contract_zh.md"))
    parser.add_argument("--v1-checkpoint", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_predictor/model_best.pt"))
    parser.add_argument("--source-snapshot", type=Path, default=Path("repro_outputs/persistent_context_v2_p3_source_snapshot_v1/source_snapshot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_markov_contact_audit_v1"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allocation-rationale", default="single idle GPU for frozen-v1 inference; simulator and ridge remain CPU-bound")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    args = parser.parse_args()
    if sha256(args.design) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen design hash mismatch")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    checks = validate_design(design)
    if args.limit_train is None:
        args.limit_train = len(design["train_segment_indices"])
    if args.limit_eval is None:
        args.limit_eval = len(design["eval_segment_indices"])
    if args.mode == "inspect":
        print(json.dumps({"contract_id": CONTRACT_ID, "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract), "checks": checks}, indent=2))
        return
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.source_snapshot.is_file():
        raise FileNotFoundError(f"missing source snapshot: {args.source_snapshot}")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        device = torch.device(args.device)
        manifest = {
            "contract_id": CONTRACT_ID,
            "command": " ".join(sys.argv),
            "git_revision": git_revision(),
            "design_sha256": sha256(args.design),
            "contract_sha256": sha256(args.contract),
            "source_snapshot": str(args.source_snapshot),
            "source_snapshot_sha256": sha256(args.source_snapshot),
            "data_sha256": sha256(args.data),
            "v1_checkpoint_sha256": sha256(args.v1_checkpoint),
            "started_unix": time.time(),
            "resource_start": resource_snapshot(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "allocation_rationale": args.allocation_rationale,
            "dirty_status": __import__("subprocess").run(["git", "status", "--short"], capture_output=True, text=True).stdout.splitlines(),
        }
        dump_json(manifest_path, manifest)
    seed_all(1_310_200)
    if args.mode in ("generate", "run"):
        if (output_dir / "train_data.npz").exists() or (output_dir / "eval_data.npz").exists():
            raise RuntimeError("output data already exists; use a new output directory")
        generate_data(args, design, output_dir)
    summary = None
    if args.mode in ("audit", "run"):
        summary = run_audit(args, design, output_dir, torch.device(args.device))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["finished_unix"] = time.time()
        manifest["resource_end"] = resource_snapshot(torch.device(args.device))
        manifest["exit_status"] = 0
        dump_json(manifest_path, manifest)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
