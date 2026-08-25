"""Exploratory, read-only feasibility analysis for matrix task-interaction gating.

This module deliberately reuses the old train/dev/formal outcomes only as D0
exploration. Its outputs are not a new formal result.
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

from research.persistent_context_v2.matrix_learned_gate import (
    feature_row as v1_factor_feature_row,
    fit_ridge,
    gate_outcome,
)
from research.persistent_context_v2.pushobj_matrix_stage0 import (
    ACTION_DIM,
    FRAMESKIP,
    MODEL_HORIZON,
    POPULATION_PRIOR_MATRIX,
    load_matrix_world_model,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import prepare_waypoint
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl,
    dump_json,
    make_env,
    make_preprocessor,
    obs_batch,
    read_jsonl,
    resource_snapshot,
    sha256,
)


D0_ID = "persistent-context-v2-matrix-task-interaction-d0-exploratory-v1"
FACTOR_FEATURES = (
    "intercept",
    "normalized_gain",
    "normalized_rotation",
    "normalized_gain_squared",
    "gain_rotation_interaction",
    "normalized_rotation_squared",
)
GEOMETRY_ACTION_FEATURES = (
    "agent_to_block_distance",
    "block_to_goal_distance",
    "agent_block_goal_alignment",
    "absolute_block_goal_angle_error",
    "command_rms_disagreement",
    "first_action_cosine_distance",
    "log_mean_action_norm_ratio",
    "log_action_variation_ratio",
)
MODEL_INTERACTION_FEATURES = (
    "prior_preference_for_prior_action",
    "context_preference_for_context_action",
    "context_sensitivity_of_prior_action",
    "context_sensitivity_of_context_action",
)
ALL_FEATURES = FACTOR_FEATURES + GEOMETRY_ACTION_FEATURES + MODEL_INTERACTION_FEATURES
FEATURE_SETS = {
    "F0_factor_only": len(FACTOR_FEATURES),
    "F1_geometry_action": len(FACTOR_FEATURES) + len(GEOMETRY_ACTION_FEATURES),
    "F2_model_interaction": len(ALL_FEATURES),
}
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_SEED = 1_203_001
BOOTSTRAP_RESAMPLES = 20_000


def wrapped_angle_error(left: float, right: float) -> float:
    delta = abs(float(left) - float(right)) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def geometry_action_features(row: dict) -> np.ndarray:
    """Compute only features available before E2 environment execution."""
    initial = np.asarray(row["e2"]["initial_state"], dtype=np.float64)
    goal = np.asarray(row["e2"]["goal_state"], dtype=np.float64)
    population_commands = np.asarray(row["e2"]["population"]["commands"], dtype=np.float64)
    context_commands = np.asarray(row["e2"]["context"]["commands"], dtype=np.float64)
    agent_to_block = initial[2:4] - initial[0:2]
    block_to_goal = goal[2:4] - initial[2:4]
    command_difference = context_commands - population_commands
    population_norm = np.linalg.norm(population_commands, axis=1)
    context_norm = np.linalg.norm(context_commands, axis=1)
    population_variation = np.linalg.norm(np.diff(population_commands, axis=0), axis=1).mean()
    context_variation = np.linalg.norm(np.diff(context_commands, axis=0), axis=1).mean()
    return np.asarray(
        [
            np.linalg.norm(agent_to_block),
            np.linalg.norm(block_to_goal),
            safe_cosine(agent_to_block, block_to_goal),
            wrapped_angle_error(initial[4], goal[4]),
            np.sqrt(np.mean(command_difference**2)),
            1.0 - safe_cosine(population_commands[0], context_commands[0]),
            math.log((float(context_norm.mean()) + 1e-8) / (float(population_norm.mean()) + 1e-8)),
            math.log((float(context_variation) + 1e-8) / (float(population_variation) + 1e-8)),
        ],
        dtype=np.float64,
    )


def model_interaction_features(scores: dict[str, float]) -> np.ndarray:
    pp = float(scores["J_prior_a_prior"])
    pc = float(scores["J_prior_a_context"])
    cp = float(scores["J_context_a_prior"])
    cc = float(scores["J_context_a_context"])
    return np.asarray([pc - pp, cp - cc, cp - pp, cc - pc], dtype=np.float64)


def predecision_feature_row(row: dict, design: dict, scores: dict[str, float]) -> np.ndarray:
    return np.concatenate(
        [v1_factor_feature_row(row, design), geometry_action_features(row), model_interaction_features(scores)]
    )


def fixed_action_goal_loss(wm, preprocessor, obs_0, obs_g, commands, context_matrix) -> float:
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
        goal_z = wm.encode_obs(transformed_g)
        prediction, _ = wm.rollout(transformed_0, actions)
        loss = objective(prediction, goal_z, step=0)
    return float(loss.reshape(-1)[0].item())


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state_digest() -> str:
    payload = pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()))
    return hashlib.sha256(payload).hexdigest()


def score_row(wm, preprocessor, env, segments, row: dict) -> tuple[dict[str, float], dict]:
    episode = row["e2"]
    initial, goal_obs, nominal_states, _ = prepare_waypoint(
        env, segments[int(episode["segment_index"])], int(episode["env_seed"])
    )
    start_obs, _ = env.prepare(int(episode["env_seed"]), initial)
    initial_error = float(np.max(np.abs(np.asarray(initial) - np.asarray(episode["initial_state"]))))
    goal_error = float(np.max(np.abs(np.asarray(nominal_states[-1]) - np.asarray(episode["goal_state"]))))
    if initial_error > 1e-6 or goal_error > 1e-6:
        raise RuntimeError(f"scene replay mismatch: initial={initial_error} goal={goal_error}")
    prior_commands = episode["population"]["commands"]
    context_commands = episode["context"]["commands"]
    context_matrix = np.asarray(row["posterior"]["mean_matrix"], dtype=np.float64)
    scores = {
        "J_prior_a_prior": fixed_action_goal_loss(
            wm, preprocessor, start_obs, goal_obs, prior_commands, POPULATION_PRIOR_MATRIX
        ),
        "J_prior_a_context": fixed_action_goal_loss(
            wm, preprocessor, start_obs, goal_obs, context_commands, POPULATION_PRIOR_MATRIX
        ),
        "J_context_a_prior": fixed_action_goal_loss(
            wm, preprocessor, start_obs, goal_obs, prior_commands, context_matrix
        ),
        "J_context_a_context": fixed_action_goal_loss(
            wm, preprocessor, start_obs, goal_obs, context_commands, context_matrix
        ),
    }
    return scores, {"initial_replay_max_abs": initial_error, "goal_replay_max_abs": goal_error}


def extract(args) -> dict:
    started = time.time()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.replace:
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design = json.loads(args.design.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index if device.index is not None else torch.cuda.current_device())
    resource_start = resource_snapshot(device)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    _, wrapper, _ = load_matrix_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    raw_output = args.output_dir / "features.jsonl"
    if raw_output.exists():
        raw_output.unlink()
    state_before = model_sha256(wrapper)
    rng_before = rng_state_digest()
    max_initial_error = 0.0
    max_goal_error = 0.0
    counts = {}
    for split in ("train", "dev", "formal"):
        input_path = args.data_dir / split / "raw.jsonl"
        rows = [row for row in read_jsonl(input_path) if row.get("record_type") == "paired_sequence"]
        rows.sort(key=lambda row: int(row["sequence_id"]))
        if args.limit_per_split:
            rows = rows[: int(args.limit_per_split)]
        counts[split] = len(rows)
        for row in rows:
            python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
            cuda_index = device.index if device.index is not None else torch.cuda.current_device()
            cuda_state = torch.cuda.get_rng_state(cuda_index) if device.type == "cuda" else None
            scores, replay = score_row(wrapper, preprocessor, env, segments, row)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, cuda_index)
            features = predecision_feature_row(row, design, scores)
            if not np.all(np.isfinite(features)):
                raise RuntimeError("non-finite feature")
            max_initial_error = max(max_initial_error, replay["initial_replay_max_abs"])
            max_goal_error = max(max_goal_error, replay["goal_replay_max_abs"])
            append_jsonl(
                raw_output,
                {
                    "record_type": "d0_exploratory_feature",
                    "d0_id": D0_ID,
                    "split": split,
                    "sequence_id": int(row["sequence_id"]),
                    "factor_index": int(row["factor_index"]),
                    "feature_names": ALL_FEATURES,
                    "features": features,
                    "model_scores": scores,
                    "posterior_covariance_trace": float(
                        np.trace(np.asarray(row["posterior"]["covariance"], dtype=np.float64))
                    ),
                    "exploratory_outcome_delta": float(
                        row["e2"]["population"]["metrics"]["pose_auc10"]
                        - row["e2"]["context"]["metrics"]["pose_auc10"]
                    ),
                    "population_pose_auc10": float(row["e2"]["population"]["metrics"]["pose_auc10"]),
                    "context_pose_auc10": float(row["e2"]["context"]["metrics"]["pose_auc10"]),
                },
            )
    state_after = model_sha256(wrapper)
    rng_after = rng_state_digest()
    resource_end = resource_snapshot(device)
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        resource_end["cuda_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(cuda_index))
    manifest = {
        "d0_id": D0_ID,
        "evidence_level": "exploratory_reuse_of_old_outcomes_not_new_formal",
        "command": " ".join(__import__("sys").argv),
        "device": str(device),
        "started_unix": started,
        "finished_unix": time.time(),
        "counts": counts,
        "feature_names": ALL_FEATURES,
        "design": str(args.design),
        "design_sha256": sha256(args.design),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "input_raw_sha256": {
            split: sha256(args.data_dir / split / "raw.jsonl") for split in ("train", "dev", "formal")
        },
        "features_sha256": sha256(raw_output),
        "source_snapshot": str(args.source_snapshot) if args.source_snapshot else None,
        "source_snapshot_sha256": sha256(args.source_snapshot) if args.source_snapshot else None,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "rng_digest_before": rng_before,
        "rng_digest_after": rng_after,
        "rng_unchanged": rng_before == rng_after,
        "initial_replay_max_abs": max_initial_error,
        "goal_replay_max_abs": max_goal_error,
        "resource_start": resource_start,
        "resource_end": resource_end,
    }
    if not manifest["model_state_unchanged"] or not manifest["rng_unchanged"]:
        raise RuntimeError("read-only extraction changed model or global RNG state")
    dump_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def load_d0_rows(path: Path) -> dict[str, list[dict]]:
    grouped = {split: [] for split in ("train", "dev", "formal")}
    for row in read_jsonl(path):
        if row.get("record_type") == "d0_exploratory_feature":
            grouped[row["split"]].append(row)
    for split in grouped:
        grouped[split].sort(key=lambda row: int(row["sequence_id"]))
    return grouped


def arrays(rows: list[dict], feature_count: int):
    x = np.asarray([row["features"][:feature_count] for row in rows], dtype=np.float64)
    population = np.asarray([row["population_pose_auc10"] for row in rows], dtype=np.float64)
    context = np.asarray([row["context_pose_auc10"] for row in rows], dtype=np.float64)
    return x, population, context


def normalize_extra_features(train_x: np.ndarray, *others: np.ndarray):
    mean = np.zeros(train_x.shape[1], dtype=np.float64)
    scale = np.ones(train_x.shape[1], dtype=np.float64)
    if train_x.shape[1] > len(FACTOR_FEATURES):
        extra = train_x[:, len(FACTOR_FEATURES) :]
        mean[len(FACTOR_FEATURES) :] = extra.mean(axis=0)
        values = extra.std(axis=0)
        scale[len(FACTOR_FEATURES) :] = np.where(values > 1e-12, values, 1.0)
    transformed = [(value - mean) / scale for value in (train_x, *others)]
    return transformed, mean, scale


def bootstrap_ci(values: np.ndarray, stream: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(stream))
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def policy_summary(population: np.ndarray, context: np.ndarray, decisions: np.ndarray, stream: int) -> dict:
    outcome = gate_outcome(population, context, decisions)
    delta = population - outcome
    return {
        "population_mean": float(population.mean()),
        "gate_mean": float(outcome.mean()),
        "mean_delta": float(delta.mean()),
        "relative_improvement": float(delta.mean() / population.mean()),
        "bootstrap_ci95_delta": bootstrap_ci(delta, stream),
        "selection_rate": float(np.mean(decisions)),
        "harm_fraction": float(np.mean(delta < -1e-12)),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "unit_deltas": delta.tolist(),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def within_factor_correlations(rows: list[dict]) -> dict[str, float | None]:
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    y = np.asarray([row["exploratory_outcome_delta"] for row in rows], dtype=np.float64)
    factors = np.asarray([row["factor_index"] for row in rows], dtype=np.int64)
    centered_x = x.copy()
    centered_y = y.copy()
    for factor in np.unique(factors):
        mask = factors == factor
        centered_x[mask] -= centered_x[mask].mean(axis=0)
        centered_y[mask] -= centered_y[mask].mean()
    return {name: correlation(centered_x[:, index], centered_y) for index, name in enumerate(ALL_FEATURES)}


def select_and_fit(train_rows: list[dict], dev_rows: list[dict], formal_rows: list[dict], feature_count: int):
    train_x, train_population, train_context = arrays(train_rows, feature_count)
    dev_x, dev_population, dev_context = arrays(dev_rows, feature_count)
    formal_x, formal_population, formal_context = arrays(formal_rows, feature_count)
    (train_x, dev_x, formal_x), mean, scale = normalize_extra_features(train_x, dev_x, formal_x)
    train_y = train_population - train_context
    dev_y = dev_population - dev_context
    candidates = []
    for alpha in RIDGE_ALPHAS:
        try:
            beta = fit_ridge(train_x, train_y, alpha)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(train_x.T @ train_x + alpha * np.diag([0.0] + [1.0] * (feature_count - 1))) @ train_x.T @ train_y
        prediction = dev_x @ beta
        decision = prediction > 0.0
        outcome = gate_outcome(dev_population, dev_context, decision)
        candidates.append({
            "alpha": float(alpha),
            "dev_mean_delta": float(np.mean(dev_population - outcome)),
            "dev_selection_rate": float(np.mean(decision)),
            "dev_prediction_mse": float(np.mean((prediction - dev_y) ** 2)),
        })
    selected = max(candidates, key=lambda row: (row["dev_mean_delta"], row["alpha"]))
    fit_x = np.concatenate([train_x, dev_x], axis=0)
    fit_y = np.concatenate([train_y, dev_y], axis=0)
    try:
        beta = fit_ridge(fit_x, fit_y, selected["alpha"])
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(fit_x.T @ fit_x + selected["alpha"] * np.diag([0.0] + [1.0] * (feature_count - 1))) @ fit_x.T @ fit_y
    prediction = formal_x @ beta
    decisions = prediction > 0.0
    return {
        "selected_alpha": selected["alpha"],
        "alpha_candidates": candidates,
        "beta": beta.tolist(),
        "extra_feature_mean": mean.tolist(),
        "extra_feature_scale": scale.tolist(),
        "formal_predictions": prediction.tolist(),
        "formal_decisions": decisions.tolist(),
        "formal_true_delta": (formal_population - formal_context).tolist(),
        "formal_prediction_correlation": correlation(prediction, formal_population - formal_context),
        "exploratory_formal_policy": policy_summary(formal_population, formal_context, decisions, 10 + feature_count),
    }


def analyze(args) -> dict:
    rows = load_d0_rows(args.features)
    if any(len(rows[split]) != 32 for split in rows):
        raise RuntimeError(f"D0 full analysis requires 32 rows per split: { {k: len(v) for k, v in rows.items()} }")
    models = {
        name: select_and_fit(rows["train"], rows["dev"], rows["formal"], count)
        for name, count in FEATURE_SETS.items()
    }
    f0 = models["F0_factor_only"]
    f0_outcome = np.asarray(f0["exploratory_formal_policy"]["unit_deltas"], dtype=np.float64)
    for index, name in enumerate(("F1_geometry_action", "F2_model_interaction")):
        current = models[name]
        current_outcome = np.asarray(current["exploratory_formal_policy"]["unit_deltas"], dtype=np.float64)
        paired_gain = current_outcome - f0_outcome
        current["vs_F0_factor_only"] = {
            "mean_additional_population_minus_cost_delta": float(paired_gain.mean()),
            "bootstrap_ci95": bootstrap_ci(paired_gain, 100 + index),
            "harm_fraction_difference": float(
                current["exploratory_formal_policy"]["harm_fraction"]
                - f0["exploratory_formal_policy"]["harm_fraction"]
            ),
            "decision_mismatch_count": int(
                np.sum(np.asarray(current["formal_decisions"]) != np.asarray(f0["formal_decisions"]))
            ),
        }
    all_rows = rows["train"] + rows["dev"] + rows["formal"]
    covariance = np.asarray([row["posterior_covariance_trace"] for row in all_rows], dtype=np.float64)
    split_rotations = {}
    for train_split, dev_split, test_split in (
        ("train", "dev", "formal"),
        ("dev", "formal", "train"),
        ("formal", "train", "dev"),
    ):
        key = f"train={train_split},dev={dev_split},test={test_split}"
        rotation_models = {
            name: select_and_fit(rows[train_split], rows[dev_split], rows[test_split], count)
            for name, count in FEATURE_SETS.items()
        }
        baseline = np.asarray(
            rotation_models["F0_factor_only"]["exploratory_formal_policy"]["unit_deltas"], dtype=np.float64
        )
        for index, name in enumerate(("F1_geometry_action", "F2_model_interaction")):
            current = np.asarray(rotation_models[name]["exploratory_formal_policy"]["unit_deltas"], dtype=np.float64)
            paired = current - baseline
            rotation_models[name]["vs_F0_factor_only"] = {
                "mean_additional_population_minus_cost_delta": float(paired.mean()),
                "bootstrap_ci95": bootstrap_ci(paired, 500 + 10 * len(split_rotations) + index),
                "harm_fraction_difference": float(
                    rotation_models[name]["exploratory_formal_policy"]["harm_fraction"]
                    - rotation_models["F0_factor_only"]["exploratory_formal_policy"]["harm_fraction"]
                ),
            }
        split_rotations[key] = rotation_models
    result = {
        "d0_id": D0_ID,
        "evidence_level": "exploratory_reuse_of_old_outcomes_not_new_formal",
        "warning": "These results may freeze a future design but cannot support a new scientific claim.",
        "analysis_source_snapshot": str(args.source_snapshot) if args.source_snapshot else None,
        "analysis_source_snapshot_sha256": sha256(args.source_snapshot) if args.source_snapshot else None,
        "counts": {split: len(value) for split, value in rows.items()},
        "feature_sets": {name: list(ALL_FEATURES[:count]) for name, count in FEATURE_SETS.items()},
        "models": models,
        "split_rotation_diagnostics": split_rotations,
        "within_factor_centered_correlations": within_factor_correlations(all_rows),
        "posterior_covariance_trace": {
            "min": float(covariance.min()),
            "max": float(covariance.max()),
            "std": float(covariance.std()),
        },
        "leakage_audit": {
            "features_are_predecision_allowlist": True,
            "forbidden_feature_names_present": sorted(
                set(ALL_FEATURES) & {"true_factor", "outcome", "pose_auc10", "success", "segment_id", "seed"}
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(args.output, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--data-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate"))
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    extract_parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_learned_gate_design.json"))
    extract_parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    extract_parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--limit-per-split", type=int)
    extract_parser.add_argument("--source-snapshot", type=Path)
    extract_parser.add_argument("--replace", action="store_true")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--features", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--source-snapshot", type=Path)
    args = parser.parse_args()
    if args.mode == "extract":
        extract(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
