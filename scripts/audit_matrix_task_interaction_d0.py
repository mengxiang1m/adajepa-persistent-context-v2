#!/usr/bin/env python3
"""Independent CPU audit of the exploratory matrix task-interaction feature table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FACTOR_NAMES = (
    "intercept",
    "normalized_gain",
    "normalized_rotation",
    "normalized_gain_squared",
    "gain_rotation_interaction",
    "normalized_rotation_squared",
)
GEOMETRY_ACTION_NAMES = (
    "agent_to_block_distance",
    "block_to_goal_distance",
    "agent_block_goal_alignment",
    "absolute_block_goal_angle_error",
    "command_rms_disagreement",
    "first_action_cosine_distance",
    "log_mean_action_norm_ratio",
    "log_action_variation_ratio",
)
MODEL_NAMES = (
    "prior_preference_for_prior_action",
    "context_preference_for_context_action",
    "context_sensitivity_of_prior_action",
    "context_sensitivity_of_context_action",
)
EXPECTED_NAMES = FACTOR_NAMES + GEOMETRY_ACTION_NAMES + MODEL_NAMES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cosine(left, right) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def angle_error(left: float, right: float) -> float:
    delta = abs(float(left) - float(right)) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def replay_features(original: dict, feature_row: dict, design: dict) -> np.ndarray:
    z = np.asarray(original["posterior"]["mean_z"], dtype=np.float64)
    gain = float(np.linalg.norm(z))
    rotation = float(math.degrees(math.atan2(z[1], z[0])))
    g = (gain - float(design["gain_center"])) / float(design["gain_scale"])
    r = rotation / float(design["rotation_scale_degrees"])
    factor = np.asarray([1.0, g, r, g * g, g * r, r * r])
    initial = np.asarray(original["e2"]["initial_state"], dtype=np.float64)
    goal = np.asarray(original["e2"]["goal_state"], dtype=np.float64)
    population = np.asarray(original["e2"]["population"]["commands"], dtype=np.float64)
    context = np.asarray(original["e2"]["context"]["commands"], dtype=np.float64)
    difference = context - population
    population_norm = np.linalg.norm(population, axis=1)
    context_norm = np.linalg.norm(context, axis=1)
    population_variation = np.linalg.norm(np.diff(population, axis=0), axis=1).mean()
    context_variation = np.linalg.norm(np.diff(context, axis=0), axis=1).mean()
    agent_to_block = initial[2:4] - initial[0:2]
    block_to_goal = goal[2:4] - initial[2:4]
    geometry = np.asarray(
        [
            np.linalg.norm(agent_to_block),
            np.linalg.norm(block_to_goal),
            cosine(agent_to_block, block_to_goal),
            angle_error(initial[4], goal[4]),
            np.sqrt(np.mean(difference**2)),
            1.0 - cosine(population[0], context[0]),
            math.log((float(context_norm.mean()) + 1e-8) / (float(population_norm.mean()) + 1e-8)),
            math.log((float(context_variation) + 1e-8) / (float(population_variation) + 1e-8)),
        ]
    )
    scores = feature_row["model_scores"]
    pp, pc = float(scores["J_prior_a_prior"]), float(scores["J_prior_a_context"])
    cp, cc = float(scores["J_context_a_prior"]), float(scores["J_context_a_context"])
    model = np.asarray([pc - pp, cp - cc, cp - pp, cc - pc])
    return np.concatenate([factor, geometry, model])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate"))
    parser.add_argument("--d0-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_learned_gate_design.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.d0_dir / "manifest.json").read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    failures = []
    feature_path = args.d0_dir / "features.jsonl"
    if file_sha256(feature_path) != manifest["features_sha256"]:
        failures.append("feature_hash")
    if file_sha256(args.design) != manifest["design_sha256"]:
        failures.append("design_hash")
    if not manifest.get("model_state_unchanged") or not manifest.get("rng_unchanged"):
        failures.append("read_only_state")
    if manifest.get("initial_replay_max_abs", 1.0) > 1e-6 or manifest.get("goal_replay_max_abs", 1.0) > 1e-6:
        failures.append("scene_replay")
    feature_rows = read_jsonl(feature_path)
    by_key = {(row["split"], int(row["sequence_id"])): row for row in feature_rows}
    max_feature_error = 0.0
    max_outcome_error = 0.0
    count_failures = 0
    finite_failures = 0
    for split in ("train", "dev", "formal"):
        original_path = args.data_dir / split / "raw.jsonl"
        if file_sha256(original_path) != manifest["input_raw_sha256"][split]:
            failures.append(f"{split}_raw_hash")
        originals = [row for row in read_jsonl(original_path) if row.get("record_type") == "paired_sequence"]
        if len(originals) != int(manifest["counts"][split]):
            count_failures += 1
        for original in originals:
            key = (split, int(original["sequence_id"]))
            if key not in by_key:
                count_failures += 1
                continue
            feature = by_key[key]
            if tuple(feature["feature_names"]) != EXPECTED_NAMES:
                failures.append("feature_names")
            replayed = replay_features(original, feature, design)
            recorded = np.asarray(feature["features"], dtype=np.float64)
            if not np.all(np.isfinite(recorded)) or not all(np.isfinite(list(feature["model_scores"].values()))):
                finite_failures += 1
            max_feature_error = max(max_feature_error, float(np.max(np.abs(replayed - recorded))))
            outcome = float(original["e2"]["population"]["metrics"]["pose_auc10"]) - float(
                original["e2"]["context"]["metrics"]["pose_auc10"]
            )
            max_outcome_error = max(max_outcome_error, abs(outcome - float(feature["exploratory_outcome_delta"])))
    if len(by_key) != len(feature_rows) or count_failures:
        failures.append("counts_or_keys")
    if finite_failures:
        failures.append("non_finite")
    if max_feature_error > 1e-12:
        failures.append("feature_replay")
    if max_outcome_error > 1e-12:
        failures.append("outcome_replay")
    audit = {
        "valid": not failures,
        "failures": sorted(set(failures)),
        "evidence_level": "exploratory_not_new_formal",
        "feature_row_count": len(feature_rows),
        "unique_key_count": len(by_key),
        "count_failure_count": count_failures,
        "finite_failure_count": finite_failures,
        "feature_replay_max_abs": max_feature_error,
        "outcome_replay_max_abs": max_outcome_error,
        "features_sha256": file_sha256(feature_path),
        "source_snapshot_sha256": manifest.get("source_snapshot_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
