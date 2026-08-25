"""Independent raw-artifact audit for PushObj rotation Stage 0.

This file deliberately does not import the runner.  It recomputes pose costs,
intervention consistency, paired statistics, bootstrap CI, and descriptive evidence
directly from the append-only state/action rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_DESIGN_ID = "persistent-context-v2-pushobj-tool-rotation-stage0-v1"
EXPECTED_DESIGN_SHA256 = "a1c1f077890d2ec591871eab86e0c26cab263557840d2ce8871f69f65a8aa299"
EXPECTED_FACTORS = (-22.5, -7.5, 7.5, 22.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wrap_angle(values, target):
    difference = np.abs(np.asarray(values, dtype=np.float64) - float(target)) % (2 * np.pi)
    return np.minimum(difference, 2 * np.pi - difference)


def raw_pose_auc(states, goal, window):
    states = np.asarray(states, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    post = states[1 : int(window) + 1]
    if len(post) != int(window):
        raise AssertionError(f"expected {window} post-action states, found {len(post)}")
    position = np.linalg.norm(post[:, 2:4] - goal[None, 2:4], axis=1)
    angle = wrap_angle(post[:, 4], goal[4])
    return float(np.mean(position / 20.0 + angle / (np.pi / 9.0)))


def rotate(actions, degrees):
    theta = math.radians(float(degrees))
    matrix = np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    return np.asarray(actions, dtype=np.float64) @ matrix.T


def bootstrap(deltas, seed=6401, resamples=20000):
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    means = deltas[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def audit(raw_path: Path, manifest_path: Path, design_path: Path):
    design_hash = sha256(design_path)
    if design_hash != EXPECTED_DESIGN_SHA256:
        raise AssertionError(f"design hash mismatch: {design_hash}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["design_id"] != EXPECTED_DESIGN_ID:
        raise AssertionError("manifest design id mismatch")
    if manifest["design_sha256"] != design_hash:
        raise AssertionError("manifest design hash mismatch")

    rows = [row for row in read_rows(raw_path) if row.get("record_type") == "paired_scenario"]
    keys = [(row["candidate"], row["factor_deg"], row["segment_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate paired scenario keys")
    candidates = {row["candidate"] for row in rows}
    windows = {int(row["window"]) for row in rows}
    if len(candidates) != 1 or len(windows) != 1:
        raise AssertionError("raw file mixes candidate or window")
    candidate = next(iter(candidates))
    window = next(iter(windows))
    expected_window = 25 if candidate == "A_released" else 10
    if window != expected_window:
        raise AssertionError(f"candidate/window mismatch: {candidate}/{window}")

    factors = {factor: 0 for factor in EXPECTED_FACTORS}
    prior_costs = []
    oracle_costs = []
    plan_l2 = []
    metric_disagreement = []
    effective_action_disagreement = []
    initial_state_disagreement = []
    for row in rows:
        if row["design_id"] != EXPECTED_DESIGN_ID:
            raise AssertionError("row design id mismatch")
        factor = float(row["factor_deg"])
        if factor not in factors:
            raise AssertionError(f"unexpected factor {factor}")
        factors[factor] += 1
        goal = row["goal_state"]
        prior = raw_pose_auc(row["prior"]["states"], goal, window)
        oracle = raw_pose_auc(row["oracle"]["states"], goal, window)
        prior_costs.append(prior)
        oracle_costs.append(oracle)
        metric_name = f"pose_auc{window}"
        metric_disagreement.extend(
            [
                abs(prior - float(row["prior"]["metrics"][metric_name])),
                abs(oracle - float(row["oracle"]["metrics"][metric_name])),
            ]
        )
        for arm in ("prior", "oracle"):
            expected_effective = rotate(row[arm]["commands"], factor)
            observed_effective = np.asarray(row[arm]["effective_actions"], dtype=np.float64)
            effective_action_disagreement.append(
                float(np.max(np.abs(expected_effective - observed_effective)))
            )
            observed_states = np.asarray(row[arm]["states"], dtype=np.float64)
            initial_state_disagreement.append(
                float(np.max(np.abs(observed_states[0] - np.asarray(row["initial_state"]))))
            )
        recomputed_l2 = float(
            np.linalg.norm(
                np.asarray(row["prior"]["commands"])
                - np.asarray(row["oracle"]["commands"])
            )
        )
        plan_l2.append(recomputed_l2)
        if abs(recomputed_l2 - float(row["plan_command_l2"])) > 1e-6:
            raise AssertionError("stored plan L2 mismatch")

    prior_costs = np.asarray(prior_costs)
    oracle_costs = np.asarray(oracle_costs)
    deltas = prior_costs - oracle_costs
    ci = bootstrap(deltas)
    identity = manifest.get("identity_audit", {})
    identity_max = max(
        float(identity.get("action_max_abs", math.inf)),
        float(identity.get("rollout_max_abs", math.inf)),
    )
    plan_changed = float(np.mean(np.asarray(plan_l2) > 1e-6)) if rows else 0.0
    relative = float(deltas.mean() / prior_costs.mean()) if rows else -math.inf
    direction = float(np.mean(deltas > 0.0)) if rows else 0.0
    validity_audits = {
        "complete_32_pairs": len(rows) == 32 and all(value == 8 for value in factors.values()),
        "identity": identity_max <= 1e-6,
        "intervention_reached_planner": plan_changed > 0.0,
        "raw_metric_consistency": max(metric_disagreement, default=math.inf) <= 1e-8,
        "effective_action_consistency": max(effective_action_disagreement, default=math.inf)
        <= 1e-6,
        "paired_initial_state_consistency": max(initial_state_disagreement, default=math.inf)
        <= 1e-6,
    }
    return {
        "audit": "independent_raw_recomputation_v1",
        "candidate": candidate,
        "window": window,
        "n_pairs": len(rows),
        "factor_counts": {str(key): value for key, value in factors.items()},
        "prior_mean": float(prior_costs.mean()) if rows else None,
        "oracle_mean": float(oracle_costs.mean()) if rows else None,
        "mean_delta": float(deltas.mean()) if rows else None,
        "relative_improvement": relative,
        "bootstrap_ci95_delta": ci,
        "direction_fraction": direction,
        "plan_changed_fraction": plan_changed,
        "max_metric_disagreement": max(metric_disagreement, default=None),
        "max_effective_action_disagreement": max(effective_action_disagreement, default=None),
        "max_initial_state_disagreement": max(initial_state_disagreement, default=None),
        "identity_max_abs": identity_max,
        "validity_audits": validity_audits,
        "valid": all(validity_audits.values()),
        "descriptive_assessment": {
            "mean_effect_direction": "positive" if deltas.mean() > 0 else "negative" if deltas.mean() < 0 else "zero",
            "ci_relation_to_zero": "above_zero" if ci[0] > 0 else "below_zero" if ci[1] < 0 else "includes_zero",
            "majority_of_pairs_positive": bool(direction > 0.5),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("docs/research/persistent_context_v2_pushobj_rotation_stage0_design.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.raw, args.manifest, args.design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
