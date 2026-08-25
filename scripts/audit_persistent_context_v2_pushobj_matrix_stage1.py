#!/usr/bin/env python3
"""Independent raw audit for Bayesian matrix history Stage 1."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.persistent_context_v2.pushobj_matrix_stage0 import (
    POPULATION_PRIOR_MATRIX,
    apply_action_matrix,
    array_sha256,
)
from research.persistent_context_v2.pushobj_matrix_stage1 import (
    CONDITIONS,
    CONTRACT_ID,
    EXPECTED_DESIGN_SHA256,
    N_EPISODES,
    N_SEQUENCES,
    POLICIES,
    SEGMENT_INDICES,
    evidence_lookup,
    factor_index_for,
    history_payload,
    infer_matrix_observations,
    observations_sha256,
    read_jsonl,
    scenario,
    sha256,
    summarize,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import deadline_success
from research.persistent_context_v2.pushobj_rotation_stage0 import pose_metrics


def close(left, right):
    return bool(np.isclose(float(left), float(right), rtol=1e-8, atol=1e-9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_matrix_stage1"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    runner = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    failures = {key: [] for key in ("manifest", "count", "scenario", "evidence", "metric", "matrix", "hash", "pairing", "posterior", "cross_condition")}
    if manifest.get("contract_id") != CONTRACT_ID or manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("contract_or_design")
    design_path = Path(manifest.get("design_path", ""))
    if not design_path.exists() or sha256(design_path) != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("design_file")
    if max(manifest.get("identity_audit", {}).values(), default=np.inf) > 1e-6:
        failures["manifest"].append("identity")
    evidence_by_condition = {}
    evaluations_by_condition = {}
    for condition in CONDITIONS:
        raw_path = args.output_dir / f"{condition}_raw.jsonl"
        rows = read_jsonl(raw_path)
        evidence_rows = [row for row in rows if row.get("record_type") == "evidence_episode"]
        evaluation_rows = [row for row in rows if row.get("record_type") == "evaluation_episode"]
        evidence_by_condition[condition] = evidence_lookup(raw_path)
        evaluations_by_condition[condition] = {(int(row["sequence_id"]), int(row["episode_index"])): row for row in evaluation_rows}
        if len(evidence_rows) != N_SEQUENCES * N_EPISODES or len(evaluation_rows) != N_SEQUENCES * N_EPISODES:
            failures["count"].append(condition)
        for row in evidence_rows:
            sequence_id, episode_index = int(row["sequence_id"]), int(row["episode_index"])
            expected_meta = scenario(condition, sequence_id, episode_index)
            if row["segment_index"] != SEGMENT_INDICES[sequence_id * N_EPISODES + episode_index] or int(row["factor_index"]) != factor_index_for(condition, sequence_id, episode_index):
                failures["scenario"].append(f"{condition}:{sequence_id}:{episode_index}:meta")
            if not np.allclose(row["true_matrix"], expected_meta["true_matrix"], rtol=0, atol=1e-7):
                failures["scenario"].append(f"{condition}:{sequence_id}:{episode_index}:factor")
            commands = np.asarray(row["commands"], dtype=np.float32)
            states = np.asarray(row["states"], dtype=np.float32)
            effective = np.asarray(row["effective_actions"], dtype=np.float32)
            expected_effective = apply_action_matrix(commands, np.asarray(row["true_matrix"], dtype=np.float32)).astype(np.float32)
            if not np.array_equal(effective, expected_effective):
                failures["matrix"].append(f"{condition}:{sequence_id}:{episode_index}:evidence")
            observations, accepted = infer_matrix_observations(commands, states)
            if not np.allclose(observations, row["matrix_observations"], rtol=0, atol=1e-12) or accepted != row["accepted_indices"] or observations_sha256(observations) != row["observation_sha256"]:
                failures["evidence"].append(f"{condition}:{sequence_id}:{episode_index}")
            if row["command_sha256"] != array_sha256(commands) or row["state_sha256"] != array_sha256(states):
                failures["hash"].append(f"{condition}:{sequence_id}:{episode_index}:evidence")
        for row in evaluation_rows:
            sequence_id, episode_index = int(row["sequence_id"]), int(row["episode_index"])
            for policy in POLICIES:
                payload = row["policies"][policy]
                metrics = pose_metrics(np.asarray(payload["states"]), np.asarray(row["goal_state"]), 10)
                if any(not close(metrics[key], payload["metrics"][key]) for key in metrics) or bool(payload["deadline_success"]) != deadline_success(np.asarray(payload["states"]), np.asarray(row["goal_state"])):
                    failures["metric"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}")
                commands = np.asarray(payload["commands"], dtype=np.float32)
                effective = np.asarray(payload["effective_actions"], dtype=np.float32)
                states = np.asarray(payload["states"], dtype=np.float32)
                expected_effective = apply_action_matrix(commands, np.asarray(row["true_matrix"], dtype=np.float32)).astype(np.float32)
                if not np.array_equal(effective, expected_effective):
                    failures["matrix"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}")
                if payload["command_sha256"] != array_sha256(commands) or payload["effective_action_sha256"] != array_sha256(effective) or payload["state_sha256"] != array_sha256(states):
                    failures["hash"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}")
                if len(payload["planner"]["trace"]) != 10 or len(states) != 11:
                    failures["pairing"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}:budget")
            if not np.allclose(row["policies"]["population_prior"]["context_matrix"], POPULATION_PRIOR_MATRIX, rtol=0, atol=1e-12) or not np.allclose(row["policies"]["current_only"]["context_matrix"], POPULATION_PRIOR_MATRIX, rtol=0, atol=1e-12):
                failures["pairing"].append(f"{condition}:{sequence_id}:{episode_index}:prior")
            if not np.allclose(row["policies"]["true_factor_oracle"]["context_matrix"], row["true_matrix"], rtol=0, atol=1e-7):
                failures["pairing"].append(f"{condition}:{sequence_id}:{episode_index}:oracle")
            if episode_index == 1:
                for policy in ("correct_history", "shuffled_history", "wrong_sequence_history"):
                    expected = history_payload(evidence_by_condition[condition], policy, sequence_id, episode_index, N_SEQUENCES)
                    observed = row["policies"][policy]
                    if not np.allclose(expected["context_matrix"], observed["context_matrix"], rtol=0, atol=1e-12) or expected["donors"] != observed["donors"] or expected["history_observation_sha256"] != observed["history_observation_sha256"] or expected["history_observation_count"] != observed["history_observation_count"]:
                        failures["posterior"].append(f"{condition}:{sequence_id}:{policy}")
    for sequence_id in range(N_SEQUENCES):
        for episode_index in range(N_EPISODES):
            left = evidence_by_condition["persistent"][(sequence_id, episode_index)]
            right = evidence_by_condition["no_persistence"][(sequence_id, episode_index)]
            if episode_index == 0 and any(left[key] != right[key] for key in ("command_sha256", "state_sha256", "observation_sha256")):
                failures["cross_condition"].append(f"{sequence_id}:E1_evidence")
        left_eval = evaluations_by_condition["persistent"][(sequence_id, 0)]["policies"]["current_only"]
        right_eval = evaluations_by_condition["no_persistence"][(sequence_id, 0)]["policies"]["current_only"]
        if left_eval["command_sha256"] != right_eval["command_sha256"] or left_eval["state_sha256"] != right_eval["state_sha256"]:
            failures["cross_condition"].append(f"{sequence_id}:E1_eval")
    recomputed = summarize(args.output_dir, N_SEQUENCES)
    checks = {"raw_recomputes_runner_summary_exactly": recomputed == runner, "all_failure_counts_zero": all(not values for values in failures.values()), "runner_valid": bool(recomputed.get("valid"))}
    result = {"schema": "persistent-context-v2-pushobj-bayesian-matrix-stage1-audit-v1", "passed": all(checks.values()), "checks": checks, "failure_counts": {key: len(value) for key, value in failures.items()}, "failure_examples": {key: value[:20] for key, value in failures.items() if value}, "recomputed_summary": recomputed}
    (args.output_dir / "independent_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
