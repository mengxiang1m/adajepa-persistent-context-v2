#!/usr/bin/env python3
"""Independent raw-artifact audit for PushObj rotation Stage 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.persistent_context_v2.pushobj_rotation_stage0 import pose_metrics, read_jsonl, sha256
from research.persistent_context_v2.pushobj_rotation_stage1 import (
    CONDITIONS,
    CONTRACT_ID,
    EXPECTED_DESIGN_SHA256,
    N_EPISODES,
    N_SEQUENCES,
    POLICIES,
    array_sha256,
    build_estimator,
    donor_sequence,
    factor_for,
    formal_segment_indices,
    summarize,
)


def close(a, b, atol=1e-10):
    return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=atol))


def audit(output_dir: Path) -> dict:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    runner = json.loads((output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    n_sequences = int(manifest["sequences"])
    episodes = int(manifest["episodes"])
    recomputed = summarize(output_dir, n_sequences, episodes)
    failures = {
        "manifest": [],
        "evidence": [],
        "scenario": [],
        "estimator": [],
        "donor": [],
        "metrics": [],
        "identity": [],
    }
    if manifest.get("contract_id") != CONTRACT_ID or manifest.get("mode") not in ("smoke", "formal"):
        failures["manifest"].append("contract_or_mode")
    if manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("design_hash")
    if sha256(Path(manifest["design_path"])) != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("design_file")
    if max(manifest.get("identity_audit", {}).values(), default=np.inf) > 1e-6:
        failures["identity"].append("wrapper_identity")
    expected_segments = formal_segment_indices().reshape(N_SEQUENCES, N_EPISODES)
    for condition in CONDITIONS:
        raw = read_jsonl(output_dir / f"{condition}_raw.jsonl")
        evidence_rows = [row for row in raw if row.get("record_type") == "evidence_episode"]
        evaluation_rows = [row for row in raw if row.get("record_type") == "evaluation_episode"]
        evidence = {
            (int(row["sequence_id"]), int(row["episode_index"])): row
            for row in evidence_rows
        }
        evaluations = {
            (int(row["sequence_id"]), int(row["episode_index"])): row
            for row in evaluation_rows
        }
        if len(evidence) != n_sequences * episodes:
            failures["evidence"].append(f"{condition}:count")
        if len(evaluations) != n_sequences * episodes:
            failures["scenario"].append(f"{condition}:count")
        for sequence_id in range(n_sequences):
            for episode_index in range(episodes):
                key = (sequence_id, episode_index)
                if key not in evidence or key not in evaluations:
                    continue
                ev = evidence[key]
                row = evaluations[key]
                expected_factor = factor_for(condition, sequence_id, episode_index)
                expected_segment = int(expected_segments[sequence_id, episode_index])
                for candidate, label in ((ev, "evidence"), (row, "evaluation")):
                    if (
                        float(candidate["factor_deg"]) != expected_factor
                        or int(candidate["segment_index"]) != expected_segment
                        or candidate.get("contract_id") != CONTRACT_ID
                    ):
                        failures["scenario"].append(f"{condition}:{sequence_id}:{episode_index}:{label}")
                expected_evidence_hash = array_sha256(np.asarray(ev["commands"], dtype=np.float32)) + ":" + array_sha256(np.asarray(ev["states"], dtype=np.float32))
                if ev.get("evidence_sha256") != expected_evidence_hash:
                    failures["evidence"].append(f"{condition}:{sequence_id}:{episode_index}:hash")
                for policy in ("correct_history", "shuffled_history", "wrong_sequence_history"):
                    estimator, donors = build_estimator(
                        evidence, policy, sequence_id, episode_index, n_sequences
                    )
                    saved = row["policies"][policy]
                    if not close(saved["context_degrees"], estimator.estimate_degrees, 1e-8):
                        failures["estimator"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}:angle")
                    for name, value in estimator.as_dict().items():
                        if name in ("transition_count", "accepted_count"):
                            match = int(saved["estimator"][name]) == int(value)
                        else:
                            match = close(saved["estimator"][name], value, 1e-8)
                        if not match:
                            failures["estimator"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}:{name}")
                    if saved["donors"] != donors:
                        failures["donor"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}:records")
                    for h, donor in enumerate(donors):
                        expected_donor = donor_sequence(policy, sequence_id, h, n_sequences)
                        if donor["donor_sequence_id"] != expected_donor or donor["history_episode"] > episode_index:
                            failures["donor"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}:lifetime")
                policies = row["policies"]
                if (
                    policies["current_only"]["command_sha256"] != policies["population_prior"]["command_sha256"]
                    or policies["current_only"]["state_sha256"] != policies["population_prior"]["state_sha256"]
                ):
                    failures["identity"].append(f"{condition}:{sequence_id}:{episode_index}:current")
                if episode_index == 0:
                    hashes = {
                        (policies[p]["command_sha256"], policies[p]["state_sha256"])
                        for p in POLICIES
                        if p != "true_factor_oracle"
                    }
                    if len(hashes) != 1:
                        failures["identity"].append(f"{condition}:{sequence_id}:episode1")
                for policy in POLICIES:
                    payload = policies[policy]
                    metrics = pose_metrics(np.asarray(payload["states"]), np.asarray(row["goal_state"]), 25)
                    if any(not close(metrics[name], payload["metrics"][name], 1e-9) for name in metrics):
                        failures["metrics"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}")
                    if payload["command_sha256"] != array_sha256(np.asarray(payload["commands"], dtype=np.float32)):
                        failures["identity"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}:command_hash")
                    if payload["state_sha256"] != array_sha256(np.asarray(payload["states"], dtype=np.float32)):
                        failures["identity"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}:state_hash")
    summary_match = runner == recomputed
    checks = {
        "raw_recomputes_runner_summary_exactly": summary_match,
        "all_failure_counts_zero": all(not values for values in failures.values()),
        "runner_structural_valid": bool(recomputed.get("valid")),
    }
    return {
        "schema": "persistent-context-v2-pushobj-rotation-stage1-independent-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failure_counts": {key: len(value) for key, value in failures.items()},
        "failure_examples": {key: value[:20] for key, value in failures.items() if value},
        "recomputed_summary": recomputed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repro_outputs/persistent_context_v2_pushobj_rotation_stage1"),
    )
    args = parser.parse_args()
    result = audit(args.output_dir)
    target = args.output_dir / "independent_audit.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
