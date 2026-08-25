#!/usr/bin/env python3
"""Independent raw audit for early-waypoint rotation history Stage 1."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import deadline_success
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage1 import (
    CONDITIONS, CONTRACT_ID, EXPECTED_DESIGN_SHA256, POLICIES,
    array_sha256, build_estimator, factor_for, formal_segment_indices, scenario, summarize,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import pose_metrics, read_jsonl, sha256


def close(a, b, atol=1e-9):
    return bool(np.isclose(float(a), float(b), rtol=1e-8, atol=atol))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage1"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    runner = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    n_sequences, episodes = int(manifest["sequences"]), int(manifest["episodes"])
    with Path(manifest["data"]).open("rb") as handle: segments = pickle.load(handle)["segments"]
    failures = {k: [] for k in ("manifest", "count", "scenario", "estimator", "donor", "metric", "hash", "identity")}
    if manifest.get("contract_id") != CONTRACT_ID or manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("contract_or_design")
    if sha256(Path(manifest["design_path"])) != EXPECTED_DESIGN_SHA256 or max(manifest.get("identity_audit", {}).values(), default=np.inf) > 1e-6:
        failures["manifest"].append("design_file_or_identity")
    for condition in CONDITIONS:
        raw = read_jsonl(args.output_dir / f"{condition}_raw.jsonl")
        evidence_rows = [r for r in raw if r.get("record_type") == "evidence_episode"]
        eval_rows = [r for r in raw if r.get("record_type") == "evaluation_episode"]
        evidence = {(int(r["sequence_id"]), int(r["episode_index"])): r for r in evidence_rows}
        evaluations = {(int(r["sequence_id"]), int(r["episode_index"])): r for r in eval_rows}
        if len(evidence) != n_sequences * episodes or len(evaluations) != n_sequences * episodes:
            failures["count"].append(condition)
        for sequence_id in range(n_sequences):
            for episode_index in range(episodes):
                key = (sequence_id, episode_index)
                if key not in evidence or key not in evaluations: continue
                expected = scenario(segments, condition, sequence_id, episode_index, n_sequences, episodes)
                ev, row = evidence[key], evaluations[key]
                for candidate in (ev, row):
                    if int(candidate["segment_index"]) != expected["segment_index"] or float(candidate["factor_deg"]) != expected["factor_deg"]:
                        failures["scenario"].append(f"{condition}:{sequence_id}:{episode_index}")
                expected_hash = array_sha256(np.asarray(ev["commands"], dtype=np.float32)) + ":" + array_sha256(np.asarray(ev["states"], dtype=np.float32))
                if ev["evidence_sha256"] != expected_hash: failures["hash"].append(f"{condition}:{sequence_id}:{episode_index}:evidence")
                for policy in ("correct_history", "shuffled_history", "wrong_sequence_history"):
                    estimator, donors = build_estimator(evidence, policy, sequence_id, episode_index, n_sequences)
                    saved = row["policies"][policy]
                    if not close(saved["context_degrees"], estimator.estimate_degrees, 1e-8): failures["estimator"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}")
                    if saved["donors"] != donors: failures["donor"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}")
                policies = row["policies"]
                if policies["current_only"]["state_sha256"] != policies["population_prior"]["state_sha256"]:
                    failures["identity"].append(f"{condition}:{sequence_id}:{episode_index}:current")
                if episode_index == 0:
                    ids = {(policies[p]["command_sha256"], policies[p]["state_sha256"]) for p in POLICIES if p != "true_factor_oracle"}
                    if len(ids) != 1: failures["identity"].append(f"{condition}:{sequence_id}:e1")
                for policy in POLICIES:
                    payload = policies[policy]
                    metrics = pose_metrics(np.asarray(payload["states"]), np.asarray(row["goal_state"]), 10)
                    if any(not close(metrics[k], payload["metrics"][k]) for k in metrics) or bool(payload["deadline_success"]) != deadline_success(np.asarray(payload["states"]), np.asarray(row["goal_state"])):
                        failures["metric"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}")
                    if payload["command_sha256"] != array_sha256(np.asarray(payload["commands"], dtype=np.float32)) or payload["state_sha256"] != array_sha256(np.asarray(payload["states"], dtype=np.float32)):
                        failures["hash"].append(f"{condition}:{sequence_id}:{episode_index}:{policy}")
    recomputed = summarize(args.output_dir, n_sequences, episodes)
    checks = {"raw_recomputes_runner_summary_exactly": recomputed == runner, "all_failure_counts_zero": all(not x for x in failures.values()), "runner_valid": bool(recomputed.get("valid"))}
    result = {"schema": "persistent-context-v2-pushobj-rotation-early-waypoint-stage1-audit-v1", "passed": all(checks.values()), "checks": checks,
              "failure_counts": {k: len(v) for k, v in failures.items()}, "failure_examples": {k: v[:20] for k, v in failures.items() if v}, "recomputed_summary": recomputed}
    (args.output_dir / "independent_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
