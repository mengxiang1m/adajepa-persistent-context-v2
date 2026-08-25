#!/usr/bin/env python3
"""Independent raw audit for early-waypoint rotation Stage 0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    CONTRACT_ID,
    EXPECTED_DESIGN_SHA256,
    SEGMENT_INDICES,
    array_sha256,
    deadline_success,
    summarize,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import pose_metrics, read_jsonl, sha256


def close(a, b, atol=1e-10):
    return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=atol))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage0"),
    )
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    runner = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    rows = [row for row in read_jsonl(args.output_dir / "raw.jsonl") if row.get("record_type") == "paired_waypoint"]
    failures = {"manifest": [], "scenario": [], "metrics": [], "hash": [], "pairing": []}
    if manifest.get("contract_id") != CONTRACT_ID:
        failures["manifest"].append("contract")
    if manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("design_manifest")
    if sha256(Path(manifest["design_path"])) != EXPECTED_DESIGN_SHA256:
        failures["manifest"].append("design_file")
    if max(manifest.get("identity_audit", {}).values(), default=np.inf) > 1e-6:
        failures["manifest"].append("identity")
    if len(rows) != 32 or len({int(row["ordinal"]) for row in rows}) != 32:
        failures["scenario"].append("count")
    for row in rows:
        ordinal = int(row["ordinal"])
        if int(row["segment_index"]) != SEGMENT_INDICES[ordinal]:
            failures["scenario"].append(f"{ordinal}:segment")
        if float(row["nominal_block_displacement_at_10"]) < 10.0:
            failures["scenario"].append(f"{ordinal}:displacement")
        if row.get("contract_id") != CONTRACT_ID:
            failures["scenario"].append(f"{ordinal}:contract")
        for policy in ("prior", "oracle"):
            payload = row[policy]
            metrics = pose_metrics(np.asarray(payload["states"]), np.asarray(row["goal_state"]), 10)
            if any(not close(metrics[name], payload["metrics"][name], 1e-9) for name in metrics):
                failures["metrics"].append(f"{ordinal}:{policy}:pose")
            success = deadline_success(np.asarray(payload["states"]), np.asarray(row["goal_state"]))
            if bool(payload["deadline_success"]) != success:
                failures["metrics"].append(f"{ordinal}:{policy}:success")
            if payload["command_sha256"] != array_sha256(np.asarray(payload["commands"], dtype=np.float32)):
                failures["hash"].append(f"{ordinal}:{policy}:command")
            if payload["state_sha256"] != array_sha256(np.asarray(payload["states"], dtype=np.float32)):
                failures["hash"].append(f"{ordinal}:{policy}:state")
        if row["prior"]["planner"]["trace"][0]["iteration"] != row["oracle"]["planner"]["trace"][0]["iteration"]:
            failures["pairing"].append(f"{ordinal}:budget")
    recomputed = summarize(args.output_dir / "raw.jsonl", manifest["identity_audit"])
    checks = {
        "raw_recomputes_runner_summary_exactly": recomputed == runner,
        "all_failure_counts_zero": all(not values for values in failures.values()),
        "runner_valid": bool(recomputed.get("valid")),
    }
    result = {
        "schema": "persistent-context-v2-pushobj-rotation-early-waypoint-stage0-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failure_counts": {key: len(value) for key, value in failures.items()},
        "failure_examples": {key: value[:20] for key, value in failures.items() if value},
        "recomputed_summary": recomputed,
    }
    target = args.output_dir / "independent_audit.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
