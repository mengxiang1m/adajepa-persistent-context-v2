#!/usr/bin/env python3
"""Independent audit for D1; deliberately does not import the D1 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DESIGN_ID = "persistent-context-v2-matrix-soft-context-d1-exploratory-v1"
PRIOR = np.eye(2, dtype=np.float64) * 0.9327804920294028


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_auc10(states, goal) -> float:
    states = np.asarray(states, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    post = states[1:11]
    position = np.linalg.norm(post[:, 2:4] - goal[None, 2:4], axis=1)
    delta = np.abs(post[:, 4] - goal[4]) % (2.0 * math.pi)
    angle = np.minimum(delta, 2.0 * math.pi - delta)
    return float(np.mean(position / 20.0 + angle / (math.pi / 9.0)))


def audit(args) -> dict:
    design = json.loads(args.design.read_text(encoding="utf-8"))
    alphas = [float(value) for value in design["alphas"]]
    rows = [row for row in read_jsonl(args.output_dir / "raw.jsonl") if row.get("record_type") == "d1_alpha_treatment"]
    originals = {}
    for split in design["splits"]:
        for row in read_jsonl(args.data_dir / split / "raw.jsonl"):
            if row.get("record_type") == "paired_sequence":
                originals[(split, int(row["sequence_id"]))] = row
    failures = []
    keys = [(str(row["split"]), int(row["sequence_id"]), float(row["alpha"])) for row in rows]
    if len(keys) != len(set(keys)):
        failures.append("duplicate treatment key")
    sequence_keys = sorted({key[:2] for key in keys})
    missing = [(split, sequence, alpha) for split, sequence in sequence_keys for alpha in alphas if (split, sequence, alpha) not in set(keys)]
    if missing:
        failures.append(f"incomplete alpha grid: {len(missing)} missing")
    max_matrix_error = max_metric_error = max_endpoint_command_error = max_endpoint_state_error = 0.0
    endpoint_hash_mismatches = stored_hash_mismatches = 0
    lookup = {}
    for row in rows:
        key = (str(row["split"]), int(row["sequence_id"]), float(row["alpha"]))
        lookup[key] = row
        posterior = np.asarray(row["posterior_matrix"], dtype=np.float64)
        expected_matrix = (1.0 - key[2]) * PRIOR + key[2] * posterior
        max_matrix_error = max(max_matrix_error, float(np.max(np.abs(expected_matrix - np.asarray(row["context_matrix"], dtype=np.float64)))))
        recomputed = pose_auc10(row["states"], row["goal_state"])
        max_metric_error = max(max_metric_error, abs(recomputed - float(row["metrics"]["pose_auc10"])))
        commands_hash = hashlib.sha256(np.asarray(row["commands"], dtype=np.float32).tobytes()).hexdigest()
        states_hash = hashlib.sha256(np.asarray(row["states"], dtype=np.float32).tobytes()).hexdigest()
        stored_hash_mismatches += int(commands_hash != row["command_sha256"] or states_hash != row["state_sha256"])
        if key[2] in (0.0, 1.0):
            original = originals[key[:2]]["e2"]["population" if key[2] == 0.0 else "context"]
            command_error = float(np.max(np.abs(np.asarray(row["commands"]) - np.asarray(original["commands"]))))
            state_error = float(np.max(np.abs(np.asarray(row["states"]) - np.asarray(original["states"]))))
            max_endpoint_command_error = max(max_endpoint_command_error, command_error)
            max_endpoint_state_error = max(max_endpoint_state_error, state_error)
            endpoint_hash_mismatches += int(row["command_sha256"] != original["command_sha256"] or row["state_sha256"] != original["state_sha256"])
    if max_matrix_error > 1e-12:
        failures.append(f"matrix replay error {max_matrix_error}")
    if max_metric_error > 1e-12:
        failures.append(f"metric replay error {max_metric_error}")
    if endpoint_hash_mismatches or max_endpoint_command_error > 1e-6 or max_endpoint_state_error > 1e-6:
        failures.append("endpoint identity failure")
    if stored_hash_mismatches:
        failures.append(f"stored array hash mismatch: {stored_hash_mismatches}")
    summary_path = args.output_dir / "summary.json"
    summary_errors = {}
    if summary_path.exists() and sequence_keys and not missing:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        costs = np.asarray([[lookup[(split, sequence, alpha)]["metrics"]["pose_auc10"] for alpha in alphas] for split, sequence in sequence_keys])
        for index, alpha in enumerate(alphas):
            expected = float(np.mean(costs[:, 0] - costs[:, index]))
            observed = float(summary["by_alpha"][str(alpha)]["mean_delta_vs_population"])
            summary_errors[str(alpha)] = abs(expected - observed)
        if max(summary_errors.values(), default=0.0) > 1e-12:
            failures.append("summary replay error")
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("design_id") != DESIGN_ID or design.get("design_id") != DESIGN_ID:
        failures.append("design id mismatch")
    result = {
        "design_id": DESIGN_ID,
        "valid": not failures,
        "failures": failures,
        "rows": len(rows),
        "unique_treatment_keys": len(set(keys)),
        "unique_sequences": len(sequence_keys),
        "matrix_replay_max_abs": max_matrix_error,
        "metric_replay_max_abs": max_metric_error,
        "endpoint_command_max_abs": max_endpoint_command_error,
        "endpoint_state_max_abs": max_endpoint_state_error,
        "endpoint_hash_mismatches": endpoint_hash_mismatches,
        "stored_array_hash_mismatches": stored_hash_mismatches,
        "summary_mean_delta_abs_errors": summary_errors,
        "raw_sha256": file_sha256(args.output_dir / "raw.jsonl"),
        "design_sha256": file_sha256(args.design),
    }
    args.audit_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_soft_context_d1_design.json"))
    parser.add_argument("--audit-output", type=Path, required=True)
    result = audit(parser.parse_args())
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
