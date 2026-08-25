#!/usr/bin/env python3
"""Freeze the author-T pool used by the delay-history formal without reading policy outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def segment_hash(segment: dict) -> str:
    digest = hashlib.sha256()
    for key in ("states", "actions"):
        value = np.asarray(segment[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def displacement(segment: dict) -> float:
    states = np.asarray(segment["states"], dtype=np.float64)
    if len(states) <= 10:
        return 0.0
    return float(np.linalg.norm(states[10, 2:4] - states[0, 2:4]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    used: set[int] = set()
    sources = []
    # Only the top-level provenance field is consumed. No metric, policy, cost,
    # command, state, factor, or outcome value participates in selection.
    for path in sorted((args.repo / "repro_outputs").rglob("*.jsonl")):
        indexes = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and "segment_index" in row:
                        indexes.append(int(row["segment_index"]))
        except OSError:
            continue
        if indexes:
            used.update(indexes)
            sources.append({
                "path": str(path.relative_to(args.repo)),
                "file_sha256": sha256(path),
                "row_count_with_segment_index": len(indexes),
                "unique_segment_indices": len(set(indexes)),
            })
    eligible = [index for index, segment in enumerate(segments) if index not in used and displacement(segment) >= 10.0]
    permutation = np.random.default_rng(int(selection["selection_seed"])).permutation(eligible).tolist()
    expected = (
        [index for pair in selection["smoke_segment_indices_by_sequence_e1_e2"] for index in pair]
        + [index for pair in selection["formal_segment_indices_by_sequence_e1_e2"] for index in pair]
        + selection["reserve_segment_indices"]
        + selection["unused_eligible_segment_indices"]
    )
    failures = []
    if len(used) != int(selection["used_unique_segment_indices_before_selection"]): failures.append("used count mismatch")
    if len(eligible) != int(selection["eligible_count_before_selection"]): failures.append("eligible count mismatch")
    if permutation != expected: failures.append("seeded selection replay mismatch")
    if len(expected) != len(set(expected)): failures.append("selected index duplicate")
    groups = {
        "smoke": [index for pair in selection["smoke_segment_indices_by_sequence_e1_e2"] for index in pair],
        "formal": [index for pair in selection["formal_segment_indices_by_sequence_e1_e2"] for index in pair],
        "reserve": selection["reserve_segment_indices"],
        "unused": selection["unused_eligible_segment_indices"],
    }
    payload = {
        "contract_id": selection["contract_id"],
        "valid": not failures,
        "failures": failures,
        "selection_path": str(args.selection),
        "selection_sha256": sha256(args.selection),
        "data_path": str(args.data),
        "data_sha256": sha256(args.data),
        "outcome_fields_used_for_selection": [],
        "used_unique_segment_indices": len(used),
        "eligible_count": len(eligible),
        "source_files": sources,
        "groups": {
            name: [{
                "segment_index": int(index),
                "segment_sha256": segment_hash(segments[index]),
                "nominal_block_displacement_at_10": displacement(segments[index]),
            } for index in indexes]
            for name, indexes in groups.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": payload["valid"], "failures": failures,
        "used_unique": len(used), "eligible": len(eligible),
        "smoke": len(groups["smoke"]), "formal": len(groups["formal"]),
        "reserve": len(groups["reserve"]), "unused": len(groups["unused"]),
        "selection_sha256": payload["selection_sha256"],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
