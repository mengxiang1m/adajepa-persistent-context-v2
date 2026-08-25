#!/usr/bin/env python3
"""Create the outcome-blind D4 development selection from unused author segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def prior_exclusions(selection: dict) -> tuple[set[str], set[str]]:
    items = []
    for split in ("smoke", "formal"):
        items.extend(row[episode] for row in selection[split] for episode in ("e1", "e2"))
    items.extend(item for rows in selection["reserve"].values() for item in rows)
    return ({str(item["segment_sha256"]) for item in items}, {str(item["provenance_key"]) for item in items})


def build_selection(design: dict, audit: dict, excluded: dict) -> dict:
    loaded: dict[str, list[dict]] = {}
    hash_counts: Counter[str] = Counter()
    for shape in design["shapes"]:
        with Path(audit["pools"][shape]["path"]).open("rb") as handle:
            segments = pickle.load(handle)["segments"]
        loaded[shape] = segments
        hash_counts.update(segment_hash(segment) for segment in segments)

    excluded_hashes, excluded_provenance = prior_exclusions(excluded)
    rng = random.Random(int(design["selection_seed"]))
    candidates: dict[str, list[tuple[int, str, str]]] = {}
    for shape in design["shapes"]:
        values = []
        for index in audit["pools"][shape]["eligible_indices"]:
            segment = loaded[shape][int(index)]
            digest = segment_hash(segment)
            provenance = f"{int(segment['ep_idx'])}:{int(segment['offset'])}"
            if hash_counts[digest] != 1 or digest in excluded_hashes or provenance in excluded_provenance:
                continue
            values.append((int(index), digest, provenance))
        rng.shuffle(values)
        candidates[shape] = values

    cursors = {shape: 0 for shape in design["shapes"]}
    used_hashes: set[str] = set()
    used_provenance: set[str] = set()

    def take(shape: str) -> dict:
        while cursors[shape] < len(candidates[shape]):
            index, digest, provenance = candidates[shape][cursors[shape]]
            cursors[shape] += 1
            if digest in used_hashes or provenance in used_provenance:
                continue
            used_hashes.add(digest)
            used_provenance.add(provenance)
            return {"shape": shape, "segment_index": index, "segment_sha256": digest, "provenance_key": provenance}
        raise RuntimeError(f"candidate pool exhausted for {shape}")

    smoke_schedule = [(pair, pair % len(design["factors"]), 0) for pair in range(len(design["shape_pairs"]))]
    development_schedule = [
        (pair, factor, replicate)
        for pair in range(len(design["shape_pairs"]))
        for factor in range(len(design["factors"]))
        for replicate in range(int(design["development"]["replicates_per_shape_pair_factor"]))
    ]
    rng.shuffle(development_schedule)

    def make_rows(schedule: list[tuple[int, int, int]]) -> list[dict]:
        rows = []
        for sequence_id, (pair_index, factor_index, replicate) in enumerate(schedule):
            e1_shape, e2_shape = design["shape_pairs"][pair_index]
            rows.append({
                "sequence_id": sequence_id,
                "shape_pair_index": pair_index,
                "factor_index": factor_index,
                "replicate": replicate,
                "e1": take(e1_shape),
                "e2": take(e2_shape),
            })
        return rows

    smoke = make_rows(smoke_schedule)
    development = make_rows(development_schedule)
    return {
        "contract_id": design["contract_id"],
        "selection_seed": int(design["selection_seed"]),
        "excluded_selection_sha256": design["excluded_selection_sha256"],
        "algorithm": "shuffle eligible unique segments after prior hash/provenance exclusions; greedily enforce selected hash/provenance uniqueness",
        "excluded_segment_hashes": len(excluded_hashes),
        "excluded_provenance_keys": len(excluded_provenance),
        "smoke": smoke,
        "development": development,
        "selected_segment_count": len(used_hashes),
        "unique_provenance_count": len(used_provenance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--excluded-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    if sha256(args.audit) != design["author_pool_audit_sha256"]:
        raise RuntimeError("author pool audit hash mismatch")
    if sha256(args.excluded_selection) != design["excluded_selection_sha256"]:
        raise RuntimeError("excluded selection hash mismatch")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    excluded = json.loads(args.excluded_selection.read_text(encoding="utf-8"))
    result = build_selection(design, audit, excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "smoke_sequences": len(result["smoke"]),
        "development_sequences": len(result["development"]),
        "selected_segments": result["selected_segment_count"],
        "excluded_segment_hashes": result["excluded_segment_hashes"],
    }, indent=2))


if __name__ == "__main__":
    main()
