#!/usr/bin/env python3
"""Create the outcome-blind frozen cross-shape segment selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np


def segment_hash(segment: dict) -> str:
    digest = hashlib.sha256()
    for key in ("states", "actions"):
        value = np.asarray(segment[key])
        digest.update(key.encode("ascii")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    loaded, hash_counts = {}, Counter()
    for shape in design["shapes"]:
        with Path(audit["pools"][shape]["path"]).open("rb") as handle:
            segments = pickle.load(handle)["segments"]
        loaded[shape] = segments
        hash_counts.update(segment_hash(segment) for segment in segments)
    rng = random.Random(int(design["selection_seed"]))
    candidates = {}
    for shape in design["shapes"]:
        values = []
        for index in audit["pools"][shape]["eligible_indices"]:
            segment = loaded[shape][int(index)]; digest = segment_hash(segment)
            if hash_counts[digest] != 1:
                continue
            values.append((int(index), digest, f"{int(segment['ep_idx'])}:{int(segment['offset'])}"))
        rng.shuffle(values); candidates[shape] = values
    cursors = {shape: 0 for shape in design["shapes"]}; used_hashes, used_provenance = set(), set()

    def take(shape: str) -> dict:
        while cursors[shape] < len(candidates[shape]):
            index, digest, provenance = candidates[shape][cursors[shape]]; cursors[shape] += 1
            if digest in used_hashes or provenance in used_provenance:
                continue
            used_hashes.add(digest); used_provenance.add(provenance)
            return {"shape": shape, "segment_index": index, "segment_sha256": digest, "provenance_key": provenance}
        raise RuntimeError(f"candidate pool exhausted for {shape}")

    smoke_schedule = [(pair_index, pair_index % len(design["factors"]), 0) for pair_index in range(len(design["shape_pairs"]))]
    formal_schedule = [(pair_index, factor_index, replicate)
                       for pair_index in range(len(design["shape_pairs"]))
                       for factor_index in range(len(design["factors"]))
                       for replicate in range(int(design["formal"]["replicates_per_shape_pair_factor"]))]
    rng.shuffle(formal_schedule)

    def rows(schedule):
        result = []
        for sequence_id, (pair_index, factor_index, replicate) in enumerate(schedule):
            e1_shape, e2_shape = design["shape_pairs"][pair_index]
            result.append({"sequence_id": sequence_id, "shape_pair_index": pair_index, "factor_index": factor_index,
                           "replicate": replicate, "no_persistence_factor_index": (factor_index + 1) % len(design["factors"]),
                           "e1": take(e1_shape), "e2": take(e2_shape)})
        return result

    smoke, formal = rows(smoke_schedule), rows(formal_schedule)
    reserve = {shape: [take(shape) for _ in range(int(design["reserve_segments_per_shape"]))] for shape in design["shapes"]}
    output = {"contract_id": design["contract_id"], "selection_seed": design["selection_seed"],
              "algorithm": "shuffle each shape eligible list; reject globally duplicated segment hashes; greedily enforce globally unique selected hash and ep_idx:offset",
              "author_pool_audit_sha256": design["author_pool_audit_sha256"], "smoke": smoke, "formal": formal, "reserve": reserve}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"smoke_sequences": len(smoke), "formal_sequences": len(formal),
                      "selected_segments": len(used_hashes), "unique_provenance": len(used_provenance),
                      "reserve_segments": sum(map(len, reserve.values()))}, indent=2))


if __name__ == "__main__":
    main()
