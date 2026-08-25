#!/usr/bin/env python3
"""Read-only audit of author-released PushObj shape target pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def segment_sha256(segment: dict) -> str:
    digest = hashlib.sha256()
    for key in ("states", "actions"):
        value = np.asarray(segment[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def manifest_data_bindings(repo: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    root = repo / "repro_outputs"
    if not root.exists():
        return {}
    for path in root.rglob("manifest.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        data = value.get("data")
        if isinstance(data, str):
            counts[data] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pools = {}
    hash_locations: defaultdict[str, list[list[object]]] = defaultdict(list)
    provenance_locations: defaultdict[str, list[list[object]]] = defaultdict(list)
    for path in sorted(args.data_root.glob("val_*/plan_targets.pkl")):
        with path.open("rb") as handle:
            data = pickle.load(handle)
        shape = str(data["shape"])
        segments = data["segments"]
        eligible = []
        hashes = []
        provenance = []
        for index, segment in enumerate(segments):
            states = np.asarray(segment["states"])
            if len(states) > 10 and float(np.linalg.norm(states[10, 2:4] - states[0, 2:4])) >= 10.0:
                eligible.append(index)
            digest = segment_sha256(segment)
            hashes.append(digest)
            hash_locations[digest].append([shape, index])
            key = f"{int(segment['ep_idx'])}:{int(segment['offset'])}"
            provenance.append(key)
            provenance_locations[key].append([shape, index])
        pools[shape] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "metadata": {key: data.get(key) for key in ("n_samples", "traj_len", "shape", "seed")},
            "total_segments": len(segments),
            "eligible_segments": len(eligible),
            "eligible_indices": eligible,
            "within_pool_exact_duplicate_segments": len(hashes) - len(set(hashes)),
            "within_pool_duplicate_provenance_keys": len(provenance) - len(set(provenance)),
        }
    cross_exact = {key: value for key, value in hash_locations.items() if len({row[0] for row in value}) > 1}
    cross_provenance = {key: value for key, value in provenance_locations.items() if len({row[0] for row in value}) > 1}
    result = {
        "schema_version": 1,
        "eligibility": "nominal block displacement at step 10 >= 10 pixels",
        "pools": pools,
        "total_pools": len(pools),
        "total_segments": sum(row["total_segments"] for row in pools.values()),
        "total_eligible_segments": sum(row["eligible_segments"] for row in pools.values()),
        "cross_pool_exact_duplicate_hash_count": len(cross_exact),
        "cross_pool_exact_duplicate_locations": cross_exact,
        "cross_pool_shared_provenance_key_count": len(cross_provenance),
        "cross_pool_shared_provenance_locations": cross_provenance,
        "recorded_manifest_data_bindings": manifest_data_bindings(args.repo),
        "interpretation": "Exact state+action hashes test asset duplication. ep_idx:offset overlap alone is not exact trajectory duplication across independently shaped datasets.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {"total_pools": result["total_pools"], "total_segments": result["total_segments"],
               "total_eligible_segments": result["total_eligible_segments"],
               "cross_pool_exact_duplicate_hash_count": result["cross_pool_exact_duplicate_hash_count"],
               "cross_pool_shared_provenance_key_count": result["cross_pool_shared_provenance_key_count"],
               "pools": {key: {name: row[name] for name in ("file_sha256", "total_segments", "eligible_segments", "within_pool_exact_duplicate_segments")}
                         for key, row in pools.items()},
               "recorded_manifest_data_bindings": result["recorded_manifest_data_bindings"]}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
