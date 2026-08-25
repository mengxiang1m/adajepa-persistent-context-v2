#!/usr/bin/env python3
"""Audit eligible val-T segments not exposed to an existing matrix outcome."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def segment_indices(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "segment_index" and isinstance(item, (int, float)):
                yield int(item)
            elif key == "segment_indices" and isinstance(item, list):
                yield from (int(index) for index in item)
            else:
                yield from segment_indices(item)
    elif isinstance(value, list):
        for item in value:
            yield from segment_indices(item)


def load_values(path: Path):
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        else:
            yield json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.data.open("rb") as handle:
        data = pickle.load(handle)
    segments = data["segments"]
    eligible = {
        index for index, segment in enumerate(segments)
        if len(segment["states"]) > 10
        and float(np.linalg.norm(np.asarray(segment["states"])[10, 2:4] - np.asarray(segment["states"])[0, 2:4])) >= 10.0
    }
    used, sources = set(), {}
    roots = [args.repo / "repro_outputs", args.repo / "docs" / "research"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in (".json", ".jsonl") or "matrix" not in str(path).lower():
                continue
            for value in load_values(path):
                for index in segment_indices(value):
                    if 0 <= index < len(segments):
                        used.add(index)
                        sources.setdefault(index, set()).add(str(path))
    unused = sorted(eligible - used)
    result = {
        "data": str(args.data),
        "data_metadata": {key: data.get(key) for key in ("n_samples", "traj_len", "shape", "seed")},
        "total_segments": len(segments),
        "eligible_segments": len(eligible),
        "matrix_exposed_segments": len(used),
        "eligible_matrix_exposed_segments": len(eligible & used),
        "eligible_matrix_unexposed_segments": len(unused),
        "eligible_matrix_unexposed_indices": unused,
        "exposure_source_counts": {str(index): len(sources[index]) for index in sorted(sources)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("eligible_matrix_unexposed_indices", "exposure_source_counts")}, indent=2))


if __name__ == "__main__":
    main()
