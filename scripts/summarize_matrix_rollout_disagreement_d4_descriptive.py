#!/usr/bin/env python3
"""Post-result descriptive supplement for the completed D4 development study.

This script does not change the preregistered primary score or direction.  It
adds complete effect and heterogeneity reporting required by project governance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_RAW_SHA256 = "7862f1dad8f8ceb00b77a040f5ef78e2a0c1a1b6290d91b995fb8246d9beb2d7"
BOOTSTRAP_STREAM_SEED = 1_222_000
BOOTSTRAP_RESAMPLES = 20_000
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "d4_sequence":
                rows.append(row)
    return rows


def bootstrap_mean_ci(values, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    return [float(value) for value in np.quantile(values[indexes].mean(axis=1), (0.025, 0.975))]


def summarize_group(rows: list[dict], indexes: list[int], stream: int) -> dict:
    population = np.asarray([rows[index]["e2"]["policies"]["population"]["metrics"]["pose_auc10"] for index in indexes])
    context = np.asarray([rows[index]["e2"]["policies"]["context"]["metrics"]["pose_auc10"] for index in indexes])
    benefit = population - context
    return {
        "n": len(indexes),
        "population_mean_pose_auc10": float(population.mean()),
        "context_mean_pose_auc10": float(context.mean()),
        "mean_benefit": float(benefit.mean()),
        "relative_improvement": float(benefit.mean() / population.mean()),
        "bootstrap_ci95_mean_benefit": bootstrap_mean_ci(benefit, BOOTSTRAP_STREAM_SEED + stream),
        "positive_count": int(np.sum(benefit > TOLERANCE)),
        "tie_count": int(np.sum(np.abs(benefit) <= TOLERANCE)),
        "negative_count": int(np.sum(benefit < -TOLERANCE)),
        "harm_fraction": float(np.mean(benefit < -TOLERANCE)),
        "sequence_ids": [int(rows[index]["sequence_id"]) for index in indexes],
        "unit_benefit": benefit.tolist(),
    }


def summarize(raw: Path) -> dict:
    if sha256(raw) != EXPECTED_RAW_SHA256:
        raise RuntimeError("completed D4 raw hash mismatch")
    rows = sorted(read_rows(raw), key=lambda row: int(row["sequence_id"]))
    if len(rows) != 96:
        raise RuntimeError("descriptive supplement requires 96 rows")
    result = {
        "label": "post_result_descriptive_supplement_not_a_new_primary_analysis",
        "raw_sha256": sha256(raw),
        "bootstrap_seed": BOOTSTRAP_STREAM_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "overall": summarize_group(rows, list(range(len(rows))), 0),
        "by_shape_pair": {},
        "by_factor": {},
        "by_execution_order": {},
    }
    for pair in range(6):
        indexes = [index for index, row in enumerate(rows) if int(row["shape_pair_index"]) == pair]
        shapes = rows[indexes[0]]["selection"]
        label = f"{shapes['e1']['shape']}->{shapes['e2']['shape']}"
        result["by_shape_pair"][label] = summarize_group(rows, indexes, 100 + pair)
    for factor in range(8):
        indexes = [index for index, row in enumerate(rows) if int(row["factor_index"]) == factor]
        values = rows[indexes[0]]["factor"]
        label = f"rotation={values['rotation_degrees']:+g},gain={values['gain']:g}"
        result["by_factor"][label] = summarize_group(rows, indexes, 200 + factor)
    for first in ("population", "context"):
        indexes = [index for index, row in enumerate(rows) if row["e2"]["execution_order"][0] == first]
        result["by_execution_order"][f"{first}_first"] = summarize_group(rows, indexes, 300 + (first == "context"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
