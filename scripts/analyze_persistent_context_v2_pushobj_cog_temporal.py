"""Descriptive stepwise error analysis; never used for checkpoint selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.persistent_context_v2.pushobj_cog_predictor import ANGLE_SCALE, signed_angle_delta
from research.persistent_context_v2.pushobj_rotation_stage0 import dump_json, read_jsonl


def trajectory_step_error(predicted, executed):
    predicted = np.asarray(predicted, dtype=np.float64)
    executed = np.asarray(executed, dtype=np.float64)
    position = np.linalg.norm(predicted[1:, 2:4] - executed[1:, 2:4], axis=1) / 20.0
    angle = np.abs(signed_angle_delta(predicted[1:, 4], executed[1:, 4])) / ANGLE_SCALE
    return position + angle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_temporal"))
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.output_dir / "raw.jsonl") if row.get("record_type") == "cog_temporal_pair"]
    names = ("population_prior_context", "v1_true_cog_context", "v2_temporal_true_cog_context")
    step_errors = {
        name: np.asarray([trajectory_step_error(row["policies"][name]["predicted_states"], row["policies"][name]["states"]) for row in rows])
        for name in names
    }
    behavior = {
        name: np.asarray([row["policies"][name]["metrics"]["pose_auc10"] for row in rows])
        for name in names
    }
    pop_error = step_errors[names[0]].mean(axis=1)
    result = {
        "descriptive_only": True,
        "n_pairs": len(rows),
        "mean_prediction_error_by_step": {name: value.mean(axis=0).tolist() for name, value in step_errors.items()},
        "mean_last_five_prediction_error": {name: float(value[:, 5:].mean()) for name, value in step_errors.items()},
        "prediction_error_reduction": {
            "v1_vs_population": float(pop_error.mean() - step_errors[names[1]].mean()),
            "v2_vs_population": float(pop_error.mean() - step_errors[names[2]].mean()),
        },
        "correlation_prediction_reduction_behavior_delta": {
            "v1": float(np.corrcoef(pop_error - step_errors[names[1]].mean(axis=1), behavior[names[0]] - behavior[names[1]])[0, 1]),
            "v2": float(np.corrcoef(pop_error - step_errors[names[2]].mean(axis=1), behavior[names[0]] - behavior[names[2]])[0, 1]),
        },
    }
    dump_json(args.output_dir / "descriptive_step_analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
