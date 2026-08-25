"""Independent CPU replay audit for the CoG temporal predictor v2 experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.pushobj_cog_predictor import apply_residual, encode_trajectory, load_model as load_v1_model
from research.persistent_context_v2.pushobj_cog_stage0 import array_sha256, rollout_physics
from research.persistent_context_v2.pushobj_cog_temporal_predictor import (
    CONTRACT_ID,
    EXPECTED_DESIGN_SHA256,
    EXPECTED_DEV_INPUT_SHA256,
    EXPECTED_DEV_TARGET_SHA256,
    EXPECTED_TRAIN_INPUT_SHA256,
    EXPECTED_TRAIN_TARGET_SHA256,
    EXPECTED_V1_CHECKPOINT_SHA256,
    FORMAL_FACTORS,
    FORMAL_SEGMENTS,
    load_temporal,
    summarize,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import dump_json, make_env, pose_metrics, read_jsonl, sha256


def close_tree(a, b, atol=1e-7):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close_tree(a[key], b[key], atol) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close_tree(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= atol
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_temporal_design.json"))
    parser.add_argument("--train-data-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_predictor"))
    parser.add_argument("--v1-checkpoint", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_predictor/model_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_temporal"))
    parser.add_argument("--output-name", default="independent_audit.json")
    args = parser.parse_args()
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    design = json.loads(args.design.read_text(encoding="utf-8"))
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    runner_summary = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    rows = [row for row in read_jsonl(args.output_dir / "raw.jsonl") if row.get("record_type") == "cog_temporal_pair"]
    train = dict(np.load(args.train_data_dir / "train_data.npz"))
    dev = dict(np.load(args.train_data_dir / "dev_data.npz"))
    v1_model, _ = load_v1_model(args.v1_checkpoint, torch.device("cpu"))
    v2_model, v2_checkpoint = load_temporal(args.output_dir / "model_best.pt", torch.device("cpu"))

    check(design["contract_id"] == CONTRACT_ID, "design contract")
    check(sha256(args.design) == EXPECTED_DESIGN_SHA256, "design hash")
    check(sha256(args.v1_checkpoint) == EXPECTED_V1_CHECKPOINT_SHA256, "v1 checkpoint hash")
    check(array_sha256(train["inputs"]) == EXPECTED_TRAIN_INPUT_SHA256, "train input hash")
    check(array_sha256(train["targets"]) == EXPECTED_TRAIN_TARGET_SHA256, "train target hash")
    check(array_sha256(dev["inputs"]) == EXPECTED_DEV_INPUT_SHA256, "dev input hash")
    check(array_sha256(dev["targets"]) == EXPECTED_DEV_TARGET_SHA256, "dev target hash")
    old = json.loads((Path("docs/research/persistent_context_v2_pushobj_cog_predictor_design.json")).read_text(encoding="utf-8"))
    used = set(old["train_segment_indices"] + old["dev_segment_indices"] + old["formal_segment_indices"] + old["excluded_cog_stage0_segments"])
    check(not (set(FORMAL_SEGMENTS) & used), "formal old-split overlap")
    best = min(v2_checkpoint["curve"], key=lambda row: row["true_context_mse"])
    check(v2_checkpoint["best_step"] == best["step"], "v2 checkpoint selection step")
    check(abs(v2_checkpoint["best_dev_true_context_mse"] - best["true_context_mse"]) <= 1e-12, "v2 checkpoint selection value")
    with torch.no_grad():
        zero = v2_model(torch.from_numpy(dev["inputs"][:128]), torch.zeros(128))
    zero_context_max_abs = float(zero.abs().max().item())
    check(zero_context_max_abs == 0.0, "zero context identity")
    check(len(rows) == 32, "formal count")
    check([int(row["segment_index"]) for row in rows] == list(FORMAL_SEGMENTS), "formal order")
    check(all(sum(float(row["factor_cog_x"]) == factor for row in rows) == 8 for factor in FORMAL_FACTORS), "factor balance")
    check(manifest.get("raw_sha256") == sha256(args.output_dir / "raw.jsonl"), "raw hash")
    check(manifest.get("v2_checkpoint_sha256") == sha256(args.output_dir / "model_best.pt"), "v2 checkpoint hash")

    env = make_env()
    execution_max_abs = prediction_max_abs = metric_max_abs = 0.0
    prediction_max_abs_by_policy = {}
    for row in rows:
        initial = np.asarray(row["initial_state"], dtype=np.float32)
        goal = np.asarray(row["goal_state"], dtype=np.float32)
        factor = float(row["factor_cog_x"])
        for name, policy in row["policies"].items():
            commands = np.asarray(policy["commands"], dtype=np.float32)
            replay = rollout_physics(env, row["shape"], initial, int(row["env_seed"]), commands, factor)
            stored_execution = np.asarray(policy["states"], dtype=np.float32)
            execution_max_abs = max(execution_max_abs, float(np.max(np.abs(replay - stored_execution))))
            recomputed_metrics = pose_metrics(replay, goal, 10)
            metric_max_abs = max(metric_max_abs, max(abs(recomputed_metrics[key] - policy["metrics"][key]) for key in recomputed_metrics))
            if name == "simulator_oracle":
                predicted = replay
            else:
                nominal = rollout_physics(env, row["shape"], initial, int(row["env_seed"]), commands, 0.0)
                x = torch.from_numpy(encode_trajectory(commands, nominal))[None]
                context = torch.tensor([float(policy["context_cog_x"])])
                model = v1_model if name == "v1_true_cog_context" else v2_model
                with torch.no_grad():
                    residual = model(x, context)[0].numpy()
                predicted = apply_residual(nominal, residual)
            stored_prediction = np.asarray(policy["predicted_states"], dtype=np.float32)
            policy_prediction_error = float(np.max(np.abs(predicted - stored_prediction)))
            prediction_max_abs = max(prediction_max_abs, policy_prediction_error)
            prediction_max_abs_by_policy[name] = max(prediction_max_abs_by_policy.get(name, 0.0), policy_prediction_error)
    check(execution_max_abs <= 1e-6, "execution replay")
    # Formal inference used CUDA/cuDNN GRU kernels; the independent audit uses
    # CPU GRU kernels. Their float32 recurrence accumulates small ulp-level
    # differences across ten steps. 0.005 is <1e-5 of the 512-pixel workspace.
    check(prediction_max_abs <= 5e-3, "prediction replay")
    check(metric_max_abs <= 1e-7, "metric replay")
    recomputed_summary = summarize(args.output_dir / "raw.jsonl")
    check(close_tree(recomputed_summary, runner_summary), "summary replay")
    audit = {
        "contract_id": CONTRACT_ID,
        "valid": not failures,
        "failures": failures,
        "zero_context_max_abs": zero_context_max_abs,
        "execution_replay_max_abs": execution_max_abs,
        "prediction_replay_max_abs": prediction_max_abs,
        "prediction_replay_max_abs_by_policy": prediction_max_abs_by_policy,
        "metric_replay_max_abs": metric_max_abs,
        "design_sha256": sha256(args.design),
        "v1_checkpoint_sha256": sha256(args.v1_checkpoint),
        "v2_checkpoint_sha256": sha256(args.output_dir / "model_best.pt"),
        "raw_sha256": sha256(args.output_dir / "raw.jsonl"),
        "recomputed_summary": recomputed_summary,
    }
    dump_json(args.output_dir / args.output_name, audit)
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["valid"] else 1)


if __name__ == "__main__":
    main()
