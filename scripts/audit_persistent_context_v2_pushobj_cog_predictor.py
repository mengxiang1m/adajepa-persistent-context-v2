"""Independent replay audit for the PushObj CoG-conditioned predictor experiment."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.pushobj_cog_predictor import (
    CONTRACT_ID,
    EXPECTED_DESIGN_SHA256,
    FORMAL_FACTORS,
    CoGFiLMResidual,
    apply_residual,
    encode_trajectory,
    summarize,
)
from research.persistent_context_v2.pushobj_cog_stage0 import array_sha256, rollout_physics
from research.persistent_context_v2.pushobj_rotation_stage0 import dump_json, make_env, pose_metrics, read_jsonl, sha256


def _close(a, b, atol=1e-7):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[key], b[key], atol) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) if math.isnan(a) or math.isnan(b) else abs(a - b) <= atol
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_predictor_design.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_predictor"))
    parser.add_argument("--output-name", default="independent_audit.json")
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((args.output_dir / "data_manifest.json").read_text(encoding="utf-8"))
    runner_summary = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    rows = [row for row in read_jsonl(args.output_dir / "raw.jsonl") if row.get("record_type") == "cog_predictor_pair"]
    checkpoint = torch.load(args.output_dir / "model_best.pt", map_location="cpu", weights_only=False)
    model = CoGFiLMResidual()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    train = dict(np.load(args.output_dir / "train_data.npz"))
    dev = dict(np.load(args.output_dir / "dev_data.npz"))
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    check(design.get("contract_id") == CONTRACT_ID, "design contract")
    check(sha256(args.design) == EXPECTED_DESIGN_SHA256, "design hash")
    train_ids, dev_ids, formal_ids = map(set, (design["train_segment_indices"], design["dev_segment_indices"], design["formal_segment_indices"]))
    excluded = set(design["excluded_cog_stage0_segments"])
    check(not (train_ids & dev_ids or train_ids & formal_ids or dev_ids & formal_ids), "split overlap")
    check(not ((train_ids | dev_ids | formal_ids) & excluded), "excluded overlap")
    check(len(train["inputs"]) == 96 * 16 * 5 and len(dev["inputs"]) == 24 * 8 * 4, "data counts")
    check(array_sha256(train["inputs"]) == data_manifest["train_input_sha256"], "train input hash")
    check(array_sha256(train["targets"]) == data_manifest["train_target_sha256"], "train target hash")
    check(array_sha256(dev["inputs"]) == data_manifest["dev_input_sha256"], "dev input hash")
    check(array_sha256(dev["targets"]) == data_manifest["dev_target_sha256"], "dev target hash")
    check(checkpoint["design_sha256"] == EXPECTED_DESIGN_SHA256, "checkpoint design")
    best = min(checkpoint["curve"], key=lambda row: row["true_context_mse"])
    check(checkpoint["best_step"] == best["step"], "checkpoint selection step")
    check(abs(checkpoint["best_dev_true_context_mse"] - best["true_context_mse"]) <= 1e-12, "checkpoint selection value")
    with torch.no_grad():
        zero = model(torch.from_numpy(dev["inputs"][:128]), torch.zeros(128))
    zero_context_max_abs = float(zero.abs().max().item())
    check(zero_context_max_abs == 0.0, "zero context identity")
    check(len(rows) == 32, "formal count")
    check(len({row["segment_index"] for row in rows}) == 32, "formal unique")
    check(all(sum(float(row["factor_cog_x"]) == f for row in rows) == 8 for f in FORMAL_FACTORS), "formal balance")
    check(manifest.get("raw_sha256") == sha256(args.output_dir / "raw.jsonl"), "raw hash")

    env = make_env()
    execution_max_abs = prediction_max_abs = metric_max_abs = 0.0
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
                predicted = rollout_physics(env, row["shape"], initial, int(row["env_seed"]), commands, factor)
            else:
                nominal = rollout_physics(env, row["shape"], initial, int(row["env_seed"]), commands, 0.0)
                x = torch.from_numpy(encode_trajectory(commands, nominal))[None]
                context = torch.tensor([float(policy["context_cog_x"])])
                with torch.no_grad():
                    residual = model(x, context)[0].numpy()
                predicted = apply_residual(nominal, residual)
            stored_prediction = np.asarray(policy["predicted_states"], dtype=np.float32)
            prediction_max_abs = max(prediction_max_abs, float(np.max(np.abs(predicted - stored_prediction))))
    check(execution_max_abs <= 1e-6, "execution replay")
    # Formal inference ran on CUDA while this independent audit deliberately
    # recomputes on CPU.  Float32 GEMM kernels differ by a few ulps.
    check(prediction_max_abs <= 1e-4, "prediction replay")
    check(metric_max_abs <= 1e-7, "metric replay")
    recomputed_summary = summarize(args.output_dir / "raw.jsonl")
    check(_close(recomputed_summary, runner_summary), "summary replay")

    audit = {
        "contract_id": CONTRACT_ID,
        "valid": not failures,
        "failures": failures,
        "zero_context_max_abs": zero_context_max_abs,
        "execution_replay_max_abs": execution_max_abs,
        "prediction_replay_max_abs": prediction_max_abs,
        "metric_replay_max_abs": metric_max_abs,
        "design_sha256": sha256(args.design),
        "checkpoint_sha256": sha256(args.output_dir / "model_best.pt"),
        "raw_sha256": sha256(args.output_dir / "raw.jsonl"),
        "recomputed_summary": recomputed_summary,
    }
    dump_json(args.output_dir / args.output_name, audit)
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["valid"] else 1)


if __name__ == "__main__":
    main()
