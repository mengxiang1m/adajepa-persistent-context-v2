"""Temporal single-FiLM residual predictor for hidden PushObj center of gravity."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.persistent_context_v2.pushobj_cog_predictor import (
    ANGLE_SCALE,
    CONTEXT_SCALE,
    CoGFiLMResidual,
    apply_residual,
    encode_trajectory,
    load_model as load_v1_model,
    plan_learned_cem,
    plan_simulator_oracle,
    prediction_mse,
    trajectory_pose_error,
)
from research.persistent_context_v2.pushobj_cog_stage0 import (
    array_sha256,
    prepare_waypoint_physics,
    rollout_physics,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    WINDOW,
    deadline_success,
    nominal_block_displacement_at_10,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl,
    dump_json,
    git_revision,
    make_env,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)


CONTRACT_ID = "persistent-context-v2-pushobj-cog-temporal-film-predictor-v2"
EXPECTED_DESIGN_SHA256 = "12b06a368fb7c25b9ecf716db6a65d71fccf4b664ed009d611b60d5165617413"
EXPECTED_V1_CHECKPOINT_SHA256 = "39bb54d90a863c012dbd05e9ea448d8e213966a1d47c5f116faebc6f0a06c403"
EXPECTED_TRAIN_INPUT_SHA256 = "8cd9fbabe9b596a8fabec275d5171765255fd4719d3f8d966eb9a9de80a2da43"
EXPECTED_TRAIN_TARGET_SHA256 = "84eced4e3dfec761c0eb399d77e18fdd525ee20f679cb36f1782669e96f8bd4e"
EXPECTED_DEV_INPUT_SHA256 = "47f3773811d0ce35e273b04b61a92bded2811df0b04c5fa8d38a84e6346c2ca4"
EXPECTED_DEV_TARGET_SHA256 = "b5b2893419139252fb5a25ecb8d2647729a93216c465bf7cf4c58fd44995bd07"
FORMAL_FACTORS = (-22.5, -7.5, 7.5, 22.5)
FORMAL_SEGMENTS = (
    187, 303, 41, 260, 479, 66, 149, 347,
    478, 492, 298, 477, 54, 270, 2, 153,
    264, 413, 403, 467, 312, 180, 15, 490,
    302, 224, 325, 166, 95, 5, 335, 229,
)
BOOTSTRAP_SEED = 1_270_300
BOOTSTRAP_RESAMPLES = 20_000


class TemporalCoGFiLMResidual(nn.Module):
    """Causal GRU over nominal transitions with exactly one FiLM context path."""

    def __init__(self, step_embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.step_encoder = nn.Sequential(nn.Linear(18, step_embedding_dim), nn.SiLU())
        self.temporal = nn.GRU(step_embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.context_film = nn.Linear(1, 2 * hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))

    @staticmethod
    def temporal_input(flat_trajectory: torch.Tensor) -> torch.Tensor:
        states = flat_trajectory[:, :88].reshape(-1, 11, 8)
        actions = flat_trajectory[:, 88:].reshape(-1, 10, 2)
        return torch.cat([states[:, :-1], states[:, 1:], actions], dim=-1)

    def _condition(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.context_film(context).chunk(2, dim=-1)
        return hidden * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, flat_trajectory: torch.Tensor, cog_x: torch.Tensor) -> torch.Tensor:
        steps = self.step_encoder(self.temporal_input(flat_trajectory))
        hidden, _ = self.temporal(steps)
        context = cog_x.reshape(-1, 1) / CONTEXT_SCALE
        zero = torch.zeros_like(context)
        residual = self.head(self._condition(hidden, context)) - self.head(self._condition(hidden, zero))
        return residual.reshape(-1, 30)


def verify_training_data(train: dict, dev: dict):
    checks = {
        "train_input": array_sha256(train["inputs"]) == EXPECTED_TRAIN_INPUT_SHA256,
        "train_target": array_sha256(train["targets"]) == EXPECTED_TRAIN_TARGET_SHA256,
        "dev_input": array_sha256(dev["inputs"]) == EXPECTED_DEV_INPUT_SHA256,
        "dev_target": array_sha256(dev["targets"]) == EXPECTED_DEV_TARGET_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen v1 data mismatch: {checks}")
    return checks


def train_temporal(train_data_dir: Path, output_dir: Path, device: torch.device, design_sha: str):
    train = dict(np.load(train_data_dir / "train_data.npz"))
    dev = dict(np.load(train_data_dir / "dev_data.npz"))
    data_checks = verify_training_data(train, dev)
    seed_all(1_270_200)
    model = TemporalCoGFiLMResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(1_270_201)
    train_x = torch.from_numpy(train["inputs"])
    train_y = torch.from_numpy(train["targets"])
    train_c = torch.from_numpy(train["contexts"])
    curve, best_mse, best_step, best_state = [], math.inf, -1, None
    for step in range(1, 3001):
        index = torch.randint(len(train_x), (256,), generator=generator)
        x, y, c = train_x[index].to(device), train_y[index].to(device), train_c[index].to(device)
        model.train()
        prediction = model(x, c)
        loss = torch.square(prediction - y).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        if step % 100 == 0:
            model.eval()
            metrics = prediction_mse(model, dev, device)
            row = {"step": step, "train_batch_mse": float(loss.item()), "gradient_norm_before_clip": gradient_norm, **metrics}
            curve.append(row)
            if metrics["true_context_mse"] < best_mse:
                best_mse = metrics["true_context_mse"]
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"TRAIN step={step} loss={loss.item():.6g} dev={metrics['true_context_mse']:.6g}", flush=True)
    if best_state is None:
        raise RuntimeError("no finite temporal checkpoint")
    checkpoint = {
        "contract_id": CONTRACT_ID,
        "design_sha256": design_sha,
        "model_state": best_state,
        "best_step": best_step,
        "best_dev_true_context_mse": best_mse,
        "curve": curve,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "data_checks": data_checks,
    }
    torch.save(checkpoint, output_dir / "model_best.pt")
    model.load_state_dict(best_state)
    model.eval()
    final = prediction_mse(model, dev, device)
    final.update({"best_step": best_step, "parameter_count": checkpoint["parameter_count"], "data_checks": data_checks})
    dump_json(output_dir / "training_summary.json", final)
    return final


def load_temporal(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TemporalCoGFiLMResidual().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def formal_scenarios(segments):
    rows = []
    for ordinal, segment_index in enumerate(FORMAL_SEGMENTS):
        displacement = nominal_block_displacement_at_10(segments[segment_index])
        if displacement < 10:
            raise RuntimeError("invalid temporal formal segment")
        rows.append({
            "ordinal": ordinal,
            "segment_index": int(segment_index),
            "factor_cog_x": FORMAL_FACTORS[ordinal // 8],
            "within_factor": ordinal % 8,
            "env_seed": 1_272_000 + ordinal,
            "cem_seed": 1_273_000 + ordinal,
            "nominal_block_displacement_at_10": float(displacement),
        })
    return rows


def _learned_policy(env, model, shape, initial_state, meta, nominal_commands, goal_state, context, device):
    commands, predicted, nominal_prediction, trace = plan_learned_cem(
        env, model, shape, initial_state, meta["env_seed"], nominal_commands, goal_state,
        context, meta["cem_seed"], device,
    )
    executed = rollout_physics(env, shape, initial_state, meta["env_seed"], commands, meta["factor_cog_x"])
    return {
        "context_cog_x": float(context), "commands": commands, "predicted_states": predicted,
        "nominal_prediction_states": nominal_prediction, "states": executed,
        "metrics": pose_metrics(executed, goal_state, WINDOW), "predicted_metrics": pose_metrics(predicted, goal_state, WINDOW),
        "deadline_success": deadline_success(executed, goal_state), "command_sha256": array_sha256(commands),
        "prediction_execution_pose_error": trajectory_pose_error(predicted, executed), "trace": trace,
    }


def run_formal(data_path: Path, output_dir: Path, v1_model, v2_model, device: torch.device, limit=32):
    with data_path.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    env = make_env()
    raw_path = output_dir / "raw.jsonl"
    completed = {int(row["ordinal"]) for row in read_jsonl(raw_path) if row.get("record_type") == "cog_temporal_pair"}
    for meta in formal_scenarios(segments)[:limit]:
        if meta["ordinal"] in completed:
            continue
        started = time.perf_counter()
        segment = segments[meta["segment_index"]]
        shape, initial_state, nominal_commands, nominal_states = prepare_waypoint_physics(env, segment, meta["env_seed"])
        goal_state = nominal_states[-1]
        policies = {
            "population_prior_context": _learned_policy(env, v2_model, shape, initial_state, meta, nominal_commands, goal_state, 0.0, device),
            "v1_true_cog_context": _learned_policy(env, v1_model, shape, initial_state, meta, nominal_commands, goal_state, meta["factor_cog_x"], device),
            "v2_temporal_true_cog_context": _learned_policy(env, v2_model, shape, initial_state, meta, nominal_commands, goal_state, meta["factor_cog_x"], device),
        }
        commands, predicted, trace = plan_simulator_oracle(
            env, shape, initial_state, meta["env_seed"], nominal_commands, goal_state,
            meta["factor_cog_x"], meta["cem_seed"],
        )
        executed = rollout_physics(env, shape, initial_state, meta["env_seed"], commands, meta["factor_cog_x"])
        policies["simulator_oracle"] = {
            "context_cog_x": float(meta["factor_cog_x"]), "commands": commands, "predicted_states": predicted,
            "states": executed, "metrics": pose_metrics(executed, goal_state, WINDOW), "predicted_metrics": pose_metrics(predicted, goal_state, WINDOW),
            "deadline_success": deadline_success(executed, goal_state), "command_sha256": array_sha256(commands),
            "prediction_execution_pose_error": trajectory_pose_error(predicted, executed),
            "prediction_execution_max_abs": float(np.max(np.abs(predicted - executed))), "trace": trace,
        }
        append_jsonl(raw_path, {
            "record_type": "cog_temporal_pair", "contract_id": CONTRACT_ID, **meta, "shape": shape,
            "initial_state": initial_state, "goal_state": goal_state, "nominal_commands": nominal_commands,
            "nominal_states": nominal_states, "policies": policies, "elapsed_s": time.perf_counter() - started,
        })
        print(
            f"FORMAL {meta['ordinal'] + 1}/32 factor={meta['factor_cog_x']:+g} "
            f"pop={policies['population_prior_context']['metrics']['pose_auc10']:.4f} "
            f"v1={policies['v1_true_cog_context']['metrics']['pose_auc10']:.4f} "
            f"v2={policies['v2_temporal_true_cog_context']['metrics']['pose_auc10']:.4f} "
            f"oracle={policies['simulator_oracle']['metrics']['pose_auc10']:.4f}", flush=True,
        )


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    return [float(x) for x in np.quantile(values[indexes].mean(axis=1), [0.025, 0.975])]


def _comparison(reference, treatment):
    delta = reference - treatment
    return {
        "mean_delta": float(delta.mean()),
        "relative_improvement": float(delta.mean() / reference.mean()),
        "bootstrap_ci95_delta": bootstrap_ci(delta),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)),
    }


def summarize(raw_path: Path):
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "cog_temporal_pair"]
    names = ("population_prior_context", "v1_true_cog_context", "v2_temporal_true_cog_context", "simulator_oracle")
    values = {name: np.asarray([row["policies"][name]["metrics"]["pose_auc10"] for row in rows]) for name in names}
    v1_delta = values[names[0]] - values[names[1]]
    v2_delta = values[names[0]] - values[names[2]]
    oracle_delta = values[names[0]] - values[names[3]]
    result = {
        "contract_id": CONTRACT_ID,
        "n_pairs": len(rows),
        "primary_metric": "pose_auc10_to_waypoint",
        "means": {name: float(value.mean()) for name, value in values.items()},
        "v1_vs_population": _comparison(values[names[0]], values[names[1]]),
        "v2_vs_population": _comparison(values[names[0]], values[names[2]]),
        "v2_vs_v1": _comparison(values[names[1]], values[names[2]]),
        "oracle_vs_population": _comparison(values[names[0]], values[names[3]]),
        "oracle_gap_recovery": {
            "v1": float(v1_delta.mean() / oracle_delta.mean()) if abs(oracle_delta.mean()) > 1e-12 else math.nan,
            "v2": float(v2_delta.mean() / oracle_delta.mean()) if abs(oracle_delta.mean()) > 1e-12 else math.nan,
        },
        "deadline_success": {name: float(np.mean([row["policies"][name]["deadline_success"] for row in rows])) for name in names},
        "prediction_execution_pose_error": {name: float(np.mean([row["policies"][name]["prediction_execution_pose_error"] for row in rows])) for name in names},
        "plan_changed_from_population_fraction": {
            name: float(np.mean([row["policies"][names[0]]["command_sha256"] != row["policies"][name]["command_sha256"] for row in rows]))
            for name in names[1:]
        },
        "by_factor": {},
    }
    for factor in FORMAL_FACTORS:
        mask = np.asarray([float(row["factor_cog_x"]) == factor for row in rows])
        result["by_factor"][str(factor)] = {
            "n": int(mask.sum()),
            "population_mean": float(values[names[0]][mask].mean()),
            "v1_mean": float(values[names[1]][mask].mean()),
            "v2_mean": float(values[names[2]][mask].mean()),
            "oracle_mean": float(values[names[3]][mask].mean()),
            "v1_relative_improvement": float(v1_delta[mask].mean() / values[names[0]][mask].mean()),
            "v2_relative_improvement": float(v2_delta[mask].mean() / values[names[0]][mask].mean()),
            "v2_positive_fraction": float(np.mean(v2_delta[mask] > 1e-12)),
        }
    result["structural_checks"] = {
        "complete": len(rows) == 32,
        "unique_segments": len({row["segment_index"] for row in rows}) == len(rows),
        "factor_balance": len(rows) == 32 and all(sum(float(row["factor_cog_x"]) == factor for row in rows) == 8 for factor in FORMAL_FACTORS),
        "waypoint_displacement": all(row["nominal_block_displacement_at_10"] >= 10 for row in rows),
        "v2_plan_changed": result["plan_changed_from_population_fraction"]["v2_temporal_true_cog_context"] > 0,
    }
    result["valid"] = all(result["structural_checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "formal", "summarize", "all"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_temporal_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_temporal_contract_zh.md"))
    parser.add_argument("--train-data-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_predictor"))
    parser.add_argument("--v1-checkpoint", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_predictor/model_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_temporal"))
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sha256(args.design) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen temporal design hash mismatch")
    if sha256(args.v1_checkpoint) != EXPECTED_V1_CHECKPOINT_SHA256:
        raise RuntimeError("frozen v1 checkpoint hash mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "contract_id": CONTRACT_ID, "git_revision": git_revision(), "design_path": str(args.design), "design_sha256": sha256(args.design),
        "contract_path": str(args.contract), "contract_sha256": sha256(args.contract), "data_path": str(args.data), "data_sha256": sha256(args.data),
        "v1_checkpoint_path": str(args.v1_checkpoint), "v1_checkpoint_sha256": sha256(args.v1_checkpoint),
        "command": " ".join(__import__("sys").argv), "started_unix": time.time(), "resource_start": resource_snapshot(device), "device": str(device),
    }
    dump_json(manifest_path, manifest)
    if args.mode in ("train", "all"):
        manifest["training"] = train_temporal(args.train_data_dir, args.output_dir, device, sha256(args.design))
        manifest["v2_checkpoint_sha256"] = sha256(args.output_dir / "model_best.pt")
        dump_json(manifest_path, manifest)
    if args.mode in ("formal", "all"):
        v1_model, v1_checkpoint = load_v1_model(args.v1_checkpoint, device)
        v2_model, v2_checkpoint = load_temporal(args.output_dir / "model_best.pt", device)
        if v1_checkpoint["design_sha256"] != "e09973efeaf0bd291a35cd0f4627888aace591134e05eb0a273cf05ff1947c1f":
            raise RuntimeError("v1 checkpoint parent mismatch")
        if v2_checkpoint["design_sha256"] != EXPECTED_DESIGN_SHA256:
            raise RuntimeError("v2 checkpoint design mismatch")
        run_formal(args.data, args.output_dir, v1_model, v2_model, device, args.limit)
    if args.mode in ("summarize", "all") and args.limit == 32:
        result = summarize(args.output_dir / "raw.jsonl")
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(device)
    if (args.output_dir / "raw.jsonl").exists():
        manifest["raw_sha256"] = sha256(args.output_dir / "raw.jsonl")
    dump_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
