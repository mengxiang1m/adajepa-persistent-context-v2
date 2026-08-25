"""Factor-diverse FiLM residual trajectory predictor for hidden PushObj CoG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.persistent_context_v2.pushobj_cog_stage0 import (
    ACTION_COUNT,
    COG_Y,
    INITIAL_SIGMA,
    NUM_SAMPLES,
    OPT_STEPS,
    POPULATION_PRIOR_COG_X,
    TOPK,
    array_sha256,
    pose_auc10,
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


CONTRACT_ID = "persistent-context-v2-pushobj-cog-film-residual-predictor-v1"
EXPECTED_DESIGN_SHA256 = "e09973efeaf0bd291a35cd0f4627888aace591134e05eb0a273cf05ff1947c1f"
TRAIN_FACTORS = (-30.0, -15.0, 0.0, 15.0, 30.0)
DEV_FACTORS = (-25.0, -10.0, 10.0, 25.0)
FORMAL_FACTORS = (-22.5, -7.5, 7.5, 22.5)
TRAIN_VARIANTS = 16
DEV_VARIANTS = 8
NOISE_SIGMAS = (0.08, 0.16, 0.24, 0.32)
CONTEXT_SCALE = 30.0
POSITION_SCALE = 20.0
ANGLE_SCALE = math.pi / 9.0
BOOTSTRAP_SEED = 1_170_300
BOOTSTRAP_RESAMPLES = 20_000


def signed_angle_delta(a, b):
    return np.arctan2(np.sin(np.asarray(a) - np.asarray(b)), np.cos(np.asarray(a) - np.asarray(b)))


def encode_trajectory(commands: np.ndarray, nominal_states: np.ndarray) -> np.ndarray:
    states = np.asarray(nominal_states, dtype=np.float32)
    commands = np.asarray(commands, dtype=np.float32)
    if states.shape != (ACTION_COUNT + 1, 7) or commands.shape != (ACTION_COUNT, 2):
        raise ValueError(f"unexpected shapes: states={states.shape}, commands={commands.shape}")
    encoded = np.concatenate(
        [
            (states[:, 0:4] - 256.0) / 256.0,
            np.sin(states[:, 4:5]),
            np.cos(states[:, 4:5]),
            states[:, 5:7] / 500.0,
        ],
        axis=1,
    )
    return np.concatenate([encoded.reshape(-1), commands.reshape(-1)]).astype(np.float32)


def residual_target(true_states: np.ndarray, nominal_states: np.ndarray) -> np.ndarray:
    true_states = np.asarray(true_states, dtype=np.float32)
    nominal_states = np.asarray(nominal_states, dtype=np.float32)
    position = (true_states[1:, 2:4] - nominal_states[1:, 2:4]) / POSITION_SCALE
    angle = signed_angle_delta(true_states[1:, 4], nominal_states[1:, 4])[:, None] / ANGLE_SCALE
    return np.concatenate([position, angle], axis=1).reshape(-1).astype(np.float32)


def apply_residual(nominal_states: np.ndarray, residual: np.ndarray) -> np.ndarray:
    states = np.asarray(nominal_states, dtype=np.float32).copy()
    correction = np.asarray(residual, dtype=np.float32).reshape(ACTION_COUNT, 3)
    states[1:, 2:4] += correction[:, :2] * POSITION_SCALE
    states[1:, 4] = np.mod(states[1:, 4] + correction[:, 2] * ANGLE_SCALE, 2.0 * np.pi)
    return states


class CoGFiLMResidual(nn.Module):
    """One FiLM path; subtracting the zero-context branch gives exact prior identity."""

    def __init__(self, input_dim: int = 108, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.context_film = nn.Linear(1, 2 * hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, ACTION_COUNT * 3))

    def _condition(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.context_film(context).chunk(2, dim=-1)
        return features * (1.0 + scale) + shift

    def forward(self, trajectory: torch.Tensor, cog_x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(trajectory)
        context = cog_x.reshape(-1, 1) / CONTEXT_SCALE
        zero = torch.zeros_like(context)
        return self.head(self._condition(features, context)) - self.head(self._condition(features, zero))


def _action_variants(nominal: np.ndarray, count: int, rng: np.random.Generator):
    yield np.asarray(nominal, dtype=np.float32)
    for index in range(1, count):
        sigma = NOISE_SIGMAS[(index - 1) % len(NOISE_SIGMAS)]
        yield (np.asarray(nominal, dtype=np.float32) + rng.normal(0.0, sigma, size=nominal.shape)).astype(np.float32)


def generate_split(env, segments, indices, factors, variants, seed):
    rng = np.random.default_rng(seed)
    inputs, targets, contexts, segment_ids, variant_ids = [], [], [], [], []
    for segment_id in indices:
        segment = segments[int(segment_id)]
        shape, initial_state, nominal_commands, _ = prepare_waypoint_physics(env, segment, 1_171_000 + int(segment_id))
        for variant_id, commands in enumerate(_action_variants(nominal_commands, variants, rng)):
            nominal_states = rollout_physics(env, shape, initial_state, 1_171_000 + int(segment_id), commands, 0.0)
            encoded = encode_trajectory(commands, nominal_states)
            for factor in factors:
                true_states = rollout_physics(env, shape, initial_state, 1_171_000 + int(segment_id), commands, factor)
                inputs.append(encoded)
                targets.append(residual_target(true_states, nominal_states))
                contexts.append(factor)
                segment_ids.append(segment_id)
                variant_ids.append(variant_id)
    return {
        "inputs": np.asarray(inputs, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "contexts": np.asarray(contexts, dtype=np.float32),
        "segment_ids": np.asarray(segment_ids, dtype=np.int32),
        "variant_ids": np.asarray(variant_ids, dtype=np.int16),
    }


def generate_data(data_path: Path, design: dict, output_dir: Path):
    with data_path.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    env = make_env()
    train = generate_split(env, segments, design["train_segment_indices"], TRAIN_FACTORS, TRAIN_VARIANTS, 1_170_100)
    dev = generate_split(env, segments, design["dev_segment_indices"], DEV_FACTORS, DEV_VARIANTS, 1_170_101)
    np.savez_compressed(output_dir / "train_data.npz", **train)
    np.savez_compressed(output_dir / "dev_data.npz", **dev)
    payload = {
        "train_samples": len(train["inputs"]),
        "dev_samples": len(dev["inputs"]),
        "train_factors": list(TRAIN_FACTORS),
        "dev_factors": list(DEV_FACTORS),
        "train_input_sha256": array_sha256(train["inputs"]),
        "train_target_sha256": array_sha256(train["targets"]),
        "dev_input_sha256": array_sha256(dev["inputs"]),
        "dev_target_sha256": array_sha256(dev["targets"]),
    }
    dump_json(output_dir / "data_manifest.json", payload)
    return payload


@torch.no_grad()
def prediction_mse(model, data, device, batch_size=1024):
    total_true = total_pop = count = 0.0
    for start in range(0, len(data["inputs"]), batch_size):
        end = min(start + batch_size, len(data["inputs"]))
        x = torch.from_numpy(data["inputs"][start:end]).to(device)
        y = torch.from_numpy(data["targets"][start:end]).to(device)
        c = torch.from_numpy(data["contexts"][start:end]).to(device)
        pred = model(x, c)
        total_true += float(torch.square(pred - y).sum().item())
        total_pop += float(torch.square(y).sum().item())
        count += float(y.numel())
    return {"true_context_mse": total_true / count, "population_context_mse": total_pop / count}


def train_model(output_dir: Path, device: torch.device, design_sha: str):
    train = dict(np.load(output_dir / "train_data.npz"))
    dev = dict(np.load(output_dir / "dev_data.npz"))
    seed_all(1_170_200)
    model = CoGFiLMResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(1_170_201)
    curve, best_mse, best_step, best_state = [], math.inf, -1, None
    train_x = torch.from_numpy(train["inputs"])
    train_y = torch.from_numpy(train["targets"])
    train_c = torch.from_numpy(train["contexts"])
    for step in range(1, 3001):
        index = torch.randint(len(train_x), (256,), generator=generator)
        x, y, c = train_x[index].to(device), train_y[index].to(device), train_c[index].to(device)
        model.train()
        pred = model(x, c)
        loss = torch.square(pred - y).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            model.eval()
            metrics = prediction_mse(model, dev, device)
            row = {"step": step, "train_batch_mse": float(loss.item()), **metrics}
            curve.append(row)
            if metrics["true_context_mse"] < best_mse:
                best_mse, best_step = metrics["true_context_mse"], step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"TRAIN step={step} loss={loss.item():.6g} dev={metrics['true_context_mse']:.6g}", flush=True)
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    checkpoint = {
        "contract_id": CONTRACT_ID,
        "design_sha256": design_sha,
        "model_state": best_state,
        "best_step": best_step,
        "best_dev_true_context_mse": best_mse,
        "curve": curve,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    torch.save(checkpoint, output_dir / "model_best.pt")
    model.load_state_dict(best_state)
    model.eval()
    final = prediction_mse(model, dev, device)
    final.update({"best_step": best_step, "parameter_count": checkpoint["parameter_count"]})
    dump_json(output_dir / "training_summary.json", final)
    return final


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CoGFiLMResidual().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_batch(env, model, shape, initial_state, env_seed, candidates, context, device):
    nominal, inputs = [], []
    for commands in np.asarray(candidates):
        states = rollout_physics(env, shape, initial_state, env_seed, commands, 0.0)
        nominal.append(states)
        inputs.append(encode_trajectory(commands, states))
    nominal = np.asarray(nominal, dtype=np.float32)
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(device)
        c = torch.full((len(x),), float(context), dtype=torch.float32, device=device)
        residual = model(x, c).cpu().numpy()
    predicted = np.stack([apply_residual(states, correction) for states, correction in zip(nominal, residual)])
    return predicted, nominal


def _pose_auc_batch(states: np.ndarray, goal_state: np.ndarray):
    position = np.linalg.norm(states[:, 1:, 2:4] - goal_state[None, None, 2:4], axis=2)
    angle = np.abs(signed_angle_delta(states[:, 1:, 4], goal_state[4]))
    return (position / 20.0 + angle / ANGLE_SCALE).mean(axis=1)


def trajectory_pose_error(predicted: np.ndarray, executed: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    executed = np.asarray(executed, dtype=np.float64)
    position = np.linalg.norm(predicted[1:, 2:4] - executed[1:, 2:4], axis=1)
    angle = np.abs(signed_angle_delta(predicted[1:, 4], executed[1:, 4]))
    return float(np.mean(position / 20.0 + angle / ANGLE_SCALE))


def plan_learned_cem(env, model, shape, initial_state, env_seed, nominal_commands, goal_state, context, cem_seed, device):
    rng = np.random.default_rng(cem_seed)
    mu = np.asarray(nominal_commands, dtype=np.float64).copy()
    sigma = np.full_like(mu, INITIAL_SIGMA)
    trace = []
    for iteration in range(OPT_STEPS):
        candidates = rng.normal(mu, sigma, size=(NUM_SAMPLES, ACTION_COUNT, 2))
        candidates[0] = mu
        predicted, _ = predict_batch(env, model, shape, initial_state, env_seed, candidates, context, device)
        losses = _pose_auc_batch(predicted, goal_state)
        elite_index = np.argsort(losses)[:TOPK]
        elite = candidates[elite_index]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0, ddof=1), 1e-4)
        trace.append({"iteration": iteration, "best_loss": float(losses[elite_index[0]]), "mean_loss": float(losses.mean())})
    commands = mu.astype(np.float32)
    predicted, nominal = predict_batch(env, model, shape, initial_state, env_seed, commands[None], context, device)
    return commands, predicted[0], nominal[0], trace


def plan_simulator_oracle(env, shape, initial_state, env_seed, nominal_commands, goal_state, factor, cem_seed):
    rng = np.random.default_rng(cem_seed)
    mu = np.asarray(nominal_commands, dtype=np.float64).copy()
    sigma = np.full_like(mu, INITIAL_SIGMA)
    trace = []
    for iteration in range(OPT_STEPS):
        candidates = rng.normal(mu, sigma, size=(NUM_SAMPLES, ACTION_COUNT, 2))
        candidates[0] = mu
        losses = np.asarray([pose_auc10(rollout_physics(env, shape, initial_state, env_seed, c, factor), goal_state) for c in candidates])
        elite_index = np.argsort(losses)[:TOPK]
        elite = candidates[elite_index]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0, ddof=1), 1e-4)
        trace.append({"iteration": iteration, "best_loss": float(losses[elite_index[0]]), "mean_loss": float(losses.mean())})
    commands = mu.astype(np.float32)
    predicted = rollout_physics(env, shape, initial_state, env_seed, commands, factor)
    return commands, predicted, trace


def formal_scenarios(design, segments):
    rows = []
    for ordinal, segment_index in enumerate(design["formal_segment_indices"]):
        displacement = nominal_block_displacement_at_10(segments[segment_index])
        if displacement < 10:
            raise RuntimeError("invalid early-waypoint formal segment")
        rows.append({
            "ordinal": ordinal,
            "segment_index": int(segment_index),
            "factor_cog_x": FORMAL_FACTORS[ordinal // 8],
            "within_factor": ordinal % 8,
            "env_seed": 1_172_000 + ordinal,
            "cem_seed": 1_173_000 + ordinal,
            "nominal_block_displacement_at_10": float(displacement),
        })
    return rows


def run_formal(data_path: Path, design: dict, output_dir: Path, model, device, limit=32):
    with data_path.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    env = make_env()
    raw_path = output_dir / "raw.jsonl"
    complete = {int(row["ordinal"]) for row in read_jsonl(raw_path) if row.get("record_type") == "cog_predictor_pair"}
    for meta in formal_scenarios(design, segments)[:limit]:
        if meta["ordinal"] in complete:
            continue
        started = time.perf_counter()
        segment = segments[meta["segment_index"]]
        shape, initial_state, nominal_commands, nominal_states = prepare_waypoint_physics(env, segment, meta["env_seed"])
        goal_state = nominal_states[-1]
        policies = {}
        for name, context in (("population_prior_context", 0.0), ("true_cog_context", meta["factor_cog_x"])):
            commands, predicted, nominal_prediction, trace = plan_learned_cem(
                env, model, shape, initial_state, meta["env_seed"], nominal_commands, goal_state,
                context, meta["cem_seed"], device,
            )
            executed = rollout_physics(env, shape, initial_state, meta["env_seed"], commands, meta["factor_cog_x"])
            policies[name] = {
                "context_cog_x": float(context), "commands": commands, "predicted_states": predicted,
                "nominal_prediction_states": nominal_prediction, "states": executed,
                "metrics": pose_metrics(executed, goal_state, WINDOW), "predicted_metrics": pose_metrics(predicted, goal_state, WINDOW),
                "deadline_success": deadline_success(executed, goal_state), "command_sha256": array_sha256(commands),
                "prediction_execution_pose_error": trajectory_pose_error(predicted, executed), "trace": trace,
            }
        commands, predicted, trace = plan_simulator_oracle(
            env, shape, initial_state, meta["env_seed"], nominal_commands, goal_state,
            meta["factor_cog_x"], meta["cem_seed"],
        )
        executed = rollout_physics(env, shape, initial_state, meta["env_seed"], commands, meta["factor_cog_x"])
        policies["simulator_oracle"] = {
            "context_cog_x": float(meta["factor_cog_x"]), "commands": commands, "predicted_states": predicted,
            "states": executed, "metrics": pose_metrics(executed, goal_state, WINDOW),
            "predicted_metrics": pose_metrics(predicted, goal_state, WINDOW), "deadline_success": deadline_success(executed, goal_state),
            "command_sha256": array_sha256(commands), "prediction_execution_max_abs": float(np.max(np.abs(predicted - executed))), "trace": trace,
        }
        row = {
            "record_type": "cog_predictor_pair", "contract_id": CONTRACT_ID, **meta, "shape": shape,
            "initial_state": initial_state, "goal_state": goal_state, "nominal_commands": nominal_commands,
            "nominal_states": nominal_states, "policies": policies, "elapsed_s": time.perf_counter() - started,
        }
        append_jsonl(raw_path, row)
        print(f"FORMAL {meta['ordinal'] + 1}/32 factor={meta['factor_cog_x']:+g} pop={policies['population_prior_context']['metrics']['pose_auc10']:.4f} learned={policies['true_cog_context']['metrics']['pose_auc10']:.4f} oracle={policies['simulator_oracle']['metrics']['pose_auc10']:.4f}", flush=True)


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    return [float(x) for x in np.quantile(values[indexes].mean(axis=1), [0.025, 0.975])]


def summarize(raw_path: Path):
    rows = [row for row in read_jsonl(raw_path) if row.get("record_type") == "cog_predictor_pair"]
    names = ("population_prior_context", "true_cog_context", "simulator_oracle")
    values = {name: np.asarray([row["policies"][name]["metrics"]["pose_auc10"] for row in rows]) for name in names}
    learned_delta = values[names[0]] - values[names[1]]
    oracle_delta = values[names[0]] - values[names[2]]
    result = {
        "contract_id": CONTRACT_ID, "n_pairs": len(rows), "primary_metric": "pose_auc10_to_waypoint",
        "means": {name: float(value.mean()) for name, value in values.items()},
        "learned_vs_population": {
            "mean_delta": float(learned_delta.mean()), "relative_improvement": float(learned_delta.mean() / values[names[0]].mean()),
            "bootstrap_ci95_delta": bootstrap_ci(learned_delta), "positive_fraction": float(np.mean(learned_delta > 1e-12)),
            "tie_fraction": float(np.mean(np.abs(learned_delta) <= 1e-12)), "negative_fraction": float(np.mean(learned_delta < -1e-12)),
        },
        "oracle_vs_population": {
            "mean_delta": float(oracle_delta.mean()), "relative_improvement": float(oracle_delta.mean() / values[names[0]].mean()),
            "bootstrap_ci95_delta": bootstrap_ci(oracle_delta), "positive_fraction": float(np.mean(oracle_delta > 1e-12)),
        },
        "oracle_gap_recovery": float(learned_delta.mean() / oracle_delta.mean()) if abs(oracle_delta.mean()) > 1e-12 else math.nan,
        "deadline_success": {name: float(np.mean([row["policies"][name]["deadline_success"] for row in rows])) for name in names},
        "prediction_execution_pose_error": {
            name: float(np.mean([trajectory_pose_error(np.asarray(row["policies"][name]["predicted_states"]), np.asarray(row["policies"][name]["states"])) for row in rows]))
            for name in names
        },
        "plan_changed_fraction": float(np.mean([row["policies"][names[0]]["command_sha256"] != row["policies"][names[1]]["command_sha256"] for row in rows])),
        "by_factor": {},
    }
    for factor in FORMAL_FACTORS:
        mask = np.asarray([float(row["factor_cog_x"]) == factor for row in rows])
        delta = learned_delta[mask]
        result["by_factor"][str(factor)] = {
            "n": int(mask.sum()), "population_mean": float(values[names[0]][mask].mean()), "learned_mean": float(values[names[1]][mask].mean()),
            "oracle_mean": float(values[names[2]][mask].mean()), "learned_mean_delta": float(delta.mean()),
            "learned_relative_improvement": float(delta.mean() / values[names[0]][mask].mean()), "learned_positive_fraction": float(np.mean(delta > 1e-12)),
        }
    result["structural_checks"] = {
        "complete": len(rows) == 32, "unique_segments": len({row["segment_index"] for row in rows}) == len(rows),
        "factor_balance": len(rows) == 32 and all(sum(float(row["factor_cog_x"]) == f for row in rows) == 8 for f in FORMAL_FACTORS),
        "waypoint_displacement": all(row["nominal_block_displacement_at_10"] >= 10 for row in rows),
        "plan_changed": result["plan_changed_fraction"] > 0,
    }
    result["valid"] = all(result["structural_checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "train", "formal", "summarize", "all"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_predictor_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_cog_predictor_contract_zh.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_cog_predictor"))
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sha256(args.design) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen design hash mismatch")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "contract_id": CONTRACT_ID, "git_revision": git_revision(), "design_path": str(args.design), "design_sha256": sha256(args.design),
        "contract_path": str(args.contract), "contract_sha256": sha256(args.contract), "data_path": str(args.data), "data_sha256": sha256(args.data),
        "command": " ".join(__import__("sys").argv), "started_unix": time.time(), "resource_start": resource_snapshot(device), "device": str(device),
    }
    dump_json(manifest_path, manifest)
    if args.mode in ("generate", "all"):
        manifest["generated_data"] = generate_data(args.data, design, args.output_dir)
        dump_json(manifest_path, manifest)
    if args.mode in ("train", "all"):
        manifest["training"] = train_model(args.output_dir, device, sha256(args.design))
        manifest["checkpoint_sha256"] = sha256(args.output_dir / "model_best.pt")
        dump_json(manifest_path, manifest)
    if args.mode in ("formal", "all"):
        model, checkpoint = load_model(args.output_dir / "model_best.pt", device)
        if checkpoint["design_sha256"] != sha256(args.design):
            raise RuntimeError("checkpoint design mismatch")
        run_formal(args.data, design, args.output_dir, model, device, args.limit)
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
