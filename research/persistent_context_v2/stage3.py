"""Stage-3 FiLM-conditioned learned dynamics and closed-loop context gate."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .stage1 import _sattolo


CONTRACT_ID = "persistent-context-v2-film-world-model-v1"
TRAIN_FACTORS = (0.50, 0.65, 0.80, 0.95, 1.05, 1.20, 1.35, 1.50)
DEVELOPMENT_FACTORS = (0.575, 0.725, 0.875, 1.125, 1.275, 1.425)
FORMAL_FACTORS = (0.6125, 0.7625, 0.9125, 1.0875, 1.2375, 1.3875)
CONDITIONS = ("persistent", "no_persistence")
POLICIES = (
    "population_context",
    "current_only_context",
    "persistent_rls_context",
    "shuffled_rls_context",
    "wrong_sequence_rls_context",
    "true_context",
    "analytic_true_factor_oracle",
)


@dataclass(frozen=True)
class Stage3Config:
    train_samples: int = 32_768
    dev_prediction_samples: int = 8_192
    hidden_dim: int = 16
    train_steps: int = 2_000
    eval_interval: int = 100
    batch_size: int = 512
    learning_rate: float = 3e-3
    weight_decay: float = 1e-6
    train_data_seed: int = 2026082231
    dev_data_seed: int = 2026082232
    torch_seed: int = 2026082233
    dev_behavior_seed: int = 2026082234
    dev_bootstrap_seed: int = 2026082235
    dev_sequences: int = 256
    dev_bootstrap_resamples: int = 5_000
    formal_master_seed: int = 2026082241
    formal_bootstrap_seed: int = 2026082242
    formal_sequences: int = 384
    formal_bootstrap_resamples: int = 20_000
    n_episodes: int = 8
    tolerance: float = 0.20
    noise_std: float = 0.015
    target_min: float = 0.65
    target_max: float = 0.85
    action_limit: float = 1.5
    planner_candidates: int = 401


class FiLMScalarDynamics(nn.Module):
    """One FiLM operation; no adapter, router, token, or residual context path."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.action_projection = nn.Linear(1, hidden_dim)
        self.context_film = nn.Linear(1, 2 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 1)

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = self.action_projection(action)
        scale, shift = self.context_film(context).chunk(2, dim=-1)
        conditioned = features * (1.0 + scale) + shift
        return self.output_projection(conditioned)


class Stage3RLS:
    def __init__(self, sum_u2: float = 0.0, sum_uy: float = 0.0, count: int = 0):
        self.sum_u2, self.sum_uy, self.count = float(sum_u2), float(sum_uy), int(count)

    def copy(self) -> "Stage3RLS":
        return Stage3RLS(self.sum_u2, self.sum_uy, self.count)

    def mean(self, noise_std: float) -> float:
        prior_mean, prior_variance = float(np.mean(TRAIN_FACTORS)), float(np.var(TRAIN_FACTORS))
        prior_precision, observation_precision = 1.0 / prior_variance, 1.0 / noise_std ** 2
        return float((prior_precision * prior_mean + observation_precision * self.sum_uy) / (prior_precision + observation_precision * self.sum_u2))

    def update(self, action: float, response: float) -> None:
        self.sum_u2 += action * action
        self.sum_uy += action * response
        self.count += 1

    def digest(self) -> str:
        return hashlib.sha256(np.asarray([self.sum_u2, self.sum_uy, self.count], dtype=np.float64).tobytes()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(factors: Sequence[float], count: int, seed: int, config: Stage3Config) -> Mapping[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    factor_array = np.asarray(factors, dtype=np.float32)
    context = factor_array[rng.integers(len(factor_array), size=count)]
    action = rng.uniform(-config.action_limit, config.action_limit, size=count).astype(np.float32)
    noise = rng.normal(0.0, config.noise_std, size=count).astype(np.float32)
    response = context * action + noise
    return {"action": action[:, None], "context": context[:, None], "response": response[:, None]}


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def train_model(config: Stage3Config, device: torch.device) -> Tuple[FiLMScalarDynamics, List[Mapping[str, float]], Mapping[str, float]]:
    torch.manual_seed(config.torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.torch_seed)
    train = _dataset(TRAIN_FACTORS, config.train_samples, config.train_data_seed, config)
    dev = _dataset(DEVELOPMENT_FACTORS, config.dev_prediction_samples, config.dev_data_seed, config)
    train_tensors = {key: _tensor(value, device) for key, value in train.items()}
    dev_tensors = {key: _tensor(value, device) for key, value in dev.items()}
    model = FiLMScalarDynamics(config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.torch_seed + 1)
    curve: List[Mapping[str, float]] = []
    best_mse, best_step, best_state = float("inf"), -1, None
    model.train()
    for step in range(1, config.train_steps + 1):
        indices = torch.randint(0, config.train_samples, (config.batch_size,), generator=generator).to(device)
        predicted = model(train_tensors["action"][indices], train_tensors["context"][indices])
        loss = torch.mean((predicted - train_tensors["response"][indices]) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config.eval_interval == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                dev_true = model(dev_tensors["action"], dev_tensors["context"])
                dev_population = model(dev_tensors["action"], torch.ones_like(dev_tensors["context"]))
                true_mse = float(torch.mean((dev_true - dev_tensors["response"]) ** 2).item())
                population_mse = float(torch.mean((dev_population - dev_tensors["response"]) ** 2).item())
            curve.append({"step": step, "train_batch_mse": float(loss.item()), "dev_true_context_mse": true_mse, "dev_population_context_mse": population_mse})
            if true_mse < best_mse:
                best_mse, best_step = true_mse, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            model.train()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    diagnostics = {"best_step": best_step, "best_dev_true_context_mse": best_mse, "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
    return model, curve, diagnostics


@torch.no_grad()
def _plan_batch(model: FiLMScalarDynamics, targets: np.ndarray, contexts: np.ndarray, config: Stage3Config, device: torch.device, analytic_factors: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(-config.action_limit, config.action_limit, config.planner_candidates, dtype=np.float32)
    n = len(targets)
    candidates = np.broadcast_to(grid[None, :], (n, len(grid)))
    if analytic_factors is None:
        action_tensor = _tensor(candidates.reshape(-1, 1), device)
        context_tensor = _tensor(np.repeat(contexts.astype(np.float32), len(grid))[:, None], device)
        predictions = model(action_tensor, context_tensor).reshape(n, len(grid)).detach().cpu().numpy()
    else:
        predictions = analytic_factors[:, None] * candidates
    errors = (predictions - targets[:, None]) ** 2
    indices = np.argmin(errors, axis=1)
    chosen_actions = candidates[np.arange(n), indices].astype(np.float64)
    chosen_predictions = predictions[np.arange(n), indices].astype(np.float64)
    chosen_errors = errors[np.arange(n), indices].astype(np.float64)
    return chosen_actions, chosen_predictions, chosen_errors


def _behavior_scenarios(factors: Sequence[float], count: int, seed: int, config: Stage3Config) -> Mapping[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    factor_ids = rng.integers(len(factors), size=count)
    signs = np.where(rng.integers(2, size=count) == 0, -1.0, 1.0)
    targets = signs * rng.uniform(config.target_min, config.target_max, size=count)
    noises = rng.normal(0.0, config.noise_std, size=count)
    return {"factor_ids": factor_ids, "factors": np.asarray(factors)[factor_ids], "targets": targets, "noises": noises}


def _cost(response: np.ndarray, target: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    miss = response - target
    unsafe = np.abs(miss) > tolerance
    return (miss / tolerance) ** 2 + unsafe.astype(np.float64), unsafe


def _bootstrap_ci(values: np.ndarray, resamples: int, seed: int, stream: int) -> Tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    parts, remaining = [], resamples
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        parts.append(values[indices].mean(axis=1))
        remaining -= count
    return tuple(float(item) for item in np.quantile(np.concatenate(parts), [0.025, 0.975]))


def _bootstrap_relative_ci(current: np.ndarray, treatment: np.ndarray, resamples: int, seed: int, stream: int) -> Tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    parts, remaining = [], resamples
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(current), size=(count, len(current)))
        current_mean, treatment_mean = current[indices].mean(axis=1), treatment[indices].mean(axis=1)
        parts.append((current_mean - treatment_mean) / current_mean)
        remaining -= count
    return tuple(float(item) for item in np.quantile(np.concatenate(parts), [0.025, 0.975]))


def development_gate(model: FiLMScalarDynamics, config: Stage3Config, device: torch.device) -> Tuple[Mapping[str, object], List[Mapping[str, object]]]:
    prediction = _dataset(DEVELOPMENT_FACTORS, config.dev_prediction_samples, config.dev_data_seed, config)
    with torch.no_grad():
        actions = _tensor(prediction["action"], device)
        true_contexts = _tensor(prediction["context"], device)
        responses = _tensor(prediction["response"], device)
        true_pred = model(actions, true_contexts)
        pop_pred = model(actions, torch.ones_like(true_contexts))
        true_mse = float(torch.mean((true_pred - responses) ** 2).item())
        pop_mse = float(torch.mean((pop_pred - responses) ** 2).item())
    scenarios = _behavior_scenarios(DEVELOPMENT_FACTORS, config.dev_sequences, config.dev_behavior_seed, config)
    pop_actions, pop_predictions, pop_pred_errors = _plan_batch(model, scenarios["targets"], np.ones(config.dev_sequences), config, device)
    true_actions, true_predictions, true_pred_errors = _plan_batch(model, scenarios["targets"], scenarios["factors"], config, device)
    pop_response = scenarios["factors"] * pop_actions + scenarios["noises"]
    true_response = scenarios["factors"] * true_actions + scenarios["noises"]
    pop_cost, pop_unsafe = _cost(pop_response, scenarios["targets"], config.tolerance)
    true_cost, true_unsafe = _cost(true_response, scenarios["targets"], config.tolerance)
    relative = float((pop_cost.mean() - true_cost.mean()) / pop_cost.mean())
    relative_ci = _bootstrap_relative_ci(pop_cost, true_cost, config.dev_bootstrap_resamples, config.dev_bootstrap_seed, 10)
    action_change = float(np.mean(np.abs(pop_actions - true_actions) > (3.0 / (config.planner_candidates - 1)) / 2.0))
    criteria = {
        "true_prediction_mse_at_most_10pct_population": true_mse <= 0.10 * pop_mse,
        "true_behavior_relative_at_least_50pct_and_ci_lower_40pct": relative >= 0.50 and relative_ci[0] >= 0.40,
        "population_unsafe_between_25_and_75pct": 0.25 <= float(pop_unsafe.mean()) <= 0.75,
        "planner_action_change_at_least_80pct": action_change >= 0.80,
        "finite_and_paired_audit": all(np.isfinite(array).all() for array in (pop_actions, true_actions, pop_predictions, true_predictions, pop_cost, true_cost)),
    }
    criteria = {key: bool(value) for key, value in criteria.items()}
    rows = []
    for index in range(config.dev_sequences):
        scenario_hash = hashlib.sha256(np.asarray([scenarios["factors"][index], scenarios["targets"][index], scenarios["noises"][index]], dtype=np.float64).tobytes()).hexdigest()
        for policy, action, predicted, pred_error, response, cost, unsafe in (
            ("population_context", pop_actions[index], pop_predictions[index], pop_pred_errors[index], pop_response[index], pop_cost[index], pop_unsafe[index]),
            ("true_context", true_actions[index], true_predictions[index], true_pred_errors[index], true_response[index], true_cost[index], true_unsafe[index]),
        ):
            rows.append({"sequence_id": index, "policy": policy, "factor": scenarios["factors"][index], "target": scenarios["targets"][index], "noise": scenarios["noises"][index], "context": 1.0 if policy == "population_context" else scenarios["factors"][index], "action": action, "predicted_response": predicted, "predicted_squared_error": pred_error, "response": response, "early_task_cost": cost, "unsafe": int(unsafe), "scenario_hash": scenario_hash})
    return {
        "prediction_mse": {"population_context": pop_mse, "true_context": true_mse, "ratio": true_mse / pop_mse},
        "behavior": {"population_mean_cost": float(pop_cost.mean()), "true_mean_cost": float(true_cost.mean()), "relative_improvement": relative, "relative_improvement_ci95": list(relative_ci), "population_unsafe_fraction": float(pop_unsafe.mean()), "true_unsafe_fraction": float(true_unsafe.mean()), "action_change_fraction": action_change},
        "criteria": criteria,
        "passed": all(criteria.values()),
    }, rows


def _formal_scenario(config: Stage3Config, condition: str) -> Mapping[str, np.ndarray]:
    code = CONDITIONS.index(condition)
    factor_rng = np.random.default_rng(np.random.SeedSequence([config.formal_master_seed, code, 1002]))
    task_rng = np.random.default_rng(np.random.SeedSequence([config.formal_master_seed, 1000]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([config.formal_master_seed, 1001]))
    if condition == "persistent":
        first = factor_rng.integers(len(FORMAL_FACTORS), size=config.formal_sequences)
        factor_ids = np.repeat(first[:, None], config.n_episodes, axis=1)
    else:
        factor_ids = factor_rng.integers(len(FORMAL_FACTORS), size=(config.formal_sequences, config.n_episodes))
    signs = np.where(task_rng.integers(2, size=(config.formal_sequences, config.n_episodes)) == 0, -1.0, 1.0)
    targets = signs * task_rng.uniform(config.target_min, config.target_max, size=(config.formal_sequences, config.n_episodes))
    noises = noise_rng.normal(0.0, config.noise_std, size=(config.formal_sequences, config.n_episodes))
    return {"factor_ids": factor_ids, "targets": targets, "noises": noises}


def _donors(config: Stage3Config, condition: str) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([config.formal_master_seed, 100 + CONDITIONS.index(condition)]))
    result = np.empty((config.n_episodes, config.formal_sequences), dtype=np.int64)
    result[0] = np.arange(config.formal_sequences)
    for episode in range(1, config.n_episodes):
        result[episode] = _sattolo(config.formal_sequences, rng)
    return result


def _run_policy(model: FiLMScalarDynamics, config: Stage3Config, device: torch.device, condition: str, policy: str, scenario: Mapping[str, np.ndarray], donors: np.ndarray) -> List[MutableMapping[str, object]]:
    local = [Stage3RLS() for _ in range(config.formal_sequences)]
    wrong = np.roll(np.arange(config.formal_sequences), -1)
    rows: List[MutableMapping[str, object]] = []
    for episode in range(config.n_episodes):
        before = [item.copy() for item in local]
        factors = np.asarray(FORMAL_FACTORS)[scenario["factor_ids"][:, episode]]
        targets, noises = scenario["targets"][:, episode], scenario["noises"][:, episode]
        contexts = np.empty(config.formal_sequences, dtype=np.float64)
        donor_ids = np.arange(config.formal_sequences)
        counts = np.zeros(config.formal_sequences, dtype=np.int64)
        hashes = ["none"] * config.formal_sequences
        for sequence_id in range(config.formal_sequences):
            if policy in ("population_context", "current_only_context"):
                context = Stage3RLS()
            elif policy == "persistent_rls_context":
                context = before[sequence_id]
            elif policy == "shuffled_rls_context":
                donor_ids[sequence_id] = donors[episode, sequence_id]
                context = before[int(donor_ids[sequence_id])]
            elif policy == "wrong_sequence_rls_context":
                donor_ids[sequence_id] = wrong[sequence_id]
                context = before[int(donor_ids[sequence_id])]
            elif policy in ("true_context", "analytic_true_factor_oracle"):
                context = None
            else:
                raise ValueError(policy)
            if policy == "population_context":
                contexts[sequence_id], hashes[sequence_id] = float(np.mean(TRAIN_FACTORS)), "population-prior"
            elif policy == "true_context" or policy == "analytic_true_factor_oracle":
                contexts[sequence_id], hashes[sequence_id] = factors[sequence_id], "true-factor"
            else:
                assert context is not None
                contexts[sequence_id], counts[sequence_id], hashes[sequence_id] = context.mean(config.noise_std), context.count, context.digest()
        if policy == "analytic_true_factor_oracle":
            actions, predictions, pred_errors = _plan_batch(model, targets, contexts, config, device, analytic_factors=factors)
        else:
            actions, predictions, pred_errors = _plan_batch(model, targets, contexts, config, device)
        responses = factors * actions + noises
        costs, unsafe = _cost(responses, targets, config.tolerance)
        estimates_after = contexts.copy()
        counts_after = counts.copy()
        for sequence_id in range(config.formal_sequences):
            if policy == "current_only_context":
                updated = Stage3RLS(); updated.update(actions[sequence_id], responses[sequence_id])
                estimates_after[sequence_id], counts_after[sequence_id] = updated.mean(config.noise_std), updated.count
            elif policy in ("persistent_rls_context", "shuffled_rls_context", "wrong_sequence_rls_context"):
                local[sequence_id].update(actions[sequence_id], responses[sequence_id])
                estimates_after[sequence_id], counts_after[sequence_id] = local[sequence_id].mean(config.noise_std), local[sequence_id].count
        for sequence_id in range(config.formal_sequences):
            scenario_hash = hashlib.sha256(np.asarray([factors[sequence_id], targets[sequence_id], noises[sequence_id]], dtype=np.float64).tobytes()).hexdigest()
            rows.append({"condition": condition, "sequence_id": sequence_id, "episode_id": episode + 1, "policy": policy, "factor_id": int(scenario["factor_ids"][sequence_id, episode]), "factor": factors[sequence_id], "target": targets[sequence_id], "noise": noises[sequence_id], "context_before": contexts[sequence_id], "context_after": estimates_after[sequence_id], "context_hash_before": hashes[sequence_id], "donor_sequence_id": int(donor_ids[sequence_id]), "history_count_before": int(counts[sequence_id]), "history_count_after": int(counts_after[sequence_id]), "planner_candidates": config.planner_candidates, "predicted_response_at_action": predictions[sequence_id], "predicted_squared_target_error": pred_errors[sequence_id], "action": actions[sequence_id], "response": responses[sequence_id], "miss": responses[sequence_id] - targets[sequence_id], "early_task_cost": costs[sequence_id], "unsafe": int(unsafe[sequence_id]), "scenario_hash": scenario_hash})
    return rows


def _sequence_values(rows: Sequence[Mapping[str, object]], config: Stage3Config, field: str) -> Mapping[str, Mapping[str, np.ndarray]]:
    buckets = {(condition, policy, sequence_id): [] for condition in CONDITIONS for policy in POLICIES for sequence_id in range(config.formal_sequences)}
    for row in rows:
        if int(row["episode_id"]) >= 2:
            buckets[(row["condition"], row["policy"], int(row["sequence_id"]))].append(float(row[field]))
    return {condition: {policy: np.asarray([np.mean(buckets[(condition, policy, sequence_id)]) for sequence_id in range(config.formal_sequences)]) for policy in POLICIES} for condition in CONDITIONS}


def _formal_audit(rows: Sequence[Mapping[str, object]], config: Stage3Config) -> Mapping[str, object]:
    failures = []
    expected = len(CONDITIONS) * len(POLICIES) * config.formal_sequences * config.n_episodes
    if len(rows) != expected:
        failures.append(f"row_count:{len(rows)}!={expected}")
    lookup = {(row["condition"], row["policy"], int(row["sequence_id"]), int(row["episode_id"])): row for row in rows}
    for sequence_id in range(config.formal_sequences):
        for condition in CONDITIONS:
            current, persistent = lookup[(condition, "current_only_context", sequence_id, 1)], lookup[(condition, "persistent_rls_context", sequence_id, 1)]
            if any(current[key] != persistent[key] for key in ("scenario_hash", "context_before", "action", "predicted_response_at_action", "response", "early_task_cost", "unsafe")):
                failures.append(f"episode1:{condition}:{sequence_id}")
        for episode in range(1, config.n_episodes + 1):
            left, right = lookup[("persistent", "current_only_context", sequence_id, episode)], lookup[("no_persistence", "current_only_context", sequence_id, episode)]
            if left["target"] != right["target"] or left["noise"] != right["noise"]:
                failures.append(f"condition_pair:{sequence_id}:{episode}")
    for row in rows:
        if int(row["planner_candidates"]) != config.planner_candidates or not all(np.isfinite(float(row[key])) for key in ("context_before", "predicted_response_at_action", "action", "response", "early_task_cost")):
            failures.append(f"budget_or_finite:{row['condition']}:{row['policy']}:{row['sequence_id']}:{row['episode_id']}")
        if int(row["episode_id"]) >= 2 and row["policy"] in ("shuffled_rls_context", "wrong_sequence_rls_context") and int(row["donor_sequence_id"]) == int(row["sequence_id"]):
            failures.append(f"self_donor:{row['condition']}:{row['policy']}:{row['sequence_id']}:{row['episode_id']}")
    return {"passed": not failures, "failures": failures, "raw_row_count": len(rows), "expected_raw_row_count": expected}


def _formal_summary(rows: Sequence[Mapping[str, object]], config: Stage3Config, audit: Mapping[str, object]) -> Mapping[str, object]:
    costs = _sequence_values(rows, config, "early_task_cost")
    pc = costs["persistent"]["current_only_context"]
    pop = costs["persistent"]["population_context"]
    rls = costs["persistent"]["persistent_rls_context"]
    true = costs["persistent"]["true_context"]
    analytic = costs["persistent"]["analytic_true_factor_oracle"]
    npc = costs["no_persistence"]["current_only_context"]
    nprls = costs["no_persistence"]["persistent_rls_context"]
    rls_delta, no_delta = pc - rls, npc - nprls
    did = rls_delta - no_delta
    true_delta, analytic_delta = pop - true, pop - analytic
    shuffled_delta = pc - costs["persistent"]["shuffled_rls_context"]
    wrong_delta = pc - costs["persistent"]["wrong_sequence_rls_context"]
    current_mean, population_mean = float(pc.mean()), float(pop.mean())
    rls_relative, no_relative = float(rls_delta.mean() / current_mean), float(no_delta.mean() / npc.mean())
    shuffled_relative, wrong_relative = float(shuffled_delta.mean() / current_mean), float(wrong_delta.mean() / current_mean)
    lookup = {(row["condition"], row["policy"], int(row["sequence_id"]), int(row["episode_id"])): row for row in rows}
    action_changes = []
    for sequence_id in range(config.formal_sequences):
        for episode in range(2, config.n_episodes + 1):
            pop_row, true_row = lookup[("persistent", "population_context", sequence_id, episode)], lookup[("persistent", "true_context", sequence_id, episode)]
            action_changes.append(abs(float(pop_row["action"]) - float(true_row["action"])) > (3.0 / (config.planner_candidates - 1)) / 2.0)
    action_change = float(np.mean(action_changes))
    rls_ci = _bootstrap_ci(rls_delta, config.formal_bootstrap_resamples, config.formal_bootstrap_seed, 100)
    true_ci = _bootstrap_ci(true_delta, config.formal_bootstrap_resamples, config.formal_bootstrap_seed, 200)
    did_ci = _bootstrap_ci(did, config.formal_bootstrap_resamples, config.formal_bootstrap_seed, 900)
    true_recovery = float(true_delta.mean() / analytic_delta.mean())
    rls_true_recovery = float(rls_delta.mean() / true_delta.mean())
    criteria = {
        "true_context_relative_at_least_50pct_and_ci_lower_above_40pct_population": float(true_delta.mean() / population_mean) >= 0.50 and true_ci[0] > 0.40 * population_mean,
        "true_context_recovers_at_least_90pct_analytic_gap": analytic_delta.mean() > 0 and true_recovery >= 0.90,
        "persistent_rls_relative_at_least_50pct_and_ci_lower_above_40pct_current": rls_relative >= 0.50 and rls_ci[0] > 0.40 * current_mean,
        "rls_recovers_at_least_90pct_true_context_improvement": true_delta.mean() > 0 and rls_true_recovery >= 0.90,
        "rls_did_ci_lower_above_30pct_current": did_ci[0] > 0.30 * current_mean,
        "positive_sequence_fraction_at_least_80pct": float(np.mean(rls_delta > 0)) >= 0.80,
        "negative_controls_not_comparable": no_relative <= 0.05 and shuffled_relative <= 0.05 and wrong_relative <= 0.05 and shuffled_relative < 0.5 * rls_relative and wrong_relative < 0.5 * rls_relative,
        "true_vs_population_action_change_at_least_80pct": action_change >= 0.80,
        "all_sequences_and_audits_valid": bool(audit["passed"]),
    }
    criteria = {key: bool(value) for key, value in criteria.items()}
    if not audit["passed"]:
        verdict, disposition = "INVALID_EXECUTION", "REPAIR_ONLY"
    elif all(criteria.values()):
        verdict, disposition = "CONTEXT_CONDITIONED_WORLD_MODEL_SUPPORTED", "GO_REAL_BENCHMARK_TRANSFER_CONTRACT"
    else:
        verdict, disposition = "CONTEXT_CONDITIONED_WORLD_MODEL_NOT_ESTABLISHED", "NO_GO_TRANSFER"
    return {
        "means": {policy: float(costs["persistent"][policy].mean()) for policy in POLICIES},
        "persistent_rls": {"relative_improvement": rls_relative, "mean_difference": float(rls_delta.mean()), "difference_ci95": list(rls_ci), "positive_sequence_fraction": float(np.mean(rls_delta > 0)), "true_context_gap_recovery": rls_true_recovery},
        "true_context": {"relative_improvement": float(true_delta.mean() / population_mean), "mean_difference": float(true_delta.mean()), "difference_ci95": list(true_ci), "analytic_gap_recovery": true_recovery},
        "did": {"mean": float(did.mean()), "ci95": list(did_ci), "relative_to_current": float(did.mean() / current_mean)},
        "negative_controls": {"no_persistence_relative_improvement": no_relative, "shuffled_relative_improvement": shuffled_relative, "wrong_sequence_relative_improvement": wrong_relative},
        "action_change_fraction_true_vs_population": action_change,
        "criteria": criteria,
        "decision": {"verdict": verdict, "disposition": disposition},
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_stage3(output_dir: Path, config: Stage3Config = Stage3Config(), command: str = "", device_name: str = "auto") -> Mapping[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else (device_name if device_name != "auto" else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    model, curve, training = train_model(config, device)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "config": asdict(config), "train_factors": TRAIN_FACTORS, "development_factors": DEVELOPMENT_FACTORS}, checkpoint_path)
    _write_csv(output_dir / "training_curve.csv", curve)
    dev, dev_rows = development_gate(model, config, device)
    _write_csv(output_dir / "development_raw.csv", dev_rows)
    formal_generated = False
    formal = None
    formal_audit = None
    formal_rows: List[MutableMapping[str, object]] = []
    if dev["passed"]:
        for condition in CONDITIONS:
            scenario, donors = _formal_scenario(config, condition), _donors(config, condition)
            for policy in POLICIES:
                formal_rows.extend(_run_policy(model, config, device, condition, policy, scenario, donors))
        _write_csv(output_dir / "formal_raw.csv", formal_rows)
        formal_audit = _formal_audit(formal_rows, config)
        formal = _formal_summary(formal_rows, config, formal_audit)
        formal_generated = True
        decision = formal["decision"]
    else:
        decision = {"verdict": "MODEL_CONTEXT_USE_NOT_ESTABLISHED", "disposition": "STOP_BEFORE_FORMAL"}
    torch_resource = {"device": str(device), "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0}
    (output_dir / "torch_resource.json").write_text(json.dumps(torch_resource, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"schema": "persistent-context-v2-stage3-summary-v1", "contract_id": CONTRACT_ID, "config": asdict(config), "splits": {"train_factors": TRAIN_FACTORS, "development_factors": DEVELOPMENT_FACTORS, "formal_factors": FORMAL_FACTORS}, "training": training, "development_gate": dev, "formal_outcomes_generated": formal_generated, "formal_audit": formal_audit, "formal": formal, "decision": decision, "torch_resource": torch_resource, "wall_time_seconds": time.time() - started}
    summary_path = output_dir / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]; contract_path = repo_root / "docs/research/persistent_context_v2_stage3_contract_zh.md"; source_path = Path(__file__).resolve()
    manifest = {"schema": "persistent-context-v2-stage3-manifest-v1", "contract_id": CONTRACT_ID, "command": command, "device": str(device), "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "config": asdict(config), "contract_sha256": _sha256(contract_path), "source_sha256": _sha256(source_path), "checkpoint_sha256": _sha256(checkpoint_path), "development_raw_sha256": _sha256(output_dir / "development_raw.csv"), "formal_raw_sha256": _sha256(output_dir / "formal_raw.csv") if formal_generated else None, "summary_sha256": _sha256(summary_path), "started_unix": started, "finished_unix": time.time(), "deviations": []}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    return summary
