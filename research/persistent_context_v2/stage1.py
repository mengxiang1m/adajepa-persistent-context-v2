"""Frozen Stage-1 history-value oracle for the V2 docking task."""

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

from .stage0 import FORMAL_FACTORS, TRAIN_FACTORS, population_prior


CONTRACT_ID = "persistent-context-v2-history-oracle-v1"
CONDITIONS = ("persistent", "no_persistence")
POLICIES = (
    "population_prior",
    "current_only",
    "correct_history",
    "shuffled_history",
    "wrong_sequence_history",
    "true_factor_oracle",
)


@dataclass(frozen=True)
class Stage1Config:
    n_sequences: int = 384
    n_episodes: int = 8
    tolerance: float = 0.20
    noise_std: float = 0.015
    target_min: float = 0.65
    target_max: float = 0.85
    action_limit: float = 1.5
    master_seed: int = 2026082211
    bootstrap_seed: int = 2026082212
    bootstrap_resamples: int = 20_000


class CategoricalPosterior:
    def __init__(self, factors: Sequence[float] = FORMAL_FACTORS, log_weights: np.ndarray = None, count: int = 0):
        self.factors = np.asarray(factors, dtype=np.float64)
        self.log_weights = np.full(len(self.factors), -np.log(len(self.factors))) if log_weights is None else np.asarray(log_weights, dtype=np.float64).copy()
        self.count = int(count)

    def copy(self) -> "CategoricalPosterior":
        return CategoricalPosterior(self.factors, self.log_weights, self.count)

    def mean(self) -> float:
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights /= weights.sum()
        return float(weights @ self.factors)

    def update(self, action: float, response: float, noise_std: float) -> None:
        residual = response - self.factors * action
        self.log_weights += -0.5 * (residual / noise_std) ** 2
        self.log_weights -= np.max(self.log_weights)
        self.log_weights -= np.log(np.exp(self.log_weights).sum())
        self.count += 1

    def digest(self) -> str:
        payload = np.concatenate([self.log_weights, np.asarray([self.count], dtype=np.float64)])
        return hashlib.sha256(payload.tobytes()).hexdigest()


def _scenario(config: Stage1Config, condition: str) -> Mapping[str, np.ndarray]:
    code = CONDITIONS.index(condition)
    # Factor lifetime is the condition treatment. Episode nuisance and noise
    # must be identical across condition-paired sequence indices for the DiD.
    factor_rng = np.random.default_rng(np.random.SeedSequence([config.master_seed, code, 1002]))
    task_rng = np.random.default_rng(np.random.SeedSequence([config.master_seed, 1000]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([config.master_seed, 1001]))
    if condition == "persistent":
        first = factor_rng.integers(len(FORMAL_FACTORS), size=config.n_sequences)
        factor_ids = np.repeat(first[:, None], config.n_episodes, axis=1)
    else:
        factor_ids = factor_rng.integers(len(FORMAL_FACTORS), size=(config.n_sequences, config.n_episodes))
    signs = np.where(task_rng.integers(2, size=(config.n_sequences, config.n_episodes)) == 0, -1.0, 1.0)
    targets = signs * task_rng.uniform(config.target_min, config.target_max, size=(config.n_sequences, config.n_episodes))
    noises = noise_rng.normal(0.0, config.noise_std, size=(config.n_sequences, config.n_episodes))
    return {"factor_ids": factor_ids, "targets": targets, "noises": noises}


def _sattolo(n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.arange(n)
    for index in range(n - 1, 0, -1):
        swap = int(rng.integers(index))
        values[index], values[swap] = values[swap], values[index]
    return values


def _history_donors(config: Stage1Config, condition: str) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([config.master_seed, 100 + CONDITIONS.index(condition)]))
    donors = np.empty((config.n_episodes, config.n_sequences), dtype=np.int64)
    donors[0] = np.arange(config.n_sequences)
    for episode in range(1, config.n_episodes):
        donors[episode] = _sattolo(config.n_sequences, rng)
    return donors


def _transition(config: Stage1Config, factor: float, target: float, noise: float, estimate: float) -> Mapping[str, float]:
    action = float(np.clip(target / estimate, -config.action_limit, config.action_limit))
    response = float(factor * action + noise)
    miss = response - target
    unsafe = abs(miss) > config.tolerance
    cost = (miss / config.tolerance) ** 2 + float(unsafe)
    return {"action": action, "response": response, "miss": miss, "early_task_cost": cost, "unsafe": int(unsafe)}


def _run_policy(config: Stage1Config, condition: str, policy: str, scenario: Mapping[str, np.ndarray], shuffled_donors: np.ndarray) -> List[MutableMapping[str, object]]:
    local = [CategoricalPosterior() for _ in range(config.n_sequences)]
    rows: List[MutableMapping[str, object]] = []
    wrong = np.roll(np.arange(config.n_sequences), -1)
    for episode in range(config.n_episodes):
        before = [item.copy() for item in local]
        for sequence_id in range(config.n_sequences):
            factor_id = int(scenario["factor_ids"][sequence_id, episode])
            factor = float(FORMAL_FACTORS[factor_id])
            target = float(scenario["targets"][sequence_id, episode])
            noise = float(scenario["noises"][sequence_id, episode])
            donor_id = sequence_id
            if policy in ("population_prior", "current_only"):
                context = CategoricalPosterior()
            elif policy == "correct_history":
                context = before[sequence_id]
            elif policy == "shuffled_history":
                donor_id = int(shuffled_donors[episode, sequence_id])
                context = before[donor_id]
            elif policy == "wrong_sequence_history":
                donor_id = int(wrong[sequence_id])
                context = before[donor_id]
            elif policy == "true_factor_oracle":
                context = None
            else:
                raise ValueError(policy)
            if policy == "true_factor_oracle":
                estimate = factor
                count_before = 0
                context_hash = "true-factor"
            elif policy == "population_prior":
                estimate = population_prior()
                count_before = 0
                context_hash = "population-prior"
            else:
                assert context is not None
                estimate = context.mean()
                count_before = context.count
                context_hash = context.digest()
            result = _transition(config, factor, target, noise, estimate)
            if policy == "current_only":
                updated = CategoricalPosterior()
                updated.update(result["action"], result["response"], config.noise_std)
                count_after = updated.count
                posterior_after = updated.mean()
            elif policy in ("correct_history", "shuffled_history", "wrong_sequence_history"):
                local[sequence_id].update(result["action"], result["response"], config.noise_std)
                count_after = local[sequence_id].count
                posterior_after = local[sequence_id].mean()
            else:
                count_after = 0
                posterior_after = estimate
            scenario_payload = np.asarray([factor, target, noise], dtype=np.float64)
            rows.append({
                "condition": condition,
                "sequence_id": sequence_id,
                "episode_id": episode + 1,
                "policy": policy,
                "factor_id": factor_id,
                "factor": factor,
                "target": target,
                "noise": noise,
                "donor_sequence_id": donor_id,
                "history_count_before": count_before,
                "history_count_after": count_after,
                "estimate_before": estimate,
                "estimate_after": posterior_after,
                **result,
                "context_hash_before": context_hash,
                "scenario_hash": hashlib.sha256(scenario_payload.tobytes()).hexdigest(),
            })
    return rows


def _bootstrap_ci(values: np.ndarray, config: Stage1Config, stream: int) -> Tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.bootstrap_seed, stream])))
    samples: List[np.ndarray] = []
    remaining = config.bootstrap_resamples
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples.append(values[indices].mean(axis=1))
        remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(samples), [0.025, 0.975]))


def _sequence_values(rows: Sequence[Mapping[str, object]], condition: str, field: str) -> Dict[str, np.ndarray]:
    result = {}
    for policy in POLICIES:
        values = []
        for sequence_id in range(max(int(row["sequence_id"]) for row in rows) + 1):
            selected = [float(row[field]) for row in rows if row["condition"] == condition and row["policy"] == policy and int(row["sequence_id"]) == sequence_id and int(row["episode_id"]) >= 2]
            values.append(float(np.mean(selected)))
        result[policy] = np.asarray(values, dtype=np.float64)
    return result


def _effect_summary(current: np.ndarray, treatment: np.ndarray, config: Stage1Config, stream: int) -> Mapping[str, object]:
    difference = current - treatment
    current_mean = float(current.mean())
    return {
        "current_mean": current_mean,
        "treatment_mean": float(treatment.mean()),
        "mean_difference": float(difference.mean()),
        "difference_ci95": list(_bootstrap_ci(difference, config, stream)),
        "relative_improvement": float(difference.mean() / current_mean),
        "positive_sequence_fraction": float(np.mean(difference > 0.0)),
    }


def _audit(rows: Sequence[Mapping[str, object]], config: Stage1Config) -> Mapping[str, object]:
    failures = []
    expected = len(CONDITIONS) * len(POLICIES) * config.n_sequences * config.n_episodes
    if len(rows) != expected:
        failures.append(f"row_count:{len(rows)}!={expected}")
    for condition in CONDITIONS:
        for sequence_id in range(config.n_sequences):
            pair = [row for row in rows if row["condition"] == condition and int(row["sequence_id"]) == sequence_id and int(row["episode_id"]) == 1 and row["policy"] in ("current_only", "correct_history")]
            if len(pair) != 2 or any(pair[0][key] != pair[1][key] for key in ("scenario_hash", "action", "response", "early_task_cost", "unsafe")):
                failures.append(f"{condition}:{sequence_id}:episode1_identity")
        persistent_ids = {}
        if condition == "persistent":
            for sequence_id in range(config.n_sequences):
                persistent_ids[sequence_id] = {int(row["factor_id"]) for row in rows if row["condition"] == condition and row["policy"] == "current_only" and int(row["sequence_id"]) == sequence_id}
            if any(len(items) != 1 for items in persistent_ids.values()):
                failures.append("persistent_factor_lifetime")
    bad_donors = [row for row in rows if int(row["episode_id"]) >= 2 and row["policy"] in ("shuffled_history", "wrong_sequence_history") and int(row["donor_sequence_id"]) == int(row["sequence_id"])]
    if bad_donors:
        failures.append(f"self_donors:{len(bad_donors)}")
    return {"passed": not failures, "failures": failures, "raw_row_count": len(rows), "expected_raw_row_count": expected}


def _summarize(rows: Sequence[Mapping[str, object]], config: Stage1Config, audit: Mapping[str, object]) -> Mapping[str, object]:
    costs = {condition: _sequence_values(rows, condition, "early_task_cost") for condition in CONDITIONS}
    unsafe = {condition: _sequence_values(rows, condition, "unsafe") for condition in CONDITIONS}
    effects = {}
    stream = 100
    for condition in CONDITIONS:
        effects[condition] = {}
        for policy in ("correct_history", "shuffled_history", "wrong_sequence_history", "true_factor_oracle"):
            effects[condition][policy] = _effect_summary(costs[condition]["current_only"], costs[condition][policy], config, stream)
            stream += 1
    persistent_delta = costs["persistent"]["current_only"] - costs["persistent"]["correct_history"]
    no_persistence_delta = costs["no_persistence"]["current_only"] - costs["no_persistence"]["correct_history"]
    did = persistent_delta - no_persistence_delta
    current_mean = float(costs["persistent"]["current_only"].mean())
    true_gap = current_mean - float(costs["persistent"]["true_factor_oracle"].mean())
    recovery = float(persistent_delta.mean() / true_gap)
    true_relative = effects["persistent"]["true_factor_oracle"]["relative_improvement"]
    true_relative_ci = tuple(value / current_mean for value in effects["persistent"]["true_factor_oracle"]["difference_ci95"])
    correct = effects["persistent"]["correct_history"]
    shuffled = effects["persistent"]["shuffled_history"]
    wrong = effects["persistent"]["wrong_sequence_history"]
    no_persist = effects["no_persistence"]["correct_history"]
    criteria = {
        "true_oracle_relative_improvement_at_least_25pct_and_ci_lower_20pct": true_relative >= 0.25 and true_relative_ci[0] >= 0.20,
        "persistent_correct_relative_at_least_30pct_and_ci_lower_above_20pct_current": correct["relative_improvement"] >= 0.30 and correct["difference_ci95"][0] > 0.20 * current_mean,
        "did_ci_lower_above_15pct_current": _bootstrap_ci(did, config, 900)[0] > 0.15 * current_mean,
        "history_recovers_at_least_half_true_gap": true_gap > 0.0 and recovery >= 0.50,
        "persistent_positive_sequence_fraction_at_least_75pct": correct["positive_sequence_fraction"] >= 0.75,
        "no_persistence_relative_improvement_at_most_5pct": no_persist["relative_improvement"] <= 0.05,
        "shuffled_and_wrong_not_comparable": shuffled["relative_improvement"] <= 0.05 and wrong["relative_improvement"] <= 0.05 and shuffled["relative_improvement"] < 0.5 * correct["relative_improvement"] and wrong["relative_improvement"] < 0.5 * correct["relative_improvement"],
        "all_sequences_and_audits_valid": bool(audit["passed"]),
    }
    if not audit["passed"]:
        verdict, disposition = "INVALID_EXECUTION", "REPAIR_ONLY"
    elif all(criteria.values()):
        verdict, disposition = "HISTORY_VALUE_SUPPORTED", "GO_EXPLICIT_CONTEXT_PILOT"
    else:
        verdict, disposition = "HISTORY_VALUE_NOT_ESTABLISHED", "NO_GO_CONTEXT_METHOD"
    return {
        "effects": effects,
        "persistent_current_mean_cost": current_mean,
        "persistent_true_oracle_mean_cost": float(costs["persistent"]["true_factor_oracle"].mean()),
        "persistent_population_prior_mean_cost": float(costs["persistent"]["population_prior"].mean()),
        "persistent_current_mean_unsafe": float(unsafe["persistent"]["current_only"].mean()),
        "persistent_correct_mean_unsafe": float(unsafe["persistent"]["correct_history"].mean()),
        "did": {"mean": float(did.mean()), "ci95": list(_bootstrap_ci(did, config, 900)), "relative_to_persistent_current": float(did.mean() / current_mean)},
        "true_oracle_gap": true_gap,
        "history_gap_recovery": recovery,
        "true_oracle_relative_improvement_ci95": list(true_relative_ci),
        "criteria": criteria,
        "decision": {"verdict": verdict, "disposition": disposition},
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage1(output_dir: Path, config: Stage1Config = Stage1Config(), command: str = "") -> Mapping[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: List[MutableMapping[str, object]] = []
    for condition in CONDITIONS:
        scenario = _scenario(config, condition)
        donors = _history_donors(config, condition)
        for policy in POLICIES:
            rows.extend(_run_policy(config, condition, policy, scenario, donors))
    raw_path = output_dir / "raw_results.csv"
    _write_csv(raw_path, rows)
    audit = _audit(rows, config)
    analysis = _summarize(rows, config, audit)
    summary = {
        "schema": "persistent-context-v2-stage1-summary-v1",
        "contract_id": CONTRACT_ID,
        "config": asdict(config),
        "train_factors": TRAIN_FACTORS,
        "formal_factors": FORMAL_FACTORS,
        "population_prior_mean": population_prior(),
        "bootstrap": {"algorithm": "paired sequence percentile bootstrap", "seed": config.bootstrap_seed, "resamples": config.bootstrap_resamples, "quantiles": [0.025, 0.975]},
        "audit": audit,
        **analysis,
        "wall_time_seconds": time.time() - started,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    contract_path = repo_root / "docs/research/persistent_context_v2_stage1_contract_zh.md"
    source_path = Path(__file__).resolve()
    manifest = {
        "schema": "persistent-context-v2-stage1-manifest-v1",
        "contract_id": CONTRACT_ID,
        "command": command,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "config": asdict(config),
        "contract_sha256": _sha256(contract_path),
        "source_sha256": _sha256(source_path),
        "raw_results_sha256": _sha256(raw_path),
        "summary_sha256": _sha256(summary_path),
        "started_unix": started,
        "finished_unix": time.time(),
        "deviations": ["repair1: persistent/no-persistence now share exact target and noise arrays; v1 unpaired-condition outputs were preserved and invalidated"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    return summary
