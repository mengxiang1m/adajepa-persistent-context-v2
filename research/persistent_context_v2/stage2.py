"""Minimal non-privileged Bayesian/RLS sequence-context pilot."""

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
from .stage1 import CategoricalPosterior, _sattolo


CONTRACT_ID = "persistent-context-v2-explicit-rls-v1"
CONDITIONS = ("persistent", "no_persistence")
POLICIES = (
    "population_prior", "current_only_rls", "persistent_rls", "shuffled_rls",
    "wrong_sequence_rls", "categorical_history_oracle", "true_factor_oracle",
)


@dataclass(frozen=True)
class Stage2Config:
    n_sequences: int = 384
    n_episodes: int = 8
    tolerance: float = 0.20
    noise_std: float = 0.015
    target_min: float = 0.65
    target_max: float = 0.85
    action_limit: float = 1.5
    master_seed: int = 2026082221
    bootstrap_seed: int = 2026082222
    bootstrap_resamples: int = 20_000


class RLSContext:
    """Three-scalar sufficient statistics with a train-calibrated prior."""

    def __init__(self, sum_u2: float = 0.0, sum_uy: float = 0.0, count: int = 0):
        self.sum_u2 = float(sum_u2)
        self.sum_uy = float(sum_uy)
        self.count = int(count)

    def copy(self) -> "RLSContext":
        return RLSContext(self.sum_u2, self.sum_uy, self.count)

    def mean(self, noise_std: float) -> float:
        prior_mean = population_prior()
        prior_variance = float(np.var(TRAIN_FACTORS))
        prior_precision = 1.0 / prior_variance
        observation_precision = 1.0 / (noise_std ** 2)
        numerator = prior_precision * prior_mean + observation_precision * self.sum_uy
        denominator = prior_precision + observation_precision * self.sum_u2
        return float(numerator / denominator)

    def update(self, action: float, response: float) -> None:
        self.sum_u2 += action * action
        self.sum_uy += action * response
        self.count += 1

    def digest(self) -> str:
        payload = np.asarray([self.sum_u2, self.sum_uy, self.count], dtype=np.float64)
        return hashlib.sha256(payload.tobytes()).hexdigest()


def _scenario(config: Stage2Config, condition: str) -> Mapping[str, np.ndarray]:
    code = CONDITIONS.index(condition)
    factor_rng = np.random.default_rng(np.random.SeedSequence([config.master_seed, code, 1002]))
    task_rng = np.random.default_rng(np.random.SeedSequence([config.master_seed, 1000]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([config.master_seed, 1001]))
    if condition == "persistent":
        ids = factor_rng.integers(len(FORMAL_FACTORS), size=config.n_sequences)
        factor_ids = np.repeat(ids[:, None], config.n_episodes, axis=1)
    else:
        factor_ids = factor_rng.integers(len(FORMAL_FACTORS), size=(config.n_sequences, config.n_episodes))
    signs = np.where(task_rng.integers(2, size=(config.n_sequences, config.n_episodes)) == 0, -1.0, 1.0)
    targets = signs * task_rng.uniform(config.target_min, config.target_max, size=(config.n_sequences, config.n_episodes))
    noises = noise_rng.normal(0.0, config.noise_std, size=(config.n_sequences, config.n_episodes))
    return {"factor_ids": factor_ids, "targets": targets, "noises": noises}


def _donors(config: Stage2Config, condition: str) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([config.master_seed, 100 + CONDITIONS.index(condition)]))
    result = np.empty((config.n_episodes, config.n_sequences), dtype=np.int64)
    result[0] = np.arange(config.n_sequences)
    for episode in range(1, config.n_episodes):
        result[episode] = _sattolo(config.n_sequences, rng)
    return result


def _transition(config: Stage2Config, factor: float, target: float, noise: float, estimate: float) -> Mapping[str, float]:
    action = float(np.clip(target / estimate, -config.action_limit, config.action_limit))
    response = float(factor * action + noise)
    miss = response - target
    unsafe = abs(miss) > config.tolerance
    return {"action": action, "response": response, "miss": miss, "early_task_cost": (miss / config.tolerance) ** 2 + float(unsafe), "unsafe": int(unsafe)}


def _run_policy(config: Stage2Config, condition: str, policy: str, scenario: Mapping[str, np.ndarray], donors: np.ndarray) -> List[MutableMapping[str, object]]:
    rls = [RLSContext() for _ in range(config.n_sequences)]
    categorical = [CategoricalPosterior() for _ in range(config.n_sequences)]
    wrong = np.roll(np.arange(config.n_sequences), -1)
    rows = []
    for episode in range(config.n_episodes):
        before_rls = [item.copy() for item in rls]
        before_cat = [item.copy() for item in categorical]
        for sequence_id in range(config.n_sequences):
            factor_id = int(scenario["factor_ids"][sequence_id, episode])
            factor = float(FORMAL_FACTORS[factor_id])
            target = float(scenario["targets"][sequence_id, episode])
            noise = float(scenario["noises"][sequence_id, episode])
            donor = sequence_id
            context_type = "none"
            if policy == "population_prior":
                estimate, count_before, context_hash = population_prior(), 0, "population-prior"
            elif policy == "current_only_rls":
                context = RLSContext()
                estimate, count_before, context_hash = context.mean(config.noise_std), 0, context.digest()
                context_type = "rls"
            elif policy == "persistent_rls":
                context = before_rls[sequence_id]
                estimate, count_before, context_hash = context.mean(config.noise_std), context.count, context.digest()
                context_type = "rls"
            elif policy == "shuffled_rls":
                donor = int(donors[episode, sequence_id])
                context = before_rls[donor]
                estimate, count_before, context_hash = context.mean(config.noise_std), context.count, context.digest()
                context_type = "rls"
            elif policy == "wrong_sequence_rls":
                donor = int(wrong[sequence_id])
                context = before_rls[donor]
                estimate, count_before, context_hash = context.mean(config.noise_std), context.count, context.digest()
                context_type = "rls"
            elif policy == "categorical_history_oracle":
                context_cat = before_cat[sequence_id]
                estimate, count_before, context_hash = context_cat.mean(), context_cat.count, context_cat.digest()
                context_type = "categorical"
            elif policy == "true_factor_oracle":
                estimate, count_before, context_hash = factor, 0, "true-factor"
            else:
                raise ValueError(policy)
            result = _transition(config, factor, target, noise, estimate)
            if policy == "current_only_rls":
                updated = RLSContext()
                updated.update(result["action"], result["response"])
                count_after, estimate_after = updated.count, updated.mean(config.noise_std)
            elif policy in ("persistent_rls", "shuffled_rls", "wrong_sequence_rls"):
                rls[sequence_id].update(result["action"], result["response"])
                count_after, estimate_after = rls[sequence_id].count, rls[sequence_id].mean(config.noise_std)
            elif policy == "categorical_history_oracle":
                categorical[sequence_id].update(result["action"], result["response"], config.noise_std)
                count_after, estimate_after = categorical[sequence_id].count, categorical[sequence_id].mean()
            else:
                count_after, estimate_after = 0, estimate
            scenario_payload = np.asarray([factor, target, noise], dtype=np.float64)
            rows.append({
                "condition": condition, "sequence_id": sequence_id, "episode_id": episode + 1,
                "policy": policy, "factor_id": factor_id, "factor": factor, "target": target,
                "noise": noise, "donor_sequence_id": donor, "context_type": context_type,
                "history_count_before": count_before, "history_count_after": count_after,
                "estimate_before": estimate, "estimate_after": estimate_after, **result,
                "context_hash_before": context_hash,
                "scenario_hash": hashlib.sha256(scenario_payload.tobytes()).hexdigest(),
            })
    return rows


def _bootstrap_ci(values: np.ndarray, config: Stage2Config, stream: int) -> Tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.bootstrap_seed, stream])))
    parts, remaining = [], config.bootstrap_resamples
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        parts.append(values[indices].mean(axis=1))
        remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(parts), [0.025, 0.975]))


def _sequence_costs(rows: Sequence[Mapping[str, object]], config: Stage2Config) -> Mapping[str, Mapping[str, np.ndarray]]:
    buckets = {(condition, policy, sequence_id): [] for condition in CONDITIONS for policy in POLICIES for sequence_id in range(config.n_sequences)}
    for row in rows:
        if int(row["episode_id"]) >= 2:
            buckets[(row["condition"], row["policy"], int(row["sequence_id"]))].append(float(row["early_task_cost"]))
    return {
        condition: {policy: np.asarray([np.mean(buckets[(condition, policy, sequence_id)]) for sequence_id in range(config.n_sequences)]) for policy in POLICIES}
        for condition in CONDITIONS
    }


def _audit(rows: Sequence[Mapping[str, object]], config: Stage2Config) -> Mapping[str, object]:
    failures = []
    expected = len(CONDITIONS) * len(POLICIES) * config.n_sequences * config.n_episodes
    if len(rows) != expected:
        failures.append(f"row_count:{len(rows)}!={expected}")
    lookup = {(row["condition"], int(row["sequence_id"]), int(row["episode_id"]), row["policy"]): row for row in rows}
    for condition in CONDITIONS:
        for sequence_id in range(config.n_sequences):
            left = lookup[(condition, sequence_id, 1, "current_only_rls")]
            right = lookup[(condition, sequence_id, 1, "persistent_rls")]
            if any(left[key] != right[key] for key in ("scenario_hash", "action", "response", "early_task_cost", "unsafe")):
                failures.append(f"{condition}:{sequence_id}:episode1")
    for row in rows:
        estimate = float(row["estimate_before"])
        if not np.isfinite(estimate) or estimate <= 0:
            failures.append(f"nonfinite:{row['condition']}:{row['sequence_id']}:{row['episode_id']}:{row['policy']}")
        if int(row["episode_id"]) >= 2 and row["policy"] in ("shuffled_rls", "wrong_sequence_rls") and int(row["donor_sequence_id"]) == int(row["sequence_id"]):
            failures.append(f"self_donor:{row['condition']}:{row['sequence_id']}:{row['episode_id']}:{row['policy']}")
    return {"passed": not failures, "failures": failures, "raw_row_count": len(rows), "expected_raw_row_count": expected}


def _summarize(rows: Sequence[Mapping[str, object]], config: Stage2Config, audit: Mapping[str, object]) -> Mapping[str, object]:
    costs = _sequence_costs(rows, config)
    p_current = costs["persistent"]["current_only_rls"]
    p_rls = costs["persistent"]["persistent_rls"]
    np_current = costs["no_persistence"]["current_only_rls"]
    np_rls = costs["no_persistence"]["persistent_rls"]
    p_delta, np_delta = p_current - p_rls, np_current - np_rls
    did = p_delta - np_delta
    current_mean = float(p_current.mean())
    categorical_delta = p_current - costs["persistent"]["categorical_history_oracle"]
    true_delta = p_current - costs["persistent"]["true_factor_oracle"]
    shuffled_delta = p_current - costs["persistent"]["shuffled_rls"]
    wrong_delta = p_current - costs["persistent"]["wrong_sequence_rls"]
    relative = float(p_delta.mean() / current_mean)
    no_relative = float(np_delta.mean() / np_current.mean())
    shuffled_relative = float(shuffled_delta.mean() / current_mean)
    wrong_relative = float(wrong_delta.mean() / current_mean)
    criteria = {
        "persistent_rls_relative_at_least_30pct_and_ci_lower_above_20pct_current": relative >= 0.30 and _bootstrap_ci(p_delta, config, 100)[0] > 0.20 * current_mean,
        "rls_did_ci_lower_above_15pct_current": _bootstrap_ci(did, config, 900)[0] > 0.15 * current_mean,
        "rls_recovers_80pct_categorical_and_50pct_true_gap": categorical_delta.mean() > 0 and p_delta.mean() / categorical_delta.mean() >= 0.80 and true_delta.mean() > 0 and p_delta.mean() / true_delta.mean() >= 0.50,
        "positive_sequence_fraction_at_least_75pct": float(np.mean(p_delta > 0)) >= 0.75,
        "no_persistence_relative_improvement_at_most_5pct": no_relative <= 0.05,
        "shuffled_and_wrong_not_comparable": shuffled_relative <= 0.05 and wrong_relative <= 0.05 and shuffled_relative < 0.5 * relative and wrong_relative < 0.5 * relative,
        "categorical_oracle_relative_improvement_at_least_30pct": float(categorical_delta.mean() / current_mean) >= 0.30,
        "all_sequences_and_audits_valid": bool(audit["passed"]),
    }
    criteria = {key: bool(value) for key, value in criteria.items()}
    if not audit["passed"]:
        verdict, disposition = "INVALID_EXECUTION", "REPAIR_ONLY"
    elif all(criteria.values()):
        verdict, disposition = "EXPLICIT_CONTEXT_SUPPORTED", "GO_CONTEXT_CONDITIONED_MODEL_DESIGN"
    else:
        verdict, disposition = "EXPLICIT_CONTEXT_NOT_ESTABLISHED", "NO_GO_CONTEXT_MODEL"
    effects = {
        "persistent_rls": {"current_mean": current_mean, "treatment_mean": float(p_rls.mean()), "mean_difference": float(p_delta.mean()), "difference_ci95": list(_bootstrap_ci(p_delta, config, 100)), "relative_improvement": relative, "positive_sequence_fraction": float(np.mean(p_delta > 0))},
        "no_persistence_rls": {"current_mean": float(np_current.mean()), "treatment_mean": float(np_rls.mean()), "mean_difference": float(np_delta.mean()), "relative_improvement": no_relative},
        "shuffled_rls_relative_improvement": shuffled_relative,
        "wrong_sequence_rls_relative_improvement": wrong_relative,
        "categorical_oracle_mean": float(costs["persistent"]["categorical_history_oracle"].mean()),
        "true_factor_oracle_mean": float(costs["persistent"]["true_factor_oracle"].mean()),
        "categorical_gap_recovery": float(p_delta.mean() / categorical_delta.mean()),
        "true_gap_recovery": float(p_delta.mean() / true_delta.mean()),
        "did": {"mean": float(did.mean()), "ci95": list(_bootstrap_ci(did, config, 900)), "relative_to_current": float(did.mean() / current_mean)},
    }
    return {"effects": effects, "criteria": criteria, "decision": {"verdict": verdict, "disposition": disposition}}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage2(output_dir: Path, config: Stage2Config = Stage2Config(), command: str = "") -> Mapping[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time(); rows = []
    for condition in CONDITIONS:
        scenario, donors = _scenario(config, condition), _donors(config, condition)
        for policy in POLICIES:
            rows.extend(_run_policy(config, condition, policy, scenario, donors))
    raw_path = output_dir / "raw_results.csv"; _write_csv(raw_path, rows)
    audit = _audit(rows, config); analysis = _summarize(rows, config, audit)
    summary = {"schema": "persistent-context-v2-stage2-summary-v1", "contract_id": CONTRACT_ID, "config": asdict(config), "train_factors": TRAIN_FACTORS, "formal_factors_audit_only": FORMAL_FACTORS, "population_prior_mean": population_prior(), "population_prior_variance": float(np.var(TRAIN_FACTORS)), "audit": audit, **analysis, "wall_time_seconds": time.time() - started}
    summary_path = output_dir / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]; contract_path = repo_root / "docs/research/persistent_context_v2_stage2_contract_zh.md"; source_path = Path(__file__).resolve(); stage1_path = source_path.with_name("stage1.py")
    manifest = {"schema": "persistent-context-v2-stage2-manifest-v1", "contract_id": CONTRACT_ID, "command": command, "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "config": asdict(config), "contract_sha256": _sha256(contract_path), "source_sha256": _sha256(source_path), "stage1_dependency_sha256": _sha256(stage1_path), "raw_results_sha256": _sha256(raw_path), "summary_sha256": _sha256(summary_path), "started_unix": started, "finished_unix": time.time(), "deviations": ["repair1: persistent/no-persistence now share exact target and noise arrays; v1 unpaired-condition outputs were preserved and invalidated"]}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    return summary
