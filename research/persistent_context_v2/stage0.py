"""Frozen Stage-0 dynamic-range calibration for the V2 docking task."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


CONTRACT_ID = "persistent-context-v2-stage0-docking-v1"
TRAIN_FACTORS = (0.55, 0.70, 0.85, 1.15, 1.30, 1.45)
DEVELOPMENT_FACTORS = (0.60, 0.80, 1.20, 1.40)
FORMAL_FACTORS = (0.575, 0.725, 0.90, 1.10, 1.275, 1.425)
CANDIDATE_TOLERANCES = (0.20, 0.15, 0.10)
POLICIES = ("population_prior", "true_factor_oracle")


@dataclass(frozen=True)
class Stage0Config:
    n_sequences: int = 512
    noise_std: float = 0.015
    target_min: float = 0.65
    target_max: float = 0.85
    action_limit: float = 1.5
    master_seed: int = 2026082201
    bootstrap_seed: int = 2026082202
    bootstrap_resamples: int = 20_000


def population_prior() -> float:
    return float(np.mean(TRAIN_FACTORS))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_hash(values: Sequence[float]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _scenario(config: Stage0Config, sequence_id: int) -> Mapping[str, float]:
    seed = np.random.SeedSequence([config.master_seed, sequence_id])
    factor_seed, task_seed, noise_seed = seed.spawn(3)
    factor_rng = np.random.default_rng(factor_seed)
    task_rng = np.random.default_rng(task_seed)
    noise_rng = np.random.default_rng(noise_seed)
    factor = DEVELOPMENT_FACTORS[int(factor_rng.integers(len(DEVELOPMENT_FACTORS)))]
    sign = -1.0 if task_rng.integers(2) == 0 else 1.0
    target = sign * task_rng.uniform(config.target_min, config.target_max)
    noise = noise_rng.normal(0.0, config.noise_std)
    return {"factor": float(factor), "target": float(target), "noise": float(noise)}


def _row(config: Stage0Config, tolerance: float, sequence_id: int, policy: str, scenario: Mapping[str, float]) -> Dict[str, object]:
    estimate = population_prior() if policy == "population_prior" else scenario["factor"]
    action = float(np.clip(scenario["target"] / estimate, -config.action_limit, config.action_limit))
    response = float(scenario["factor"] * action + scenario["noise"])
    miss = response - scenario["target"]
    unsafe = abs(miss) > tolerance
    cost = (miss / tolerance) ** 2 + float(unsafe)
    scenario_payload = np.asarray([scenario["factor"], scenario["target"], scenario["noise"]], dtype=np.float64)
    return {
        "candidate_tolerance": tolerance,
        "sequence_id": sequence_id,
        "policy": policy,
        "factor": scenario["factor"],
        "target": scenario["target"],
        "noise": scenario["noise"],
        "estimate": estimate,
        "action": action,
        "response": response,
        "miss": miss,
        "early_task_cost": cost,
        "unsafe": int(unsafe),
        "scenario_hash": hashlib.sha256(scenario_payload.tobytes()).hexdigest(),
    }


def _bootstrap_relative_ci(prior: np.ndarray, oracle: np.ndarray, config: Stage0Config, stream: int) -> Tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.bootstrap_seed, stream])))
    samples: List[np.ndarray] = []
    remaining = config.bootstrap_resamples
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(prior), size=(count, len(prior)))
        p_mean = prior[indices].mean(axis=1)
        o_mean = oracle[indices].mean(axis=1)
        samples.append((p_mean - o_mean) / p_mean)
        remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(samples), [0.025, 0.975]))


def _summarize(rows: Sequence[Mapping[str, object]], tolerance: float, config: Stage0Config, stream: int) -> Mapping[str, object]:
    selected = [row for row in rows if float(row["candidate_tolerance"]) == tolerance]
    by_policy = {policy: sorted((row for row in selected if row["policy"] == policy), key=lambda row: int(row["sequence_id"])) for policy in POLICIES}
    prior = np.asarray([row["early_task_cost"] for row in by_policy["population_prior"]], dtype=np.float64)
    oracle = np.asarray([row["early_task_cost"] for row in by_policy["true_factor_oracle"]], dtype=np.float64)
    prior_unsafe = float(np.mean([row["unsafe"] for row in by_policy["population_prior"]]))
    relative = float((prior.mean() - oracle.mean()) / prior.mean())
    ci = _bootstrap_relative_ci(prior, oracle, config, stream)
    identity_ok = all(
        left["scenario_hash"] == right["scenario_hash"]
        for left, right in zip(by_policy["population_prior"], by_policy["true_factor_oracle"])
    )
    criteria = {
        "population_unsafe_fraction_between_25_and_75pct": 0.25 <= prior_unsafe <= 0.75,
        "oracle_relative_cost_improvement_at_least_25pct": relative >= 0.25,
        "relative_improvement_ci_lower_at_least_20pct": ci[0] >= 0.20,
        "oracle_mean_cost_strictly_lower": float(oracle.mean()) < float(prior.mean()),
        "all_512_pairs_valid_and_identical": len(prior) == config.n_sequences and len(oracle) == config.n_sequences and identity_ok,
    }
    return {
        "tolerance": tolerance,
        "population_prior_mean_cost": float(prior.mean()),
        "true_factor_oracle_mean_cost": float(oracle.mean()),
        "population_prior_unsafe_fraction": prior_unsafe,
        "true_factor_oracle_unsafe_fraction": float(np.mean([row["unsafe"] for row in by_policy["true_factor_oracle"]])),
        "relative_cost_improvement": relative,
        "relative_cost_improvement_ci95": list(ci),
        "criteria": criteria,
        "qualified": all(criteria.values()),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_stage0(output_dir: Path, config: Stage0Config = Stage0Config(), command: str = "") -> Mapping[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: List[Mapping[str, object]] = []
    for sequence_id in range(config.n_sequences):
        scenario = _scenario(config, sequence_id)
        for tolerance in CANDIDATE_TOLERANCES:
            for policy in POLICIES:
                rows.append(_row(config, tolerance, sequence_id, policy, scenario))
    raw_path = output_dir / "raw_results.csv"
    _write_csv(raw_path, rows)
    candidates = [_summarize(rows, tolerance, config, stream=100 + index) for index, tolerance in enumerate(CANDIDATE_TOLERANCES)]
    selected = next((item for item in candidates if item["qualified"]), None)
    decision = {
        "verdict": "TASK_DYNAMIC_RANGE_ESTABLISHED" if selected else "TASK_DYNAMIC_RANGE_NOT_ESTABLISHED",
        "disposition": "GO_STAGE1_CONTRACT" if selected else "STOP_TASK_FAMILY",
        "selected_tolerance": None if selected is None else selected["tolerance"],
    }
    summary = {
        "schema": "persistent-context-v2-stage0-summary-v1",
        "contract_id": CONTRACT_ID,
        "config": asdict(config),
        "train_factors": TRAIN_FACTORS,
        "development_factors": DEVELOPMENT_FACTORS,
        "formal_factors_sealed_not_run": FORMAL_FACTORS,
        "population_prior_mean": population_prior(),
        "population_prior_hash": _value_hash(TRAIN_FACTORS),
        "candidates": candidates,
        "decision": decision,
        "formal_outcomes_generated": False,
        "wall_time_seconds": time.time() - started,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    contract_path = repo_root / "docs/research/persistent_context_v2_stage0_contract_zh.md"
    source_path = Path(__file__).resolve()
    manifest = {
        "schema": "persistent-context-v2-stage0-manifest-v1",
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
        "deviations": [],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    return summary
