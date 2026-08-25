"""Train and evaluate the frozen low-dimensional matrix surrogate gate."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from research.persistent_context_v2.pushobj_rotation_stage0 import dump_json, git_revision, read_jsonl, sha256


CONTRACT_ID = "persistent-context-v2-matrix-learned-surrogate-gate-v1"
EXPECTED_DESIGN_SHA256 = "607d5e943635e34c10883ac16b37162c212e1b0e30fd075bcb1f7e6136f3d756"


def load_design(path: Path) -> dict:
    if sha256(path) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen learned-gate design hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["contract_id"] != CONTRACT_ID:
        raise RuntimeError("contract id mismatch")
    return value


def load_rows(path: Path, expected_split: str) -> list[dict]:
    rows = [row for row in read_jsonl(path) if row.get("record_type") == "paired_sequence"]
    rows.sort(key=lambda row: int(row["sequence_id"]))
    if len(rows) != 32 or any(row["split"] != expected_split for row in rows):
        raise RuntimeError(f"invalid {expected_split} raw")
    return rows


def posterior_parameters(row: dict) -> tuple[float, float]:
    mean_z = np.asarray(row["posterior"]["mean_z"], dtype=np.float64)
    return float(np.linalg.norm(mean_z)), float(math.degrees(math.atan2(mean_z[1], mean_z[0])))


def feature_row(row: dict, design: dict) -> np.ndarray:
    gain, rotation = posterior_parameters(row)
    g = (gain - float(design["gain_center"])) / float(design["gain_scale"])
    r = rotation / float(design["rotation_scale_degrees"])
    return np.asarray([1.0, g, r, g * g, g * r, r * r], dtype=np.float64)


def arrays(rows: list[dict], design: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([feature_row(row, design) for row in rows])
    population = np.asarray([row["e2"]["population"]["metrics"]["pose_auc10"] for row in rows], dtype=np.float64)
    context = np.asarray([row["e2"]["context"]["metrics"]["pose_auc10"] for row in rows], dtype=np.float64)
    return x, population, context


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + float(alpha) * penalty, x.T @ y)


def bootstrap_ci(values: np.ndarray, design: dict, stream: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(design["bootstrap_seed"]) + int(stream))
    indexes = rng.integers(0, len(values), size=(int(design["bootstrap_resamples"]), len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def effect(population: np.ndarray, treatment: np.ndarray, design: dict, stream: int) -> dict:
    population = np.asarray(population, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    delta = population - treatment
    return {
        "population_mean": float(population.mean()),
        "treatment_mean": float(treatment.mean()),
        "mean_delta": float(delta.mean()),
        "relative_improvement": float(delta.mean() / population.mean()),
        "bootstrap_ci95_delta": bootstrap_ci(delta, design, stream),
        "positive_fraction": float(np.mean(delta > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
        "negative_fraction": float(np.mean(delta < -1e-12)),
        "harm_fraction": float(np.mean(delta < -1e-12)),
        "unit_deltas": delta.tolist(),
    }


def functional_decisions(rows: list[dict], design: dict) -> np.ndarray:
    decisions = []
    for row in rows:
        gain, rotation = posterior_parameters(row)
        decisions.append(gain < float(design["gain_center"]) or abs(rotation) >= 15.0)
    return np.asarray(decisions, dtype=bool)


def gate_outcome(population: np.ndarray, context: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(decisions, dtype=bool), context, population)


def select_alpha(train_rows: list[dict], dev_rows: list[dict], design: dict) -> tuple[float, list[dict]]:
    train_x, train_population, train_context = arrays(train_rows, design)
    train_y = train_population - train_context
    dev_x, dev_population, dev_context = arrays(dev_rows, design)
    candidates = []
    for alpha in design["ridge_alphas"]:
        beta = fit_ridge(train_x, train_y, float(alpha))
        predictions = dev_x @ beta
        decisions = predictions > 0.0
        outcome = gate_outcome(dev_population, dev_context, decisions)
        candidates.append({
            "alpha": float(alpha),
            "beta": beta.tolist(),
            "dev_mean_delta": float(np.mean(dev_population - outcome)),
            "dev_selection_rate": float(np.mean(decisions)),
            "dev_prediction_mse": float(np.mean((predictions - (dev_population - dev_context)) ** 2)),
        })
    chosen = max(candidates, key=lambda row: (row["dev_mean_delta"], row["alpha"]))
    return float(chosen["alpha"]), candidates


def confusion(predicted: np.ndarray, true_delta: np.ndarray) -> dict:
    predicted = np.asarray(predicted, dtype=bool)
    beneficial = np.asarray(true_delta, dtype=np.float64) > 1e-12
    harmful = np.asarray(true_delta, dtype=np.float64) < -1e-12
    tp = int(np.sum(predicted & beneficial))
    fp = int(np.sum(predicted & harmful))
    fn = int(np.sum(~predicted & beneficial))
    tn = int(np.sum(~predicted & harmful))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "accuracy_excluding_ties": float((tp + tn) / (tp + fp + fn + tn)) if tp + fp + fn + tn else 0.0,
    }


def evaluate(args) -> dict:
    started = time.time()
    design = load_design(args.design)
    train_path = args.data_dir / "train" / "raw.jsonl"
    dev_path = args.data_dir / "dev" / "raw.jsonl"
    formal_path = args.data_dir / "formal" / "raw.jsonl"
    train_rows = load_rows(train_path, "train")
    dev_rows = load_rows(dev_path, "dev")
    selected_alpha, candidates = select_alpha(train_rows, dev_rows, design)
    fit_rows = train_rows + dev_rows
    fit_x, fit_population, fit_context = arrays(fit_rows, design)
    beta = fit_ridge(fit_x, fit_population - fit_context, selected_alpha)
    model = {
        "contract_id": CONTRACT_ID,
        "selected_alpha": selected_alpha,
        "beta": beta.tolist(),
        "feature_names": design["feature_basis"],
        "gain_center": design["gain_center"],
        "gain_scale": design["gain_scale"],
        "rotation_scale_degrees": design["rotation_scale_degrees"],
        "training_units": 64,
        "alpha_candidates": candidates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "gate_model.json", model)

    # Formal outcomes are loaded only after model selection and refit are fixed.
    formal_rows = load_rows(formal_path, "formal")
    formal_x, population, context = arrays(formal_rows, design)
    true_delta = population - context
    predictions = formal_x @ beta
    learned = predictions > 0.0
    functional = functional_decisions(formal_rows, design)
    inverted = ~learned
    random = np.zeros(len(formal_rows), dtype=bool)
    selected_count = int(np.sum(learned))
    permutation = np.random.default_rng(int(design["random_control_seed"])).permutation(len(formal_rows))
    random[permutation[:selected_count]] = True
    best = np.minimum(population, context)
    policy_outcomes = {
        "always_context": context,
        "learned_gate": gate_outcome(population, context, learned),
        "functional_gate": gate_outcome(population, context, functional),
        "inverted_learned_gate": gate_outcome(population, context, inverted),
        "selection_matched_random_gate": gate_outcome(population, context, random),
        "best_of_two_behavior_ceiling": best,
    }
    policies = {name: effect(population, value, design, 100 + index) for index, (name, value) in enumerate(policy_outcomes.items())}
    learned_effect = policies["learned_gate"]
    best_effect = policies["best_of_two_behavior_ceiling"]
    always_minus_learned = context - policy_outcomes["learned_gate"]
    result = {
        "contract_id": CONTRACT_ID,
        "evaluation_type": design["evaluation_type"],
        "selected_alpha": selected_alpha,
        "beta": beta.tolist(),
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "n_formal": len(formal_rows),
        "formal_context_selection_rate": float(np.mean(learned)),
        "formal_functional_selection_rate": float(np.mean(functional)),
        "policies": policies,
        "learned_vs_always_context": {
            "mean_delta_always_minus_learned": float(always_minus_learned.mean()),
            "relative_improvement_over_always": float(always_minus_learned.mean() / context.mean()),
            "bootstrap_ci95": bootstrap_ci(always_minus_learned, design, 200),
        },
        "best_of_two_opportunity_recovery": float(learned_effect["mean_delta"] / best_effect["mean_delta"]),
        "formal_prediction": {
            "mse": float(np.mean((predictions - true_delta) ** 2)),
            "mae": float(np.mean(np.abs(predictions - true_delta))),
            "correlation": float(np.corrcoef(predictions, true_delta)[0, 1]),
            **confusion(learned, true_delta),
        },
        "by_factor": {},
        "valid": True,
    }
    for factor_index, factor in enumerate(design["formal_factors"]):
        indexes = np.asarray([int(row["factor_index"]) == factor_index for row in formal_rows], dtype=bool)
        key = f"theta={float(factor['rotation_degrees']):+g},gain={float(factor['gain']):g}"
        result["by_factor"][key] = {
            "n": int(np.sum(indexes)),
            "learned_selection_rate": float(np.mean(learned[indexes])),
            "true_context_mean_delta": float(np.mean(true_delta[indexes])),
            "learned_gate": effect(population[indexes], policy_outcomes["learned_gate"][indexes], design, 300 + factor_index),
        }
    decision_path = args.output_dir / "formal_decisions.jsonl"
    if decision_path.exists():
        decision_path.unlink()
    from research.persistent_context_v2.pushobj_rotation_stage0 import append_jsonl
    for index, row in enumerate(formal_rows):
        append_jsonl(decision_path, {
            "sequence_id": int(row["sequence_id"]),
            "factor_index": int(row["factor_index"]),
            "features": formal_x[index],
            "predicted_delta": float(predictions[index]),
            "learned_use_context": bool(learned[index]),
            "functional_use_context": bool(functional[index]),
            "random_use_context": bool(random[index]),
            "population": float(population[index]),
            "context": float(context[index]),
            "true_delta": float(true_delta[index]),
        })
    dump_json(args.output_dir / "runner_summary.json", result)
    manifest = {
        "contract_id": CONTRACT_ID,
        "git_revision": git_revision(),
        "design_path": str(args.design),
        "design_sha256": sha256(args.design),
        "contract_path": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "raw_paths": {"train": str(train_path), "dev": str(dev_path), "formal": str(formal_path)},
        "raw_sha256": {"train": sha256(train_path), "dev": sha256(dev_path), "formal": sha256(formal_path)},
        "model_sha256": sha256(args.output_dir / "gate_model.json"),
        "decisions_sha256": sha256(decision_path),
        "command": " ".join(__import__("sys").argv),
        "started_unix": started,
        "finished_unix": time.time(),
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_matrix_learned_gate/evaluation"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_matrix_learned_gate_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_matrix_learned_gate_contract_zh.md"))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
