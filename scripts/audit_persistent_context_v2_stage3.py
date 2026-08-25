#!/usr/bin/env python3
"""Independent checkpoint, planner, RLS, raw-result, and gate audit for Stage 3."""

from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

POLICIES = ("population_context", "current_only_context", "persistent_rls_context", "shuffled_rls_context", "wrong_sequence_rls_context", "true_context", "analytic_true_factor_oracle")
CONDITIONS = ("persistent", "no_persistence")


class AuditFiLM(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__(); self.action_projection = nn.Linear(1, hidden_dim); self.context_film = nn.Linear(1, 2 * hidden_dim); self.output_projection = nn.Linear(hidden_dim, 1)
    def forward(self, action, context):
        features = self.action_projection(action); scale, shift = self.context_film(context).chunk(2, dim=-1); return self.output_projection(features * (1.0 + scale) + shift)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int, stream: int) -> tuple:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream]))); parts, remaining = [], resamples
    while remaining:
        count = min(1000, remaining); indices = rng.integers(0, len(values), size=(count, len(values))); parts.append(values[indices].mean(axis=1)); remaining -= count
    return tuple(float(item) for item in np.quantile(np.concatenate(parts), [0.025, 0.975]))


def bootstrap_relative_ci(current, treatment, resamples, seed, stream):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream]))); parts, remaining = [], resamples
    while remaining:
        count = min(1000, remaining); indices = rng.integers(0, len(current), size=(count, len(current))); cm, tm = current[indices].mean(axis=1), treatment[indices].mean(axis=1); parts.append((cm - tm) / cm); remaining -= count
    return tuple(float(item) for item in np.quantile(np.concatenate(parts), [0.025, 0.975]))


def dataset(factors, count, seed, config):
    rng = np.random.default_rng(seed); factor_array = np.asarray(factors, dtype=np.float32); context = factor_array[rng.integers(len(factor_array), size=count)]; action = rng.uniform(-config["action_limit"], config["action_limit"], size=count).astype(np.float32); noise = rng.normal(0.0, config["noise_std"], size=count).astype(np.float32); return action[:, None], context[:, None], (context * action + noise)[:, None]


@torch.no_grad()
def audit_planner(model, rows, config, device, failures, prefix):
    grid = np.linspace(-config["action_limit"], config["action_limit"], config["planner_candidates"], dtype=np.float32)
    batch_size = 1024
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]; targets = np.asarray([float(row["target"]) for row in batch], dtype=np.float32); contexts = np.asarray([float(row["context"] if "context" in row else row["context_before"]) for row in batch], dtype=np.float32)
        actions = np.broadcast_to(grid[None, :], (len(batch), len(grid)))
        analytic = all(row["policy"] == "analytic_true_factor_oracle" for row in batch)
        if analytic:
            factors = np.asarray([float(row["factor"]) for row in batch], dtype=np.float32); predictions = factors[:, None] * actions
        else:
            action_tensor = torch.as_tensor(actions.reshape(-1, 1), device=device); context_tensor = torch.as_tensor(np.repeat(contexts, len(grid))[:, None], device=device); predictions = model(action_tensor, context_tensor).reshape(len(batch), len(grid)).cpu().numpy()
        indices = np.argmin((predictions - targets[:, None]) ** 2, axis=1); expected_actions = actions[np.arange(len(batch)), indices]; expected_predictions = predictions[np.arange(len(batch)), indices]
        for offset, row in enumerate(batch):
            raw_pred_key = "predicted_response" if "predicted_response" in row else "predicted_response_at_action"
            if not np.isclose(expected_actions[offset], float(row["action"]), atol=1e-7) or not np.isclose(expected_predictions[offset], float(row[raw_pred_key]), rtol=2e-5, atol=2e-6): failures.append(f"{prefix}:{start+offset}")


def rls_mean(history, train_factors, noise_std):
    prior_mean, prior_variance = float(np.mean(train_factors)), float(np.var(train_factors)); prior_precision, obs_precision = 1.0 / prior_variance, 1.0 / noise_std ** 2
    sum_u2 = sum(float(row["action"]) ** 2 for row in history); sum_uy = sum(float(row["action"]) * float(row["response"]) for row in history)
    return float((prior_precision * prior_mean + obs_precision * sum_uy) / (prior_precision + obs_precision * sum_u2))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cpu"); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite: {args.output}")
    summary_path, manifest_path, checkpoint_path = args.run_dir / "summary.json", args.run_dir / "run_manifest.json", args.run_dir / "checkpoint.pt"
    summary = json.loads(summary_path.read_text()); manifest = json.loads(manifest_path.read_text()); config = summary["config"]; device = torch.device(args.device)
    payload = torch.load(checkpoint_path, map_location=device); model = AuditFiLM(int(config["hidden_dim"])).to(device); model.load_state_dict(payload["model_state"]); model.eval()
    hash_checks = {"checkpoint": sha256(checkpoint_path) == manifest["checkpoint_sha256"], "development_raw": sha256(args.run_dir / "development_raw.csv") == manifest["development_raw_sha256"], "formal_raw": sha256(args.run_dir / "formal_raw.csv") == manifest["formal_raw_sha256"], "summary": sha256(summary_path) == manifest["summary_sha256"]}
    curve = list(csv.DictReader((args.run_dir / "training_curve.csv").open(encoding="utf-8", newline=""))); best_curve = min(curve, key=lambda row: float(row["dev_true_context_mse"])); checkpoint_selection = int(best_curve["step"]) == int(summary["training"]["best_step"]) and np.isclose(float(best_curve["dev_true_context_mse"]), float(summary["training"]["best_dev_true_context_mse"]))
    actions, contexts, responses = dataset(summary["splits"]["development_factors"], int(config["dev_prediction_samples"]), int(config["dev_data_seed"]), config)
    with torch.no_grad():
        at = torch.as_tensor(actions, device=device); ct = torch.as_tensor(contexts, device=device); rt = torch.as_tensor(responses, device=device); true_mse = float(torch.mean((model(at, ct) - rt) ** 2).item()); pop_mse = float(torch.mean((model(at, torch.ones_like(ct)) - rt) ** 2).item())
    with (args.run_dir / "development_raw.csv").open(encoding="utf-8", newline="") as handle: dev_rows = list(csv.DictReader(handle))
    dev_planner_failures = []; audit_planner(model, [row for row in dev_rows if row["policy"] != "analytic_true_factor_oracle"], config, device, dev_planner_failures, "dev_planner")
    dev_group = defaultdict(list)
    for row in dev_rows: dev_group[row["policy"]].append(float(row["early_task_cost"]))
    dev_pop, dev_true = np.asarray(dev_group["population_context"]), np.asarray(dev_group["true_context"]); dev_relative = float((dev_pop.mean() - dev_true.mean()) / dev_pop.mean()); dev_relative_ci = bootstrap_relative_ci(dev_pop, dev_true, int(config["dev_bootstrap_resamples"]), int(config["dev_bootstrap_seed"]), 10)
    dev_lookup = defaultdict(dict)
    for row in dev_rows: dev_lookup[int(row["sequence_id"])][row["policy"]] = row
    dev_identity = all(pair["population_context"]["scenario_hash"] == pair["true_context"]["scenario_hash"] for pair in dev_lookup.values()); step = 3.0 / (int(config["planner_candidates"]) - 1); dev_action_change = float(np.mean([abs(float(pair["population_context"]["action"]) - float(pair["true_context"]["action"])) > step / 2 for pair in dev_lookup.values()])); dev_pop_unsafe = float(np.mean([int(row["unsafe"]) for row in dev_rows if row["policy"] == "population_context"]))
    dev_criteria = {"true_prediction_mse_at_most_10pct_population": true_mse <= .10 * pop_mse, "true_behavior_relative_at_least_50pct_and_ci_lower_40pct": dev_relative >= .50 and dev_relative_ci[0] >= .40, "population_unsafe_between_25_and_75pct": .25 <= dev_pop_unsafe <= .75, "planner_action_change_at_least_80pct": dev_action_change >= .80, "finite_and_paired_audit": dev_identity and not dev_planner_failures}; dev_criteria = {key: bool(value) for key, value in dev_criteria.items()}
    with (args.run_dir / "formal_raw.csv").open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    n, episodes = int(config["formal_sequences"]), int(config["n_episodes"]); lookup = {(row["condition"], row["policy"], int(row["sequence_id"]), int(row["episode_id"])): row for row in rows}
    structural_failures = defaultdict(list)
    for condition in CONDITIONS:
        for sequence_id in range(n):
            current, persistent = lookup[(condition, "current_only_context", sequence_id, 1)], lookup[(condition, "persistent_rls_context", sequence_id, 1)]
            if any(current[key] != persistent[key] for key in ("scenario_hash", "context_before", "action", "predicted_response_at_action", "response", "early_task_cost", "unsafe")): structural_failures["episode1"].append(f"{condition}:{sequence_id}")
    for sequence_id in range(n):
        if len({lookup[("persistent", "current_only_context", sequence_id, episode)]["factor_id"] for episode in range(1, episodes + 1)}) != 1: structural_failures["factor"].append(str(sequence_id))
        for episode in range(1, episodes + 1):
            left, right = lookup[("persistent", "current_only_context", sequence_id, episode)], lookup[("no_persistence", "current_only_context", sequence_id, episode)]
            if left["target"] != right["target"] or left["noise"] != right["noise"]: structural_failures["condition_pair"].append(f"{sequence_id}:{episode}")
            if len({lookup[("persistent", policy, sequence_id, episode)]["scenario_hash"] for policy in POLICIES}) != 1: structural_failures["policy_pair"].append(f"{sequence_id}:{episode}")
    train_factors = summary["splits"]["train_factors"]
    for row in rows:
        policy, condition, sequence_id, episode = row["policy"], row["condition"], int(row["sequence_id"]), int(row["episode_id"])
        if int(row["planner_candidates"]) != int(config["planner_candidates"]): structural_failures["budget"].append(f"{condition}:{policy}:{sequence_id}:{episode}")
        if policy in ("current_only_context", "persistent_rls_context", "shuffled_rls_context", "wrong_sequence_rls_context"):
            donor = int(row["donor_sequence_id"]); source = sequence_id if policy in ("current_only_context", "persistent_rls_context") else donor
            history = [] if policy == "current_only_context" else [lookup[(condition, policy, source, previous)] for previous in range(1, episode)]
            expected = rls_mean(history, train_factors, float(config["noise_std"]))
            if not np.isclose(expected, float(row["context_before"]), rtol=1e-12, atol=1e-12): structural_failures["rls"].append(f"{condition}:{policy}:{sequence_id}:{episode}")
            if int(row["history_count_before"]) != len(history): structural_failures["count"].append(f"{condition}:{policy}:{sequence_id}:{episode}")
            if episode >= 2 and policy in ("shuffled_rls_context", "wrong_sequence_rls_context") and donor == sequence_id: structural_failures["donor"].append(f"{condition}:{policy}:{sequence_id}:{episode}")
        response = float(row["factor"]) * float(row["action"]) + float(row["noise"]); miss = response - float(row["target"]); cost = (miss / float(config["tolerance"])) ** 2 + float(abs(miss) > float(config["tolerance"]))
        if not np.isclose(response, float(row["response"])) or not np.isclose(cost, float(row["early_task_cost"])): structural_failures["outcome"].append(f"{condition}:{policy}:{sequence_id}:{episode}")
    formal_planner_failures = []
    model_rows = [row for row in rows if row["policy"] != "analytic_true_factor_oracle"]; analytic_rows = [row for row in rows if row["policy"] == "analytic_true_factor_oracle"]
    audit_planner(model, model_rows, config, device, formal_planner_failures, "formal_model_planner"); audit_planner(model, analytic_rows, config, device, formal_planner_failures, "formal_analytic_planner")
    buckets = defaultdict(list)
    for row in rows:
        if int(row["episode_id"]) >= 2: buckets[(row["condition"], row["policy"], int(row["sequence_id"]))].append(float(row["early_task_cost"]))
    values = {condition: {policy: np.asarray([np.mean(buckets[(condition, policy, sequence_id)]) for sequence_id in range(n)]) for policy in POLICIES} for condition in CONDITIONS}
    pc, pop, rls, true, analytic = (values["persistent"][policy] for policy in ("current_only_context", "population_context", "persistent_rls_context", "true_context", "analytic_true_factor_oracle")); npc, nprls = values["no_persistence"]["current_only_context"], values["no_persistence"]["persistent_rls_context"]
    rls_delta, no_delta, true_delta, analytic_delta = pc - rls, npc - nprls, pop - true, pop - analytic; did = rls_delta - no_delta; current_mean, population_mean = float(pc.mean()), float(pop.mean()); rls_relative, no_relative = float(rls_delta.mean() / current_mean), float(no_delta.mean() / npc.mean()); shuffled_relative = float((pc - values["persistent"]["shuffled_rls_context"]).mean() / current_mean); wrong_relative = float((pc - values["persistent"]["wrong_sequence_rls_context"]).mean() / current_mean)
    action_change = float(np.mean([abs(float(lookup[("persistent", "population_context", sequence_id, episode)]["action"]) - float(lookup[("persistent", "true_context", sequence_id, episode)]["action"])) > step / 2 for sequence_id in range(n) for episode in range(2, episodes + 1)])); seed, resamples = int(config["formal_bootstrap_seed"]), int(config["formal_bootstrap_resamples"]); rls_ci, true_ci, did_ci = bootstrap_ci(rls_delta, resamples, seed, 100), bootstrap_ci(true_delta, resamples, seed, 200), bootstrap_ci(did, resamples, seed, 900)
    formal_structural = {"raw_row_count": len(rows) == 2 * len(POLICIES) * n * episodes, "checkpoint_and_file_hashes": all(hash_checks.values()), "checkpoint_selected_by_min_dev_mse": checkpoint_selection, "development_recomputed": dev_criteria == summary["development_gate"]["criteria"], "development_planner": not dev_planner_failures, "formal_pairing_lifetime_rls_budget_outcome": not any(structural_failures.values()), "formal_planner_all_401_candidates": not formal_planner_failures}
    formal_structural = {key: bool(value) for key, value in formal_structural.items()}
    criteria = {"true_context_relative_at_least_50pct_and_ci_lower_above_40pct_population": float(true_delta.mean() / population_mean) >= .50 and true_ci[0] > .40 * population_mean, "true_context_recovers_at_least_90pct_analytic_gap": true_delta.mean() / analytic_delta.mean() >= .90, "persistent_rls_relative_at_least_50pct_and_ci_lower_above_40pct_current": rls_relative >= .50 and rls_ci[0] > .40 * current_mean, "rls_recovers_at_least_90pct_true_context_improvement": rls_delta.mean() / true_delta.mean() >= .90, "rls_did_ci_lower_above_30pct_current": did_ci[0] > .30 * current_mean, "positive_sequence_fraction_at_least_80pct": float(np.mean(rls_delta > 0)) >= .80, "negative_controls_not_comparable": no_relative <= .05 and shuffled_relative <= .05 and wrong_relative <= .05 and shuffled_relative < .5 * rls_relative and wrong_relative < .5 * rls_relative, "true_vs_population_action_change_at_least_80pct": action_change >= .80, "all_sequences_and_audits_valid": all(formal_structural.values())}; criteria = {key: bool(value) for key, value in criteria.items()}
    verdict = "CONTEXT_CONDITIONED_WORLD_MODEL_SUPPORTED" if all(criteria.values()) else "CONTEXT_CONDITIONED_WORLD_MODEL_NOT_ESTABLISHED"
    audit = {"schema": "persistent-context-v2-stage3-independent-audit-v1", "passed": bool(all(formal_structural.values()) and criteria == summary["formal"]["criteria"] and verdict == summary["decision"]["verdict"]), "development": {"prediction_mse": {"true": true_mse, "population": pop_mse, "ratio": true_mse / pop_mse}, "relative_improvement": dev_relative, "relative_ci95": dev_relative_ci, "action_change_fraction": dev_action_change, "criteria": dev_criteria}, "formal_structural": formal_structural, "formal_criteria": criteria, "recomputed": {"persistent_current_mean": current_mean, "persistent_rls_mean": float(rls.mean()), "persistent_true_context_mean": float(true.mean()), "analytic_true_mean": float(analytic.mean()), "rls_relative_improvement": rls_relative, "rls_difference_ci95": rls_ci, "true_relative_improvement": float(true_delta.mean() / population_mean), "true_difference_ci95": true_ci, "rls_true_context_gap_recovery": float(rls_delta.mean() / true_delta.mean()), "true_analytic_gap_recovery": float(true_delta.mean() / analytic_delta.mean()), "did_mean": float(did.mean()), "did_ci95": did_ci, "no_persistence_relative": no_relative, "shuffled_relative": shuffled_relative, "wrong_relative": wrong_relative, "action_change_fraction": action_change}, "failure_counts": {key: len(value) for key, value in structural_failures.items()} | {"development_planner": len(dev_planner_failures), "formal_planner": len(formal_planner_failures)}, "recomputed_verdict": verdict}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n"); print(json.dumps(audit, indent=2, sort_keys=True)); return 0 if audit["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
