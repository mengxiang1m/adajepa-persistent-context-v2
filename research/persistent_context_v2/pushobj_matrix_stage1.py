"""Formal Bayesian rotation-gain matrix history experiment for PushObj."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.pushobj_matrix_stage0 import (
    POPULATION_PRIOR_MATRIX,
    apply_action_matrix,
    array_sha256,
    execute_matrix,
    factor_matrix,
    identity_audit,
    load_matrix_world_model,
    plan_matrix_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    WINDOW,
    deadline_success,
    prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl,
    dump_json,
    git_revision,
    make_env,
    make_preprocessor,
    pose_metrics,
    read_jsonl,
    resource_snapshot,
    seed_all,
    sha256,
)
from research.persistent_context_v2.pushobj_rotation_stage1 import pd_coefficients


CONTRACT_ID = "persistent-context-v2-pushobj-bayesian-matrix-history-stage1-v1"
EXPECTED_DESIGN_SHA256 = "96e0ebfc32ed6e081d6238acf43d99156bc386d0c04b421fc2f41116f8f64b37"
FORMAL_FACTORS = (
    (-25.0, 0.82), (-25.0, 1.18),
    (-10.0, 0.82), (-10.0, 1.18),
    (10.0, 0.82), (10.0, 1.18),
    (25.0, 0.82), (25.0, 1.18),
)
TRAIN_ROTATIONS = (-30.0, -15.0, 0.0, 15.0, 30.0)
TRAIN_GAINS = (0.75, 1.0, 1.25)
POLICIES = (
    "population_prior",
    "current_only",
    "correct_history",
    "shuffled_history",
    "wrong_sequence_history",
    "true_factor_oracle",
)
CONDITIONS = ("persistent", "no_persistence")
N_SEQUENCES = 32
N_EPISODES = 2
SEGMENT_INDICES = (
    507, 822, 885, 699, 706, 790, 833, 631,
    746, 690, 531, 739, 788, 875, 593, 506,
    503, 511, 714, 973, 558, 583, 520, 584,
    854, 697, 700, 600, 981, 750, 856, 767,
    549, 807, 781, 762, 537, 979, 512, 649,
    834, 686, 570, 592, 878, 821, 972, 534,
    712, 702, 768, 855, 840, 800, 950, 747,
    742, 763, 585, 587, 594, 602, 930, 901,
)
DONOR_SEED = 1_040_200
BOOTSTRAP_SEED = 1_040_301
BOOTSTRAP_RESAMPLES = 20_000
COMMAND_NORM_MIN = 1e-4
GAIN_RATIO_MIN = 0.45
GAIN_RATIO_MAX = 1.55
OBSERVATION_NOISE_STD = 0.01
PRIOR_COVARIANCE_RIDGE = 1e-4


def z_from_factor(rotation_degrees: float, gain: float) -> np.ndarray:
    theta = math.radians(float(rotation_degrees))
    return np.asarray([float(gain) * math.cos(theta), float(gain) * math.sin(theta)], dtype=np.float64)


def matrix_from_z(z) -> np.ndarray:
    c, s = np.asarray(z, dtype=np.float64)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def prior_parameters():
    samples = np.asarray([z_from_factor(theta, gain) for theta in TRAIN_ROTATIONS for gain in TRAIN_GAINS])
    mean = samples.mean(axis=0)
    covariance = np.cov(samples.T, bias=True) + PRIOR_COVARIANCE_RIDGE * np.eye(2)
    return mean, covariance


@dataclass
class BayesianMatrixContext:
    precision: np.ndarray = field(default_factory=lambda: np.linalg.inv(prior_parameters()[1]))
    information: np.ndarray = field(default_factory=lambda: np.linalg.inv(prior_parameters()[1]) @ prior_parameters()[0])
    observation_count: int = 0

    def update_observations(self, observations) -> None:
        observations = np.asarray(observations, dtype=np.float64).reshape(-1, 2)
        if len(observations) == 0:
            return
        observation_precision = 1.0 / (OBSERVATION_NOISE_STD ** 2)
        self.precision = self.precision + len(observations) * observation_precision * np.eye(2)
        self.information = self.information + observation_precision * observations.sum(axis=0)
        self.observation_count += int(len(observations))

    @property
    def covariance(self):
        return np.linalg.inv(self.precision)

    @property
    def mean_z(self):
        return np.linalg.solve(self.precision, self.information)

    @property
    def mean_matrix(self):
        return matrix_from_z(self.mean_z)

    @property
    def gain(self):
        return float(np.linalg.norm(self.mean_z))

    @property
    def rotation_degrees(self):
        return float(math.degrees(math.atan2(self.mean_z[1], self.mean_z[0])))

    def as_dict(self):
        return {"mean_z": self.mean_z, "mean_matrix": self.mean_matrix, "covariance": self.covariance, "precision": self.precision, "information": self.information, "observation_count": self.observation_count, "gain": self.gain, "rotation_degrees": self.rotation_degrees}


def infer_matrix_observations(commands, states):
    commands = np.asarray(commands, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    if states.shape[0] != commands.shape[0] + 1 or states.shape[1] < 7:
        raise ValueError("commands/states do not form PushObj proprio transitions")
    p_coef, v_coef, target_coef = pd_coefficients()
    observations = []
    accepted_indices = []
    for index, command in enumerate(commands):
        command_norm_sq = float(np.dot(command, command))
        if command_norm_sq < COMMAND_NORM_MIN ** 2:
            continue
        p0, p1, v0 = states[index, :2], states[index + 1, :2], states[index, 5:7]
        target_position = (p1 - p_coef * p0 - v_coef * v0) / target_coef
        inferred_effective = (target_position - p0) / 100.0
        gain_ratio = float(np.linalg.norm(inferred_effective) / math.sqrt(command_norm_sq))
        if not np.isfinite(gain_ratio) or not (GAIN_RATIO_MIN <= gain_ratio <= GAIN_RATIO_MAX):
            continue
        c = float(np.dot(command, inferred_effective) / command_norm_sq)
        s = float((command[0] * inferred_effective[1] - command[1] * inferred_effective[0]) / command_norm_sq)
        if np.isfinite(c) and np.isfinite(s):
            observations.append([c, s])
            accepted_indices.append(index)
    return np.asarray(observations, dtype=np.float64).reshape(-1, 2), accepted_indices


def observations_sha256(observations) -> str:
    return hashlib.sha256(np.asarray(observations, dtype=np.float64).tobytes()).hexdigest()


def factor_index_for(condition: str, sequence_id: int, episode_index: int) -> int:
    base = int(sequence_id) % len(FORMAL_FACTORS)
    if condition == "persistent" or int(episode_index) == 0:
        return base
    if condition == "no_persistence":
        return (base + 3) % len(FORMAL_FACTORS)
    raise ValueError(condition)


def scenario(condition: str, sequence_id: int, episode_index: int):
    factor_index = factor_index_for(condition, sequence_id, episode_index)
    rotation, gain = FORMAL_FACTORS[factor_index]
    segment_index = SEGMENT_INDICES[int(sequence_id) * N_EPISODES + int(episode_index)]
    return {"condition": condition, "sequence_id": int(sequence_id), "episode_index": int(episode_index), "episode": int(episode_index) + 1, "segment_index": int(segment_index), "factor_index": factor_index, "rotation_degrees": rotation, "gain": gain, "true_matrix": factor_matrix(rotation, gain), "env_seed": 1_050_000 + int(sequence_id) * 100 + int(episode_index), "cem_seed": 1_060_000 + int(sequence_id) * 100 + int(episode_index)}


def completed_keys(path, record_type):
    return {(int(row["sequence_id"]), int(row["episode_index"])) for row in read_jsonl(path) if row.get("record_type") == record_type}


def evidence_lookup(path):
    return {(int(row["sequence_id"]), int(row["episode_index"])): row for row in read_jsonl(path) if row.get("record_type") == "evidence_episode"}


def matched_observations(observations, count):
    observations = np.asarray(observations, dtype=np.float64).reshape(-1, 2)
    if count == 0:
        return observations[:0]
    if len(observations) == 0:
        raise RuntimeError("history donor has no accepted observations")
    return observations[np.arange(int(count)) % len(observations)]


def history_payload(evidence, policy, sequence_id, episode_index, n_sequences):
    prior = BayesianMatrixContext()
    if policy in ("population_prior", "current_only") or episode_index == 0:
        return {"context_matrix": prior.mean_matrix, "estimator": None, "donors": [], "history_observation_count": 0, "history_observation_sha256": observations_sha256(np.empty((0, 2)))}
    if policy == "true_factor_oracle":
        raise ValueError("true factor is handled by caller")
    own = np.asarray(evidence[(sequence_id, 0)]["matrix_observations"], dtype=np.float64)
    count = len(own)
    donors = []
    if policy == "correct_history":
        selected = own
        donors = [sequence_id] * count
    elif policy == "wrong_sequence_history":
        donor = (sequence_id + 1) % n_sequences
        selected = matched_observations(evidence[(donor, 0)]["matrix_observations"], count)
        donors = [donor] * count
    elif policy == "shuffled_history":
        pool = []
        for donor in range(n_sequences):
            if donor == sequence_id:
                continue
            for observation in evidence[(donor, 0)]["matrix_observations"]:
                pool.append((donor, observation))
        rng = np.random.default_rng(np.random.SeedSequence([DONOR_SEED, sequence_id]))
        selected_indexes = rng.permutation(len(pool))[:count]
        selected = np.asarray([pool[index][1] for index in selected_indexes], dtype=np.float64)
        donors = [int(pool[index][0]) for index in selected_indexes]
    else:
        raise ValueError(policy)
    context = BayesianMatrixContext()
    context.update_observations(selected)
    return {"context_matrix": context.mean_matrix, "estimator": context.as_dict(), "donors": donors, "history_observation_count": count, "history_observation_sha256": observations_sha256(selected)}


def generate_evidence(args, condition, wrapper, preprocessor, segments, env, sequence_ids):
    raw_path = args.output_dir / f"{condition}_raw.jsonl"
    completed = completed_keys(raw_path, "evidence_episode")
    for sequence_id in sequence_ids:
        for episode_index in range(N_EPISODES):
            if (sequence_id, episode_index) in completed:
                continue
            meta = scenario(condition, sequence_id, episode_index)
            started = time.perf_counter()
            initial, goal_obs, nominal_states, nominal_actions = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
            start_obs, _ = env.prepare(meta["env_seed"], initial)
            commands, planner = plan_matrix_waypoint(wrapper, preprocessor, start_obs, goal_obs, POPULATION_PRIOR_MATRIX, meta["cem_seed"])
            states, effective, contacts, coverages = execute_matrix(env, initial, meta["env_seed"], commands, meta["true_matrix"])
            observations, accepted_indices = infer_matrix_observations(commands, states)
            row = {"record_type": "evidence_episode", "contract_id": CONTRACT_ID, **meta, "initial_state": initial, "goal_state": nominal_states[-1], "nominal_actions": nominal_actions, "commands": commands, "effective_actions": effective, "states": states, "contacts": contacts, "coverages": coverages, "matrix_observations": observations, "accepted_indices": accepted_indices, "observation_sha256": observations_sha256(observations), "command_sha256": array_sha256(commands), "state_sha256": array_sha256(states), "planner": planner, "elapsed_s": time.perf_counter() - started, "resource": resource_snapshot(next(wrapper.parameters()).device)}
            append_jsonl(raw_path, row)
            print(f"EVIDENCE {condition} s={sequence_id} e={episode_index + 1} factor={meta['factor_index']} accepted={len(observations)}", flush=True)
    return evidence_lookup(raw_path)


def run_evaluations(args, condition, wrapper, preprocessor, segments, env, sequence_ids, evidence):
    raw_path = args.output_dir / f"{condition}_raw.jsonl"
    completed = completed_keys(raw_path, "evaluation_episode")
    for sequence_id in sequence_ids:
        for episode_index in range(N_EPISODES):
            if (sequence_id, episode_index) in completed:
                continue
            meta = scenario(condition, sequence_id, episode_index)
            started = time.perf_counter()
            initial, goal_obs, nominal_states, _ = prepare_waypoint(env, segments[meta["segment_index"]], meta["env_seed"])
            start_obs, _ = env.prepare(meta["env_seed"], initial)
            payloads = {}
            for policy in POLICIES:
                if policy == "true_factor_oracle":
                    payloads[policy] = {"context_matrix": np.asarray(meta["true_matrix"]), "estimator": None, "donors": [], "history_observation_count": 0, "history_observation_sha256": "true-factor"}
                else:
                    payloads[policy] = history_payload(evidence, policy, sequence_id, episode_index, len(sequence_ids))
            planned = {}
            policies = {}
            for policy in POLICIES:
                context = np.asarray(payloads[policy]["context_matrix"], dtype=np.float64)
                key = tuple(np.round(context.reshape(-1), 12))
                if key not in planned:
                    commands, planner = plan_matrix_waypoint(wrapper, preprocessor, start_obs, goal_obs, context, meta["cem_seed"])
                    states, effective, contacts, coverages = execute_matrix(env, initial, meta["env_seed"], commands, meta["true_matrix"])
                    planned[key] = {"commands": commands, "effective_actions": effective, "states": states, "contacts": contacts, "coverages": coverages, "metrics": pose_metrics(states, nominal_states[-1], WINDOW), "deadline_success": deadline_success(states, nominal_states[-1]), "command_sha256": array_sha256(commands), "effective_action_sha256": array_sha256(effective), "state_sha256": array_sha256(states), "planner": planner}
                policies[policy] = {**payloads[policy], **planned[key]}
            row = {"record_type": "evaluation_episode", "contract_id": CONTRACT_ID, **meta, "initial_state": initial, "goal_state": nominal_states[-1], "policies": policies, "elapsed_s": time.perf_counter() - started, "resource": resource_snapshot(next(wrapper.parameters()).device)}
            append_jsonl(raw_path, row)
            print(f"EVAL {condition} s={sequence_id} e={episode_index + 1} current={policies['current_only']['metrics']['pose_auc10']:.4f} history={policies['correct_history']['metrics']['pose_auc10']:.4f}", flush=True)


def bootstrap_ci(values, stream):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(stream))
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def effect(current, treatment, stream):
    current, treatment = np.asarray(current), np.asarray(treatment)
    delta = current - treatment
    return {"current_mean": float(current.mean()), "treatment_mean": float(treatment.mean()), "mean_delta": float(delta.mean()), "relative_improvement": float(delta.mean() / current.mean()), "bootstrap_ci95_delta": bootstrap_ci(delta, stream), "positive_fraction": float(np.mean(delta > 1e-12)), "tie_fraction": float(np.mean(np.abs(delta) <= 1e-12)), "negative_fraction": float(np.mean(delta < -1e-12)), "sequence_deltas": delta.tolist()}


def evaluation_rows(path):
    return [row for row in read_jsonl(path) if row.get("record_type") == "evaluation_episode"]


def evidence_rows(path):
    return [row for row in read_jsonl(path) if row.get("record_type") == "evidence_episode"]


def structural_checks(condition, eval_rows, ev_rows, n_sequences):
    evaluations = {(int(row["sequence_id"]), int(row["episode_index"])): row for row in eval_rows}
    evidence = {(int(row["sequence_id"]), int(row["episode_index"])): row for row in ev_rows}
    e1_identity = True
    current_prior_identity = True
    counts_equal = True
    donors_isolated = True
    evidence_positive = True
    factor_lifetime = True
    for sequence_id in range(n_sequences):
        factors = [factor_index_for(condition, sequence_id, episode) for episode in range(N_EPISODES)]
        factor_lifetime &= (factors[0] == factors[1]) if condition == "persistent" else (factors[0] != factors[1])
        evidence_positive &= len(evidence[(sequence_id, 0)]["matrix_observations"]) > 0
        for episode_index in range(N_EPISODES):
            policies = evaluations[(sequence_id, episode_index)]["policies"]
            current_prior_identity &= policies["current_only"]["command_sha256"] == policies["population_prior"]["command_sha256"] and policies["current_only"]["state_sha256"] == policies["population_prior"]["state_sha256"]
            if episode_index == 0:
                hashes = {(policies[policy]["command_sha256"], policies[policy]["state_sha256"]) for policy in POLICIES if policy != "true_factor_oracle"}
                e1_identity &= len(hashes) == 1
            else:
                history_counts = [policies[policy]["history_observation_count"] for policy in ("correct_history", "shuffled_history", "wrong_sequence_history")]
                counts_equal &= len(set(history_counts)) == 1 and history_counts[0] > 0
                donors_isolated &= all(donor != sequence_id for policy in ("shuffled_history", "wrong_sequence_history") for donor in policies[policy]["donors"])
    return {"complete_evidence": len(ev_rows) == n_sequences * N_EPISODES and len(evidence) == n_sequences * N_EPISODES, "complete_evaluation": len(eval_rows) == n_sequences * N_EPISODES and len(evaluations) == n_sequences * N_EPISODES, "episode_one_identity": bool(e1_identity), "current_population_identity": bool(current_prior_identity), "history_counts_equal": bool(counts_equal), "donors_isolated": bool(donors_isolated), "evidence_positive": bool(evidence_positive), "factor_lifetime": bool(factor_lifetime)}


def summarize(output_dir, n_sequences=N_SEQUENCES):
    eval_by_condition = {condition: evaluation_rows(Path(output_dir) / f"{condition}_raw.jsonl") for condition in CONDITIONS}
    ev_by_condition = {condition: evidence_rows(Path(output_dir) / f"{condition}_raw.jsonl") for condition in CONDITIONS}
    values = {}
    for condition, rows in eval_by_condition.items():
        e2 = {int(row["sequence_id"]): row for row in rows if int(row["episode_index"]) == 1}
        values[condition] = {policy: np.asarray([e2[sequence_id]["policies"][policy]["metrics"]["pose_auc10"] for sequence_id in range(n_sequences)]) for policy in POLICIES}
    persistent = effect(values["persistent"]["current_only"], values["persistent"]["correct_history"], 100)
    no_persistence = effect(values["no_persistence"]["current_only"], values["no_persistence"]["correct_history"], 200)
    true_effect = effect(values["persistent"]["current_only"], values["persistent"]["true_factor_oracle"], 300)
    did_values = np.asarray(persistent["sequence_deltas"]) - np.asarray(no_persistence["sequence_deltas"])
    persistent_e2 = {int(row["sequence_id"]): row for row in eval_by_condition["persistent"] if int(row["episode_index"]) == 1}
    matrix_errors, gain_errors, angle_errors = [], [], []
    for sequence_id in range(n_sequences):
        row = persistent_e2[sequence_id]
        estimate = np.asarray(row["policies"]["correct_history"]["context_matrix"], dtype=np.float64)
        truth = np.asarray(row["true_matrix"], dtype=np.float64)
        matrix_errors.append(float(np.linalg.norm(estimate - truth)))
        estimator = row["policies"]["correct_history"]["estimator"]
        gain_errors.append(abs(float(estimator["gain"]) - float(row["gain"])))
        angle_errors.append(abs((float(estimator["rotation_degrees"]) - float(row["rotation_degrees"]) + 180) % 360 - 180))
    result = {"contract_id": CONTRACT_ID, "primary_metric": "E2_pose_auc10_to_waypoint", "n_sequences_per_condition": n_sequences, "episodes_per_sequence": N_EPISODES, "persistent_correct_history": persistent, "no_persistence_correct_history": no_persistence, "persistent_true_factor": true_effect, "persistent_shuffled_history": effect(values["persistent"]["current_only"], values["persistent"]["shuffled_history"], 400), "persistent_wrong_sequence_history": effect(values["persistent"]["current_only"], values["persistent"]["wrong_sequence_history"], 500), "did": {"mean": float(did_values.mean()), "bootstrap_ci95": bootstrap_ci(did_values, 600), "positive_fraction": float(np.mean(did_values > 1e-12)), "sequence_values": did_values.tolist()}, "true_gap_recovery": float(np.mean(persistent["sequence_deltas"]) / np.mean(true_effect["sequence_deltas"])), "deadline_success": {}, "estimator": {"matrix_frobenius_mae": float(np.mean(matrix_errors)), "matrix_frobenius_max": float(np.max(matrix_errors)), "gain_mae": float(np.mean(gain_errors)), "gain_max": float(np.max(gain_errors)), "angle_mae_degrees": float(np.mean(angle_errors)), "angle_max_degrees": float(np.max(angle_errors))}, "by_persistent_factor": {}, "structural_checks": {condition: structural_checks(condition, eval_by_condition[condition], ev_by_condition[condition], n_sequences) for condition in CONDITIONS}}
    for condition in CONDITIONS:
        e2_rows = [row for row in eval_by_condition[condition] if int(row["episode_index"]) == 1]
        result["deadline_success"][condition] = {policy: float(np.mean([row["policies"][policy]["deadline_success"] for row in e2_rows])) for policy in POLICIES}
    for factor_index, (rotation, gain) in enumerate(FORMAL_FACTORS):
        ids = [sequence_id for sequence_id in range(n_sequences) if sequence_id % len(FORMAL_FACTORS) == factor_index]
        result["by_persistent_factor"][f"theta={rotation:+g},gain={gain:g}"] = effect(values["persistent"]["current_only"][ids], values["persistent"]["correct_history"][ids], 700 + factor_index)
    result["valid"] = all(all(checks.values()) for checks in result["structural_checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run", "summarize"))
    parser.add_argument("--limit-sequences", type=int, default=N_SEQUENCES)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"))
    parser.add_argument("--data", type=Path, default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"))
    parser.add_argument("--design", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_matrix_stage1_design.json"))
    parser.add_argument("--contract", type=Path, default=Path("docs/research/persistent_context_v2_pushobj_matrix_stage1_contract_zh.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/persistent_context_v2_pushobj_matrix_stage1"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        segments = pickle.load(handle)["segments"]
    if args.mode == "inspect":
        payload = {"contract_id": CONTRACT_ID, "design_sha256": sha256(args.design), "contract_sha256": sha256(args.contract), "prior_mean_z": prior_parameters()[0], "prior_covariance": prior_parameters()[1], "population_prior_matrix": matrix_from_z(prior_parameters()[0]), "scenarios": [scenario(condition, sequence_id, episode_index) for condition in CONDITIONS for sequence_id in range(N_SEQUENCES) for episode_index in range(N_EPISODES)]}
        dump_json(args.output_dir / "selection_audit.json", payload)
        print(json.dumps(payload, indent=2, default=lambda value: np.asarray(value).tolist()))
        return
    manifest_path = args.output_dir / "manifest.json"
    if args.mode == "summarize":
        result = summarize(args.output_dir, N_SEQUENCES)
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
        return
    if sha256(args.design) != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("frozen design hash mismatch")
    seed_all(1_040_401)
    device = torch.device(args.device)
    if not manifest_path.exists():
        manifest = {"contract_id": CONTRACT_ID, "git_revision": git_revision(), "design_path": str(args.design), "design_sha256": sha256(args.design), "contract_path": str(args.contract), "contract_sha256": sha256(args.contract), "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "data": str(args.data), "data_sha256": sha256(args.data), "command": " ".join(__import__("sys").argv), "started_unix": time.time(), "resource_start": resource_snapshot(device)}
        dump_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base, wrapper, _ = load_matrix_world_model(args.checkpoint, device)
    preprocessor, env = make_preprocessor(), make_env()
    sequence_ids = list(range(args.limit_sequences))
    if "identity_audit" not in manifest:
        first = scenario("persistent", 0, 0)
        initial, _, _, _ = prepare_waypoint(env, segments[first["segment_index"]], first["env_seed"])
        start_obs, _ = env.prepare(first["env_seed"], initial)
        manifest["identity_audit"] = identity_audit(base, wrapper, preprocessor, start_obs)
        dump_json(manifest_path, manifest)
    for condition in CONDITIONS:
        evidence = generate_evidence(args, condition, wrapper, preprocessor, segments, env, sequence_ids)
        run_evaluations(args, condition, wrapper, preprocessor, segments, env, sequence_ids, evidence)
    if args.limit_sequences == N_SEQUENCES:
        result = summarize(args.output_dir, N_SEQUENCES)
        dump_json(args.output_dir / "runner_summary.json", result)
        print(json.dumps(result, indent=2))
    manifest["finished_unix"] = time.time()
    manifest["resource_end"] = resource_snapshot(device)
    dump_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
