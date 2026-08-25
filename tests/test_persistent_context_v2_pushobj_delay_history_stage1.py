import numpy as np

from research.persistent_context_v2.pushobj_delay_history_stage1 import (
    CANDIDATES,
    DiscreteDelayPosterior,
    FACTORS,
    factor_for,
    history_payload,
    shuffled_commands,
)
from research.persistent_context_v2.pushobj_delay_stage0 import apply_discrete_delay
from research.persistent_context_v2.pushobj_rotation_stage1 import pd_coefficients


def states_from_effective(effective):
    effective = np.asarray(effective, dtype=np.float64)
    p_coef, v_coef, target_coef = pd_coefficients()
    states = np.zeros((len(effective) + 1, 7), dtype=np.float64)
    states[0, :2] = [3.0, -2.0]
    states[0, 5:7] = [0.25, -0.1]
    for index, action in enumerate(effective):
        p0, v0 = states[index, :2], states[index, 5:7]
        target = p0 + 100.0 * action
        states[index + 1, :2] = p_coef * p0 + v_coef * v0 + target_coef * target
        # Only position is needed by the next independent transition in this fixture.
        states[index + 1, 5:7] = 0.0
    return states


def evidence_row(delay, offset=0.0):
    commands = np.stack([np.linspace(-0.4, 0.5, 10), np.linspace(0.3, -0.6, 10)], axis=1) + offset
    states = states_from_effective(apply_discrete_delay(commands, delay))
    return {"commands": commands, "states": states, "evidence_sha256": f"evidence-{delay}-{offset}"}


def test_exact_proprio_history_recovers_every_candidate_delay():
    for delay in CANDIDATES:
        row = evidence_row(delay)
        posterior = DiscreteDelayPosterior()
        posterior.update(row["commands"], row["states"])
        assert posterior.map_delay == delay
        assert posterior.evidence_count == 10
        assert posterior.episode_count == 1
        assert posterior.as_dict()["posterior_probability"][delay] > 0.95


def test_factor_schedule_is_paired_balanced_and_changes_only_no_persistence_e2():
    persistent_e2 = [factor_for("persistent", sequence, 1) for sequence in range(4)]
    no_persistence_e2 = [factor_for("no_persistence", sequence, 1) for sequence in range(4)]
    assert persistent_e2 == list(FACTORS)
    assert sorted(no_persistence_e2) == sorted(FACTORS)
    assert all(a != b for a, b in zip(persistent_e2, no_persistence_e2))
    assert [factor_for("persistent", sequence, 0) for sequence in range(4)] == [
        factor_for("no_persistence", sequence, 0) for sequence in range(4)
    ]


def test_history_controls_have_explicit_donors_and_zero_current_evidence():
    evidence = {sequence: evidence_row(delay, sequence * 0.01) for sequence, delay in enumerate(FACTORS)}
    correct = history_payload(evidence, "correct_history_map", 0)
    wrong = history_payload(evidence, "wrong_sequence_history", 0)
    current = history_payload(evidence, "current_only", 0)
    assert correct["context_delay_steps"] == FACTORS[0]
    assert correct["donors"][0]["sequence_id"] == 0
    assert wrong["context_delay_steps"] == FACTORS[1]
    assert wrong["donors"][0]["sequence_id"] == 1
    assert current["context_delay_steps"] == 2
    assert all(payload["current_episode_evidence_count"] == 0 for payload in (correct, wrong, current))


def test_shuffle_is_deterministic_and_breaks_time_alignment():
    commands = np.arange(20, dtype=np.float64).reshape(10, 2)
    first, order1 = shuffled_commands(commands, 2)
    second, order2 = shuffled_commands(commands, 2)
    assert order1 == order2
    assert not np.array_equal(order1, np.arange(10))
    assert np.array_equal(first, second)
    assert sorted(order1) == list(range(10))
