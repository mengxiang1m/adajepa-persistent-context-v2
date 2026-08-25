import math

import numpy as np

from research.persistent_context_v2.pushobj_rotation_stage0 import rotate_actions
from research.persistent_context_v2.pushobj_rotation_stage1 import (
    FORMAL_FACTORS_DEG,
    RotationMLE,
    donor_sequence,
    factor_for,
    formal_segment_indices,
    pd_coefficients,
)


def _free_motion_states(commands, degrees):
    commands = np.asarray(commands, dtype=np.float64)
    effective = rotate_actions(commands, degrees).astype(np.float64)
    states = np.zeros((len(commands) + 1, 7), dtype=np.float64)
    states[0, :2] = [240.0, 270.0]
    states[0, 5:7] = [0.0, 0.0]
    k_p, k_v, dt = 100.0, 20.0, 0.01
    for action_index, action in enumerate(effective):
        position = states[action_index, :2].copy()
        velocity = states[action_index, 5:7].copy()
        target = position + 100.0 * action
        for _ in range(10):
            velocity += (k_p * (target - position) - k_v * velocity) * dt
            position += velocity * dt
        states[action_index + 1, :2] = position
        states[action_index + 1, 5:7] = velocity
    return states


def test_pd_coefficients_reconstruct_target_and_rotation_mle():
    p_coef, v_coef, target_coef = pd_coefficients()
    assert np.isclose(p_coef + target_coef, 1.0)
    assert target_coef > 0.0
    commands = np.asarray([[0.2, -0.1], [-0.05, 0.3], [0.11, 0.17]])
    states = _free_motion_states(commands, 17.25)
    estimator = RotationMLE()
    estimator.update(commands, states)
    assert estimator.transition_count == 3
    assert estimator.accepted_count == 3
    assert abs(estimator.estimate_degrees - 17.25) < 1e-8


def test_estimator_rejects_dynamics_inconsistent_transition_without_contact_label():
    commands = np.asarray([[0.2, 0.1], [-0.2, 0.05]])
    states = _free_motion_states(commands, -10.0)
    states[1, :2] += 30.0
    estimator = RotationMLE()
    estimator.update(commands, states)
    assert estimator.transition_count == 2
    assert estimator.accepted_count < 2


def test_formal_selection_and_factor_lifetimes_are_frozen_and_balanced():
    indices = formal_segment_indices()
    assert len(indices) == 128
    assert len(set(indices.tolist())) == 128
    assert np.all((indices >= 500) & (indices < 1000))
    persistent = np.asarray(
        [[factor_for("persistent", sequence, episode) for episode in range(4)] for sequence in range(32)]
    )
    changing = np.asarray(
        [[factor_for("no_persistence", sequence, episode) for episode in range(4)] for sequence in range(32)]
    )
    assert all(len(set(row.tolist())) == 1 for row in persistent)
    assert all(len(set(row.tolist())) == 4 for row in changing)
    for episode in range(4):
        values, counts = np.unique(changing[:, episode], return_counts=True)
        assert values.tolist() == list(FORMAL_FACTORS_DEG)
        assert counts.tolist() == [8, 8, 8, 8]
    assert np.array_equal(persistent[:, 0], changing[:, 0])


def test_wrong_and_shuffled_donors_never_self_reference():
    for n_sequences in (4, 32):
        for sequence in range(n_sequences):
            wrong = [donor_sequence("wrong_sequence_history", sequence, h, n_sequences) for h in range(3)]
            shuffled = [donor_sequence("shuffled_history", sequence, h, n_sequences) for h in range(3)]
            assert all(donor != sequence for donor in wrong + shuffled)
            assert len(set(wrong)) == 1
            assert len(set(shuffled)) == 3
