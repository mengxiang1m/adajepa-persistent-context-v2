import json
from pathlib import Path

import numpy as np

from research.persistent_context_v2.pushobj_rotation_stage0_audit import (
    raw_pose_auc,
    rotate,
    wrap_angle,
)


def test_auditor_rotation_matches_column_vector_convention():
    np.testing.assert_allclose(rotate([[1.0, 0.0]], 90), [[0.0, 1.0]], atol=1e-8)


def test_auditor_pose_cost_recomputes_from_states():
    states = np.zeros((3, 7))
    states[1, 2] = 20.0
    states[2, 4] = np.pi / 9
    assert np.isclose(raw_pose_auc(states, np.zeros(7), 2), 1.0)


def test_auditor_wraps_angles():
    np.testing.assert_allclose(wrap_angle([2 * np.pi - 0.2], 0), [0.2])

