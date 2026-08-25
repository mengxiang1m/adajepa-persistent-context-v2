import numpy as np

from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    ACTION_COUNT,
    FACTORS_DEG,
    SEGMENT_INDICES,
    deadline_success,
    nominal_block_displacement_at_10,
    scenario_rows,
)


def _segments():
    rows = []
    for index in range(500):
        states = np.zeros((26, 7), dtype=np.float32)
        states[10, 2] = 12.0
        rows.append({"states": states, "actions": np.zeros((25, 2)), "shape": "T"})
    return rows


def test_frozen_waypoint_selection_is_unique_and_balanced():
    rows = scenario_rows(_segments())
    assert len(rows) == 32
    assert len(set(SEGMENT_INDICES)) == 32
    assert [row["factor_deg"] for row in rows[::8]] == list(FACTORS_DEG)
    assert all(row["nominal_block_displacement_at_10"] == 12.0 for row in rows)


def test_waypoint_displacement_uses_block_not_agent():
    segment = _segments()[0]
    segment["states"][10, :2] = [100.0, 100.0]
    assert nominal_block_displacement_at_10(segment) == 12.0


def test_deadline_success_uses_block_pose_tolerances():
    goal = np.zeros(7)
    states = np.zeros((ACTION_COUNT + 1, 7))
    states[-1, :2] = [1000.0, 1000.0]
    states[-1, 2:4] = [19.0, 0.0]
    states[-1, 4] = np.pi / 10.0
    assert deadline_success(states, goal)
    states[-1, 2] = 21.0
    assert not deadline_success(states, goal)
