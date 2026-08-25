import math

import numpy as np
import torch

from research.persistent_context_v2.pushobj_rotation_stage0 import (
    DEV_FACTORS_DEG,
    RotationWorldModel,
    early_contact_pool,
    pose_metrics,
    rotate_actions,
    select_scenarios,
    wrapped_angle_error,
)


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def rollout(self, obs_0, act):
        return {"visual": act, "proprio": act}, None

    def encode_obs(self, obs):
        return obs


def _segments(n=600):
    result = []
    for index in range(n):
        states = np.zeros((26, 5), dtype=np.float32)
        states[:, 2] = np.linspace(0, 30 if index % 2 == 0 else 0, 26)
        result.append({"states": states, "actions": np.zeros((25, 2)), "shape": "T"})
    return result


def test_rotation_direction_and_norm():
    action = np.asarray([[1.0, 0.0]], dtype=np.float32)
    rotated = rotate_actions(action, 90.0)
    np.testing.assert_allclose(rotated, [[0.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=-1), [1.0], atol=1e-6)


def test_wrapper_identity_is_exact_and_nonzero_context_rotates_physical_action():
    wrapper = RotationWorldModel(_Base(), torch.tensor([0.1, -0.2]), torch.tensor([2.0, 4.0]))
    action = torch.randn(3, 5, 10)
    wrapper.set_context(0.0)
    assert wrapper.effective_normalized_actions(action) is action
    zero = torch.tensor([[[0.0, 0.0] * 5]])
    wrapper.set_context(90.0)
    got = wrapper.effective_normalized_actions(zero).reshape(1, 1, 5, 2)[0, 0, 0]
    # normalized zero means physical [0.1,-0.2], rotated to [0.2,0.1]
    expected = torch.tensor([(0.2 - 0.1) / 2.0, (0.1 + 0.2) / 4.0])
    torch.testing.assert_close(got, expected)


def test_wrapped_angle_and_pose_metric_excludes_initial():
    err = wrapped_angle_error(np.asarray([2 * math.pi - 0.1]), 0.0)
    np.testing.assert_allclose(err, [0.1])
    states = np.zeros((3, 7), dtype=np.float32)
    states[0, 2] = 1000.0
    states[1, 2] = 20.0
    states[2, 4] = math.pi / 9
    goal = np.zeros(7, dtype=np.float32)
    metrics = pose_metrics(states, goal, 2)
    np.testing.assert_allclose(metrics["pose_auc2"], 1.0, atol=1e-6)


def test_candidate_selection_is_disjoint_and_frozen():
    segments = _segments()
    for candidate in ("A_released", "B_early_contact"):
        rows = select_scenarios(segments, candidate)
        assert len(rows) == 32
        assert len({row["segment_index"] for row in rows}) == 32
        assert [row["factor_deg"] for row in rows[::8]] == list(DEV_FACTORS_DEG)
        assert all(0 <= row["segment_index"] < 500 for row in rows)
    assert len(early_contact_pool(segments)) == 250
