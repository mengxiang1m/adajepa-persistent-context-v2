import unittest

import numpy as np
import torch

from research.persistent_context_v2.pointmaze_lag_stage0 import (
    DEV_FACTORS,
    POPULATION_PRIOR,
    LagWorldModel,
    _adjacency,
    _graph_distance,
    local_waypoint_start_goal,
)
from research.persistent_context_v2.pointmaze_transfer import DataStats


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class PointMazeLagStage0Test(unittest.TestCase):
    def _model(self):
        stats = DataStats(
            action_mean=torch.zeros(2),
            action_std=torch.ones(2),
            state_mean=torch.zeros(4),
            state_std=torch.ones(4),
        )
        return LagWorldModel(_Base(), stats)

    def test_population_prior_and_dev_factors_are_frozen(self):
        self.assertEqual(POPULATION_PRIOR, 0.4)
        self.assertEqual(DEV_FACTORS, (0.1, 0.3, 0.5, 0.7))

    def test_lag_filter_resets_and_has_expected_recurrence(self):
        model = self._model()
        model.set_context(0.5)
        action = torch.zeros(1, 1, 10)
        action[..., 0::2] = 1.0
        filtered = model.lag_actions(action).reshape(5, 2)
        expected_x = torch.tensor([0.5, 0.75, 0.875, 0.9375, 0.96875])
        torch.testing.assert_close(filtered[:, 0], expected_x)
        torch.testing.assert_close(filtered[:, 1], torch.zeros(5))
        second = model.lag_actions(action).reshape(5, 2)
        torch.testing.assert_close(second, filtered)

    def test_local_waypoint_is_deterministic_and_graph_distance_two(self):
        start0, goal0 = local_waypoint_start_goal(220123)
        start1, goal1 = local_waypoint_start_goal(220123)
        np.testing.assert_array_equal(start0, start1)
        np.testing.assert_array_equal(goal0, goal1)
        graph = _adjacency()
        source = min(graph, key=lambda cell: np.linalg.norm(start0[:2] - np.asarray(cell)))
        target = min(graph, key=lambda cell: np.linalg.norm(goal0[:2] - np.asarray(cell)))
        self.assertEqual(_graph_distance(graph, source, target), 2)
        np.testing.assert_array_equal(start0[2:], np.zeros(2))
        np.testing.assert_array_equal(goal0[2:], np.zeros(2))


if __name__ == "__main__":
    unittest.main()
