import numpy as np
import torch
import unittest

from research.persistent_context_v2.pointmaze_transfer import (
    ActionFiLM,
    AnalyticCalibration,
    DEV_FACTORS,
    FORMAL_FACTORS,
    POPULATION_PRIOR,
    ScalarRLS,
    TRAIN_FACTORS,
    hard_start_goal,
)


class PointMazeTransferTest(unittest.TestCase):
    def test_factor_splits_are_disjoint_and_prior_is_calibrated(self):
        self.assertTrue(set(TRAIN_FACTORS).isdisjoint(DEV_FACTORS))
        self.assertTrue(set(TRAIN_FACTORS).isdisjoint(FORMAL_FACTORS))
        self.assertTrue(set(DEV_FACTORS).isdisjoint(FORMAL_FACTORS))
        self.assertEqual(POPULATION_PRIOR, np.mean(TRAIN_FACTORS))

    def test_analytic_calibration_obeys_command_and_actuator_clipping(self):
        adapter = AnalyticCalibration()
        command = torch.tensor([[-2.0, 0.5], [0.75, 2.0]])
        actual = adapter(command, torch.tensor(1.5))
        expected = torch.tensor([[-1.0, 0.75], [1.0, 1.0]])
        torch.testing.assert_close(actual, expected)

    def test_film_has_only_eight_parameters_and_can_represent_gain(self):
        adapter = ActionFiLM()
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 8)
        with torch.no_grad():
            adapter.gamma.weight.fill_(1.0)
            adapter.gamma.bias.zero_()
            adapter.beta.weight.zero_()
            adapter.beta.bias.zero_()
        command = torch.tensor([[0.25, -0.5]])
        torch.testing.assert_close(adapter(command, torch.tensor([1.2])), command * 1.2)

    def test_scalar_rls_recovers_gain_from_action_response(self):
        regression = np.zeros((5, 2), dtype=np.float64)
        regression[3:, :] = np.eye(2)
        commands = np.tile(np.asarray([[0.4, -0.3]], dtype=np.float64), (100, 1))
        gain = 1.3
        states = np.zeros((101, 4), dtype=np.float64)
        states[1:, 2:] = gain * commands
        estimator = ScalarRLS(prior_precision=2.0)
        estimator.update(states, commands, regression)
        self.assertLess(abs(estimator.mean - gain), 0.03)
        self.assertEqual(estimator.transition_count, 100)

    def test_hard_scenario_sampling_is_deterministic(self):
        a0, g0 = hard_start_goal(70123)
        a1, g1 = hard_start_goal(70123)
        np.testing.assert_array_equal(a0, a1)
        np.testing.assert_array_equal(g0, g1)
        self.assertGreater(np.linalg.norm(a0[:2] - g0[:2]), 2.0)
