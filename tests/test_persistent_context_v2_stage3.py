import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.stage3 import (
    FiLMScalarDynamics,
    Stage3Config,
    _formal_scenario,
    development_gate,
    run_stage3,
    train_model,
)


class PersistentContextV2Stage3Test(unittest.TestCase):
    def test_model_has_only_one_film_path_and_expected_shape(self):
        model = FiLMScalarDynamics(hidden_dim=16)
        action = torch.tensor([[0.2], [0.4]])
        context = torch.tensor([[0.7], [1.3]])
        self.assertEqual(tuple(model(action, context).shape), (2, 1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 113)

    def test_training_uses_context_on_unseen_development_factors(self):
        config = Stage3Config(
            train_samples=4_096,
            dev_prediction_samples=1_024,
            train_steps=500,
            eval_interval=50,
            batch_size=256,
            dev_sequences=64,
            dev_bootstrap_resamples=200,
            planner_candidates=101,
        )
        model, _, diagnostics = train_model(config, torch.device("cpu"))
        gate, _ = development_gate(model, config, torch.device("cpu"))
        self.assertEqual(diagnostics["parameter_count"], 113)
        self.assertLess(gate["prediction_mse"]["ratio"], 0.20)
        self.assertGreater(gate["behavior"]["action_change_fraction"], 0.70)

    def test_formal_conditions_share_nuisance_but_change_factor_lifetime(self):
        config = Stage3Config(formal_sequences=16, n_episodes=4)
        persistent = _formal_scenario(config, "persistent")
        negative = _formal_scenario(config, "no_persistence")
        np.testing.assert_array_equal(persistent["targets"], negative["targets"])
        np.testing.assert_array_equal(persistent["noises"], negative["noises"])
        self.assertTrue(all(len(set(row.tolist())) == 1 for row in persistent["factor_ids"]))
        self.assertTrue(any(len(set(row.tolist())) > 1 for row in negative["factor_ids"]))

    def test_end_to_end_smoke_is_audited_and_non_overwriting(self):
        config = Stage3Config(
            train_samples=4_096,
            dev_prediction_samples=1_024,
            train_steps=500,
            eval_interval=50,
            batch_size=256,
            dev_sequences=64,
            dev_bootstrap_resamples=200,
            formal_sequences=24,
            formal_bootstrap_resamples=200,
            n_episodes=4,
            planner_candidates=101,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "smoke"
            summary = run_stage3(output, config=config, command="smoke", device_name="cpu")
            self.assertTrue(summary["development_gate"]["passed"])
            self.assertTrue(summary["formal_outcomes_generated"])
            self.assertTrue(summary["formal_audit"]["passed"])
            with self.assertRaises(FileExistsError):
                run_stage3(output, config=config, command="overwrite", device_name="cpu")


if __name__ == "__main__":
    unittest.main()
