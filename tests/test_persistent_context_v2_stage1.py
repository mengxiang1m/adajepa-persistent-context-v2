import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.persistent_context_v2.stage1 import (
    CategoricalPosterior,
    Stage1Config,
    _history_donors,
    _scenario,
    run_stage1,
)


class PersistentContextV2Stage1Test(unittest.TestCase):
    def test_posterior_identifies_gain_from_transitions(self):
        posterior = CategoricalPosterior()
        for _ in range(3):
            posterior.update(0.75, 1.275 * 0.75, 0.015)
        self.assertAlmostEqual(posterior.mean(), 1.275, places=5)
        self.assertEqual(posterior.count, 3)

    def test_factor_lifetime_and_donors(self):
        config = Stage1Config(n_sequences=16, n_episodes=4, bootstrap_resamples=20)
        persistent = _scenario(config, "persistent")
        negative = _scenario(config, "no_persistence")
        np.testing.assert_array_equal(persistent["targets"], negative["targets"])
        np.testing.assert_array_equal(persistent["noises"], negative["noises"])
        self.assertTrue(all(len(set(row.tolist())) == 1 for row in persistent["factor_ids"]))
        self.assertTrue(any(len(set(row.tolist())) > 1 for row in negative["factor_ids"]))
        donors = _history_donors(config, "persistent")
        for episode in range(1, config.n_episodes):
            self.assertTrue(np.all(donors[episode] != np.arange(config.n_sequences)))

    def test_smoke_is_deterministic_audited_and_non_overwriting(self):
        config = Stage1Config(n_sequences=24, n_episodes=4, bootstrap_resamples=200)
        with tempfile.TemporaryDirectory() as temp_dir:
            first_dir = Path(temp_dir) / "first"
            second_dir = Path(temp_dir) / "second"
            first = run_stage1(first_dir, config=config, command="smoke")
            second = run_stage1(second_dir, config=config, command="smoke")
            self.assertTrue(first["audit"]["passed"])
            self.assertEqual(first["effects"], second["effects"])
            with self.assertRaises(FileExistsError):
                run_stage1(first_dir, config=config, command="overwrite")


if __name__ == "__main__":
    unittest.main()
