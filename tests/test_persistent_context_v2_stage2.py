import tempfile, unittest
from pathlib import Path
import numpy as np
from research.persistent_context_v2.stage2 import RLSContext, Stage2Config, _donors, _scenario, run_stage2


class PersistentContextV2Stage2Test(unittest.TestCase):
    def test_rls_recovers_continuous_gain_without_factor_support(self):
        context = RLSContext()
        for action in (0.5, -0.8, 0.7): context.update(action, 1.237 * action)
        self.assertAlmostEqual(context.mean(0.015), 1.237, places=3)
        self.assertEqual(context.count, 3)

    def test_scenario_lifetime_and_donor_isolation(self):
        config = Stage2Config(n_sequences=16, n_episodes=4, bootstrap_resamples=20)
        scenario = _scenario(config, "persistent")
        negative = _scenario(config, "no_persistence")
        np.testing.assert_array_equal(scenario["targets"], negative["targets"])
        np.testing.assert_array_equal(scenario["noises"], negative["noises"])
        self.assertTrue(all(len(set(row.tolist())) == 1 for row in scenario["factor_ids"]))
        donors = _donors(config, "persistent")
        self.assertTrue(all(np.all(donors[episode] != np.arange(config.n_sequences)) for episode in range(1, config.n_episodes)))

    def test_smoke_deterministic_audited_and_non_overwriting(self):
        config = Stage2Config(n_sequences=24, n_episodes=4, bootstrap_resamples=200)
        with tempfile.TemporaryDirectory() as temp_dir:
            first_dir, second_dir = Path(temp_dir) / "first", Path(temp_dir) / "second"
            first = run_stage2(first_dir, config, "smoke"); second = run_stage2(second_dir, config, "smoke")
            self.assertTrue(first["audit"]["passed"]); self.assertEqual(first["effects"], second["effects"])
            with self.assertRaises(FileExistsError): run_stage2(first_dir, config, "overwrite")


if __name__ == "__main__": unittest.main()
