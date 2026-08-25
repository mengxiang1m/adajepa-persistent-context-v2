import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.persistent_context_v2.stage0 import (
    DEVELOPMENT_FACTORS,
    FORMAL_FACTORS,
    Stage0Config,
    TRAIN_FACTORS,
    _scenario,
    population_prior,
    run_stage0,
)


class PersistentContextV2Stage0Test(unittest.TestCase):
    def test_splits_are_disjoint_and_prior_is_calibrated(self):
        self.assertFalse(set(TRAIN_FACTORS) & set(DEVELOPMENT_FACTORS))
        self.assertFalse(set(TRAIN_FACTORS) & set(FORMAL_FACTORS))
        self.assertFalse(set(DEVELOPMENT_FACTORS) & set(FORMAL_FACTORS))
        self.assertAlmostEqual(population_prior(), 1.0)

    def test_scenarios_are_deterministic_and_isolated(self):
        config = Stage0Config(n_sequences=4, bootstrap_resamples=20)
        self.assertEqual(_scenario(config, 0), _scenario(config, 0))
        self.assertNotEqual(_scenario(config, 0), _scenario(config, 1))

    def test_smoke_is_audited_deterministic_and_non_overwriting(self):
        config = Stage0Config(n_sequences=32, bootstrap_resamples=200)
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first"
            second_path = Path(temp_dir) / "second"
            first = run_stage0(first_path, config=config, command="smoke")
            second = run_stage0(second_path, config=config, command="smoke")
            self.assertEqual(first["candidates"], second["candidates"])
            self.assertFalse(first["formal_outcomes_generated"])
            self.assertTrue(all(item["criteria"]["all_512_pairs_valid_and_identical"] for item in first["candidates"]))
            with self.assertRaises(FileExistsError):
                run_stage0(first_path, config=config, command="overwrite")


if __name__ == "__main__":
    unittest.main()
