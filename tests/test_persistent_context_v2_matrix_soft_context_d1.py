import numpy as np
import pytest

from research.persistent_context_v2.matrix_soft_context_d1 import blend_context, dose_summary


def test_blend_endpoints_and_midpoint():
    prior = np.eye(2)
    posterior = np.asarray([[2.0, 1.0], [-1.0, 2.0]])
    assert np.array_equal(blend_context(prior, posterior, 0.0), prior)
    assert np.array_equal(blend_context(prior, posterior, 1.0), posterior)
    assert np.allclose(blend_context(prior, posterior, 0.25), 0.75 * prior + 0.25 * posterior)


def test_blend_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        blend_context(np.eye(2), np.eye(2), 1.1)


def test_dose_summary_detects_interior_best():
    design = {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0], "splits": ["train"], "bootstrap_seed": 1, "bootstrap_resamples": 100}
    rows = []
    for sequence, costs in enumerate(([5, 4, 3, 4, 5], [5, 5, 4, 3, 2])):
        for alpha, cost in zip(design["alphas"], costs):
            rows.append({"split": "train", "sequence_id": sequence, "factor_index": 0, "alpha": alpha, "metrics": {"pose_auc10": cost}})
    result = dose_summary(rows, design)
    assert result["n_sequences"] == 2
    assert result["exploratory_ceilings"]["interior_best_fraction"] == 0.5
    assert result["exploratory_ceilings"]["non_monotonic_fraction"] == 0.5
