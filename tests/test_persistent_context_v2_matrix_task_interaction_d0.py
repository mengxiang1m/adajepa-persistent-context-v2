import copy

import numpy as np

from research.persistent_context_v2.matrix_task_interaction_d0 import (
    FACTOR_FEATURES,
    geometry_action_features,
    model_interaction_features,
    normalize_extra_features,
    policy_summary,
)


def sample_row():
    return {
        "e2": {
            "initial_state": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "goal_state": [0.0, 0.0, 3.0, 0.0, 0.5, 0.0, 0.0],
            "population": {
                "commands": [[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                "metrics": {"pose_auc10": 3.0},
                "states": [[0.0] * 7],
            },
            "context": {
                "commands": [[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]],
                "metrics": {"pose_auc10": 2.0},
                "states": [[1.0] * 7],
            },
        }
    }


def test_geometry_action_features_do_not_read_outcomes_or_execution_states():
    row = sample_row()
    expected = geometry_action_features(row)
    changed = copy.deepcopy(row)
    changed["e2"]["population"]["metrics"]["pose_auc10"] = -999.0
    changed["e2"]["context"]["metrics"]["pose_auc10"] = 999.0
    changed["e2"]["population"]["states"] = [[123.0] * 7]
    changed["e2"]["context"]["states"] = [[-123.0] * 7]
    assert np.array_equal(expected, geometry_action_features(changed))


def test_model_interaction_features_have_expected_signs():
    features = model_interaction_features(
        {
            "J_prior_a_prior": 1.0,
            "J_prior_a_context": 1.5,
            "J_context_a_prior": 2.0,
            "J_context_a_context": 0.5,
        }
    )
    assert np.allclose(features, [0.5, 1.5, 1.0, -1.0])


def test_normalization_leaves_factor_basis_unchanged_and_handles_constant_extra():
    factor_count = len(FACTOR_FEATURES)
    x = np.ones((3, factor_count + 2), dtype=np.float64)
    x[:, 1] = [-1.0, 0.0, 1.0]
    x[:, factor_count] = [1.0, 2.0, 3.0]
    transformed, mean, scale = normalize_extra_features(x)
    result = transformed[0]
    assert np.array_equal(result[:, :factor_count], x[:, :factor_count])
    assert np.allclose(result[:, factor_count], [-1.224744871391589, 0.0, 1.224744871391589])
    assert np.array_equal(result[:, factor_count + 1], [0.0, 0.0, 0.0])
    assert scale[factor_count + 1] == 1.0
    assert mean[0] == 0.0


def test_policy_summary_counts_harm_and_ties():
    population = np.asarray([3.0, 3.0, 3.0])
    context = np.asarray([2.0, 4.0, 1.0])
    result = policy_summary(population, context, np.asarray([True, True, False]), 0)
    assert result["positive_fraction"] == 1 / 3
    assert result["harm_fraction"] == 1 / 3
    assert result["tie_fraction"] == 1 / 3
