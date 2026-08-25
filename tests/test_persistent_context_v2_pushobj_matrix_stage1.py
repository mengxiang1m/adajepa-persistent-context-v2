import numpy as np

from research.persistent_context_v2.pushobj_matrix_stage1 import (
    BayesianMatrixContext,
    FORMAL_FACTORS,
    factor_index_for,
    matrix_from_z,
    prior_parameters,
    z_from_factor,
)


def test_train_prior_matches_frozen_population_matrix():
    mean, covariance = prior_parameters()
    assert np.allclose(matrix_from_z(mean), np.eye(2) * 0.9327804920294028)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_bayesian_context_recovers_unlisted_rotation_and_gain():
    truth = z_from_factor(17.0, 1.11)
    context = BayesianMatrixContext()
    context.update_observations(np.repeat(truth[None], 10, axis=0))
    assert abs(context.gain - 1.11) < 2e-3
    assert abs(context.rotation_degrees - 17.0) < 0.1
    assert context.observation_count == 10


def test_factor_lifetime_and_no_persistence_marginals():
    persistent = [[factor_index_for("persistent", sequence, episode) for episode in range(2)] for sequence in range(32)]
    changing = [[factor_index_for("no_persistence", sequence, episode) for episode in range(2)] for sequence in range(32)]
    assert all(left == right for left, right in persistent)
    assert all(left != right for left, right in changing)
    for episode in range(2):
        assert sorted(row[episode] for row in changing) == sorted(list(range(len(FORMAL_FACTORS))) * 4)
