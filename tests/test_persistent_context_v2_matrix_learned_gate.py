import numpy as np

from research.persistent_context_v2.matrix_learned_gate import fit_ridge, gate_outcome


def test_ridge_recovers_unregularized_linear_target():
    x = np.asarray([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = 0.5 + 2.0 * x[:, 1]
    beta = fit_ridge(x, y, 0.0)
    assert np.allclose(beta, [0.5, 2.0], atol=1e-12)


def test_gate_outcome_selects_context_only_on_true_decisions():
    population = np.asarray([3.0, 4.0, 5.0])
    context = np.asarray([2.0, 6.0, 1.0])
    result = gate_outcome(population, context, np.asarray([True, False, True]))
    assert np.array_equal(result, [2.0, 4.0, 1.0])
