import numpy as np

from research.persistent_context_v2.matrix_soft_policy_d2 import decisions, dose_matrix, fit_model, predict


def test_dose_matrix_zero_and_full_context():
    x = np.asarray([[1.0, 2.0], [1.0, -1.0]])
    phi = dose_matrix(x, np.asarray([0.0, 1.0]))
    assert np.array_equal(phi[0], np.zeros(4))
    assert np.array_equal(phi[1], np.asarray([1.0, -1.0, 0.0, 0.0]))


def test_decision_tie_uses_smaller_alpha():
    prediction = np.asarray([[0.0, 1.0, 1.0, 0.5, 0.0], [0.0, -1.0, -2.0, -3.0, -4.0]])
    assert decisions(prediction).tolist() == [1, 0]


def test_fit_predict_recovers_simple_linear_dose():
    rows = []
    for value in (-2.0, -1.0, 1.0, 2.0):
        benefits = np.asarray([0.0, 0.25 * value, 0.5 * value, 0.75 * value, value])
        rows.append({"features": np.asarray([1.0, value]), "costs": 5.0 - benefits})
    model = fit_model(rows, 2, 0.01)
    prediction = predict(model, rows, 2, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.max(np.abs(prediction[:, 0])) == 0.0
    assert np.corrcoef(prediction[:, -1], np.asarray([-2.0, -1.0, 1.0, 2.0]))[0, 1] > 0.999
