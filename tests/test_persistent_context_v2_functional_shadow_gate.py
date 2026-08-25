import numpy as np

from research.persistent_context_v2.functional_shadow_gate import (
    deadzone_gate,
    delay_gate,
    effect,
    matrix_gate,
    matrix_parameters,
)


def test_functional_rules_at_boundaries():
    assert not deadzone_gate(0.1)
    assert deadzone_gate(0.10001)
    assert not delay_gate(2)
    assert delay_gate(3)
    low_gain = np.asarray([[0.82, 0.0], [0.0, 0.82]])
    high_gain_large_rotation = 1.18 * np.asarray([[np.cos(np.deg2rad(25)), -np.sin(np.deg2rad(25))], [np.sin(np.deg2rad(25)), np.cos(np.deg2rad(25))]])
    high_gain_small_rotation = 1.18 * np.asarray([[np.cos(np.deg2rad(10)), -np.sin(np.deg2rad(10))], [np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10))]])
    assert matrix_gate(low_gain)
    assert matrix_gate(high_gain_large_rotation)
    assert not matrix_gate(high_gain_small_rotation)
    gain, rotation = matrix_parameters(high_gain_large_rotation)
    assert abs(gain - 1.18) < 1e-12
    assert abs(rotation - 25.0) < 1e-12


def test_effect_uses_paired_population_minus_treatment():
    result = effect(np.asarray([2.0, 4.0]), np.asarray([1.0, 5.0]))
    assert result["mean_delta"] == 0.0
    assert result["positive_fraction"] == 0.5
    assert result["negative_fraction"] == 0.5
