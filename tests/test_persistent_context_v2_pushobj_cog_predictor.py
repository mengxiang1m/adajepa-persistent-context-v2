import numpy as np
import torch

from research.persistent_context_v2.pushobj_cog_predictor import (
    CoGFiLMResidual,
    apply_residual,
    encode_trajectory,
    residual_target,
)


def test_zero_context_is_exact_identity():
    torch.manual_seed(3)
    model = CoGFiLMResidual()
    x = torch.randn(5, 108)
    out = model(x, torch.zeros(5))
    assert torch.count_nonzero(out).item() == 0


def test_encoding_and_residual_round_trip():
    rng = np.random.default_rng(4)
    nominal = rng.normal(size=(11, 7)).astype(np.float32)
    nominal[:, :4] += 256
    nominal[:, 4] = np.mod(nominal[:, 4], 2 * np.pi)
    true = nominal.copy()
    true[1:, 2:4] += rng.normal(0, 2, size=(10, 2))
    true[1:, 4] = np.mod(true[1:, 4] + rng.normal(0, 0.1, size=10), 2 * np.pi)
    commands = rng.normal(size=(10, 2)).astype(np.float32)
    assert encode_trajectory(commands, nominal).shape == (108,)
    restored = apply_residual(nominal, residual_target(true, nominal))
    np.testing.assert_allclose(restored[:, 2:4], true[:, 2:4], atol=2e-5)
    angle_error = np.arctan2(np.sin(restored[:, 4] - true[:, 4]), np.cos(restored[:, 4] - true[:, 4]))
    np.testing.assert_allclose(angle_error, 0, atol=2e-6)
