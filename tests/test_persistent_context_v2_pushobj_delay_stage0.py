import numpy as np
import pytest
import torch

from research.persistent_context_v2.pushobj_delay_stage0 import (
    DelayWorldModel,
    apply_discrete_delay,
)


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(()))

    def rollout(self, obs, act):
        return {"action": act}, None

    def encode_obs(self, obs):
        return obs


def test_numpy_fifo_has_zero_prefix_and_exact_shift():
    commands = np.arange(20, dtype=np.float32).reshape(10, 2)
    got = apply_discrete_delay(commands, 3)
    assert np.array_equal(got[:3], np.zeros((3, 2), dtype=np.float32))
    assert np.array_equal(got[3:], commands[:-3])
    assert apply_discrete_delay(commands, 0) is commands


def test_delay_rejects_non_integer_or_negative_factor():
    commands = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_discrete_delay(commands, 1.5)
    with pytest.raises(ValueError):
        apply_discrete_delay(commands, -1)


def test_world_model_fifo_crosses_five_action_model_boundary():
    mean = torch.tensor([0.2, -0.3])
    std = torch.tensor([0.5, 2.0])
    model = DelayWorldModel(_Base(), mean, std)
    physical = torch.arange(20, dtype=torch.float32).reshape(1, 2, 5, 2) / 10.0
    normalized = ((physical - mean) / std).reshape(1, 2, 10)
    model.set_context(2)
    got = model.effective_normalized_actions(normalized)
    got_physical = (got.reshape(1, 2, 5, 2) * std + mean).reshape(1, 10, 2)
    expected = torch.zeros(1, 10, 2)
    expected[:, 2:] = physical.reshape(1, 10, 2)[:, :-2]
    assert torch.allclose(got_physical, expected, atol=1e-6)
    # Time 5 is in the second model step and must receive time 3, not zero.
    assert torch.allclose(got_physical[:, 5], physical.reshape(1, 10, 2)[:, 3])


def test_zero_delay_wrapper_is_object_identity():
    model = DelayWorldModel(_Base(), torch.zeros(2), torch.ones(2))
    action = torch.randn(3, 2, 10)
    model.set_context(0)
    assert model.effective_normalized_actions(action) is action
