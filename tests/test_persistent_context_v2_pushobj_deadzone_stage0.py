import numpy as np
import torch

from research.persistent_context_v2.pushobj_deadzone_stage0 import DeadZoneWorldModel, apply_radial_deadzone


class _Base(torch.nn.Module):
    def __init__(self): super().__init__(); self.p = torch.nn.Parameter(torch.zeros(()))
    def rollout(self, obs, act): return {"x": act}, None
    def encode_obs(self, obs): return obs


def test_radial_deadzone_threshold_direction_and_no_amplification():
    actions = np.asarray([[0.03, 0.04], [0.3, 0.4], [0.0, 0.0]], dtype=np.float32)
    got = apply_radial_deadzone(actions, 0.1)
    assert np.allclose(got[0], 0.0) and np.allclose(got[2], 0.0)
    assert np.allclose(got[1], [0.24, 0.32])
    assert np.all(np.linalg.norm(got, axis=1) <= np.linalg.norm(actions, axis=1) + 1e-8)


def test_zero_context_wrapper_is_exact_identity():
    model = DeadZoneWorldModel(_Base(), torch.zeros(2), torch.ones(2))
    action = torch.randn(3, 2, 10)
    model.set_context(0.0)
    assert model.effective_normalized_actions(action) is action
    model.set_context(0.1)
    zero = torch.zeros(1, 1, 10)
    assert torch.allclose(model.effective_normalized_actions(zero), zero)
