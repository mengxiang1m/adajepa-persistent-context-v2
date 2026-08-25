import numpy as np
import torch

from research.persistent_context_v2.pushobj_matrix_stage0 import MatrixWorldModel, apply_action_matrix, factor_matrix


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.parameter=torch.nn.Parameter(torch.zeros(()))
    def rollout(self, obs, act): return {"action":act},None
    def encode_obs(self, obs): return obs


def test_factor_matrix_matches_gain_times_rotation():
    matrix=factor_matrix(90,0.5,dtype=np.float64)
    got=apply_action_matrix(np.asarray([[2.0,0.0]]),matrix)
    assert np.allclose(got,[[0.0,1.0]],atol=1e-12)


def test_identity_action_and_wrapper_are_exact():
    actions=np.arange(12,dtype=np.float32).reshape(6,2)
    assert apply_action_matrix(actions,np.eye(2,dtype=np.float32)) is actions
    model=MatrixWorldModel(_Base(),torch.zeros(2),torch.ones(2)); normalized=torch.randn(3,2,10); model.set_context(np.eye(2))
    assert model.effective_normalized_actions(normalized) is normalized


def test_wrapper_denormalizes_transforms_and_renormalizes():
    mean=torch.tensor([.2,-.3]); std=torch.tensor([.5,2.]); model=MatrixWorldModel(_Base(),mean,std)
    physical=torch.tensor([[[[1.,2.],[3.,4.],[5.,6.],[7.,8.],[9.,10.]]]])
    normalized=((physical-mean)/std).reshape(1,1,10); matrix=np.asarray([[1.2,-.1],[.2,.8]]); model.set_context(matrix)
    got=model.effective_normalized_actions(normalized); got_physical=got.reshape(1,1,5,2)*std+mean
    expected=physical@torch.tensor(matrix,dtype=torch.float32).T
    assert torch.allclose(got_physical,expected,atol=1e-5)
