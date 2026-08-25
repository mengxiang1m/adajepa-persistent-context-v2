import torch

from research.persistent_context_v2.pushobj_cog_temporal_predictor import TemporalCoGFiLMResidual


def test_temporal_zero_context_is_exact_identity():
    torch.manual_seed(10)
    model = TemporalCoGFiLMResidual()
    trajectory = torch.randn(4, 108)
    output = model(trajectory, torch.zeros(4))
    assert output.shape == (4, 30)
    assert torch.count_nonzero(output).item() == 0


def test_temporal_model_is_causal_across_steps():
    torch.manual_seed(11)
    model = TemporalCoGFiLMResidual().eval()
    first = torch.randn(1, 108)
    second = first.clone()
    # Encoded state 10 and action 9 affect only transition step 9.
    second[:, 80:88] += 3.0
    second[:, 106:108] -= 2.0
    with torch.no_grad():
        out_first = model(first, torch.tensor([15.0])).reshape(1, 10, 3)
        out_second = model(second, torch.tensor([15.0])).reshape(1, 10, 3)
    torch.testing.assert_close(out_first[:, :9], out_second[:, :9])
    assert not torch.allclose(out_first[:, 9], out_second[:, 9])
