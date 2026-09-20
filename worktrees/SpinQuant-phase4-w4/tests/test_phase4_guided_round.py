import torch
from experiments.phase4.guided_round import weighted_inputs
from experiments.phase3.quantization import SP2Quantizer, SP2ScaleSTE


def test_weighted_gram_matches_saliency_objective():
    torch.manual_seed(42)
    x = torch.randn(32, 9)
    error = torch.randn(5, 9)
    saliency = torch.rand(32).square()
    saliency[0] = 0
    scaled = weighted_inputs(x, saliency, saliency.mean())
    actual = ((error @ scaled.T).square()).sum()
    oracle = ((error @ x.T).square() * (saliency/saliency.mean())[None, :]).sum()
    torch.testing.assert_close(actual, oracle)
    assert torch.equal(weighted_inputs(x, torch.ones(32), torch.tensor(1.)), x)


def test_sp2_sensitivity_uses_same_forward_and_clipped_ste():
    x = torch.tensor([-2., -.37, 0., .47, 2.], requires_grad=True)
    q = SP2Quantizer(1.)
    previous = q.scale.clone()
    actual = SP2ScaleSTE.apply(x, q.scale, q.levels)
    assert torch.equal(actual, q(x))
    actual.sum().backward()
    assert torch.equal(x.grad, torch.tensor([0., 1., 1., 1., 0.]))
    assert torch.equal(previous, q.scale)
