import torch
from fixed_code_sw import bounded_scale_fit


def test_constrained_fit_matches_output_error_minimum():
    qy=torch.tensor([[1.,2.,0.],[2.,-1.,0.],[-3.,4.,0.]])
    target=qy*torch.tensor([1.08,.8,9.])[None,:]
    initial=torch.ones(3,1)
    result=bounded_scale_fit((qy*target).sum(0),qy.square().sum(0),initial)
    assert torch.allclose(result[:,0],torch.tensor([1.08,1.,1.]))
    for value in torch.linspace(1,1.125,51):
        assert ((qy*result[:,0]-target).square().sum(0)<=
                (qy*value-target).square().sum(0)+1e-6).all()
    assert torch.equal(initial,torch.ones(3,1))


def test_expansion_is_bounded_and_nan_rejected():
    initial=torch.tensor([[.5],[2.]])
    result=bounded_scale_fit(torch.tensor([100.,-5.]),torch.ones(2),initial)
    assert torch.equal(result,torch.tensor([[.5625],[2.]]))
    try:bounded_scale_fit(torch.tensor([float('nan'),1.]),torch.ones(2),initial)
    except AssertionError:pass
    else:raise AssertionError('NaN accepted')
