import torch
from experiments.phase4.scale_round import fit_scale
from experiments.phase3.quantization import pack_int4, unpack_int4


def test_closed_form_matches_output_space_oracle():
    torch.manual_seed(17)
    q = torch.randint(-8, 8, (5, 12), dtype=torch.int8)
    x = torch.randn(37, 12)
    w = torch.randn(5, 12) * .02 + q.float() * .15
    old = torch.full((5, 1), .1)
    fitted, report = fit_scale(q, old, w, x)
    a, b = x @ q.float().T, x @ w.T
    oracle = ((a * b).sum(0) / a.square().sum(0)).reshape(-1, 1)
    torch.testing.assert_close(fitted, oracle)
    assert report['accepted_rows'] == 5
    assert report['fit_mse_after'] < report['fit_mse_before']


def test_bf16_fallback_and_packed_export():
    torch.manual_seed(5)
    q = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
    q[0] = 0
    x = torch.randn(64, 16).bfloat16()
    old = torch.full((8, 1), .0625)
    w = (q.float() * .063 + .003 * torch.randn(8, 16)).bfloat16()
    w[1] = (-q[1].float() * .0625).bfloat16()
    fitted, _ = fit_scale(q, old, w, x)
    assert torch.equal(fitted[:2], old[:2])
    before = (x.float() @ ((q.float() * old).bfloat16().float() - w.float()).T).square().mean(0)
    after = (x.float() @ ((q.float() * fitted).bfloat16().float() - w.float()).T).square().mean(0)
    assert torch.all(after <= before + 1e-7)
    assert fitted.shape == (8, 1) and torch.all(fitted > 0)
    restored = unpack_int4(pack_int4(q), q.shape)
    assert torch.equal(restored, q)
    assert torch.equal((restored.float() * fitted).bfloat16(), (q.float() * fitted).bfloat16())
