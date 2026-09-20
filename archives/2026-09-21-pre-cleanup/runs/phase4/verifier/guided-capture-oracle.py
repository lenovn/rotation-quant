"""CPU-only toy oracle for the new saliency capture; no Llama forward."""
from unittest.mock import patch
import torch
from experiments.phase4 import guided_round as guided
from experiments.phase3.quantization import SP2Quantizer, SP2ScaleSTE


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer = SP2Quantizer(1.0)
        self.module = torch.nn.Linear(4, 4, bias=False).bfloat16()

    def forward(self, values):
        return self.module(self.quantizer(values))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target = Wrapper()
        self.suffix = Wrapper()

    def forward(self, values):
        return self.suffix(self.target(values))


torch.set_num_threads(1)
torch.manual_seed(29)
model = Toy().eval().requires_grad_(False)
windows = [torch.randn(1, 7, 4).bfloat16() for _ in range(2)]
before = [model(x).detach().clone() for x in windows]
oracle = []
for x in windows:
    leaf = model.target(x).detach().requires_grad_(True)
    quant = SP2ScaleSTE.apply(leaf, model.suffix.quantizer.scale,
                            model.suffix.quantizer.levels)
    result = model.suffix.module(quant)
    gradient, = torch.autograd.grad(result.float().square().mean() * 1000, leaf)
    oracle.append(gradient.float().square().mean(-1).reshape(-1))
oracle = torch.cat(oracle)
seen = []


def loss_function(m, x):
    result = m(x)
    seen.append(result.detach().clone())
    return result.float().square().mean()


with patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self), \
     patch.object(torch.nn.Module, "cuda", lambda self, *a, **kw: self), \
     patch.object(guided, "wrappers", lambda m: {
         "model.layers.0.mlp.down_proj": m.target,
         "model.layers.1.mlp.down_proj": m.suffix}), \
     patch.object(guided, "token_nll", loss_function), \
     patch.object(guided, "progress", lambda *a, **kw: None):
    captured = guided.capture_saliency(model, "target", windows, None)

torch.testing.assert_close(captured, oracle, rtol=0, atol=0)
assert bool((captured > 0).any())
assert captured.shape == (14,)
assert all(torch.equal(a, b) for a, b in zip(before, seen))
assert all(not p.requires_grad and p.grad is None for p in model.parameters())
assert all(not m._forward_hooks for m in model.modules())
assert all(torch.equal(a, model(x)) for a, x in zip(before, windows))
print(dict(status="PASS", rows=14, nonzero_rows=int((captured > 0).sum()),
           forward_bitwise_equal=True, explicit_suffix_gradient_bitwise_equal=True,
           frozen_parameters=True, hooks_removed=True, full_model_forwards=0,
           gpu_used=False))
