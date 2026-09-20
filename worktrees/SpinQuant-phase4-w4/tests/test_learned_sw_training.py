import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from test_w4_aware_training import make_model
from train_utils.rotation_calibration import (
    enable_weight_scale_learning, export_rotation_scales,
    refresh_current_rotation_weight_scales, LearnableWeightScaleCallback,
)
from utils.quant_utils import ChannelScaleQuantize, STEQuantize


def test_channel_gradient_reference_and_b_ste():
    x = torch.tensor([[-2., -.22, .36, 1.9], [-.8, .17, .25, .81]], requires_grad=True)
    s = torch.tensor([[.2], [.1]], requires_grad=True)
    g = torch.tensor([[1., 2., -3., 4.], [2., -1., 3., -2.]])
    y = ChannelScaleQuantize.apply(x, s, torch.tensor(7))
    assert torch.equal(y, STEQuantize.apply(x, s, torch.tensor(7)))
    (y * g).sum().backward()
    expected = []
    for row, scale, grad in zip(x.detach(), s.detach(), g):
        terms = []
        for v, upstream in zip(row, grad):
            z = float(v / scale)
            term = -8 if z < -8 else 7 if z > 7 else round(z) - z
            terms.append(float(upstream) * term)
        expected.append([sum(terms) / math.sqrt(4 * 7)])
    assert torch.allclose(s.grad, torch.tensor(expected), atol=1e-6)
    assert torch.equal(x.grad, g)


def test_persistent_sw_checkpoint_updates_and_no_overwrite():
    model = make_model()
    refresh_current_rotation_weight_scales(model)
    initial = export_rotation_scales(model)['weight']
    params = enable_weight_scale_learning(model, initial)
    assert len(params) == 7
    optimizer = torch.optim.SGD(params + [model.R1.weight], lr=.01)
    callback = LearnableWeightScaleCallback()
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None, model=model)
    callback.on_step_begin(None, state, None, model=model, optimizer=optimizer)
    linear = model.model.layers[0].self_attn.q_proj.module
    sw = linear.quantizer.scale
    for _ in range(2):
        x = torch.tensor([[.31, -.72]], requires_grad=True)
        checkpoint(lambda t: linear(t, R1=model.R1.weight), x, use_reentrant=True).square().sum().backward()
        assert linear.quantizer.scale is sw
    assert sw.grad is not None and torch.count_nonzero(sw.grad)
    assert model.R1.weight.grad is not None
    assert linear.weight.grad is None
    optimizer.step()
    state.global_step = 1
    callback.on_step_end(None, state, None, model=model)
    assert linear.quantizer.scale is sw and sw.dtype == torch.float32
    assert any(r['changed_channels'] for r in callback.history[0]['modules'].values())
    with pytest.raises(RuntimeError, match='Cannot refresh'):
        refresh_current_rotation_weight_scales(model)
    with pytest.raises(RuntimeError, match='Cannot recalibrate'):
        linear.quantizer.begin_calibration()
    with pytest.raises(RuntimeError, match='Cannot overwrite'):
        linear.quantizer.load_scale(torch.ones_like(sw))
