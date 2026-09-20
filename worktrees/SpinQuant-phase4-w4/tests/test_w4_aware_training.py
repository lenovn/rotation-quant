import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from train_utils.quant_linear import QuantizeLinear
from train_utils.rotation_calibration import (
    LearnableScaleClampCallback,
    enable_non_downproj_scale_learning,
    export_rotation_scales,
    refresh_current_rotation_weight_scales,
)
from utils.quant_utils import ActQuantWrapper, RotationStaticActQuantizer, RotationStaticWeightQuantizer
from utils.static_quant_policy import load_rotation_static_activation_scales


def make_model(nlayers=1):
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_hidden_layers=nlayers)
    model.R1 = torch.nn.Linear(2, 2, bias=False)
    model.R1.weight.data.copy_(torch.eye(2))
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(nlayers):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.R2 = torch.nn.Linear(2, 2, bias=False)
        layer.self_attn.R2.weight.data.copy_(torch.eye(2))
        layer.mlp = torch.nn.Module()
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                linear = QuantizeLinear(2, 2, bias=False)
                linear.weight.data.copy_(torch.tensor([[1., .2], [.1, 2.]]))
                linear.weight.requires_grad_(False)
                linear.quantizer = RotationStaticWeightQuantizer()
                linear.quantizer.configure(4, perchannel=True, sym=True, weight_groupsize=-1)
                wrapper = ActQuantWrapper(linear)
                wrapper.quantizer = RotationStaticActQuantizer().bfloat16()
                setattr(parent, name, wrapper)
        model.model.layers.append(layer)
    enable_non_downproj_scale_learning(model)
    for name, module in model.named_modules():
        if isinstance(module, RotationStaticActQuantizer) and module.bits < 16:
            module.load_scale(torch.tensor([.031234567]))
    return model


def test_current_sw_fixed_across_accumulation_checkpoint_and_final_refresh(monkeypatch):
    model = make_model()
    layer = model.model.layers[0].self_attn.q_proj.module
    optimizer = torch.optim.SGD([model.R1.weight], lr=.1)
    # Callback also needs the SA optimizer group to capture its used LR.
    optimizer.add_param_group({"params": [p for n, p in model.named_parameters() if n.endswith("quantizer.scale")]})
    callback = LearnableScaleClampCallback(w4_aware=True)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    refresh_calls = []
    original_refresh = layer.quantizer.refresh_current_weight

    def record_refresh(weight):
        refresh_calls.append(1)
        original_refresh(weight)

    monkeypatch.setattr(layer.quantizer, "refresh_current_weight", record_refresh)
    callback.on_step_begin(None, state, None, model=model, optimizer=optimizer)
    first = layer.quantizer.scale.clone()
    storage = layer.quantizer.scale.data_ptr()
    assert not layer.quantizer.scale.requires_grad
    assert layer.quantizer.scale.grad_fn is None
    for _ in range(2):
        x = torch.tensor([[.3, -.7]], requires_grad=True)
        checkpoint(lambda t: layer(t, R1=model.R1.weight), x, use_reentrant=True).square().sum().backward()
        assert torch.equal(layer.quantizer.scale, first)
    assert model.R1.weight.grad is not None
    assert layer.weight.grad is None
    assert len(refresh_calls) == 1
    optimizer.step()
    # Final export refresh uses the updated R without any activation forward.
    sa_before = export_rotation_scales(model)["activation"]
    refresh_current_rotation_weight_scales(model)
    assert len(refresh_calls) == 2
    expected = layer.rotated_weight(model.R1.weight).detach().float().abs().amax(1, keepdim=True).clamp_min(1e-5) / 7
    assert torch.equal(layer.quantizer.scale, expected)
    assert not torch.equal(first, expected)
    assert layer.quantizer.scale.data_ptr() == storage
    for name, value in export_rotation_scales(model)["activation"].items():
        assert torch.equal(value, sa_before[name])


def test_refresh_uses_r2_and_output_transpose():
    model = make_model()
    layer = model.model.layers[0]
    with torch.no_grad():
        model.R1.weight.copy_(torch.tensor([[.8, -.6], [.6, .8]]))
        layer.self_attn.R2.weight.copy_(torch.tensor([[0., -1.], [1., 0.]]))
    refresh_current_rotation_weight_scales(model)
    r1, r2 = model.R1.weight.detach(), layer.self_attn.R2.weight.detach()
    for wrapper, expected in (
        (layer.self_attn.v_proj, r2.T @ layer.self_attn.v_proj.weight @ r1),
        (layer.self_attn.o_proj, r1.T @ layer.self_attn.o_proj.weight @ r2),
        (layer.mlp.down_proj, r1.T @ layer.mlp.down_proj.weight),
    ):
        assert torch.allclose(wrapper.module.quantizer.scale, expected.abs().amax(1, keepdim=True) / 7)


def test_paired_sa_restores_exact_fp32_values_and_96_trainables():
    model = make_model(16)
    paired = export_rotation_scales(model)["activation"]
    parameters = {name: module.scale for name, module in model.named_modules()
                  if isinstance(module, RotationStaticActQuantizer) and module.scale_learnable}
    for module in model.modules():
        if isinstance(module, RotationStaticActQuantizer) and module.scale_learnable:
            module.begin_calibration()
            module(torch.tensor([[200.]]))
            module.finish_calibration()
    load_rotation_static_activation_scales(model, paired, down_proj_fp16=True)
    assert len(parameters) == 96
    for name, module in model.named_modules():
        if name in parameters:
            assert module.scale is parameters[name]
            assert module.scale.dtype == torch.float32
            assert torch.equal(module.scale, paired[name])
        if name.endswith("mlp.down_proj.quantizer"):
            assert module.bits == 16
    assert sum(p.requires_grad for p in model.parameters()) == 96 + 17


def test_callback_records_actual_delta_and_warmup_lr(tmp_path):
    model = make_model()
    scales = [p for n, p in model.named_parameters() if n.endswith("quantizer.scale")]
    optimizer = torch.optim.SGD(scales, lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 2, 1))
    callback = LearnableScaleClampCallback()
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None, model=model)
    for step in range(2):
        callback.on_step_begin(None, state, None, model=model, optimizer=optimizer)
        for scale in scales:
            scale.grad = torch.full_like(scale, .01)
        optimizer.step()
        callback.on_optimizer_step(None, state, None, model=model)
        scheduler.step()
        optimizer.zero_grad()
        state.global_step = step + 1
        callback.on_step_end(None, state, None, model=model)
    path = tmp_path / "sa_update_history.json"
    callback.save_history(path)
    history = json.loads(path.read_text())
    for module in history[0]["modules"].values():
        assert module["lr_used"] == 0
        assert module["delta_abs"] == [0.]
    for module in history[1]["modules"].values():
        assert module["lr_used"] == .05
        assert module["delta_abs"][0] > 0
        assert module["dtype"] == "torch.float32"
        assert module["finite"]


def test_callback_rejects_nonfinite_sa_before_clamp():
    model = make_model()
    model.model.layers[0].self_attn.q_proj.quantizer.scale.data.fill_(float("-inf"))
    with pytest.raises(RuntimeError, match="Invalid SA after update"):
        LearnableScaleClampCallback().on_step_end(
            None, SimpleNamespace(global_step=1, is_world_process_zero=True), None, model=model
        )
