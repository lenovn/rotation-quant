from types import SimpleNamespace

import torch

from eval_utils.gptq_utils import _prefill_layer
from train_utils.quant_linear import QuantizeLinear
from train_utils.downproj_mse_calibration import _best_alpha, _mse
from train_utils.rotation_calibration import (
    PeriodicRecalibrationCallback,
    enable_non_downproj_scale_learning,
    export_rotation_scales,
)
from utils.quant_utils import (
    RotationStaticActQuantizer,
    RotationStaticWeightQuantizer,
    sym_quant_dequant,
)
from utils.quant_phase import QuantPhase


def test_activation_recalibration_observes_before_using_old_scale():
    quantizer = RotationStaticActQuantizer()
    initial = torch.tensor([[-2.0, 0.25, 1.0]])

    quantizer.begin_calibration()
    assert torch.equal(quantizer(initial), initial)
    quantizer.finish_calibration()
    old_scale = quantizer.scale.clone()

    shifted = torch.tensor([[-6.0, 0.5, 4.0]])
    quantizer.begin_calibration()
    during_recalibration = quantizer(shifted)

    assert torch.equal(
        during_recalibration,
        sym_quant_dequant(shifted, old_scale, 127),
    )
    assert torch.equal(quantizer.scale, old_scale)

    quantizer.finish_calibration()
    assert torch.allclose(quantizer.scale, torch.tensor([6.0 / 127]))


def test_weight_scale_stays_fixed_until_next_calibration():
    layer = QuantizeLinear(2, 2, bias=False)
    layer.weight.data.copy_(torch.tensor([[1.0, 0.2], [0.1, 2.0]]))
    quantizer = RotationStaticWeightQuantizer()
    quantizer.configure(bits=4, perchannel=True, sym=True, weight_groupsize=-1)
    layer.quantizer = quantizer
    x = torch.tensor([[1.0, -0.5]])
    identity = torch.eye(2)
    rotated = torch.tensor([[2**-0.5, -2**-0.5], [2**-0.5, 2**-0.5]])

    quantizer.begin_calibration()
    layer(x, R1=identity)
    quantizer.finish_calibration()
    first_scale = quantizer.scale.clone()

    layer(x, R1=rotated)
    assert torch.equal(quantizer.scale, first_scale)

    quantizer.begin_calibration()
    layer(x, R1=rotated)
    quantizer.finish_calibration()
    assert not torch.equal(quantizer.scale, first_scale)


def test_periodic_callback_leaves_final_calibration_to_main_flow():
    class Recorder:
        def __init__(self):
            self.steps = []

        def run(self, model, update_step, kind):
            self.steps.append((update_step, kind))

    recorder = Recorder()
    callback = PeriodicRecalibrationCallback(recorder, interval=20, final_update=100)
    control = object()
    for step in (1, 20, 40, 60, 80, 100):
        callback.on_step_end(
            None,
            SimpleNamespace(global_step=step),
            control,
            model=object(),
        )

    assert recorder.steps == [
        (20, "periodic"),
        (40, "periodic"),
        (60, "periodic"),
        (80, "periodic"),
    ]


def test_downproj_mse_search_clips_a_heavy_tail_and_reduces_error():
    values = torch.cat([torch.linspace(-1, 1, 4096), torch.tensor([20.0])])
    max_abs = float(values.abs().max().item())

    best_alpha, best_mse = _best_alpha(values, max_abs)

    assert 0.05 <= best_alpha < 1.0
    assert best_mse < _mse(values, max_abs / 127)


def test_lsq_static_activation_scale_receives_task_gradient():
    quantizer = RotationStaticActQuantizer()
    quantizer.load_scale(torch.tensor([1.0]))
    quantizer.enable_scale_learning()
    values = torch.tensor([[0.2, 2.0, 300.0]], requires_grad=True)

    quantizer(values).sum().backward()

    assert quantizer.scale_learnable
    assert quantizer.scale.grad is not None
    assert torch.isfinite(quantizer.scale.grad).all()
    assert quantizer.scale.grad.abs().item() > 0


def test_joint_scale_selection_excludes_down_proj():
    model = torch.nn.Module()
    model.keep = torch.nn.Module()
    model.keep.quantizer = RotationStaticActQuantizer()
    model.mlp = torch.nn.Module()
    model.mlp.down_proj = torch.nn.Module()
    model.mlp.down_proj.quantizer = RotationStaticActQuantizer()

    learned, bypassed = enable_non_downproj_scale_learning(model)

    assert [name for name, _ in learned] == ["keep.quantizer"]
    assert bypassed == ["mlp.down_proj.quantizer"]
    assert model.keep.quantizer.scale_learnable
    assert not model.mlp.down_proj.quantizer.scale_learnable
    assert model.mlp.down_proj.quantizer.bits == 16

    model.keep.quantizer.load_scale(torch.tensor([0.25]))
    exported = export_rotation_scales(model)
    assert set(exported["activation"]) == {"keep.quantizer"}


def test_gptq_direct_layer_forward_uses_prefill_phase():
    class RecordingLayer:
        def __call__(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            quant_phase=None,
        ):
            assert quant_phase is QuantPhase.PREFILL
            return (hidden_states + 1,)

    hidden_states = torch.zeros(1, 2, 3)
    output = _prefill_layer(
        RecordingLayer(), hidden_states, torch.ones(1, 2), torch.arange(2)
    )

    assert torch.equal(output, hidden_states + 1)
