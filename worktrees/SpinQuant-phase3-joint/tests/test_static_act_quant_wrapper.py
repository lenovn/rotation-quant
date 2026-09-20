import pytest
import torch

from utils.quant_phase import QuantPhase, quant_phase_context
from utils.quant_utils import ActQuantizer, ActQuantWrapper


def _identity_wrapper(features=4):
    module = torch.nn.Linear(features, features, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(features))
    return ActQuantWrapper(module)


def _freeze_phase(quantizer, phase, observation):
    quantizer.begin_observing(phase.value)
    quantizer.observe(observation, phase.value)
    quantizer.freeze(phase.value)


def _static_wrapper():
    wrapper = _identity_wrapper()
    wrapper.quantizer.configure_static("symmetric_int8")
    _freeze_phase(
        wrapper.quantizer,
        QuantPhase.PREFILL,
        torch.tensor([[-1.0, 0.0, 1.0, 0.5]]),
    )
    _freeze_phase(
        wrapper.quantizer,
        QuantPhase.DECODE,
        torch.tensor([[-8.0, 0.0, 8.0, 4.0]]),
    )
    return wrapper


@pytest.mark.parametrize("phase", [QuantPhase.PREFILL, QuantPhase.DECODE])
def test_static_wrapper_matches_direct_phase_quantizer(phase):
    wrapper = _static_wrapper()
    x = torch.tensor([[0.3, -0.6, 1.7, -3.2]])
    expected = wrapper.module(wrapper.quantizer(x, phase=phase.value))

    with quant_phase_context(phase):
        actual = wrapper(x)

    torch.testing.assert_close(actual, expected)


def test_static_wrapper_skips_dynamic_and_output_quantizer_paths(monkeypatch):
    wrapper = _static_wrapper()

    def forbidden(*args, **kwargs):
        raise AssertionError("static wrapper entered a forbidden quantizer path")

    monkeypatch.setattr(wrapper.quantizer, "find_params", forbidden)
    monkeypatch.setattr(wrapper.quantizer, "free", forbidden)
    monkeypatch.setattr(wrapper.out_quantizer, "find_params", forbidden)
    monkeypatch.setattr(wrapper.out_quantizer, "forward", forbidden)
    monkeypatch.setattr(wrapper.out_quantizer, "free", forbidden)

    with quant_phase_context(QuantPhase.PREFILL):
        result = wrapper(torch.tensor([[0.3, -0.6, 1.7, -3.2]]))

    assert result.shape == (1, 4)


def test_static_wrapper_requires_phase_context():
    wrapper = _static_wrapper()

    with pytest.raises(RuntimeError, match="No quantization phase is active"):
        wrapper(torch.ones(1, 4))


def test_static_wrapper_requires_both_phases_frozen():
    wrapper = _identity_wrapper()
    wrapper.quantizer.configure_static("symmetric_int8")
    _freeze_phase(
        wrapper.quantizer,
        QuantPhase.PREFILL,
        torch.tensor([[-1.0, 0.0, 1.0, 0.5]]),
    )

    with quant_phase_context(QuantPhase.PREFILL), pytest.raises(
        RuntimeError, match="both prefill and decode frozen"
    ):
        wrapper(torch.ones(1, 4))


def test_static_wrapper_propagates_corrupt_qparams_error():
    wrapper = _static_wrapper()
    wrapper.quantizer.prefill_scale = None

    with quant_phase_context(QuantPhase.PREFILL), pytest.raises(
        RuntimeError, match="scale buffer is missing"
    ):
        wrapper(torch.ones(1, 4))


def test_legacy_dynamic_wrapper_preserves_find_forward_free_order(monkeypatch):
    wrapper = _identity_wrapper()
    wrapper.quantizer.configure(bits=8, sym=True)
    reference_quantizer = ActQuantizer()
    reference_quantizer.configure(bits=8, sym=True)
    x = torch.tensor([[0.3, -0.6, 1.7, -3.2]])
    reference_quantizer.find_params(x)
    expected = wrapper.module(reference_quantizer(x))
    calls = []
    original_find_params = wrapper.quantizer.find_params
    original_free = wrapper.quantizer.free

    def tracked_find_params(value):
        calls.append("find_params")
        return original_find_params(value)

    def tracked_free():
        calls.append("free")
        return original_free()

    monkeypatch.setattr(wrapper.quantizer, "find_params", tracked_find_params)
    monkeypatch.setattr(wrapper.quantizer, "free", tracked_free)

    actual = wrapper(x)

    assert calls == ["find_params", "free"]
    assert wrapper.quantizer.scale is None
    assert wrapper.quantizer.zero is None
    torch.testing.assert_close(actual, expected)
