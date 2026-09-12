from unittest import mock

import pytest
import torch

from utils.quant_utils import ActQuantizer


def _freeze_phase(quantizer, phase, observation):
    quantizer.begin_observing(phase)
    quantizer.observe(observation, phase)
    quantizer.freeze(phase)


def _configured_quantizer(encoding="symmetric_int8"):
    quantizer = ActQuantizer()
    quantizer.configure_static(encoding)
    return quantizer


def test_legal_state_transitions_persist_qparams():
    quantizer = _configured_quantizer()
    assert quantizer.phase_state("prefill") == "empty"

    quantizer.begin_observing("prefill")
    assert quantizer.phase_state("prefill") == "observing"
    quantizer.observe(torch.tensor([[-2.0, 0.5, 3.0]]), "prefill")
    quantizer.freeze("prefill")

    assert quantizer.phase_state("prefill") == "frozen"
    assert quantizer.prefill_sample_count.item() == 3
    assert quantizer.prefill_scale.item() > 0
    assert quantizer.prefill_zero.item() == 0
    assert quantizer.prefill_clip_min.item() == -3
    assert quantizer.prefill_clip_max.item() == 3


def test_illegal_transitions_and_dynamic_fallback_fail_closed():
    quantizer = _configured_quantizer()

    with pytest.raises(ValueError, match="phase"):
        quantizer.begin_observing("prompt")
    with pytest.raises(RuntimeError, match="not observing"):
        quantizer.observe(torch.ones(2), "prefill")
    with pytest.raises(RuntimeError, match="not observing"):
        quantizer.freeze("prefill")

    quantizer.begin_observing("prefill")
    with pytest.raises(RuntimeError, match="must be empty"):
        quantizer.begin_observing("prefill")
    with pytest.raises(RuntimeError, match="without observations"):
        quantizer.freeze("prefill")

    quantizer.observe(torch.ones(2), "prefill")
    quantizer.freeze("prefill")
    with pytest.raises(RuntimeError, match="not observing"):
        quantizer.observe(torch.ones(2), "prefill")
    with pytest.raises(RuntimeError, match="not observing"):
        quantizer.freeze("prefill")
    with pytest.raises(RuntimeError, match="legacy configure"):
        quantizer.configure(bits=8)
    with pytest.raises(RuntimeError, match="runtime parameters"):
        quantizer.find_params(torch.ones(2))
    with pytest.raises(RuntimeError, match="legacy free"):
        quantizer.free()


def test_static_configuration_rejects_unknown_encoding_and_clip_ratio():
    quantizer = ActQuantizer()
    with pytest.raises(ValueError, match="encoding"):
        quantizer.configure_static("int8")
    with pytest.raises(ValueError, match="Clip ratio"):
        quantizer.configure_static("symmetric_int8", clip_ratio=0)


def test_phase_isolation_and_explicit_frozen_phase_requirement():
    quantizer = _configured_quantizer()
    _freeze_phase(quantizer, "prefill", torch.tensor([-1.0, 1.0]))

    assert quantizer.phase_state("prefill") == "frozen"
    assert quantizer.phase_state("decode") == "empty"
    assert quantizer.decode_sample_count.item() == 0
    with pytest.raises(ValueError, match="phase"):
        quantizer(torch.ones(2))
    with pytest.raises(RuntimeError, match="not frozen"):
        quantizer(torch.ones(2), phase="decode")
    with pytest.raises(RuntimeError, match="both prefill and decode frozen"):
        quantizer(torch.ones(2), phase="prefill")

    _freeze_phase(quantizer, "decode", torch.tensor([-8.0, 8.0]))
    assert quantizer.decode_scale.item() > quantizer.prefill_scale.item()


def test_frozen_parameters_do_not_change_during_forward():
    quantizer = _configured_quantizer("asymmetric_uint8")
    _freeze_phase(quantizer, "prefill", torch.tensor([-2.0, 2.0]))
    _freeze_phase(quantizer, "decode", torch.tensor([-1.0, 4.0]))
    before = {
        name: getattr(quantizer, f"decode_{name}").clone()
        for name in ("scale", "zero", "clip_min", "clip_max", "sample_count")
    }

    quantizer(torch.tensor([-100.0, 100.0]), phase="decode")

    for name, expected in before.items():
        torch.testing.assert_close(getattr(quantizer, f"decode_{name}"), expected)


def test_frozen_forward_performs_no_runtime_reductions():
    quantizer = _configured_quantizer()
    _freeze_phase(quantizer, "prefill", torch.tensor([-2.0, 2.0]))
    _freeze_phase(quantizer, "decode", torch.tensor([-4.0, 4.0]))

    forbidden = AssertionError("frozen forward attempted a runtime reduction")
    with mock.patch("torch.amin", side_effect=forbidden), mock.patch(
        "torch.amax", side_effect=forbidden
    ), mock.patch("torch.min", side_effect=forbidden), mock.patch(
        "torch.max", side_effect=forbidden
    ):
        result = quantizer(torch.tensor([-3.0, 0.0, 3.0]), phase="prefill")

    assert result.shape == (3,)


@pytest.mark.parametrize(
    ("encoding", "expected_maxq", "tuple_length"),
    [
        ("symmetric_int8", 127, 2),
        ("asymmetric_uint8", 255, 3),
    ],
)
def test_supported_static_encodings(encoding, expected_maxq, tuple_length):
    quantizer = _configured_quantizer(encoding)
    _freeze_phase(quantizer, "prefill", torch.tensor([-2.0, 0.0, 6.0]))
    _freeze_phase(quantizer, "decode", torch.tensor([-4.0, 0.0, 8.0]))

    quantized = quantizer.quantize(
        torch.tensor([-20.0, 0.0, 20.0]), phase="prefill"
    )
    assert quantizer.static_encoding == encoding
    assert quantizer._static_maxq.item() == expected_maxq
    assert len(quantized) == tuple_length
    integers = quantized[0]
    if encoding == "symmetric_int8":
        assert integers.min().item() >= -128
        assert integers.max().item() <= 127
    else:
        assert integers.min().item() >= 0
        assert integers.max().item() <= 255


def test_missing_frozen_qparams_fail_closed():
    quantizer = _configured_quantizer()
    _freeze_phase(quantizer, "prefill", torch.tensor([-1.0, 1.0]))
    _freeze_phase(quantizer, "decode", torch.tensor([-2.0, 2.0]))
    quantizer.prefill_scale = None

    with pytest.raises(RuntimeError, match="scale buffer is missing"):
        quantizer(torch.ones(2), phase="prefill")


def test_state_dict_round_trip_preserves_both_phases():
    quantizer = _configured_quantizer("asymmetric_uint8")
    _freeze_phase(quantizer, "prefill", torch.tensor([-1.0, 3.0]))
    _freeze_phase(quantizer, "decode", torch.tensor([-5.0, 7.0, 9.0]))
    expected_prefill = quantizer(
        torch.tensor([-2.0, 0.0, 5.0]), phase="prefill"
    )
    expected_decode = quantizer(
        torch.tensor([-2.0, 0.0, 5.0]), phase="decode"
    )

    restored = ActQuantizer()
    restored.load_state_dict(quantizer.state_dict())

    assert restored.static_enabled
    assert restored.static_encoding == "asymmetric_uint8"
    assert restored.bits == 8
    assert restored.groupsize == -1
    assert not restored.sym
    assert restored.phase_state("prefill") == "frozen"
    assert restored.phase_state("decode") == "frozen"
    assert restored.prefill_sample_count.item() == 2
    assert restored.decode_sample_count.item() == 3
    torch.testing.assert_close(
        restored(torch.tensor([-2.0, 0.0, 5.0]), phase="prefill"),
        expected_prefill,
    )
    torch.testing.assert_close(
        restored(torch.tensor([-2.0, 0.0, 5.0]), phase="decode"),
        expected_decode,
    )


def test_legacy_dynamic_state_dict_remains_strict_loadable():
    legacy = ActQuantizer()
    legacy.configure(bits=8, sym=True)
    legacy_state = {
        name: value
        for name, value in legacy.state_dict().items()
        if not name.startswith("_")
        and not name.startswith("prefill_")
        and not name.startswith("decode_")
    }

    restored = ActQuantizer()
    restored.load_state_dict(legacy_state, strict=True)
    assert not restored.static_enabled
