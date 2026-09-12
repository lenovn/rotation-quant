# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math

import torch
import transformers

from train_utils.quant_linear import QuantizeLinear
from utils import hadamard_utils
from utils.quant_phase import require_quant_phase
from utils.utils import HadamardTransform


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class ChannelScaleQuantize(torch.autograd.Function):
    """Per-row LSQ scale gradient, retaining B's identity weight STE."""
    @staticmethod
    def forward(ctx, x, scale, maxq):
        ctx.qmax = int(maxq.item())
        scaled = x.float() / scale
        ctx.save_for_backward(scaled)
        return (scale * scaled.round().clamp(-ctx.qmax - 1, ctx.qmax)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (scaled,) = ctx.saved_tensors
        lo, hi = -ctx.qmax - 1, ctx.qmax
        term = torch.where(scaled < lo, lo, torch.where(scaled > hi, hi, scaled.round() - scaled))
        grad_scale = (grad_output.float() * term).sum(dim=1, keepdim=True)
        grad_scale /= math.sqrt(scaled.shape[1] * hi)
        return grad_output, grad_scale, None


class LSQQuantize(torch.autograd.Function):
    """Signed per-tensor quantization with an LSQ step-size gradient."""

    @staticmethod
    def forward(ctx, x, scale, maxq):
        qmax = int(maxq.item())
        qmin = -qmax - 1
        scaled = x / scale
        quantized = torch.clamp(torch.round(scaled), qmin, qmax)
        ctx.save_for_backward(scaled, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return scale * quantized

    @staticmethod
    def backward(ctx, grad_output):
        scaled, scale = ctx.saved_tensors
        in_range = (scaled >= ctx.qmin) & (scaled <= ctx.qmax)
        grad_input = grad_output * in_range.to(grad_output.dtype)

        grad_scale_term = torch.where(
            scaled < ctx.qmin,
            torch.full_like(scaled, ctx.qmin),
            torch.where(
                scaled > ctx.qmax,
                torch.full_like(scaled, ctx.qmax),
                torch.round(scaled) - scaled,
            ),
        )
        grad_factor = 1.0 / math.sqrt(scaled.numel() * ctx.qmax)
        grad_scale = (
            grad_output.float() * grad_scale_term.float()
        ).sum().mul_(grad_factor).reshape_as(scale)
        return grad_input, grad_scale.to(scale.dtype), None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
    """
    Activation quantizer supporting both the legacy dynamic path and an explicit
    dual-phase static path.

    The legacy configure/find_params/forward/free lifecycle is intentionally kept
    intact. Static quantization must be selected explicitly with configure_static;
    once selected, it never falls back to runtime parameter discovery.
    """

    _PHASES = ("prefill", "decode")
    _STATE_EMPTY = 0
    _STATE_OBSERVING = 1
    _STATE_FROZEN = 2
    _STATE_NAMES = {
        _STATE_EMPTY: "empty",
        _STATE_OBSERVING: "observing",
        _STATE_FROZEN: "frozen",
    }
    _ENCODING_CODES = {
        "symmetric_int8": 0,
        "asymmetric_uint8": 1,
    }
    _ENCODING_NAMES = {value: key for key, value in _ENCODING_CODES.items()}
    _STATIC_GLOBAL_BUFFERS = (
        "_static_enabled",
        "_static_encoding_code",
        "_static_clip_ratio",
        "_static_maxq",
    )
    _STATIC_PHASE_FIELDS = (
        "state",
        "scale",
        "zero",
        "clip_min",
        "clip_max",
        "sample_count",
    )

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))

        self.register_buffer("_static_enabled", torch.tensor(False))
        self.register_buffer(
            "_static_encoding_code", torch.tensor(-1, dtype=torch.int8)
        )
        self.register_buffer("_static_clip_ratio", torch.ones(1))
        self.register_buffer("_static_maxq", torch.zeros(1, dtype=torch.int64))
        for phase in self._PHASES:
            self.register_buffer(
                f"{phase}_state", torch.tensor(self._STATE_EMPTY, dtype=torch.int8)
            )
            self.register_buffer(f"{phase}_scale", torch.full((1,), float("nan")))
            self.register_buffer(f"{phase}_zero", torch.full((1,), float("nan")))
            self.register_buffer(f"{phase}_clip_min", torch.full((1,), float("nan")))
            self.register_buffer(f"{phase}_clip_max", torch.full((1,), float("nan")))
            self.register_buffer(
                f"{phase}_sample_count", torch.zeros(1, dtype=torch.int64)
            )
        self.bits = 16

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        static_buffer_names = list(self._STATIC_GLOBAL_BUFFERS) + [
            f"{phase}_{field}"
            for phase in self._PHASES
            for field in self._STATIC_PHASE_FIELDS
        ]
        has_static_metadata = any(
            prefix + name in state_dict for name in static_buffer_names
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # Checkpoints predating static A8 contain none of the new buffers. Treat
        # that complete absence as legacy dynamic state, while partial static
        # metadata remains a strict-load error and therefore fails closed.
        if not has_static_metadata:
            for name in static_buffer_names:
                key = prefix + name
                if key in missing_keys:
                    missing_keys.remove(key)
        elif self.static_enabled:
            encoding = self.static_encoding
            self.bits = 8
            self.groupsize = -1
            self.sym = encoding == "symmetric_int8"
            self.clip_ratio = float(self._static_clip_ratio.item())

    @property
    def static_enabled(self) -> bool:
        return bool(self._static_enabled.item())

    @property
    def static_encoding(self):
        if not self.static_enabled:
            return None
        code = int(self._static_encoding_code.item())
        if code not in self._ENCODING_NAMES:
            raise RuntimeError("Static activation encoding metadata is invalid")
        return self._ENCODING_NAMES[code]

    @classmethod
    def _validate_phase(cls, phase):
        if phase not in cls._PHASES:
            raise ValueError(
                "phase must be explicitly set to 'prefill' or 'decode'"
            )
        return phase

    def _phase_buffer(self, phase, field):
        phase = self._validate_phase(phase)
        value = getattr(self, f"{phase}_{field}", None)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Static {phase} {field} buffer is missing")
        return value

    def _phase_state_code(self, phase):
        state = self._phase_buffer(phase, "state")
        if state.numel() != 1:
            raise RuntimeError(f"Static {phase} state metadata is invalid")
        code = int(state.item())
        if code not in self._STATE_NAMES:
            raise RuntimeError(f"Static {phase} state metadata is invalid")
        return code

    def phase_state(self, phase):
        return self._STATE_NAMES[self._phase_state_code(phase)]

    def _set_phase_state(self, phase, state):
        self._phase_buffer(phase, "state").fill_(state)

    def _require_static(self):
        if not self.static_enabled:
            raise RuntimeError("Static activation quantization is not configured")
        # Accessing the property validates the persisted encoding code.
        self.static_encoding

    def configure_static(
        self, encoding: str, clip_ratio: float = 1.0
    ) -> None:
        if encoding not in self._ENCODING_CODES:
            raise ValueError(
                "encoding must be 'symmetric_int8' or 'asymmetric_uint8'"
            )
        clip_ratio = float(clip_ratio)
        if not 0 < clip_ratio <= 1:
            raise ValueError("Clip ratio should be in (0, 1]")
        if any(
            self._phase_state_code(phase) != self._STATE_EMPTY
            for phase in self._PHASES
        ):
            raise RuntimeError("Static encoding cannot change after observation starts")

        code = self._ENCODING_CODES[encoding]
        self._static_enabled.fill_(True)
        self._static_encoding_code.fill_(code)
        self._static_clip_ratio.fill_(clip_ratio)
        self._static_maxq.fill_(127 if encoding == "symmetric_int8" else 255)

        # Preserve the familiar public configuration attributes for callers that
        # inspect quantizers, while static execution uses persisted metadata above.
        self.bits = 8
        self.groupsize = -1
        self.sym = encoding == "symmetric_int8"
        self.clip_ratio = clip_ratio

    def begin_observing(self, phase) -> None:
        self._require_static()
        phase = self._validate_phase(phase)
        if self._phase_state_code(phase) != self._STATE_EMPTY:
            raise RuntimeError(
                f"Static {phase} lifecycle must be empty before observing"
            )
        self._set_phase_state(phase, self._STATE_OBSERVING)

    def start_observing(self, phase) -> None:
        self.begin_observing(phase)

    def observe(self, x, phase) -> None:
        self._require_static()
        phase = self._validate_phase(phase)
        if self._phase_state_code(phase) != self._STATE_OBSERVING:
            raise RuntimeError(f"Static {phase} lifecycle is not observing")
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            raise ValueError("Static activation observations must be non-empty tensors")
        if not x.is_floating_point():
            raise TypeError("Static activation observations must be floating point")

        observed = x.detach()
        observed_min = torch.amin(observed).to(dtype=torch.float32).reshape(1)
        observed_max = torch.amax(observed).to(dtype=torch.float32).reshape(1)
        if not bool(torch.isfinite(observed_min).item()) or not bool(
            torch.isfinite(observed_max).item()
        ):
            raise ValueError("Static activation observations must be finite")

        sample_count = self._phase_buffer(phase, "sample_count")
        clip_min = self._phase_buffer(phase, "clip_min")
        clip_max = self._phase_buffer(phase, "clip_max")
        if int(sample_count.item()) == 0:
            setattr(self, f"{phase}_clip_min", observed_min.clone())
            setattr(self, f"{phase}_clip_max", observed_max.clone())
        else:
            if clip_min.device != observed_min.device:
                raise RuntimeError(
                    f"Static {phase} observations must stay on one device"
                )
            clip_min.copy_(torch.minimum(clip_min, observed_min))
            clip_max.copy_(torch.maximum(clip_max, observed_max))
        sample_count.add_(x.numel())

    def freeze(self, phase) -> None:
        self._require_static()
        phase = self._validate_phase(phase)
        if self._phase_state_code(phase) != self._STATE_OBSERVING:
            raise RuntimeError(f"Static {phase} lifecycle is not observing")

        sample_count = self._phase_buffer(phase, "sample_count")
        if sample_count.numel() != 1 or int(sample_count.item()) <= 0:
            raise RuntimeError(f"Static {phase} cannot freeze without observations")

        raw_min = self._phase_buffer(phase, "clip_min")
        raw_max = self._phase_buffer(phase, "clip_max")
        ratio = self._static_clip_ratio.to(device=raw_min.device, dtype=raw_min.dtype)
        encoding = self.static_encoding

        if encoding == "symmetric_int8":
            bound = torch.maximum(torch.abs(raw_min), torch.abs(raw_max)) * ratio
            clip_min = -bound
            clip_max = bound
            scale = torch.where(bound == 0, torch.ones_like(bound), bound / 127)
            zero = torch.zeros_like(scale)
        else:
            origin = torch.zeros_like(raw_min)
            clip_min = torch.minimum(raw_min, origin) * ratio
            clip_max = torch.maximum(raw_max, origin) * ratio
            all_zero = (clip_min == 0) & (clip_max == 0)
            clip_min = torch.where(all_zero, -torch.ones_like(clip_min), clip_min)
            clip_max = torch.where(all_zero, torch.ones_like(clip_max), clip_max)
            scale = (clip_max - clip_min) / 255
            zero = torch.round(-clip_min / scale).clamp(0, 255)

        setattr(self, f"{phase}_scale", scale.detach().clone().reshape(1))
        setattr(self, f"{phase}_zero", zero.detach().clone().reshape(1))
        setattr(self, f"{phase}_clip_min", clip_min.detach().clone().reshape(1))
        setattr(self, f"{phase}_clip_max", clip_max.detach().clone().reshape(1))
        self._set_phase_state(phase, self._STATE_FROZEN)

    def _frozen_qparams(self, phase):
        self._require_static()
        phase = self._validate_phase(phase)
        if self._phase_state_code(phase) != self._STATE_FROZEN:
            raise RuntimeError(f"Static {phase} lifecycle is not frozen")
        if any(
            self._phase_state_code(required_phase) != self._STATE_FROZEN
            for required_phase in self._PHASES
        ):
            raise RuntimeError(
                "Static activation execution requires both prefill and decode frozen"
            )

        scale = self._phase_buffer(phase, "scale")
        zero = self._phase_buffer(phase, "zero")
        clip_min = self._phase_buffer(phase, "clip_min")
        clip_max = self._phase_buffer(phase, "clip_max")
        sample_count = self._phase_buffer(phase, "sample_count")
        for name, value in (
            ("scale", scale),
            ("zero", zero),
            ("clip_min", clip_min),
            ("clip_max", clip_max),
            ("sample_count", sample_count),
        ):
            if value.numel() != 1:
                raise RuntimeError(f"Static {phase} {name} metadata is invalid")
        for name, value in (
            ("scale", scale),
            ("zero", zero),
            ("clip_min", clip_min),
            ("clip_max", clip_max),
        ):
            if not bool(torch.isfinite(value).item()):
                raise RuntimeError(f"Static {phase} {name} metadata is invalid")
        if float(scale.item()) <= 0 or int(sample_count.item()) <= 0:
            raise RuntimeError(f"Static {phase} qparams are incomplete")
        if float(clip_min.item()) > float(clip_max.item()):
            raise RuntimeError(f"Static {phase} clipping metadata is invalid")

        encoding = self.static_encoding
        expected_maxq = 127 if encoding == "symmetric_int8" else 255
        if (
            self._static_maxq.numel() != 1
            or int(self._static_maxq.item()) != expected_maxq
        ):
            raise RuntimeError("Static activation encoding metadata is inconsistent")
        zero_value = float(zero.item())
        if encoding == "symmetric_int8" and zero_value != 0:
            raise RuntimeError(f"Static {phase} symmetric zero point must be zero")
        if encoding == "asymmetric_uint8" and not 0 <= zero_value <= 255:
            raise RuntimeError(f"Static {phase} zero point is outside uint8 range")
        return scale, zero, expected_maxq

    def free(self) -> None:
        if self.static_enabled:
            raise RuntimeError(
                "Static activation quantization cannot use legacy free()"
            )
        self.zero = None
        self.scale = None

    def forward(self, x, phase=None):
        x_dtype = x.dtype
        if self.static_enabled:
            scale, zero, maxq = self._frozen_qparams(phase)
            if self.static_encoding == "symmetric_int8":
                return STEQuantize.apply(x, scale, maxq).to(x_dtype)
            return AsymSTEQuantize.apply(x, scale, zero, maxq).to(x_dtype)
        if phase is not None:
            raise RuntimeError("phase is only valid for configured static quantization")
        if self.bits == 16:
            return x
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x, phase=None):
        if self.static_enabled:
            scale, zero, maxq = self._frozen_qparams(phase)
            if self.static_encoding == "symmetric_int8":
                return sym_quant(x, scale, maxq)
            return asym_quant(x, scale, zero, maxq)
        if phase is not None:
            raise RuntimeError("phase is only valid for configured static quantization")
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0
    ) -> None:
        if self.static_enabled:
            raise RuntimeError(
                "Static activation quantization cannot fall back to legacy configure()"
            )
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(self, x) -> None:
        if self.static_enabled:
            raise RuntimeError(
                "Static activation quantization cannot discover runtime parameters"
            )
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x) -> None:
        if self.static_enabled:
            raise RuntimeError(
                "Static activation quantization cannot discover runtime parameters"
            )
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = (
                self.scale.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )
            self.zero = (
                self.zero.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )


class RotationStaticActQuantizer(torch.nn.Module):
    """Single-scale A8 used while optimizing R1/R2.

    Candidate statistics are collected before quantization.  The first
    calibration pass has no previous scale and therefore bypasses A8.  Later
    passes keep quantizing with the previous scale until ``finish_calibration``
    atomically publishes the candidate scale.
    """

    rotation_static_enabled = True
    static_enabled = True
    static_encoding = "symmetric_int8"

    def __init__(self, clip_ratio: float = 1.0) -> None:
        super().__init__()
        self.bits = 8
        self.groupsize = -1
        self.sym = True
        self.clip_ratio = float(clip_ratio)
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("maxq", torch.tensor(127))
        self.register_buffer("candidate_absmax", torch.zeros(1))
        self.register_buffer("sample_count", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("_has_scale", torch.tensor(False))
        self.register_buffer("_observing", torch.tensor(False))

    @property
    def has_scale(self) -> bool:
        return bool(self._has_scale.item())

    @property
    def observing(self) -> bool:
        return bool(self._observing.item())

    @property
    def scale_learnable(self) -> bool:
        return isinstance(self.scale, torch.nn.Parameter)

    def enable_scale_learning(self) -> None:
        if self.observing:
            raise RuntimeError("Cannot enable scale learning during calibration")
        if self.scale_learnable:
            return
        initial_scale = self._buffers.pop("scale")
        self.register_parameter(
            "scale", torch.nn.Parameter(initial_scale.detach().float().clone())
        )

    def begin_calibration(self) -> None:
        if self.observing:
            raise RuntimeError("Activation calibration is already running")
        self.candidate_absmax.zero_()
        self.sample_count.zero_()
        self._observing.fill_(True)

    def abort_calibration(self) -> None:
        self.candidate_absmax.zero_()
        self.sample_count.zero_()
        self._observing.fill_(False)

    def _observe(self, x: torch.Tensor) -> None:
        observed_absmax = torch.amax(torch.abs(x.detach())).to(
            device=self.candidate_absmax.device,
            dtype=self.candidate_absmax.dtype,
        )
        self.candidate_absmax.copy_(
            torch.maximum(self.candidate_absmax, observed_absmax.reshape(1))
        )
        self.sample_count.add_(x.numel())

    def finish_calibration(self) -> None:
        if not self.observing or int(self.sample_count.item()) == 0:
            raise RuntimeError("Activation calibration has no observations")
        bound = self.candidate_absmax * self.clip_ratio
        next_scale = torch.where(
            bound == 0, torch.ones_like(bound), bound / 127
        )
        with torch.no_grad():
            self.scale.copy_(next_scale)
            self.zero.zero_()
            self._has_scale.fill_(True)
            self._observing.fill_(False)

    def load_scale(self, scale: torch.Tensor) -> None:
        with torch.no_grad():
            self.scale.copy_(scale.detach().reshape_as(self.scale).to(self.scale))
            self.zero.zero_()
            self._has_scale.fill_(True)
            self._observing.fill_(False)

    def forward(self, x: torch.Tensor):
        if self.observing:
            self._observe(x)
            if not self.has_scale:
                return x
        if not self.has_scale:
            raise RuntimeError("Rotation-static A8 requires calibration")
        if self.scale_learnable:
            return LSQQuantize.apply(x, self.scale, self.maxq).to(x.dtype)
        return STEQuantize.apply(x, self.scale, 127).to(x.dtype)


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registered to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear) -> None:
        super(ActQuantWrapper, self).__init__()
        # assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def forward(self, x, R1=None, R2=None, transpose=False):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = (
                    HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim
                        ).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16:  # Quantize, if needed
            if getattr(self.quantizer, "rotation_static_enabled", False):
                x = self.quantizer(x).to(x_dtype)
            elif self.quantizer.static_enabled:
                phase = require_quant_phase()
                x = self.quantizer(x, phase=phase.value).to(x_dtype)
            else:
                self.quantizer.find_params(x)
                x = self.quantizer(x).to(x_dtype)
                self.quantizer.free()
        if R1 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        weight_groupsize: int = -1,
    ) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params_weight_groupwise(self, x) -> None:
        init_shape = x.shape
        x = x.reshape(
            x.shape[-2], x.shape[-1] // self.weight_groupsize, self.weight_groupsize
        )

        xmax = torch.amax(x, dim=-1, keepdim=True)
        xmin = torch.amin(x, dim=-1, keepdim=True)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, self.weight_groupsize)
        self.zero = self.zero.repeat(1, 1, self.weight_groupsize)

        if self.mse:
            best = torch.full(
                [x.shape[0], x.shape[1]], float("inf"), device=x.device
            ).type_as(x)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = sym_quant_dequant(x, scale1, self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = asym_quant_dequant(x, scale1, zero1, self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, -1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        self.scale = self.scale.reshape(init_shape)
        self.zero = self.zero.reshape(init_shape)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape

        if self.weight_groupsize > 0:
            # group-wise per-token quantization
            self.find_params_weight_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return
        elif self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
                x_dtype
            )
        return x

    # Return int value and scale in addtional to fake quantized weight
    def fake_quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            scale = self.scale.to(x.device)
            q = torch.clamp(torch.round(x / scale), -(self.maxq + 1), self.maxq)
            return (scale * q).to(x_dtype), q, scale
        else:
            return None, None, None

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class RotationStaticWeightQuantizer(WeightQuantizer):
    """W4 quantizer whose qparams change only at calibration boundaries."""

    rotation_static_enabled = True

    def __init__(self, shape: int = 1) -> None:
        super().__init__(shape=shape)
        self.register_buffer("_calibration_pending", torch.tensor(False))
        self.register_buffer("_has_scale", torch.tensor(False))
        self._previous_scale = None
        self._previous_zero = None
        self._previous_has_scale = False

    @property
    def scale_learnable(self):
        return isinstance(self.scale, torch.nn.Parameter)

    def enable_scale_learning(self):
        if not self.has_scale or bool(self._calibration_pending.item()):
            raise RuntimeError("Initialize SW before enabling learning")
        if self.bits != 4 or not self.sym or not self.perchannel or self.weight_groupsize != -1 or self.mse:
            raise ValueError("Learned SW requires symmetric per-channel W4")
        if not self.scale_learnable:
            initial = self._buffers.pop("scale")
            self.register_parameter("scale", torch.nn.Parameter(initial.detach().float().clone()))

    @property
    def has_scale(self) -> bool:
        return bool(self._has_scale.item())

    def begin_calibration(self) -> None:
        if self.scale_learnable:
            raise RuntimeError("Cannot recalibrate learned SW")
        if bool(self._calibration_pending.item()):
            raise RuntimeError("Weight calibration is already running")
        self._previous_has_scale = self.has_scale
        self._previous_scale = self.scale.detach().clone() if self.has_scale else None
        self._previous_zero = self.zero.detach().clone() if self.has_scale else None
        self._calibration_pending.fill_(True)

    def abort_calibration(self) -> None:
        if self._previous_has_scale:
            self.scale = self._previous_scale
            self.zero = self._previous_zero
            self._has_scale.fill_(True)
        else:
            self._has_scale.fill_(False)
        self._calibration_pending.fill_(False)
        self._previous_scale = None
        self._previous_zero = None

    def finish_calibration(self) -> None:
        if bool(self._calibration_pending.item()) or not self.has_scale:
            raise RuntimeError("Weight calibration did not visit this layer")
        self._previous_scale = None
        self._previous_zero = None

    def load_scale(self, scale: torch.Tensor) -> None:
        if self.scale_learnable:
            raise RuntimeError("Cannot overwrite learned SW")
        self.scale = scale.detach().clone().to(self.maxq.device)
        self.zero = torch.zeros_like(self.scale)
        self._has_scale.fill_(True)
        self._calibration_pending.fill_(False)

    @torch.no_grad()
    def refresh_current_weight(self, weight: torch.Tensor) -> None:
        """Stop-gradient symmetric W4 per-output scales at update boundaries."""
        if self.bits != 4 or not self.sym or not self.perchannel or self.weight_groupsize != -1 or self.mse:
            raise ValueError("W4-aware training requires unclipped symmetric per-channel W4")
        if self.scale_learnable:
            raise RuntimeError("Cannot refresh learned SW")
        scale = weight.detach().float().abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 7
        if not torch.isfinite(scale).all():
            raise RuntimeError("Non-finite current-R weight scale")
        # Preserve storage once initialized, including DDP's broadcast buffers.
        if self.scale.shape == scale.shape and self.scale.dtype == scale.dtype:
            self.scale.copy_(scale)
            self.zero.zero_()
        else:
            self.load_scale(scale)
        self._has_scale.fill_(True)

    def calibrate_if_pending(self, weight: torch.Tensor) -> None:
        if not bool(self._calibration_pending.item()):
            return
        super().find_params(weight.detach())
        self._has_scale.fill_(True)
        self._calibration_pending.fill_(False)

    def quantize(self, x):
        if not self.has_scale:
            raise RuntimeError("Rotation-static W4 requires calibration")
        if self.scale_learnable:
            return ChannelScaleQuantize.apply(x, self.scale, self.maxq)
        return super().quantize(x)


def add_actquant(
    module: ActQuantWrapper,
    name: str = "",
    layers=[
        torch.nn.Linear,
        QuantizeLinear,
        ActQuantWrapper,
        transformers.models.falcon.modeling_falcon.FalconLinear,
    ],
) -> None:
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) is torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) is torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + "." + name1 if name != "" else name1, layers)


def find_qlayers(
    module,
    layers=[torch.nn.Linear, ActQuantWrapper, QuantizeLinear],
    name: str = "",
):
    # fix for llama embedding layer
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
