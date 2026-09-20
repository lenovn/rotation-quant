import math

import torch

from utils.quant_utils import RotationStaticWeightQuantizer


def sp2_levels():
    first = [0.0] + [2.0 ** -exponent for exponent in range(1, 16)]
    second = [0.0] + [2.0 ** -exponent for exponent in range(1, 8)]
    return torch.tensor(sorted({left + right for left in first for right in second}), dtype=torch.float32)


def sp2_project(inputs, alpha, levels):
    normalized = (inputs.float().abs() / alpha).clamp(max=1)
    midpoints = (levels[:-1] + levels[1:]) * 0.5
    indices = torch.bucketize(normalized.contiguous(), midpoints, right=False)
    return (levels[indices] * inputs.float().sign() * alpha).to(inputs.dtype)


class SP2ScaleSTE(torch.autograd.Function):
    @staticmethod
    def forward(context, inputs, scale, levels):
        alpha = scale * 127
        normalized = inputs.float() / alpha
        quantized = sp2_project(inputs, alpha, levels)
        projected = sp2_project(normalized, torch.ones_like(alpha), levels)
        context.save_for_backward(normalized, projected, scale)
        return quantized

    @staticmethod
    def backward(context, gradient):
        normalized, projected, scale = context.saved_tensors
        inside = normalized.abs() <= 1
        scale_term = 127 * (projected - normalized * inside)
        scale_gradient = (gradient.float() * scale_term).sum()
        scale_gradient = scale_gradient / math.sqrt(normalized.numel() * 127)
        return gradient * inside, scale_gradient.reshape_as(scale), None


class SP2Quantizer(torch.nn.Module):
    rotation_static_enabled = True
    static_enabled = True
    bits = 8
    sym = True
    groupsize = -1

    def __init__(self, alpha, learnable=False):
        super().__init__()
        if not math.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError("SP2 alpha must be finite and positive")
        scale = torch.tensor([float(alpha) / 127], dtype=torch.float32)
        if learnable:
            self.scale = torch.nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        self.register_buffer("levels", sp2_levels())

    @property
    def alpha(self):
        return self.scale * 127

    def forward(self, inputs):
        if isinstance(self.scale, torch.nn.Parameter):
            return SP2ScaleSTE.apply(inputs, self.scale, self.levels)
        return sp2_project(inputs, self.alpha, self.levels)


class Phase3WeightQuantizer(RotationStaticWeightQuantizer):
    def quantize(self, inputs):
        if self.bits == 16:
            return inputs
        return super().quantize(inputs)


def int4_codes(weight, scale):
    if scale.shape != (weight.shape[0], 1):
        raise ValueError("W4 scale must be per output channel")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("W4 scale must be finite and positive")
    return (weight.float() / scale).round().clamp(-8, 7).to(torch.int8)


def pack_int4(codes):
    if codes.numel() % 2 or not ((codes >= -8) & (codes <= 7)).all():
        raise ValueError("Invalid signed INT4 codes or odd element count")
    shifted = (codes.reshape(-1).to(torch.int16) + 8).to(torch.uint8)
    return shifted[::2] | (shifted[1::2] << 4)


def unpack_int4(packed, shape):
    values = torch.stack((packed & 15, packed >> 4), dim=1).reshape(shape)
    return (values.to(torch.int16) - 8).to(torch.int8)
