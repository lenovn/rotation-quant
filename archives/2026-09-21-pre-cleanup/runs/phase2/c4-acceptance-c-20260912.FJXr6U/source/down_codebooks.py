"""Signed static A8 projection; see down_codebooks.md for sources and semantics.

PoT codebook/projection follows yhhhli/APoT_Quantization's non-additive
build_power_value and signed apot_quantization. SP2 is Eq. (8) of
arXiv:2012.04240v2, NOT the different APoT additive codebook.
"""

import math

import torch


FORMATS = ("int8", "pot", "sp2")


def build_magnitude_codebook(format, bits=8):
    if bits != 8:
        raise ValueError("This experiment implements signed A8 only")
    if format == "int8":
        return torch.arange(128, dtype=torch.float32) / 127
    if format == "pot":
        # APoT author's B=7, additive=False, after max normalization.
        values = [0.] + [2. ** -i for i in range(127)]
    elif format == "sp2":
        # Eq. (8): one sign bit, m1=4 and m2=3; duplicate sums share a value.
        first = [0.] + [2. ** -i for i in range(1, 16)]
        second = [0.] + [2. ** -i for i in range(1, 8)]
        values = [a + b for a in first for b in second]
    else:
        raise ValueError(f"Unknown format: {format}")
    return torch.tensor(sorted(set(values)), dtype=torch.float32)


def quantize(x, alpha, format, levels=None):
    """Nearest-value projection in FP32, followed by the actual GEMM input dtype.

PoT/SP2 halfway ties choose smaller magnitude. INT8 retains torch.round's
ties-to-even and the project's conventional [-128,127] integer range.
"""
    if format not in FORMATS:
        raise ValueError(f"Unknown format: {format}")
    if isinstance(alpha, (int, float)) and (not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("alpha must be finite and positive")
    xf = x.float()
    if format == "int8":
        step = alpha / 127
        return ((xf / step).round().clamp(-128, 127) * step).to(x.dtype)
    if levels is None:
        levels = build_magnitude_codebook(format)
    levels = levels.to(device=x.device, dtype=torch.float32)
    normalized = (xf.abs() / alpha).clamp(max=1)
    # Equivalent to a sorted nearest-neighbour search, without materializing
    # an entire [number_of_levels, number_of_elements] distance matrix.
    midpoints = (levels[:-1] + levels[1:]) * 0.5
    index = torch.bucketize(normalized.contiguous(), midpoints, right=False)
    return (levels[index] * xf.sign() * alpha).to(x.dtype)


class FrozenCodebookQuantizer(torch.nn.Module):
    """In-memory replacement accepted by SpinQuant's ActQuantWrapper."""

    rotation_static_enabled = True
    static_enabled = True
    bits = 8
    sym = True
    groupsize = -1

    def __init__(self, format, alpha, bits=8):
        super().__init__()
        if not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.format = format
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        self.register_buffer("levels", build_magnitude_codebook(format, bits))

    def forward(self, x):
        return quantize(x, self.alpha, self.format, self.levels)
