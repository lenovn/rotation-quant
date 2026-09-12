from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Iterator, Optional

import torch
from transformers.cache_utils import Cache, StaticCache
from transformers.utils import is_torchdynamo_compiling


class QuantPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


_ACTIVE_QUANT_PHASE: ContextVar[Optional[QuantPhase]] = ContextVar(
    "active_quant_phase", default=None
)
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_cache_position(cache_position: torch.Tensor) -> int:
    if cache_position is None:
        raise ValueError("cache_position metadata is required")
    if not isinstance(cache_position, torch.Tensor):
        raise TypeError("cache_position must be a torch.Tensor")
    if cache_position.ndim != 1:
        raise ValueError("cache_position must be one-dimensional")
    if cache_position.numel() == 0:
        raise ValueError("cache_position must not be empty")
    if cache_position.dtype not in _INTEGER_DTYPES:
        raise TypeError("cache_position must use an integer dtype")

    first_position = int(cache_position[0].item())
    if first_position < 0:
        raise ValueError("cache_position must not contain negative positions")
    expected = torch.arange(
        first_position,
        first_position + cache_position.numel(),
        dtype=cache_position.dtype,
        device=cache_position.device,
    )
    if not torch.equal(cache_position, expected):
        raise ValueError("cache_position must contain contiguous increasing positions")
    return first_position


def _normalize_cache_length(cache_length) -> int:
    if isinstance(cache_length, torch.Tensor):
        if cache_length.numel() != 1 or cache_length.dtype not in _INTEGER_DTYPES:
            raise TypeError("Cache sequence length must be one integer scalar")
        cache_length = cache_length.item()
    if isinstance(cache_length, bool) or not isinstance(cache_length, int):
        raise TypeError("Cache sequence length must be an integer")
    if cache_length < 0:
        raise ValueError("Cache sequence length must not be negative")
    return cache_length


def resolve_quant_phase(
    past_key_values: Optional[Cache], cache_position: torch.Tensor
) -> QuantPhase:
    """Resolve phase at an eager host boundary from cache/position metadata."""
    if is_torchdynamo_compiling():
        raise RuntimeError("Quantization phase resolution is forbidden while compiling")
    first_position = _validate_cache_position(cache_position)

    if isinstance(past_key_values, StaticCache):
        # StaticCache.get_seq_length scans cache occupancy. The validated first
        # position is the existing length for the static/compiled path.
        existing_length = first_position
    elif past_key_values is None:
        existing_length = 0
    elif isinstance(past_key_values, Cache):
        existing_length = _normalize_cache_length(past_key_values.get_seq_length())
    else:
        raise TypeError("past_key_values must be None or a transformers Cache")

    if not isinstance(past_key_values, StaticCache) and first_position != existing_length:
        raise ValueError(
            "cache_position start does not match the existing cache sequence length"
        )
    return QuantPhase.PREFILL if existing_length == 0 else QuantPhase.DECODE


@contextmanager
def quant_phase_context(phase: QuantPhase) -> Iterator[QuantPhase]:
    if not isinstance(phase, QuantPhase):
        raise TypeError("phase context requires an explicit QuantPhase")
    token = _ACTIVE_QUANT_PHASE.set(phase)
    try:
        yield phase
    finally:
        _ACTIVE_QUANT_PHASE.reset(token)


def require_quant_phase() -> QuantPhase:
    phase = _ACTIVE_QUANT_PHASE.get()
    if phase is None:
        raise RuntimeError("No quantization phase is active")
    return phase
