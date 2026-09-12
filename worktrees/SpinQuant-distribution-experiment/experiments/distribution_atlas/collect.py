"""Collect BF16 weight and input-activation distributions for the Atlas experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import datasets
import torch
import transformers
from torch import nn
from transformers import LlamaTokenizerFast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils import rotation_utils  # noqa: E402
from eval_utils.modeling_llama import LlamaForCausalLM  # noqa: E402
from utils import fuse_norm_utils  # noqa: E402


PROJECT_ROOT = REPO_ROOT.parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "cache/models/llama-3.2-1b-instruct"
DEFAULT_ROTATION_PATH = (
    PROJECT_ROOT
    / "runs/phase2/w16a8-joint-r-sa-r12-s42/rotation/R.bin"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs/distribution-atlas/llama32-1b-r12-bf16-wt2-s42"
)

WEIGHT_RECORDS_FILENAME = "weight_records.json"
ACTIVATION_RECORDS_FILENAME = "activation_records.json"

HISTOGRAM_BINS = 4096
BF16_VALUE_COUNT = 1 << 16
EXPECTED_WEIGHT_SOURCES = 112
EXPECTED_ACTIVATION_SOURCES = 64
NUM_WINDOWS = 128
SEQUENCE_LENGTH = 2048
SEED = 42

ProjectionSource = Tuple[str, nn.Linear]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--rotation-path", type=Path, default=DEFAULT_ROTATION_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_rotated_model(model_path: Path, rotation_path: Path) -> LlamaForCausalLM:
    if not torch.cuda.is_available():
        raise RuntimeError("The distribution atlas requires a CUDA device")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not rotation_path.is_file():
        raise FileNotFoundError(f"Rotation checkpoint does not exist: {rotation_path}")

    config = transformers.AutoConfig.from_pretrained(str(model_path), token=None)
    process_word_embeddings = False
    if getattr(config, "tie_word_embeddings", False):
        config.tie_word_embeddings = False
        process_word_embeddings = True

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(model_path),
        config=config,
        torch_dtype=torch.bfloat16,
        token=None,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # Match the formal Phase 2 evaluation order.
    model.cuda()
    model.eval()
    fuse_norm_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(
        model,
        SimpleNamespace(
            rotate_mode="hadamard",
            optimized_rotation_path=str(rotation_path),
            rotation_components="r1_r2",
        ),
    )
    # rotate_model intentionally offloads rotated Linear weights to CPU.
    model.cuda()
    model.eval()
    model.config.use_cache = False
    return model


def _weight_sources(model: LlamaForCausalLM) -> List[ProjectionSource]:
    sources: List[ProjectionSource] = []
    for layer_idx, layer in enumerate(model.model.layers):
        projections = (
            ("self_attn.q_proj", layer.self_attn.q_proj),
            ("self_attn.k_proj", layer.self_attn.k_proj),
            ("self_attn.v_proj", layer.self_attn.v_proj),
            ("self_attn.o_proj", layer.self_attn.o_proj),
            ("mlp.gate_proj", layer.mlp.gate_proj),
            ("mlp.up_proj", layer.mlp.up_proj),
            ("mlp.down_proj", layer.mlp.down_proj),
        )
        for suffix, module in projections:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{suffix} in layer {layer_idx} is not nn.Linear")
            sources.append((f"model.layers.{layer_idx}.{suffix}", module))

    if len(sources) != EXPECTED_WEIGHT_SOURCES:
        raise RuntimeError(
            f"Expected {EXPECTED_WEIGHT_SOURCES} weight sources, got {len(sources)}"
        )
    return sources


def _activation_sources(model: LlamaForCausalLM) -> List[ProjectionSource]:
    sources: List[ProjectionSource] = []
    for layer_idx, layer in enumerate(model.model.layers):
        projections = (
            ("self_attn.qkv_input", layer.self_attn.q_proj),
            ("self_attn.o_proj_input", layer.self_attn.o_proj),
            ("mlp.gate_up_input", layer.mlp.gate_proj),
            ("mlp.down_proj_input", layer.mlp.down_proj),
        )
        for suffix, module in projections:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{suffix} in layer {layer_idx} is not nn.Linear")
            sources.append((f"model.layers.{layer_idx}.{suffix}", module))

    if len(sources) != EXPECTED_ACTIVATION_SOURCES:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_ACTIVATION_SOURCES} activation sources, got {len(sources)}"
        )
    return sources


def _require_bfloat16(value: torch.Tensor, source_name: str) -> None:
    if value.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{source_name} has dtype {value.dtype}; "
            "the Atlas experiment requires torch.bfloat16"
        )


_BF16_VALUES: Optional[torch.Tensor] = None


def _bf16_value_table() -> torch.Tensor:
    global _BF16_VALUES
    if _BF16_VALUES is None:
        signed_bit_patterns = torch.arange(
            -32768, 32768, dtype=torch.int32
        ).to(torch.int16)
        _BF16_VALUES = signed_bit_patterns.view(torch.bfloat16).float()
    return _BF16_VALUES


def _histogram_edges(min_value: float, max_value: float) -> torch.Tensor:
    if min_value == max_value:
        margin = max(abs(min_value), 1.0) * 1e-6
        min_value -= margin
        max_value += margin
    return torch.linspace(
        min_value,
        max_value,
        HISTOGRAM_BINS + 1,
        dtype=torch.float64,
    )


class DistributionAccumulator:
    """Stream BF16 value counts, then accumulate a fixed-range histogram."""

    def __init__(self, source_name: str, source_kind: str) -> None:
        self.source_name = source_name
        self.source_kind = source_kind
        self.shape: Optional[Tuple[int, ...]] = None
        self.dtype: Optional[torch.dtype] = None
        self._value_counts: Optional[torch.Tensor] = None
        self._num_elements: Optional[int] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._p99_abs: Optional[float] = None
        self._p99_9_abs: Optional[float] = None
        self._p99_99_abs: Optional[float] = None
        self._edges: Optional[torch.Tensor] = None
        self._histogram_counts: Optional[torch.Tensor] = None

    def update(self, value: torch.Tensor) -> None:
        _require_bfloat16(value, self.source_name)
        if self._edges is not None:
            raise RuntimeError("Cannot update an accumulator after finalization")

        if self.shape is None:
            self.shape = tuple(int(dim) for dim in value.shape)
            self.dtype = value.dtype
        elif tuple(value.shape) != self.shape:
            raise RuntimeError(
                f"{self.source_name} changed shape from {self.shape} to {tuple(value.shape)}"
            )

        flat = value.detach().reshape(-1)
        if not bool(torch.isfinite(flat).all().item()):
            raise RuntimeError(f"{self.source_name} contains a non-finite value")

        signed_bit_patterns = flat.contiguous().view(torch.int16)
        indices = signed_bit_patterns.to(torch.int64) + 32768
        batch_counts = torch.bincount(
            indices,
            minlength=BF16_VALUE_COUNT,
        )
        if self._value_counts is None:
            self._value_counts = torch.zeros(
                BF16_VALUE_COUNT,
                dtype=torch.int64,
                device=flat.device,
            )
        elif self._value_counts.device != flat.device:
            raise RuntimeError(f"{self.source_name} changed device during collection")
        self._value_counts.add_(batch_counts)

    def finalize_statistics(self) -> None:
        if self._value_counts is None or self.shape is None or self.dtype is None:
            raise RuntimeError(f"{self.source_name} has no observations")

        counts = self._value_counts.detach().cpu()
        values = _bf16_value_table()
        observed = counts > 0
        if not bool(observed.any().item()):
            raise RuntimeError(f"{self.source_name} has no observed values")
        if bool((observed & ~torch.isfinite(values)).any().item()):
            raise RuntimeError(f"{self.source_name} contains a non-finite value")

        observed_values = values[observed]
        self._num_elements = int(counts.sum().item())
        self._min = float(observed_values.min().item())
        self._max = float(observed_values.max().item())

        safe_values = torch.where(observed, values, torch.zeros_like(values))
        weighted_counts = counts.to(torch.float64)
        value_doubles = safe_values.to(torch.float64)
        total = float(self._num_elements)
        mean = (value_doubles * weighted_counts).sum().item() / total
        second_moment = (
            value_doubles.square() * weighted_counts
        ).sum().item() / total
        variance = max(second_moment - mean * mean, 0.0)
        self._mean = float(mean)
        self._std = float(math.sqrt(variance))

        abs_values = value_doubles.abs()[observed]
        observed_counts = counts[observed]
        order = torch.argsort(abs_values)
        sorted_abs = abs_values[order]
        cumulative_counts = observed_counts[order].cumsum(0)
        self._p99_abs = self._weighted_quantile(
            sorted_abs, cumulative_counts, self._num_elements, 0.99
        )
        self._p99_9_abs = self._weighted_quantile(
            sorted_abs, cumulative_counts, self._num_elements, 0.999
        )
        self._p99_99_abs = self._weighted_quantile(
            sorted_abs, cumulative_counts, self._num_elements, 0.9999
        )

        self._edges = _histogram_edges(self._min, self._max)
        self._histogram_counts = torch.zeros(
            HISTOGRAM_BINS,
            dtype=torch.int64,
        )
        self._value_counts = None

    @staticmethod
    def _weighted_quantile(
        sorted_values: torch.Tensor,
        cumulative_counts: torch.Tensor,
        num_elements: int,
        quantile: float,
    ) -> float:
        target = quantile * (num_elements - 1)
        lower_rank = math.floor(target)
        upper_rank = math.ceil(target)

        lower_index = int(
            torch.searchsorted(
                cumulative_counts,
                torch.tensor(lower_rank, dtype=cumulative_counts.dtype),
                right=True,
            ).item()
        )
        upper_index = int(
            torch.searchsorted(
                cumulative_counts,
                torch.tensor(upper_rank, dtype=cumulative_counts.dtype),
                right=True,
            ).item()
        )
        lower_value = sorted_values[lower_index].item()
        upper_value = sorted_values[upper_index].item()
        fraction = target - lower_rank
        return float(lower_value + fraction * (upper_value - lower_value))

    def accumulate_histogram(self, value: torch.Tensor) -> None:
        _require_bfloat16(value, self.source_name)
        if self._edges is None or self._histogram_counts is None:
            raise RuntimeError(f"{self.source_name} histogram is not initialized")
        if tuple(value.shape) != self.shape:
            raise RuntimeError(
                f"{self.source_name} changed shape from {self.shape} to {tuple(value.shape)}"
            )

        flat = value.detach().reshape(-1).float()
        if not bool(torch.isfinite(flat).all().item()):
            raise RuntimeError(f"{self.source_name} contains a non-finite value")
        edges = self._edges.to(device=flat.device, dtype=flat.dtype)
        bin_indices = torch.bucketize(flat, edges, right=True) - 1
        bin_indices.clamp_(0, HISTOGRAM_BINS - 1)
        batch_counts = torch.bincount(
            bin_indices,
            minlength=HISTOGRAM_BINS,
        )
        self._histogram_counts.add_(batch_counts.cpu())

    def record(self) -> Dict[str, object]:
        if (
            self.shape is None
            or self.dtype is None
            or self._num_elements is None
            or self._min is None
            or self._max is None
            or self._mean is None
            or self._std is None
            or self._p99_abs is None
            or self._p99_9_abs is None
            or self._p99_99_abs is None
            or self._edges is None
            or self._histogram_counts is None
        ):
            raise RuntimeError(f"{self.source_name} is incomplete")

        count_sum = int(self._histogram_counts.sum().item())
        if count_sum != self._num_elements:
            raise RuntimeError(
                f"{self.source_name} histogram count {count_sum} "
                f"does not equal num_elements {self._num_elements}"
            )

        return {
            "source_name": self.source_name,
            "source_kind": self.source_kind,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "num_elements": self._num_elements,
            "min": self._min,
            "max": self._max,
            "mean": self._mean,
            "std": self._std,
            "p99_abs": self._p99_abs,
            "p99_9_abs": self._p99_9_abs,
            "p99_99_abs": self._p99_99_abs,
            "histogram": {
                "bins": HISTOGRAM_BINS,
                "edges": [float(edge) for edge in self._edges.tolist()],
                "counts": [int(count) for count in self._histogram_counts.tolist()],
            },
        }


def _load_wikitext_windows(
    tokenizer: LlamaTokenizerFast,
    num_windows: int = NUM_WINDOWS,
    sequence_length: int = SEQUENCE_LENGTH,
    seed: int = SEED,
) -> List[torch.Tensor]:
    dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    encoded = tokenizer(
        "\n\n".join(dataset["train"]["text"]),
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"].reshape(-1)
    available_windows = token_ids.numel() // sequence_length
    if available_windows < num_windows:
        raise RuntimeError(
            f"WikiText-2 train provides only {available_windows} full windows; "
            f"{num_windows} are required"
        )

    selected_window_ids = sorted(
        random.Random(seed).sample(range(available_windows), num_windows)
    )
    return [
        token_ids[
            window_id * sequence_length : (window_id + 1) * sequence_length
        ]
        .unsqueeze(0)
        .contiguous()
        for window_id in selected_window_ids
    ]


def _run_windows(
    model: LlamaForCausalLM,
    windows: Iterable[torch.Tensor],
    device: torch.device,
) -> None:
    for input_ids in windows:
        outputs = model.model(
            input_ids=input_ids.to(device, non_blocking=True),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        del outputs


def _collect_weight_records(
    sources: Sequence[ProjectionSource],
) -> List[Dict[str, object]]:
    for source_name, module in sources:
        _require_bfloat16(module.weight, source_name)

    records: List[Dict[str, object]] = []
    for source_name, module in sources:
        accumulator = DistributionAccumulator(source_name, "weight")
        accumulator.update(module.weight)
        accumulator.finalize_statistics()
        accumulator.accumulate_histogram(module.weight)
        records.append(accumulator.record())
    return records


def _register_activation_hooks(
    sources: Sequence[ProjectionSource],
    accumulators: Dict[str, DistributionAccumulator],
    collect_histogram: bool,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for source_name, module in sources:
        if collect_histogram:
            callback = lambda _module, inputs, name=source_name: accumulators[
                name
            ].accumulate_histogram(inputs[0])
        else:
            callback = lambda _module, inputs, name=source_name: accumulators[
                name
            ].update(inputs[0])
        handles.append(module.register_forward_pre_hook(callback))
    return handles


def _collect_activation_records(
    model: LlamaForCausalLM,
    sources: Sequence[ProjectionSource],
    windows: Sequence[torch.Tensor],
    device: torch.device,
) -> List[Dict[str, object]]:
    accumulators = {
        source_name: DistributionAccumulator(source_name, "activation")
        for source_name, _module in sources
    }

    handles = _register_activation_hooks(
        sources,
        accumulators,
        collect_histogram=False,
    )
    try:
        _run_windows(model, windows, device)
    finally:
        for handle in handles:
            handle.remove()

    for accumulator in accumulators.values():
        accumulator.finalize_statistics()

    handles = _register_activation_hooks(
        sources,
        accumulators,
        collect_histogram=True,
    )
    try:
        _run_windows(model, windows, device)
    finally:
        for handle in handles:
            handle.remove()

    records = [accumulators[source_name].record() for source_name, _ in sources]
    return records


def _validate_records(
    records: Sequence[Dict[str, object]],
    expected_count: int,
    source_kind: str,
) -> None:
    if len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {source_kind} records, got {len(records)}"
        )
    for record in records:
        if record["source_kind"] != source_kind:
            raise RuntimeError(
                f"Record {record['source_name']} has unexpected source kind"
            )
        histogram = record["histogram"]
        if not isinstance(histogram, dict):
            raise RuntimeError(f"Record {record['source_name']} has no histogram")
        if len(histogram["edges"]) != HISTOGRAM_BINS + 1:
            raise RuntimeError(f"Record {record['source_name']} has invalid edges")
        if len(histogram["counts"]) != HISTOGRAM_BINS:
            raise RuntimeError(f"Record {record['source_name']} has invalid counts")
        if sum(histogram["counts"]) != record["num_elements"]:
            raise RuntimeError(
                f"Record {record['source_name']} has inconsistent histogram counts"
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    model = _load_rotated_model(args.model_path, args.rotation_path)
    device = next(model.parameters()).device

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=str(args.model_path),
        model_max_length=SEQUENCE_LENGTH,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=None,
    )
    windows = _load_wikitext_windows(tokenizer)
    if any(window.shape != (1, SEQUENCE_LENGTH) for window in windows):
        raise RuntimeError("The calibration windows are not batch-1 2048-token inputs")

    weight_sources = _weight_sources(model)
    activation_sources = _activation_sources(model)
    with torch.inference_mode():
        weight_records = _collect_weight_records(weight_sources)
        activation_records = _collect_activation_records(
            model,
            activation_sources,
            windows,
            device,
        )

    _validate_records(
        weight_records,
        EXPECTED_WEIGHT_SOURCES,
        "weight",
    )
    _validate_records(
        activation_records,
        EXPECTED_ACTIVATION_SOURCES,
        "activation",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / WEIGHT_RECORDS_FILENAME).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(weight_records, output_file, indent=2)
    with (args.output_dir / ACTIVATION_RECORDS_FILENAME).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(activation_records, output_file, indent=2)

    print(
        f"Saved {len(weight_records)} weight records and "
        f"{len(activation_records)} activation records to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
