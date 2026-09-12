"""Collect per-output-channel scores for rotated BF16 Linear weights."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distribution_atlas.collect import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROTATION_PATH,
    _load_rotated_model,
    _require_bfloat16,
    _weight_sources,
)


SUMMARY_FILENAME = "weight_channel_summary.pt"
SCORE_SUMMARY_FILENAME = "weight_channel_score_summary.json"
CHANNEL_CHUNK_SIZE = 256
TAIL_SCORE = "tail_score_t"
BREADTH_SCORE = "breadth_score_b"
SCORE_KEYS = (TAIL_SCORE, BREADTH_SCORE)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--rotation-path", type=Path, default=DEFAULT_ROTATION_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _row_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    """Return a linearly interpolated quantile for each matrix row."""

    rank = quantile * (values.shape[1] - 1)
    lower_rank = math.floor(rank)
    upper_rank = math.ceil(rank)
    lower = torch.kthvalue(values, lower_rank + 1, dim=1).values
    if lower_rank == upper_rank:
        return lower
    upper = torch.kthvalue(values, upper_rank + 1, dim=1).values
    return lower + (upper - lower) * (rank - lower_rank)


def _channel_statistics(weight: torch.Tensor) -> Dict[str, torch.Tensor]:
    if weight.ndim != 2:
        raise RuntimeError(f"Expected a 2-D Linear weight, got {tuple(weight.shape)}")

    chunks: Dict[str, List[torch.Tensor]] = {
        "min": [],
        "max": [],
        "mean": [],
        "std": [],
        "rms": [],
        "max_abs": [],
        "p90_abs": [],
        "p99_abs": [],
        TAIL_SCORE: [],
        BREADTH_SCORE: [],
    }
    for start in range(0, weight.shape[0], CHANNEL_CHUNK_SIZE):
        values = weight[start : start + CHANNEL_CHUNK_SIZE].float()
        abs_values = values.abs()
        minimum = values.amin(dim=1)
        maximum = values.amax(dim=1)
        max_abs = abs_values.amax(dim=1)
        p90_abs = _row_quantile(abs_values, 0.90)
        p99_abs = _row_quantile(abs_values, 0.99)
        tiny = torch.finfo(p99_abs.dtype).tiny

        chunks["min"].append(minimum.cpu())
        chunks["max"].append(maximum.cpu())
        chunks["mean"].append(values.mean(dim=1).cpu())
        chunks["std"].append(values.std(dim=1, unbiased=False).cpu())
        chunks["rms"].append(values.square().mean(dim=1).sqrt().cpu())
        chunks["max_abs"].append(max_abs.cpu())
        chunks["p90_abs"].append(p90_abs.cpu())
        chunks["p99_abs"].append(p99_abs.cpu())
        chunks[TAIL_SCORE].append((max_abs / p99_abs.clamp_min(tiny)).cpu())
        chunks[BREADTH_SCORE].append((p99_abs / p90_abs.clamp_min(tiny)).cpu())

    return {name: torch.cat(parts) for name, parts in chunks.items()}


def _source_summary(source_name: str, weight: torch.Tensor) -> Dict[str, object]:
    _require_bfloat16(weight, source_name)
    statistics = _channel_statistics(weight)
    return {
        "source_name": source_name,
        "projection_family": source_name.rsplit(".", 1)[-1],
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "num_channels": int(weight.shape[0]),
        "elements_per_channel": int(weight.shape[1]),
        "channel_index": torch.arange(weight.shape[0], dtype=torch.int64),
        **statistics,
    }


def _score_statistics(values: torch.Tensor) -> Dict[str, float]:
    quantiles = torch.quantile(
        values.float(), torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32)
    )
    return {
        "min": float(values.min().item()),
        "median": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "max": float(values.max().item()),
    }


def _group_summary(
    group_name: str, sources: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    score_vectors: Dict[str, List[torch.Tensor]] = {
        score_key: [] for score_key in SCORE_KEYS
    }
    channel_lengths = set()
    num_channels = 0
    for source in sources:
        num_channels += int(source["num_channels"])
        channel_lengths.add(int(source["elements_per_channel"]))
        for score_key in SCORE_KEYS:
            values = source[score_key]
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"{source['source_name']} has invalid {score_key}")
            score_vectors[score_key].append(values)

    return {
        "group_name": group_name,
        "num_sources": len(sources),
        "num_channels": num_channels,
        "elements_per_channel": sorted(channel_lengths),
        TAIL_SCORE: _score_statistics(torch.cat(score_vectors[TAIL_SCORE])),
        BREADTH_SCORE: _score_statistics(
            torch.cat(score_vectors[BREADTH_SCORE])
        ),
    }


def _score_summary_payload(
    summaries: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    families: Dict[str, List[Dict[str, object]]] = {}
    for summary in summaries:
        family = str(summary["projection_family"])
        families.setdefault(family, []).append(summary)

    return {
        "granularity": "weight_output_channel",
        "thresholds_defined": False,
        "metric_definitions": {
            TAIL_SCORE: "max_abs / p99_abs",
            BREADTH_SCORE: "p99_abs / p90_abs",
        },
        "full_model": _group_summary("full_model", summaries),
        "projection_families": {
            family: _group_summary(family, sources)
            for family, sources in families.items()
        },
        "linears": {
            str(summary["source_name"]): _group_summary(
                str(summary["source_name"]), [summary]
            )
            for summary in summaries
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    model = _load_rotated_model(args.model_path, args.rotation_path)
    sources = _weight_sources(model)

    summaries: List[Dict[str, object]] = []
    with torch.inference_mode():
        for source_name, module in sources:
            summaries.append(_source_summary(source_name, module.weight))

    total_channels = sum(int(summary["num_channels"]) for summary in summaries)
    summary_payload = {
        "granularity": "weight_output_channel",
        "thresholds_defined": False,
        "metric_definitions": {
            "p90_abs": "P90(abs(weight[channel_index, :]))",
            "p99_abs": "P99(abs(weight[channel_index, :]))",
            TAIL_SCORE: "max_abs / p99_abs",
            BREADTH_SCORE: "p99_abs / p90_abs",
        },
        "total_sources": len(summaries),
        "total_channels": total_channels,
        "sources": summaries,
    }
    score_summary = _score_summary_payload(summaries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(summary_payload, args.output_dir / SUMMARY_FILENAME)
    with (args.output_dir / SCORE_SUMMARY_FILENAME).open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(score_summary, output_file, indent=2)

    print(
        f"Saved {total_channels} channel score records from "
        f"{len(summaries)} Linear weights to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
