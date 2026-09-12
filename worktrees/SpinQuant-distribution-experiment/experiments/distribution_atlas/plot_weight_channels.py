"""Plot weight-channel T and B score distributions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RECORDS_DIR = (
    PROJECT_ROOT / "runs/distribution-atlas/llama32-1b-r12-bf16-wt2-s42"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RECORDS_DIR / "weight_channel_scores"
SUMMARY_FILENAME = "weight_channel_summary.pt"
SCORE_BINS = 128
TAIL_SCORE = "tail_score_t"
BREADTH_SCORE = "breadth_score_b"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_summary(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sources = payload.get("sources")
    if payload.get("thresholds_defined") is not False:
        raise RuntimeError(f"Unexpected threshold metadata in {path}")
    if not isinstance(sources, list) or len(sources) != payload.get("total_sources"):
        raise RuntimeError(f"Invalid weight-channel summary in {path}")
    if sum(int(source["num_channels"]) for source in sources) != int(
        payload["total_channels"]
    ):
        raise RuntimeError(f"Invalid channel count in {path}")
    return payload


def _safe_filename(source_name: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name).strip("._")
    return filename or "weight_channel"


def _score_values(
    sources: Sequence[Dict[str, object]], score_key: str
) -> np.ndarray:
    tensors = []
    for source in sources:
        values = source.get(score_key)
        if not isinstance(values, torch.Tensor) or values.ndim != 1:
            raise RuntimeError(f"Invalid {score_key} for {source.get('source_name')}")
        tensors.append(values.float())
    return torch.cat(tensors).numpy()


def _score_histogram(
    linear_axis: plt.Axes,
    log_axis: plt.Axes,
    values: np.ndarray,
    score_label: str,
) -> None:
    counts, edges = np.histogram(values, bins=SCORE_BINS)
    lefts = edges[:-1]
    widths = np.diff(edges)
    log_counts = counts.astype(np.float64)
    log_counts[log_counts == 0] = np.nan

    linear_axis.bar(lefts, counts, width=widths, align="edge", edgecolor="none")
    linear_axis.set_title(f"{score_label} Linear Count")
    linear_axis.set_ylabel("Count")
    log_axis.bar(lefts, log_counts, width=widths, align="edge", edgecolor="none")
    log_axis.set_yscale("log")
    log_axis.set_title(f"{score_label} Log Count")
    log_axis.set_ylabel("Count")

    median, p90, p99 = np.quantile(values, [0.5, 0.9, 0.99])
    markers = (
        (median, "median", "black", "-"),
        (p90, "P90", "tab:orange", "--"),
        (p99, "P99", "tab:red", ":"),
    )
    for axis in (linear_axis, log_axis):
        for value, label, color, linestyle in markers:
            axis.axvline(
                value,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
            )
        axis.set_xlim(float(edges[0]), float(edges[-1]))
        axis.set_xlabel(score_label)
    linear_axis.legend()


def _plot_group(
    label: str,
    sources: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    tail_scores = _score_values(sources, TAIL_SCORE)
    breadth_scores = _score_values(sources, BREADTH_SCORE)

    figure = plt.figure(figsize=(19, 10))
    grid = figure.add_gridspec(2, 3)
    tail_linear = figure.add_subplot(grid[0, 0])
    tail_log = figure.add_subplot(grid[0, 1])
    breadth_linear = figure.add_subplot(grid[1, 0])
    breadth_log = figure.add_subplot(grid[1, 1])
    joint_axis = figure.add_subplot(grid[:, 2])

    _score_histogram(tail_linear, tail_log, tail_scores, "T = max / P99")
    _score_histogram(
        breadth_linear, breadth_log, breadth_scores, "B = P99 / P90"
    )
    density = joint_axis.hexbin(
        tail_scores,
        breadth_scores,
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    joint_axis.set_title("Paired T-B Density")
    joint_axis.set_xlabel("T = max / P99")
    joint_axis.set_ylabel("B = P99 / P90")
    figure.colorbar(density, ax=joint_axis, label="Count (log color scale)")

    figure.suptitle(f"{label} | {tail_scores.size} channels")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    payload = _load_summary(args.records_dir / SUMMARY_FILENAME)
    sources = payload["sources"]
    if not isinstance(sources, list):
        raise RuntimeError("Weight-channel sources are missing")

    plot_count = 0
    _plot_group("Full model", sources, args.output_dir / "full_model.png")
    plot_count += 1

    families: Dict[str, List[Dict[str, object]]] = {}
    for source in sources:
        family = str(source["projection_family"])
        families.setdefault(family, []).append(source)
    for family, family_sources in families.items():
        _plot_group(
            f"Projection family: {family}",
            family_sources,
            args.output_dir / "projection_families" / f"{family}.png",
        )
        plot_count += 1

    for source in sources:
        source_name = str(source["source_name"])
        _plot_group(
            source_name,
            [source],
            args.output_dir
            / "linears"
            / f"{_safe_filename(source_name)}.png",
        )
        plot_count += 1

    print(f"Saved {plot_count} weight-channel score PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
