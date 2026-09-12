"""Plot 4096-bin Atlas histograms as 256-bin two-panel PNGs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RECORDS_DIR = (
    PROJECT_ROOT / "runs/distribution-atlas/llama32-1b-r12-bf16-wt2-s42"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RECORDS_DIR.with_name(
    DEFAULT_RECORDS_DIR.name + "-plan-format"
)
WEIGHT_RECORDS_FILENAME = "weight_records.json"
ACTIVATION_RECORDS_FILENAME = "activation_records.json"
HISTOGRAM_BINS = 4096
DISPLAY_BINS = 256
EXPECTED_WEIGHT_SOURCES = 112
EXPECTED_ACTIVATION_SOURCES = 64


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_records(path: Path, expected_count: int, source_kind: str) -> List[Dict]:
    with path.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list) or len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {source_kind} records in {path}"
        )

    for record in records:
        if record.get("source_kind") != source_kind:
            raise RuntimeError(
                f"Record {record.get('source_name')} is not a {source_kind} record"
            )
        histogram = record.get("histogram")
        if not isinstance(histogram, dict):
            raise RuntimeError(f"Record {record.get('source_name')} has no histogram")
        if histogram.get("bins") != HISTOGRAM_BINS:
            raise RuntimeError(
                f"Record {record['source_name']} does not use {HISTOGRAM_BINS} bins"
            )
        edges = np.asarray(histogram.get("edges"), dtype=np.float64)
        counts = np.asarray(histogram.get("counts"), dtype=np.int64)
        if edges.size != HISTOGRAM_BINS + 1 or counts.size != HISTOGRAM_BINS:
            raise RuntimeError(f"Record {record['source_name']} has invalid histogram")
        if not np.all(np.diff(edges) > 0):
            raise RuntimeError(f"Record {record['source_name']} has unsorted edges")
        if int(counts.sum()) != int(record["num_elements"]):
            raise RuntimeError(
                f"Record {record['source_name']} has inconsistent histogram counts"
            )
    return records


def _safe_filename(source_name: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name).strip("._")
    return filename or "source"


def _mark_extrema(
    axis: plt.Axes,
    minimum: float,
    maximum: float,
) -> None:
    axis.axvline(minimum, color="tab:blue", linestyle="--", linewidth=0.8)
    axis.axvline(maximum, color="tab:red", linestyle="--", linewidth=0.8)
    axis.text(
        minimum,
        0.98,
        f"min={minimum:.5g}",
        color="tab:blue",
        rotation=90,
        ha="right",
        va="top",
        transform=axis.get_xaxis_transform(),
    )
    axis.text(
        maximum,
        0.98,
        f"max={maximum:.5g}",
        color="tab:red",
        rotation=90,
        ha="left",
        va="top",
        transform=axis.get_xaxis_transform(),
    )


def _plot_record(record: Dict, output_path: Path) -> None:
    histogram = record["histogram"]
    edges = np.asarray(histogram["edges"], dtype=np.float64)
    counts = np.asarray(histogram["counts"], dtype=np.int64)
    merge_factor = HISTOGRAM_BINS // DISPLAY_BINS
    display_counts = counts.reshape(DISPLAY_BINS, merge_factor).sum(axis=1)
    display_edges = edges[::merge_factor]
    log_counts = display_counts.astype(np.float64)
    log_counts[log_counts == 0] = np.nan

    bar_lefts = display_edges[:-1]
    bar_widths = np.diff(display_edges)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 1600 / 150),
        sharex=True,
    )
    axes[0].bar(
        bar_lefts,
        display_counts,
        width=bar_widths,
        align="edge",
        color="tab:blue",
        edgecolor="none",
    )
    axes[0].set_title("Linear Count")
    axes[0].set_ylabel("Count")
    axes[1].bar(
        bar_lefts,
        log_counts,
        width=bar_widths,
        align="edge",
        color="tab:blue",
        edgecolor="none",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Log Count")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Value")

    for axis in axes:
        _mark_extrema(axis, float(record["min"]), float(record["max"]))
        axis.set_xlim(float(edges[0]), float(edges[-1]))
    figure.suptitle(record["source_name"])
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_records(
    records: Sequence[Dict],
    output_dir: Path,
) -> int:
    for record in records:
        output_path = output_dir / f"{_safe_filename(record['source_name'])}.png"
        _plot_record(record, output_path)
    return len(records)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    weight_records = _load_records(
        args.records_dir / WEIGHT_RECORDS_FILENAME,
        EXPECTED_WEIGHT_SOURCES,
        "weight",
    )
    activation_records = _load_records(
        args.records_dir / ACTIVATION_RECORDS_FILENAME,
        EXPECTED_ACTIVATION_SOURCES,
        "activation",
    )

    weight_count = _plot_records(weight_records, args.output_dir / "weights")
    activation_count = _plot_records(
        activation_records,
        args.output_dir / "activations",
    )
    print(
        f"Saved {weight_count} weight PNGs and "
        f"{activation_count} activation PNGs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
