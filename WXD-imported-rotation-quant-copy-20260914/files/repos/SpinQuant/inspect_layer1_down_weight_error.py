"""Locate fixed-checkpoint W4 error against the matching rotated BF16 weight.

Run with the project's Python environment. CPU only; no model forward or GPTQ.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    project = Path(__file__).resolve().parents[2]
    run = project / "runs/phase2/w16a8-downa16-joint-r-sa-r12-100step-s42"
    output = run / "weight-error-layer1-down"
    output.mkdir(parents=True, exist_ok=True)
    target = "model.layers.1.mlp.down_proj"
    model_file = project / "cache/models/llama-3.2-1b-instruct/model.safetensors"
    with safe_open(model_file, framework="pt", device="cpu") as source:
        original = source.get_tensor(target + ".weight").to(torch.bfloat16)
    rotation = torch.load(run / "rotation/R.bin", map_location="cpu", weights_only=True)
    # Matches rotate_mlp_output(apply_r4=False): float64 product -> BF16.
    # fuse_layer_norms modifies gate/up but does not modify down_proj.
    w16 = (rotation["R1"].double().T @ original.double()).to(torch.bfloat16)
    del original, rotation
    artifact = torch.load(run / "gptq/w4_gptq_model.pt", map_location="cpu", weights_only=False)
    w4 = artifact["model"][target + ".module.weight"].clone()
    quantizer = artifact["w_quantizers"][target + ".module"]
    scale = quantizer.scale.detach().float().reshape(-1).clone()
    bits = int(quantizer.bits)
    del artifact, quantizer
    assert bits == 4 and w4.shape == w16.shape == (2048, 8192)
    assert scale.numel() == w4.shape[0]
    error = w4.float() - w16.float()
    assert torch.isfinite(error).all()
    absolute = error.abs()
    square = error.double().square()
    total = square.sum().item()
    row_sse, col_sse = square.sum(1), square.sum(0)
    row_order, col_order = row_sse.argsort(descending=True), col_sse.argsort(descending=True)
    reference_energy = w16.double().square()
    torch.save({"w16": w16, "w4": w4, "row_scale": scale}, output / "weight_pair.pt")

    records = {}
    for name, dim, sums, order in (("rows", 1, row_sse, row_order), ("columns", 0, col_sse, col_order)):
        mae, maximum = absolute.mean(dim), absolute.amax(dim)
        rms = (sums / error.shape[dim]).sqrt()
        ref = reference_energy.sum(dim)
        relative = (sums / ref.clamp_min(1e-30)).sqrt()
        rows = []
        for rank, index in enumerate(order.tolist(), 1):
            row = {"rank": rank, "index": index, "sse": float(sums[index]),
                   "sse_percent": 100 * float(sums[index]) / total,
                   "rmse": float(rms[index]), "mae": float(mae[index]),
                   "max_abs_error": float(maximum[index]),
                   "relative_l2_error": float(relative[index])}
            if name == "rows":
                row["quant_scale"] = float(scale[index])
            rows.append(row)
        records[name] = rows
        with (output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    values, indices = absolute.flatten().topk(100)
    elements = []
    for value, index in zip(values.tolist(), indices.tolist()):
        row, col = divmod(index, error.shape[1])
        elements.append({"row": row, "column": col, "abs_error": value,
                         "signed_error": float(error[row, col]),
                         "w16": float(w16[row, col]), "w4": float(w4[row, col])})
    with (output / "top100_elements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(elements[0]))
        writer.writeheader()
        writer.writerows(elements)

    concentration = {}
    for name, sums, order in (("rows", row_sse, row_order), ("columns", col_sse, col_order)):
        cumulative = sums[order].cumsum(0) / total
        concentration[name] = {
            "top_counts_sse_percent": {str(k): float(cumulative[k-1] * 100) for k in (1, 5, 10, 32, 64, 128)},
            "count_for_50_percent_sse": int(torch.searchsorted(cumulative, 0.5)) + 1,
            "count_for_90_percent_sse": int(torch.searchsorted(cumulative, 0.9)) + 1,
        }
    tracked = {str(c): next(row for row in records["columns"] if row["index"] == c)
               for c in (1417, 113, 7272, 2937, 7281)}
    summary = {
        "target": target, "shape": list(error.shape), "index_base": 0,
        "reference": "original BF16 down weight -> R1.T @ W in CPU float64 -> BF16; R4 off",
        "model_file": str(model_file), "rotation": str(run / "rotation/R.bin"),
        "w4_checkpoint": str(run / "gptq/w4_gptq_model.pt"),
        "metric": "squared error of stored W4 minus matching rotated BF16; not activation-weighted",
        "sse": total, "rmse": float(square.mean().sqrt()),
        "mae": float(absolute.mean()), "max_abs_error": float(absolute.max()),
        "relative_frobenius_error": float((square.sum() / reference_energy.sum()).sqrt()),
        "top_rows": records["rows"][:20], "top_columns": records["columns"][:20],
        "concentration": concentration, "activation_outlier_columns": tracked,
        "largest_error_elements": elements[:10],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    fig.suptitle("Layer 1 down_proj: where is W4 weight error?\n"
                 "Delta W = stored W4 - matching R1-rotated BF16 | no activation weighting", fontsize=17)
    # Explicit block averaging; exact row/column profiles retain narrow anomalies.
    block_mse = square.reshape(256, 8, 256, 32).mean(dim=(1, 3)).numpy()
    im = axes[0, 0].imshow(np.log10(block_mse + 1e-30), aspect="auto", origin="upper",
                            extent=(-0.5, 8191.5, 2047.5, -0.5), cmap="magma")
    axes[0, 0].set(title="A. Error map (each cell averages 8 rows x 32 columns)",
                   xlabel="Input channel / column", ylabel="Output channel / row")
    fig.colorbar(im, ax=axes[0, 0], label="log10(mean squared error)")
    for ax, sums, title, label in (
        (axes[0, 1], col_sse, "B. Column error profile", "Input channel / column"),
        (axes[0, 2], row_sse, "C. Row error profile", "Output channel / row"),
    ):
        ax.plot(np.arange(len(sums)), sums.numpy() / total * 100, lw=0.7)
        ax.set(title=title, xlabel=label, ylabel="Share of total squared error (%)")
    axes[0, 1].axvline(1417, color="#c44e52", ls="--", label="Activation outlier column 1417")
    axes[0, 1].legend(fontsize=8)
    for ax, name, title in ((axes[1, 0], "columns", "D. Highest-error columns"),
                             (axes[1, 1], "rows", "E. Highest-error rows")):
        top_rows = records[name][:12]
        ax.bar(range(12), [row["sse_percent"] for row in top_rows], color="#2878a0")
        ax.set_xticks(range(12), [str(row["index"]) for row in top_rows], rotation=60)
        ax.set(title=title, xlabel="Channel index", ylabel="Share of total squared error (%)")
    ax = axes[1, 2]
    for name, sums, order in (("Rows", row_sse, row_order), ("Columns", col_sse, col_order)):
        ax.plot(np.arange(1, len(sums)+1) / len(sums) * 100,
                sums[order].cumsum(0).numpy() / total * 100, label=name)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Uniform error")
    ax.set(title="F. Error concentration (ranked by squared error)",
           xlabel="Top fraction of rows / columns (%)", ylabel="Cumulative squared error (%)")
    ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"weight_error.{extension}", dpi=180)
    plt.close(fig)
    lines = ["# Layer 1 down_proj W4 weight error", "",
             "All indices are zero-based. Delta W = stored W4 - matching rotated BF16.",
             "Squared-error concentration describes weights alone, not PPL or activation-weighted output error.", ""]
    for name in ("columns", "rows"):
        lines += [f"## Top {name}", "", "| Rank | Index | Total SSE share | RMSE | Max absolute error |",
                  "|---|---|---|---|---|"]
        for row in records[name][:10]:
            lines.append(f"| {row['rank']} | {row['index']} | {row['sse_percent']:.4f}% | {row['rmse']:.6g} | {row['max_abs_error']:.6g} |")
        lines += ["", str(concentration[name]), ""]
    lines += ["## Previously observed activation-outlier columns", "",
              "| Column | SSE rank | Total SSE share |", "|---|---|---|"]
    for c, row in tracked.items():
        lines.append(f"| {c} | {row['rank']} / 8192 | {row['sse_percent']:.6f}% |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "concentration": concentration,
                      "top_columns": records["columns"][:5], "top_rows": records["rows"][:5],
                      "activation_outlier_columns": tracked}, indent=2), flush=True)


if __name__ == "__main__":
    main()
