"""Conditional precision restoration using the existing full validation evaluator."""
import argparse
from pathlib import Path
import shutil
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))
import torch
from experiments.phase3.common import data_windows, full_validation, wrappers, write_json
from experiments.phase3.postprocess import apply_records, load_static, reference_weights
from experiments.phase3.run import source_record


def selected(name, family):
    if family == "none":
        return False
    if family == "all":
        return True
    if family == "down":
        return name.endswith("down_proj")
    if family == "attention":
        return ".self_attn." in name
    if family == "gateup":
        return name.endswith(("gate_proj", "up_proj"))
    if family == "mlp":
        return ".mlp." in name
    if family == "down1":
        return name == "model.layers.1.mlp.down_proj"
    if family == "down_except1":
        return name.endswith("down_proj") and not selected(name, "down1")
    raise ValueError(f"Unknown restoration family: {family}")


def rtn_reference(weight, bits, groupsize):
    """Same absmax convention as existing SW initialization, grouped on input axis."""
    if bits not in (4, 8) or groupsize not in (-1, 128):
        raise ValueError("Only declared diagnostic formats are supported")
    width = weight.shape[1] if groupsize == -1 else groupsize
    if weight.shape[1] % width:
        raise ValueError("Input width must be divisible by group size")
    grouped = weight.float().reshape(weight.shape[0], -1, width)
    maximum = 2 ** (bits - 1) - 1
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-5) / maximum
    codes = (grouped / scale).round().clamp(-maximum-1, maximum)
    return (codes * scale).reshape_as(weight).to(weight.dtype)


def error_components(parent, reference, scale):
    """Exact FP32 split relative to the *existing* signed [-8,7] row grid."""
    if scale.shape != (reference.shape[0], 1):
        raise ValueError("Expected original per-output-channel SW")
    ref = reference.float()
    bounded = torch.maximum(torch.minimum(ref, 7 * scale), -8 * scale)
    clipping = bounded - ref
    grid = parent.float() - bounded
    return clipping, grid


@torch.no_grad()
def configure(model, records, reference, activation, family, replacement="reference"):
    if activation not in ("all16", "down16", "static"):
        raise ValueError(activation)
    apply_records(model, records)
    restored = []
    modes = {}
    components = {}
    for name, wrapper in wrappers(model).items():
        wrapper.quantizer.bits = 16 if activation == "all16" or (
            activation == "down16" and name.endswith("down_proj")) else 8
        wrapper.out_quantizer.bits = 16
        modes[name] = wrapper.quantizer.bits
        if selected(name, family):
            weight = reference[name].to(wrapper.module.weight.device)
            if replacement in ("remove_clipping", "remove_grid"):
                scale = records[name]["scale"].to(weight.device)
                parent = wrapper.module.weight
                clip, grid = error_components(parent, weight, scale)
                components[name] = dict(elements=weight.numel(), clipped=int((clip != 0).sum()),
                    clipping_sse=float(clip.square().sum()), grid_sse=float(grid.square().sum()),
                    cross_twice=float(2 * (clip * grid).sum()),
                    total_sse=float((parent.float()-weight.float()).square().sum()))
                error = clip if replacement == "remove_clipping" else grid
                weight = (parent.float() - error).to(weight.dtype)
            elif replacement != "reference":
                bits, group = {"rtn4c": (4, -1), "rtn4g128": (4, 128), "rtn8c": (8, -1)}[replacement]
                weight = rtn_reference(weight, bits, group)
            wrapper.module.weight.copy_(weight)
            restored.append(name)
    return dict(restored=restored, input_bits=modes, replacement=replacement, components=components)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "diagnose.py")
    _, _, _, validation, data = data_windows()
    settings = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source=source_record(), data=data,
        reference="original unquantized norm-fused BF16 weights rotated with B100 saved R; never dequantized W4",
        protocol="reset all 112 parent q/SW before every case; only specified weights/activation precision changes; same evaluator; no calibration, optimization or package export",
        rtn="same unquantized reference and absmax/maxcode scale, nearest ties-to-even rounding and signed clip; channel versus contiguous input groups128; matched deterministic solver, no data or search",
        components="Wparent-Wref=(clamp(Wref,-8SW,7SW)-Wref)+(Wparent-clamp(Wref,-8SW,7SW)); remove only one FP32 component then BF16 cast; grid includes optimized code decisions and dequant rounding, not only nearest rounding; diagnostic only",
        interpretation="conditional restoration effects; not additive attribution")
    write_json(args.output / "settings.json", settings)
    reference = reference_weights(args.reference_state)
    model, records = load_static(args.parent)
    results = {}
    for case in args.cases:
        fields = case.split(":")
        conditions = configure(model, records, reference, *fields)
        started = time.monotonic()
        print(f"START {case} restored={len(conditions['restored'])}", flush=True)
        result = full_validation(model, validation)
        results[case] = dict(conditions=conditions, evaluation=result, seconds=time.monotonic()-started)
        write_json(args.output / "results.json", results)
        print(f"RESULT {case} {result}", flush=True)
    write_json(args.output / "complete.json", dict(cases=list(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    run(parser.parse_args())
