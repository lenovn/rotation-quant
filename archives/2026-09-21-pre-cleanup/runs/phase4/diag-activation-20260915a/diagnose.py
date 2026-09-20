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
    if family == "down1":
        return name == "model.layers.1.mlp.down_proj"
    if family == "down_except1":
        return name.endswith("down_proj") and not selected(name, "down1")
    raise ValueError(f"Unknown restoration family: {family}")


@torch.no_grad()
def configure(model, records, reference, activation, family):
    if activation not in ("all16", "down16", "static"):
        raise ValueError(activation)
    apply_records(model, records)
    restored = []
    modes = {}
    for name, wrapper in wrappers(model).items():
        wrapper.quantizer.bits = 16 if activation == "all16" or (
            activation == "down16" and name.endswith("down_proj")) else 8
        wrapper.out_quantizer.bits = 16
        modes[name] = wrapper.quantizer.bits
        if selected(name, family):
            wrapper.module.weight.copy_(reference[name])
            restored.append(name)
    return dict(restored=restored, input_bits=modes)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "diagnose.py")
    _, _, _, validation, data = data_windows()
    settings = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source=source_record(), data=data,
        reference="original unquantized norm-fused BF16 weights rotated with B100 saved R; never dequantized W4",
        protocol="reset all 112 parent q/SW before every case; only specified weight/activation precision changes; same evaluator; no calibration, optimization or package export",
        interpretation="conditional restoration effects; not additive attribution")
    write_json(args.output / "settings.json", settings)
    reference = reference_weights(args.reference_state)
    model, records = load_static(args.parent)
    results = {}
    for case in args.cases:
        activation, family = case.split(":")
        conditions = configure(model, records, reference, activation, family)
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
