"""Matched down-only INT8/INT16/BF16 evaluation; no training or range search."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "worktrees/SpinQuant-phase3-joint"
sys.path.insert(0, str(SOURCE))

import torch
from transformers import LlamaTokenizerFast, set_seed
from experiments.phase3.common import DATA_PATH, MODEL_PATH, full_validation, wrappers, write_json
from experiments.phase3.postprocess import load_static

B100 = PROJECT / "runs/phase3/route-b-adam-100-20260914a"
PARENT = B100 / "checkpoint-0100/static_w4a8.pt"
CALIBRATION = B100 / "checkpoint-0100/sp2_calibration.json"
CAPTURE = PROJECT / "runs/phase3/b100-uniform-calibration-20260918c/capture_metadata.json"


class StaticSignedQuantizer(torch.nn.Module):
    """FP32 integer-grid simulation, returning the original floating dtype.

    bits=16 cannot enter the legacy wrapper's quantizer path. Both tested
    bitwidths therefore use the same explicit Linear pre-hook adapter.
    """
    def __init__(self, bits, alpha):
        super().__init__()
        if bits not in (8, 16) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Expected signed INT8/INT16 with positive finite alpha")
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.register_buffer("scale", torch.tensor([alpha / self.qmax], dtype=torch.float32))
        self.register_buffer("counts", torch.zeros(3, dtype=torch.int64))
        self.calls = 0

    def forward(self, values):
        original = values.float()
        normalized = original / self.scale
        codes = normalized.round().clamp(self.qmin, self.qmax)
        result = (codes * self.scale).to(values.dtype)
        self.counts[0] += values.numel()
        self.counts[1] += ((normalized < self.qmin) | (normalized > self.qmax)).sum()
        self.counts[2] += (result != values).sum()
        self.calls += 1
        return result


def install_down_format(model, precision, bounds):
    downs = {name: wrapper for name, wrapper in wrappers(model).items() if name.endswith("down_proj")}
    if set(downs) != set(bounds) or precision not in ("float", "int8", "int16"):
        raise ValueError("Down coverage or precision differs")
    handles, quantizers = [], {}
    for name, wrapper in downs.items():
        if wrapper.online_full_had or wrapper.online_partial_had:
            raise ValueError("This experiment requires offline R1/R2 only")
        # Explicitly bypass legacy SP2, which treats bits=16 as float.
        wrapper.quantizer.bits = 16
        if precision == "float":
            continue
        quantizer = StaticSignedQuantizer(int(precision[3:]), bounds[name]).to(wrapper.module.weight.device)
        # Registered on the Linear so evaluator's layer.to/cpu moves the scale.
        wrapper.module.add_module("phase6_input_quantizer", quantizer)
        def hook(module, inputs):
            return (module.phase6_input_quantizer(inputs[0]), *inputs[1:])
        handles.append(wrapper.module.register_forward_pre_hook(hook))
        quantizers[name] = quantizer
    return handles, quantizers


def load_bounds():
    history = json.loads(CALIBRATION.read_text())
    captured = json.loads(CAPTURE.read_text())
    if set(history) != set(captured) or len(history) != 16:
        raise ValueError("Historical capture coverage differs")
    for name, row in history.items():
        if row["full_absmax"] != captured[name]["full_absmax"] or row["sampled_rows"] != captured[name]["sampled_rows"]:
            raise ValueError("Historical calibration statistics disagree: " + name)
    return {name: row["full_absmax"] for name, row in history.items()}


def validation_windows():
    from datasets import Dataset
    tokenizer = LlamaTokenizerFast.from_pretrained(str(MODEL_PATH), local_files_only=True,
        model_max_length=2048, padding_side="right", use_fast=True,
        add_eos_token=False, add_bos_token=False)
    data = Dataset.from_file(str(DATA_PATH / "wikitext-validation.arrow"))
    ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    windows = [ids[:, start:start + 2048] for start in range(0, ids.numel() - 1, 2048)]
    if ids.numel() != 252852 or sum(window.numel() - 1 for window in windows) != 252728:
        raise ValueError("WikiText validation protocol differs from B100")
    return ids, windows


def frozen_scales(model):
    return {name: {key: value.detach().cpu().clone() for key, value in wrapper.quantizer.state_dict().items()}
            for name, wrapper in wrappers(model).items()}


@torch.no_grad()
def evaluate_arm(precision, bounds, windows, directory):
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    model, records = load_static(PARENT)
    del records
    original = frozen_scales(model)
    handles, quantizers = install_down_format(model, precision, bounds)
    scales = {name: quantizer.scale.detach().cpu().clone() for name, quantizer in quantizers.items()}
    torch.save(scales, directory / "down_scales.pt")
    torch.cuda.reset_peak_memory_stats()
    print(f"START {precision} pid={os.getpid()}", flush=True)
    result = full_validation(model, windows)
    after = frozen_scales(model)
    unchanged = original.keys() == after.keys() and all(
        original[name].keys() == after[name].keys() and all(
            torch.equal(value, after[name][key]) for key, value in row.items())
        for name, row in original.items())
    down_unchanged = all(torch.equal(scales[name], quantizer.scale.detach().cpu())
                         for name, quantizer in quantizers.items())
    if not unchanged or not down_unchanged:
        raise RuntimeError("Static scale buffers changed during evaluation")
    diagnostics = {}
    for name, quantizer in quantizers.items():
        total, clipped, changed = quantizer.counts.cpu().tolist()
        if total == 0 or quantizer.calls == 0:
            raise RuntimeError("Integer quantizer was bypassed: " + name)
        diagnostics[name] = dict(bits=quantizer.bits, scale=float(quantizer.scale.item()),
            qmin=quantizer.qmin, qmax=quantizer.qmax, calls=quantizer.calls,
            elements=total, outside_range=clipped, changed_after_bf16=changed,
            outside_range_fraction=clipped / total, changed_after_bf16_fraction=changed / total)
    result.update(precision=precision, parent=str(PARENT), dataset="Salesforce/wikitext",
        subset="wikitext-2-raw-v1", split="validation", seed=42,
        backbone_weight_bits=4, non_down_activation_bits=8,
        down_format="BF16 bypass" if precision == "float" else "static signed " + precision.upper(),
        calibration="reused train-only absmax; no search", calibration_source=str(CALIBRATION),
        alpha_by_layer=bounds, kv_format="BF16", use_cache=False, training=False,
        inference="BF16 fake-quant prefill; not native integer GEMM or NPU measurement",
        all_scale_buffers_unchanged=unchanged and down_unchanged, quantizer_diagnostics=diagnostics,
        elapsed_seconds=time.monotonic() - started,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    write_json(directory / "result.json", result)
    print(f"RESULT {precision} PPL={result['ppl']:.10f} NLL={result['nll']:.10f}", flush=True)
    for handle in handles:
        handle.remove()
    del model, quantizers
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=("float", "int8", "int16"), default=["float", "int8", "int16"])
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("Duplicate arms")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    source_copy = args.output / "source"
    source_copy.mkdir()
    shutil.copy2(__file__, source_copy / Path(__file__).name)
    for relative in ("experiments/phase3/common.py", "experiments/phase3/postprocess.py",
                     "experiments/phase3/quantization.py", "utils/quant_utils.py", "utils/eval_utils.py"):
        target = source_copy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, target)
    shutil.copy2(PROJECT / "scripts/phase2/validation_acceptance.py", source_copy / "validation_acceptance.py")
    (source_copy / "tracked.diff").write_text(subprocess.check_output(
        ["git", "-C", str(SOURCE), "diff", "HEAD"], text=True))
    settings = dict(pid=os.getpid(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        command=sys.argv, source_root=str(SOURCE), parent=str(PARENT), arms=args.arms,
        source_head=subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        calibration_metadata=json.loads((B100 / "data.json").read_text()),
        calibration_source=str(CALIBRATION), capture_crosscheck=str(CAPTURE))
    write_json(args.output / "settings.json", settings)
    try:
        bounds = load_bounds()
        ids, windows = validation_windows()
        torch.save(dict(input_ids=ids, split="validation", predicted_tokens=252728), args.output / "input_tokens.pt")
        results = {}
        for arm in args.arms:
            write_json(args.output / "progress.json", dict(stage="evaluating", arm=arm, pid=os.getpid()))
            results[arm] = evaluate_arm(arm, bounds, windows, args.output / arm)
            write_json(args.output / "results.json", results)
        write_json(args.output / "progress.json", dict(stage="completed", pid=os.getpid(), arms=args.arms))
    except BaseException as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    main()
