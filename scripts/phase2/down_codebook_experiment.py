#!/usr/bin/env python3
"""Fixed-W4 down-input codebook calibration, local damage, and test PPL."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from down_codebooks import FORMATS, FrozenCodebookQuantizer, build_magnitude_codebook, quantize


def write_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def text_windows(tokenizer, split, count, seqlen, seed):
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    available = ids.numel() // seqlen
    if count > available:
        raise ValueError(f"Only {available} disjoint {split} windows, requested {count}")
    # Disjoint windows; train is shuffled, validation uses its first windows.
    indices = torch.randperm(available, generator=torch.Generator().manual_seed(seed))[:count]
    if split != "train":
        indices = torch.arange(count)
    return [ids[:, int(i) * seqlen:(int(i) + 1) * seqlen] for i in indices], indices.tolist()


@torch.no_grad()
def collect_calibration(model, downs, windows, rows_per_window, seed, device):
    from utils.quant_phase import QuantPhase

    rows = {name: [] for name in downs}
    full_absmax = {name: 0. for name in downs}
    generator = torch.Generator().manual_seed(seed)
    selected = [torch.randperm(window.shape[1], generator=generator)[:rows_per_window]
                for window in windows]
    current = [0]

    def capture(name):
        def hook(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            full_absmax[name] = max(full_absmax[name], float(x.abs().max()))
            rows[name].append(x[selected[current[0]].to(x.device)].cpu())
            # Return None: do not perturb this or any downstream layer's input.
        return hook

    handles = [wrapper.register_forward_pre_hook(capture(name)) for name, wrapper in downs.items()]
    try:
        model.to(device)
        for i, ids in enumerate(windows):
            current[0] = i
            model.model(ids.to(device), use_cache=False, quant_phase=QuantPhase.PREFILL)
            if (i + 1) % 8 == 0:
                print(f"Calibration reference windows {i + 1}/{len(windows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(values) for name, values in rows.items()}, full_absmax


@torch.no_grad()
def calibrate_layer(x, weight, full_absmax, coarse_steps, refine_steps):
    """Same 33+17 alpha evaluations and output-MSE objective for each format."""
    weight = weight.float()
    xf = x.float()
    # Include >max: PoT can improve rounding alignment with alpha above max.
    bound = max(full_absmax, torch.finfo(torch.float32).tiny)
    coarse = torch.linspace(-16., 1., coarse_steps, dtype=torch.float64)
    result = {}
    for format in FORMATS:
        levels = build_magnitude_codebook(format).to(x.device)
        evaluations = []

        def evaluate(log_ratio):
            alpha = bound * 2. ** float(log_ratio)
            delta = quantize(x, alpha, format, levels).float() - xf
            output_delta = F.linear(delta, weight)
            mse = float(output_delta.square().mean())
            evaluations.append({"alpha": alpha, "output_mse": mse})
            return mse

        coarse_losses = [evaluate(log_ratio) for log_ratio in coarse]
        best_index = min(range(len(coarse_losses)), key=coarse_losses.__getitem__)
        lower = coarse[max(0, best_index - 1)]
        upper = coarse[min(len(coarse) - 1, best_index + 1)]
        for log_ratio in torch.linspace(float(lower), float(upper), refine_steps, dtype=torch.float64):
            evaluate(log_ratio)
        best = min(evaluations, key=lambda row: row["output_mse"])
        result[format] = dict(best, alpha_over_full_absmax=best["alpha"] / bound,
                              coarse_best_at_boundary=best_index in (0, coarse_steps - 1),
                              evaluated_candidates=evaluations)
    return result


@torch.no_grad()
def local_damage(model, downs, windows, scales, device):
    """All held-out tokens, no candidate error propagation between layers."""
    from utils.quant_phase import QuantPhase

    sums = {name: {format: {"activation_sse": 0., "output_sse": 0.,
                            "input_energy": 0., "output_energy": 0.,
                            "input_count": 0, "output_count": 0, "clipped": 0,
                            "zeroed_nonzero": 0, "nonzero_input": 0,
                            "max_abs_output_error": 0., "window_output_mse": []}
                   for format in FORMATS} for name in downs}

    def measure(name):
        def hook(wrapper, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            weight = wrapper.module.weight.float()
            xf = x.float()
            reference = F.linear(xf, weight)
            input_energy = float(xf.square().sum(dtype=torch.float64))
            output_energy = float(reference.square().sum(dtype=torch.float64))
            nonzero = x != 0
            for format in FORMATS:
                alpha = scales[format][name]
                q = quantize(x, alpha, format)
                delta = q.float() - xf
                output_delta = F.linear(delta, weight)
                row = sums[name][format]
                output_sse = float(output_delta.square().sum(dtype=torch.float64))
                row["activation_sse"] += float(delta.square().sum(dtype=torch.float64))
                row["output_sse"] += output_sse
                row["input_energy"] += input_energy
                row["output_energy"] += output_energy
                row["input_count"] += x.numel()
                row["output_count"] += output_delta.numel()
                lower = -alpha * (128 / 127 if format == "int8" else 1)
                row["clipped"] += int(((xf < lower) | (xf > alpha)).sum())
                row["zeroed_nonzero"] += int(((q == 0) & nonzero).sum())
                row["nonzero_input"] += int(nonzero.sum())
                row["max_abs_output_error"] = max(row["max_abs_output_error"], float(output_delta.abs().max()))
                row["window_output_mse"].append(output_sse / output_delta.numel())
        return hook

    handles = [wrapper.register_forward_pre_hook(measure(name)) for name, wrapper in downs.items()]
    try:
        model.to(device)
        for i, ids in enumerate(windows):
            model.model(ids.to(device), use_cache=False, quant_phase=QuantPhase.PREFILL)
            print(f"Local damage validation windows {i + 1}/{len(windows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    for formats in sums.values():
        for row in formats.values():
            row["activation_mse"] = row["activation_sse"] / row["input_count"]
            row["output_mse"] = row["output_sse"] / row["output_count"]
            row["output_nmse"] = row["output_sse"] / max(row["output_energy"], 1e-30)
            row["clipped_fraction"] = row["clipped"] / row["input_count"]
            row["zeroed_nonzero_fraction"] = row["zeroed_nonzero"] / max(row["nonzero_input"], 1)
    return sums


def run_experiment(model, testenc, device, args, options, original_evaluator, source):
    from transformers import LlamaTokenizerFast

    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    downs = {name: module for name, module in model.named_modules()
             if name.endswith("mlp.down_proj")}
    if len(downs) != len(model.model.layers):
        raise ValueError("Missing down wrappers")
    for name, wrapper in downs.items():
        if wrapper.quantizer.bits != 16 or wrapper.online_full_had or wrapper.online_partial_had:
            raise ValueError(f"Expected R1/R2-only down-A16 reference: {name}")
    non_down = {name: module.quantizer for name, module in model.named_modules()
                if hasattr(module, "quantizer") and hasattr(module, "module")
                and name not in downs and name != "lm_head"}
    exported = torch.load(args.static_scale_path, map_location="cpu", weights_only=True)["activation"]
    snapshot = {}
    for name, quantizer in non_down.items():
        scale = quantizer.scale.detach().cpu().clone()
        if quantizer.bits != 8 or not torch.equal(scale.reshape(-1), exported[name + ".quantizer"].float().reshape(-1)):
            raise ValueError(f"Loaded non-down SA differs from source export: {name}")
        snapshot[name] = scale
    if len(snapshot) != 6 * len(downs):
        raise ValueError(f"Expected six non-down SA per block, got {len(snapshot)}")

    output = options.down_output
    start = time.monotonic()
    summary = dict(source, bits=8, weight_method=options.down_weight_method,
                   reference_mode="fixed-W4/non-down-static-A8/down-A16/KV16",
                   calibration_split="train", calibration_windows=options.down_calibration_windows,
                   calibration_rows_per_window=options.down_rows_per_window,
                   local_split="validation", local_windows=options.down_local_windows,
                   eval_split="test", eval_nsamples=args.eval_nsamples, seqlen=model.seqlen,
                   seed=args.seed, non_down_scale_count=len(snapshot), down_count=len(downs),
                   coarse_steps=options.down_coarse_steps, refine_steps=options.down_refine_steps,
                   calibration_objective="mean(square((Q_bf16(X)-X) @ W4.T)) in FP32",
                   scale_range_relative_to_full_absmax=[2. ** -16, 2.],
                   results={})
    tokenizer = LlamaTokenizerFast.from_pretrained(
        source["input_model"], model_max_length=model.seqlen, padding_side="right",
        use_fast=True, add_eos_token=False, add_bos_token=False)
    train, train_indices = text_windows(tokenizer, "train", options.down_calibration_windows,
                                         model.seqlen, args.seed)
    heldout, heldout_indices = text_windows(tokenizer, "validation", options.down_local_windows,
                                            model.seqlen, args.seed)
    summary["calibration_window_indices"] = train_indices
    summary["local_window_indices"] = heldout_indices
    write_json(output / "settings.json", summary)
    reference_ppl = original_evaluator(model, testenc, device, args)
    summary["results"]["down_a16"] = {"ppl": reference_ppl, "nll": math.log(reference_ppl)}
    print(f"REFERENCE down-A16 PPL={reference_ppl:.8f}", flush=True)

    bank, maxima = collect_calibration(model, downs, train, options.down_rows_per_window, args.seed, device)
    scales = {format: {} for format in FORMATS}
    calibration = {}
    for i, (name, wrapper) in enumerate(downs.items()):
        calibration[name] = calibrate_layer(bank.pop(name).to(device), wrapper.module.weight,
                                             maxima[name], options.down_coarse_steps, options.down_refine_steps)
        for format in FORMATS:
            scales[format][name] = calibration[name][format]["alpha"]
        write_json(output / f"calibration_layer_{i:02d}.json", calibration[name])
        print(f"Calibrated layer {i}: " + ", ".join(
            f"{f} alpha={scales[f][name]:.6g} mse={calibration[name][f]['output_mse']:.6g}" for f in FORMATS), flush=True)
    write_json(output / "down_scales.json", scales)
    torch.save(scales, output / "down_scales.pt")
    codebooks = {format: build_magnitude_codebook(format).tolist() for format in FORMATS}
    write_json(output / "codebooks.json", {"magnitude_levels": codebooks,
               "unique_signed_values": {"int8": 256, "pot": 255, "sp2": 187},
               "sp2_bit_allocation": [1, 4, 3], "projection_dtype": "float32", "gemm_input_dtype": "bfloat16"})
    damage = local_damage(model, downs, heldout, scales, device)
    write_json(output / "local_damage.json", damage)

    original_quantizers = {name: wrapper.quantizer for name, wrapper in downs.items()}
    try:
        for format in FORMATS:
            for name, wrapper in downs.items():
                wrapper.quantizer = FrozenCodebookQuantizer(format, scales[format][name]).to(wrapper.module.weight.device)
            ppl = original_evaluator(model, testenc, device, args)
            summary["results"][format] = {"ppl": ppl, "nll": math.log(ppl),
                                          "delta_ppl_vs_down_a16": ppl - reference_ppl}
            write_json(output / f"ppl_{format}.json", summary["results"][format])
            print(f"ALL DOWN {format} PPL={ppl:.8f}", flush=True)
    finally:
        for name, wrapper in downs.items():
            wrapper.quantizer = original_quantizers[name]
    for name, quantizer in non_down.items():
        if not torch.equal(quantizer.scale.detach().cpu(), snapshot[name]):
            raise RuntimeError(f"Non-down SA changed during the experiment: {name}")
    summary["non_down_scales_unchanged"] = True
    summary["elapsed_seconds"] = time.monotonic() - start
    write_json(output / "summary.json", summary)
    print(json.dumps(summary["results"], indent=2), flush=True)
    return reference_ppl


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--down-output", type=Path, required=True)
    parser.add_argument("--down-weight-method", choices=("gptq", "rtn"), required=True)
    parser.add_argument("--down-calibration-windows", type=int, default=32)
    parser.add_argument("--down-local-windows", type=int, default=8)
    parser.add_argument("--down-rows-per-window", type=int, default=64)
    parser.add_argument("--down-coarse-steps", type=int, default=33)
    parser.add_argument("--down-refine-steps", type=int, default=17)
    options, ptq_argv = parser.parse_known_args()
    options.down_output.mkdir(parents=True, exist_ok=False)
    sys.argv = [sys.argv[0], *ptq_argv]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repos/SpinQuant"))
    import ptq

    # Match existing evaluator precision; local linear-error measurements use FP32.
    torch.backends.cuda.matmul.allow_tf32 = False
    original_prepare = ptq.ptq_model
    original_evaluator = ptq.eval_utils.evaluator
    source = {}

    def prepare(args, model, model_args=None):
        if not args.load_qmodel_path or args.save_qmodel_path or args.w_bits != 4:
            raise ValueError("Load one existing W4 checkpoint without GPTQ/RTN recomputation or save")
        if not args.static_down_proj_fp16 or args.rotation_components != "r1_r2":
            raise ValueError("Reference must use R1/R2 only and down-A16")
        source.update(input_model=model_args.input_model, weight_path=args.load_qmodel_path,
                      rotation_path=args.optimized_rotation_path, scale_path=args.static_scale_path)
        return original_prepare(args, model, model_args)

    def evaluate(model, testenc, device, args):
        return run_experiment(model, testenc, device, args, options, original_evaluator, source)

    ptq.ptq_model = prepare
    ptq.eval_utils.evaluator = evaluate
    try:
        ptq.train()
    finally:
        ptq.ptq_model = original_prepare
        ptq.eval_utils.evaluator = original_evaluator


if __name__ == "__main__":
    main()
