"""Calibration/held-out diagnostics at the actual layer-1 down GEMM input."""
import copy
import json
import random
from pathlib import Path

import datasets
import numpy as np
import torch
import transformers
from transformers import LlamaTokenizerFast

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import quant_utils
from utils.process_args import process_args_ptq
from utils.quant_phase import QuantPhase


def quantiles(hist, levels):
    """Exact linear quantiles from a histogram of nonnegative BF16 bit patterns."""
    counts = hist.cpu().numpy()
    cumulative = counts.cumsum()
    values = torch.arange(32768, dtype=torch.int16).view(torch.bfloat16).float().numpy()
    result = []
    for q in levels:
        pos = q * (int(cumulative[-1]) - 1)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        left = values[np.searchsorted(cumulative, lo, side="right")]
        right = values[np.searchsorted(cumulative, hi, side="right")]
        result.append(float(left + (right-left) * (pos-lo)))
    return result


@torch.no_grad()
def main():
    ma, ta, args = process_args_ptq()
    out = Path(ta.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = transformers.AutoConfig.from_pretrained(ma.input_model)
    tied = config.tie_word_embeddings
    config.tie_word_embeddings = False
    model = LlamaForCausalLM.from_pretrained(ma.input_model, config=config, torch_dtype=torch.bfloat16)
    if tied:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    prep = copy.copy(args)
    prep.w_bits, prep.load_qmodel_path = 16, None
    model = ptq_model(prep, model.cuda(), ma).cpu()
    artifact = torch.load(args.load_qmodel_path, map_location="cpu", weights_only=False)
    model.load_state_dict(artifact["model"], strict=True)
    del artifact
    for module in model.modules():
        if isinstance(module, quant_utils.ActQuantWrapper):
            module.quantizer.bits = module.out_quantizer.bits = 16
    model.eval().cuda()
    target = model.model.layers[1].mlp.down_proj
    assert args.rotation_components == "r1_r2" and not target.online_full_had and not target.online_partial_had
    assert args.k_bits == args.v_bits == 16
    tokenizer = LlamaTokenizerFast.from_pretrained(ma.input_model, add_eos_token=False, add_bos_token=False)
    dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    thresholds = {}
    pooled = torch.zeros(32768, dtype=torch.int64)
    rows, details = [], []
    current = {}
    levels = [0.99, 0.999, 0.9999, 0.99999]
    qnames = ["p99", "p99_9", "p99_99", "p99_999"]

    class Captured(Exception):
        pass

    def capture(module, inputs):
        nonlocal pooled
        raw = inputs[0].detach().reshape(2048, 8192)
        assert raw.dtype == torch.bfloat16
        x, a = raw.float(), raw.abs()
        assert torch.isfinite(x).all()
        hist = torch.bincount(a.view(torch.int16).long().flatten(), minlength=32768).cpu()
        if current["split"] == "train":
            pooled += hist
        top_channels = a.float().amax(0).topk(10)
        token_max, token_channel = a.float().max(1)
        top_tokens = token_max.topk(20).indices
        flat_top = a.float().flatten().topk(20).indices
        row = {**current, "min": float(x.min()), "max": float(x.max()), "absmax": float(a.max()),
               "abs_quantiles": dict(zip(qnames, quantiles(hist, levels))),
               "top10_channels": top_channels.indices.tolist(),
               "top10_channel_absmax": top_channels.values.tolist(),
               "top20_tokens": [{"position": int(t), "token_id": int(current_tokens[0, t]),
                                  "text": tokenizer.decode([int(current_tokens[0, t])]),
                                  "channel": int(token_channel[t]), "absmax": float(token_max[t])}
                                 for t in top_tokens],
               "top20_elements": [{"token_position": int(i)//8192, "channel": int(i)%8192,
                                    "value": float(x.flatten()[i])} for i in flat_top]}
        detail = {"split": current["split"], "sample_id": current["sample_id"],
                  "token_absmax": token_max.cpu(), "token_argmax_channel": token_channel.cpu(),
                  "channel_absmax": a.float().amax(0).cpu()}
        if current["split"] == "test":
            row["fixed_range_results"] = {}
            weight = target.module.weight.float()
            y = x @ weight.T
            xenergy, yenergy = x.double().square().sum(), y.double().square().sum()
            for name, bound in thresholds.items():
                scale = bound / 127
                outside = x.abs() > bound
                q = (x / scale).round().clamp(-128, 127)
                # Match wrapper dtype restoration before GEMM.
                xq = (q * scale).to(torch.bfloat16).float()
                diff = xq - x
                dy = xq @ weight.T - y
                row["fixed_range_results"][name] = {
                    "bound": bound, "scale": scale,
                    "outside_abs_bound_count": int(outside.sum()),
                    "outside_abs_bound_fraction": float(outside.float().mean()),
                    "outside_representable_count": int(((x < -128*scale) | (x > 127*scale)).sum()),
                    "tokens_outside": int(outside.any(1).sum()),
                    "quantized_zero_fraction": float((q == 0).float().mean()),
                    "nonzero_rounded_to_zero_fraction": float(((x != 0) & (q == 0)).float().mean()),
                    "input_sse": float(diff.double().square().sum()), "input_energy": float(xenergy),
                    "output_sse": float(dy.double().square().sum()), "output_energy": float(yenergy),
                }
                detail[name + "_channel_outside_counts"] = outside.sum(0).cpu()
                detail[name + "_token_outside_counts"] = outside.sum(1).cpu()
            # A fixed diagnostic large-value threshold, fitted only on calibration data.
            large = x.abs() > large_threshold
            detail["large_channel_counts"] = large.sum(0).cpu()
            detail["large_token_counts"] = large.sum(1).cpu()
            row["large_value_count"] = int(large.sum())
            row["large_value_tokens"] = int(large.any(1).sum())
        rows.append(row)
        details.append(detail)
        with (out / "per_text.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(current["split"], current["sample_id"], "absmax", row["absmax"], flush=True)
        raise Captured()

    # Abort after observing this actual Linear input; later layers cannot affect it.
    handle = target.module.register_forward_pre_hook(capture)
    try:
        for split, count in (("train", 128), ("test", 32)):
            tokens = tokenizer("\n\n".join(dataset[split]["text"]), return_tensors="pt").input_ids
            windows = sorted(random.Random(42).sample(range(tokens.numel()//2048), count))
            for i, window in enumerate(windows):
                current = {"split": split, "sample_id": i, "window_id": window, "token_start": window*2048}
                current_tokens = tokens[:, window*2048:(window+1)*2048]
                try:
                    model.model(current_tokens.cuda(), use_cache=False, quant_phase=QuantPhase.PREFILL)
                except Captured:
                    pass
            if split == "train":
                thresholds = {"calibration_absmax": max(r["absmax"] for r in rows),
                              "calibration_p99_99": quantiles(pooled, [0.9999])[0]}
                large_threshold = quantiles(pooled, [0.999])[0]
    finally:
        handle.remove()
    torch.save({"per_text": details, "calibration_abs_histogram": pooled}, out / "position_stats.pt")
    summary = {"target": "model.layers.1.mlp.down_proj.module input",
               "checkpoint": args.load_qmodel_path, "rotation": args.optimized_rotation_path,
               "upstream": "W4/A16/KV16", "online_full_had": target.online_full_had,
               "online_partial_had": target.online_partial_had, "sequence_length": 2048,
               "calibration_count": 128, "heldout_count": 32, "seed": 42,
               "calibration_sampling": "nonoverlapping train windows, newly fitted diagnostic ranges, not original GPTQ samples",
               "quantization": "one static scalar per candidate shared by all tokens/channels/texts; round clamp[-128,127]; BF16 dequant",
               "local_output_metric": "FP32 GEMM with same W4 weight; not model PPL or downstream A8 propagation",
               "thresholds": thresholds, "diagnostic_large_abs_threshold": large_threshold,
               "calibration_abs_quantiles": dict(zip(qnames, quantiles(pooled, levels))), "splits": {}}
    for split in ("train", "test"):
        subset = [r for r in rows if r["split"] == split]
        sets = [set(r["top10_channels"]) for r in subset]
        shared = [len(a & b) for i, a in enumerate(sets) for b in sets[i+1:]]
        counts = {c: sum(c in s for s in sets) for c in set.union(*sets)}
        summary["splits"][split] = {
            "min_range": [min(r["min"] for r in subset), max(r["min"] for r in subset)],
            "max_range": [min(r["max"] for r in subset), max(r["max"] for r in subset)],
            "absmax_range": [min(r["absmax"] for r in subset), max(r["absmax"] for r in subset)],
            "mean_shared_top10": float(np.mean(shared)),
            "top_channel_recurrence": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
            "peak_token_positions": [r["top20_tokens"][0]["position"] for r in subset],
            "abs_quantile_ranges": {q: [min(r["abs_quantiles"][q] for r in subset), max(r["abs_quantiles"][q] for r in subset)] for q in qnames}}
    held = [r for r in rows if r["split"] == "test"]
    summary["heldout_fixed_range"] = {}
    for name in thresholds:
        stats = [r["fixed_range_results"][name] for r in held]
        summary["heldout_fixed_range"][name] = {
            "bound": thresholds[name], "scale": thresholds[name]/127,
            "values": 32*2048*8192, "outside_count": sum(s["outside_abs_bound_count"] for s in stats),
            "outside_representable_count": sum(s["outside_representable_count"] for s in stats),
            "texts_outside": sum(s["outside_abs_bound_count"] > 0 for s in stats),
            "tokens_outside": sum(s["tokens_outside"] for s in stats),
            "mean_zero_fraction": float(np.mean([s["quantized_zero_fraction"] for s in stats])),
            "mean_nonzero_rounded_to_zero_fraction": float(np.mean([s["nonzero_rounded_to_zero_fraction"] for s in stats])),
            "input_relative_l2": (sum(s["input_sse"] for s in stats)/sum(s["input_energy"] for s in stats))**0.5,
            "output_relative_l2": (sum(s["output_sse"] for s in stats)/sum(s["output_energy"] for s in stats))**0.5}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["heldout_fixed_range"], indent=2), flush=True)


if __name__ == "__main__":
    main()
