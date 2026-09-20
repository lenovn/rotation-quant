"""All down_proj GEMM inputs: per-text ranges, positions, recurrence and held-out coverage."""
import copy
import json
import random
from pathlib import Path

import datasets
import torch
import transformers
from transformers import LlamaTokenizerFast

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from inspect_down_activation_ranges import quantiles
from utils import quant_utils
from utils.process_args import process_args_ptq
from utils.quant_phase import QuantPhase


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
    weight_bits = sorted({int(q.bits) for q in artifact["w_quantizers"].values()})
    del artifact
    for module in model.modules():
        if isinstance(module, quant_utils.ActQuantWrapper):
            module.quantizer.bits = module.out_quantizer.bits = 16
    model.eval().cuda()
    targets = [layer.mlp.down_proj for layer in model.model.layers]
    assert len(targets) == 16 and weight_bits == [4]
    assert args.rotate and args.rotation_components == "r1_r2" and args.k_bits == args.v_bits == 16
    assert all(not t.online_full_had and not t.online_partial_had for t in targets)
    tokenizer = LlamaTokenizerFast.from_pretrained(ma.input_model, add_eos_token=False, add_bos_token=False)
    dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    pooled = [torch.zeros(32768, dtype=torch.int64) for _ in targets]
    thresholds, rows, details = {}, [[] for _ in targets], [[] for _ in targets]
    current, current_tokens = {}, None
    levels, qnames = [0.99, 0.999, 0.9999, 0.99999], ["p99", "p99_9", "p99_99", "p99_999"]
    streams = []
    for layer in range(16):
        directory = out / f"layer_{layer:02d}"
        directory.mkdir(exist_ok=True)
        streams.append((directory / "per_text.jsonl").open("w"))

    def hook_for(layer):
        def capture(module, inputs):
            raw = inputs[0].detach().reshape(2048, 8192)
            assert raw.dtype == torch.bfloat16 and torch.isfinite(raw).all()
            a = raw.abs()
            x, absolute = raw.float(), a.float()
            hist = torch.bincount(a.view(torch.int16).long().flatten(), minlength=32768).cpu()
            if current["split"] == "train":
                pooled[layer] += hist
            cm = absolute.amax(0)
            tm, tc = absolute.max(1)
            topc, topt = cm.topk(10).indices, tm.topk(10).indices
            peak = int(absolute.flatten().argmax())
            token_top_channels = absolute.topk(10, dim=1).indices
            token_top_counts = torch.bincount(token_top_channels.flatten(), minlength=8192)
            record = {**current, "layer": layer, "min": float(x.min()), "max": float(x.max()),
                      "absmax": float(absolute.max()),
                      "abs_quantiles": dict(zip(qnames, quantiles(hist, levels))),
                      "peak_token": peak//8192, "peak_channel": peak%8192,
                      "peak_value": float(x.flatten()[peak]),
                      "absmax_excluding_token0": float(tm[1:].max()),
                      "top10_channels": topc.tolist(), "top10_channel_absmax": cm[topc].tolist(),
                      "top10_tokens": [{"position": int(t), "id": int(current_tokens[0, t]),
                                        "text": tokenizer.decode([int(current_tokens[0, t])]),
                                        "channel": int(tc[t]), "absmax": float(tm[t])} for t in topt]}
            detail = {"split": current["split"], "sample_id": current["sample_id"],
                      "channel_absmax": cm.cpu(), "token_absmax": tm.cpu(),
                      "token_argmax_channel": tc.cpu(), "token_top10_channel_counts": token_top_counts.cpu()}
            if current["split"] == "test":
                record["fixed_ranges"] = {}
                for name in ("absmax", "p99_99"):
                    bound = thresholds[layer][name]
                    scale = bound/127
                    mask = absolute > bound
                    representable = (x < -128*scale) | (x > 127*scale)
                    # Every element at this location shares the SAME static scalar.
                    q = (x / scale).round().clamp(-128, 127)
                    record["fixed_ranges"][name] = {
                        "bound": bound, "scale": scale, "outside_count": int(mask.sum()),
                        "outside_representable_count": int(representable.sum()),
                        "tokens_outside": int(mask.any(1).sum()),
                        "zero_count": int((q == 0).sum()),
                        "nonzero_to_zero_count": int(((q == 0) & (x != 0)).sum())}
                    detail[name + "_channel_outside_counts"] = mask.sum(0).cpu()
                    detail[name + "_token_outside_counts"] = mask.sum(1).cpu()
                large = absolute > thresholds[layer]["p99_9"]
                detail["large_channel_counts"] = large.sum(0).cpu()
                detail["large_token_counts"] = large.sum(1).cpu()
                record["large_count"] = int(large.sum())
            rows[layer].append(record)
            details[layer].append(detail)
            streams[layer].write(json.dumps(record, ensure_ascii=False) + "\n")
            streams[layer].flush()
        return capture

    handles = [t.module.register_forward_pre_hook(hook_for(i)) for i, t in enumerate(targets)]
    sample_windows = {}
    try:
        for split, count in (("train", 128), ("test", 32)):
            tokens = tokenizer("\n\n".join(dataset[split]["text"]), return_tensors="pt").input_ids
            windows = sorted(random.Random(42).sample(range(tokens.numel()//2048), count))
            sample_windows[split] = windows
            for i, window in enumerate(windows):
                current = {"split": split, "sample_id": i, "window_id": window, "token_start": window*2048}
                current_tokens = tokens[:, window*2048:(window+1)*2048]
                model.model(current_tokens.cuda(), use_cache=False, quant_phase=QuantPhase.PREFILL)
                print(split, i+1, "/", count, "all 16 layers captured", flush=True)
            if split == "train":
                for layer in range(16):
                    qs = quantiles(pooled[layer], [0.999, 0.9999])
                    thresholds[layer] = {"absmax": max(r["absmax"] for r in rows[layer]),
                                         "p99_9": qs[0], "p99_99": qs[1]}
                (out / "calibration_ranges.json").write_text(json.dumps(thresholds, indent=2))
    finally:
        for h in handles:
            h.remove()
        for stream in streams:
            stream.close()
    for layer in range(16):
        torch.save({"per_text": details[layer], "calibration_abs_histogram": pooled[layer]},
                   out / f"layer_{layer:02d}/position_stats.pt")
    protocol = {"checkpoint": args.load_qmodel_path, "rotation": args.optimized_rotation_path,
                "layers": 16, "weights": "W4", "upstream_activation": "A16", "kv": "KV16",
                "capture": "actual down_proj.module Linear GEMM input", "online_hadamard": False,
                "sequence_length": 2048, "channels": 8192, "calibration_count": 128,
                "heldout_count": 32, "window_ids": sample_windows,
                "dataset": "WikiText-2 raw train/test", "seed": 42,
                "context": "each window starts with empty context; no added BOS/EOS; prefill only",
                "range_policy": "each layer has its own scalar, fixed across tokens/channels/texts; absmax and pooled P99.99 candidates",
                "integer_range": [-128, 127], "large_value_threshold": "per-layer calibration pooled P99.9",
                "scope": "per-location diagnostics, not simultaneous A8 forward or PPL"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print("COMPLETE", str(out), flush=True)


if __name__ == "__main__":
    main()
