"""Measure cross-window input-channel recurrence under fixed W4/A16/KV16."""
import copy
import csv
import json
import random
from pathlib import Path

import torch
import transformers
from transformers import LlamaTokenizerFast

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, quant_utils
from utils.process_args import process_args_ptq
from utils.quant_phase import QuantPhase


@torch.no_grad()
def main():
    model_args, training_args, args = process_args_ptq()
    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transformers.set_seed(args.seed)
    config = transformers.AutoConfig.from_pretrained(model_args.input_model)
    tied = config.tie_word_embeddings
    config.tie_word_embeddings = False
    model = LlamaForCausalLM.from_pretrained(
        model_args.input_model, config=config, torch_dtype=torch.bfloat16
    )
    if tied:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # Reuse the established restoration loader, without running GPTQ again.
    prep = copy.copy(args)
    prep.w_bits = 16
    prep.load_qmodel_path = None
    model = ptq_model(prep, model.cuda(), model_args).cpu()
    artifact = torch.load(args.load_qmodel_path, map_location="cpu", weights_only=False)
    model.load_state_dict(artifact["model"], strict=True)
    weight_bits = sorted({int(q.bits) for q in artifact["w_quantizers"].values()})
    del artifact
    for module in model.modules():
        if isinstance(module, quant_utils.ActQuantWrapper):
            module.quantizer.bits = 16
            module.out_quantizer.bits = 16
    model.eval().cuda()
    model.config.use_cache = False
    target = model.model.layers[1].mlp.down_proj
    assert args.rotation_components == "r1_r2" and args.rotate
    assert args.k_bits == args.v_bits == 16 and weight_bits == [4]
    assert not target.online_full_had and not target.online_partial_had

    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_args.input_model, add_eos_token=False, add_bos_token=False
    )
    encoded = data_utils.get_wikitext2(tokenizer=tokenizer, eval_mode=True)
    length, count, k = 2048, 32, 10
    ids = sorted(random.Random(args.seed).sample(
        range(encoded.input_ids.numel() // length), count
    ))
    stats, samples = [], []

    def capture(module, inputs):
        x = inputs[0].detach().reshape(-1, config.intermediate_size).float().abs()
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite target activation")
        token_top = x.topk(k, dim=1).indices
        stats.append({
            "max_abs": x.amax(dim=0).cpu(),
            "p99_abs": torch.quantile(x, 0.99, dim=0).cpu(),
            "token_top10_rate": (torch.bincount(
                token_top.flatten(), minlength=x.shape[1]
            ).float() / x.shape[0]).cpu(),
        })

    # Hook the actual Linear input after the wrapper; this is the GEMM input.
    handle = target.module.register_forward_pre_hook(capture)
    try:
        for sample_id, window in enumerate(ids):
            tokens = encoded.input_ids[:, window * length:(window + 1) * length]
            model.model(tokens.cuda(), use_cache=False, quant_phase=QuantPhase.PREFILL)
            assert len(stats) == sample_id + 1
            row = {"sample_id": sample_id, "window_id": window,
                   "token_start": window * length,
                   "text_preview": tokenizer.decode(tokens[0, :80])}
            for metric in ("max_abs", "p99_abs"):
                values, channels = stats[-1][metric].topk(k)
                row[metric + "_top10"] = [
                    {"channel": c, "value": v} for c, v in zip(channels.tolist(), values.tolist())
                ]
            samples.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        handle.remove()

    tensors = {key: torch.stack([s[key] for s in stats]) for key in stats[0]}
    torch.save({"window_ids": ids, **tensors}, output / "channel_stats.pt")
    with (output / "per_text.jsonl").open("w") as stream:
        for row in samples:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    rankings, overlaps = {}, {}
    frequencies = {}
    for metric in ("max_abs", "p99_abs"):
        top = tensors[metric].topk(k, dim=1).indices
        freq = torch.bincount(top.flatten(), minlength=config.intermediate_size)
        frequencies[metric] = freq
        rankings[metric] = [
            {"channel": c, "texts_in_top10": int(freq[c]),
             "mean_token_top10_rate": float(tensors["token_top10_rate"][:, c].mean()),
             "min_token_top10_rate": float(tensors["token_top10_rate"][:, c].min()),
             "max_abs_over_all_texts": float(tensors["max_abs"][:, c].max())}
            for c in sorted(range(len(freq)), key=lambda c: (-int(freq[c]), c))[:30]
        ]
        sets = [set(row.tolist()) for row in top]
        intersections = [len(a & b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        overlaps[metric] = {"mean_shared_channels": sum(intersections) / len(intersections),
                            "min_shared_channels": min(intersections),
                            "max_shared_channels": max(intersections)}
    with (output / "channels.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "texts_max_top10", "texts_p99_top10", "mean_token_top10_rate"])
        for c in range(config.intermediate_size):
            writer.writerow([c, int(frequencies["max_abs"][c]), int(frequencies["p99_abs"][c]),
                             float(tensors["token_top10_rate"][:, c].mean())])
    summary = {
        "target": "model.layers.1.mlp.down_proj", "channel_index_base": 0,
        "checkpoint": args.load_qmodel_path, "rotation": args.optimized_rotation_path,
        "weight_bits": weight_bits, "activation_bits": 16, "kv_bits": 16,
        "dtype": "bfloat16", "dataset": "WikiText-2 test", "seed": args.seed,
        "samples": count, "tokens_per_sample": length, "channels": config.intermediate_size,
        "sampling": "random nonoverlapping windows of concatenated test text; not independent articles",
        "top_k": k, "rankings": rankings, "pairwise_top10_overlap": overlaps,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Layer 1 down_proj input channel recurrence", "",
             "Fixed W4, A16/KV16; 32 nonoverlapping WikiText-2 test windows of 2048 tokens. Channels are zero-based.",
             "", "| Sample | Window | Top-10 channels by max absolute activation |",
             "|---|---|---|"]
    for row in samples:
        channels = ", ".join(str(item["channel"]) for item in row["max_abs_top10"])
        lines.append(f"| {row['sample_id']} | {row['window_id']} | {channels} |")
    lines += ["", "| Channel | Texts in max Top-10 | Mean token Top-10 rate |",
              "|---|---|---|"]
    for row in rankings["max_abs"][:15]:
        lines.append(f"| {row['channel']} | {row['texts_in_top10']}/32 | {row['mean_token_top10_rate']:.2%} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "overlap": overlaps}), flush=True)


if __name__ == "__main__":
    main()
