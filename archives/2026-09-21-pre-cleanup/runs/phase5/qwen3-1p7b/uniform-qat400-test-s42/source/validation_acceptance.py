"""Two-model acceptance on WikiText-2 or saved evaluation tokens, including tail."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch


def evaluate_full_validation(model, enc, device, args, evaluator, chunk_windows=None):
    ids = enc.input_ids
    count = ids.numel()
    if count < 2:
        raise ValueError("At least two tokens required")
    original_length = model.seqlen
    if original_length < 2:
        raise ValueError("Window length must be at least two")
    if chunk_windows is not None and chunk_windows < 1:
        raise ValueError("Chunk size must be positive")
    eval_args = copy.copy(args)
    eval_args.eval_nsamples = None
    eval_args.bsz = 1
    full, tail = divmod(count, original_length)
    segments = []
    chunk_size = chunk_windows or max(full, 1)
    jobs = [(first * original_length, original_length, min(chunk_size, full - first))
            for first in range(0, full, chunk_size)]
    if tail >= 2:
        jobs.append((full * original_length, tail, 1))
    try:
        for index, (start, length, windows) in enumerate(jobs):
            print(f"Evaluation segment {index + 1}/{len(jobs)}: "
                  f"start={start}, windows={windows}, seqlen={length}", flush=True)
            model.seqlen = length
            segment = SimpleNamespace(input_ids=ids[:, start:start + length * windows])
            ppl = evaluator(model, segment, device, eval_args)
            segments.append(dict(start_token=start, seqlen=length, windows=windows,
                                 predicted_tokens=(length - 1) * windows,
                                 ppl=ppl, nll=math.log(ppl)))
    finally:
        model.seqlen = original_length
    predicted = sum(row["predicted_tokens"] for row in segments)
    nll = sum(row["nll"] * row["predicted_tokens"] for row in segments) / predicted
    return dict(token_count=count, predicted_tokens=predicted, segments=segments,
                unscored_tail_tokens=int(tail == 1), nll=nll, ppl=math.exp(nll))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--accept-mode", choices=("w16a16", "w4a16", "w4a8_down_a16", "w4a8"), required=True)
    parser.add_argument("--accept-output", type=Path, required=True)
    parser.add_argument("--accept-down-scales", type=Path)
    parser.add_argument("--accept-token-file", type=Path)
    parser.add_argument("--accept-chunk-windows", type=int)
    options, remaining = parser.parse_known_args()
    saved_tokens = None
    if options.accept_token_file:
        saved_tokens = torch.load(options.accept_token_file, map_location="cpu", weights_only=True)
        ids = saved_tokens["input_ids"]
        if ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Saved evaluation tokens must be int64 with shape [1, tokens]")
    options.accept_output.mkdir(parents=True, exist_ok=False)
    sys.argv = [sys.argv[0], *remaining]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repos/SpinQuant"))
    import ptq
    from datasets import load_dataset
    from down_codebooks import FrozenCodebookQuantizer

    torch.backends.cuda.matmul.allow_tf32 = False
    original_prepare = ptq.ptq_model
    original_evaluator = ptq.eval_utils.evaluator
    original_data = ptq.data_utils.get_wikitext2
    source = {}
    snapshot = {}

    def prepare(args, model, model_args=None):
        source.update(input_model=model_args.input_model, mode=options.accept_mode,
                      dataset="Salesforce/wikitext", subset="wikitext-2-raw-v1",
                      split="validation", window_length=2048, seed=args.seed,
                      dtype="bfloat16", kv_bits=16, use_cache=False,
                      evaluation="prefill teacher-forced next-token; disjoint windows including partial tail",
                      nll_precision="log of original evaluator float32 PPL, token-weighted across segments")
        if saved_tokens is not None:
            metadata = saved_tokens["metadata"]
            if Path(metadata["tokenizer_path"]).resolve() != Path(model_args.input_model).resolve():
                raise ValueError("Saved tokens use a different model/tokenizer path")
            source.update(dataset=metadata["dataset"], subset=metadata["subset"],
                          split=metadata["split"], input_token_path=str(options.accept_token_file.resolve()),
                          input_metadata=metadata, chunk_windows=options.accept_chunk_windows)
        if options.accept_mode == "w16a16":
            if args.w_bits != 16 or args.a_bits != 16 or args.rotate:
                raise ValueError("Baseline must be original unrotated W16A16")
            source["quantized_linear_count"] = 0
        else:
            if not args.load_qmodel_path or args.save_qmodel_path or args.w_bits != 4:
                raise ValueError("Must load existing W4 without saving or requantizing")
            model = original_prepare(args, model, model_args)
            source.update(weight_path=args.load_qmodel_path, weight_method="RTN",
                          rotation_path=args.optimized_rotation_path,
                          non_down_scale_path=args.static_scale_path,
                          down_scale_path=str(options.accept_down_scales))
            scales = (json.loads(options.accept_down_scales.read_text())["sp2"]
                      if options.accept_mode == "w4a8" else None)
            exported = torch.load(args.static_scale_path, map_location="cpu", weights_only=True)["activation"]
            wrappers = {name: module for name, module in model.named_modules()
                        if name.startswith("model.layers.") and hasattr(module, "module")
                        and hasattr(module, "quantizer")}
            down_count = 0
            for name, wrapper in wrappers.items():
                if name.endswith("mlp.down_proj"):
                    if wrapper.online_full_had or wrapper.online_partial_had:
                        raise ValueError("Unexpected down Hadamard")
                    if scales is not None:
                        wrapper.quantizer = FrozenCodebookQuantizer("sp2", scales[name]).to(wrapper.module.weight.device)
                    else:
                        wrapper.quantizer.bits = 16
                    down_count += 1
                else:
                    q = wrapper.quantizer
                    if (q.bits != 8 or not q.rotation_static_enabled or q.scale.numel() != 1
                            or q.observing or not q.has_scale
                            or not torch.equal(q.scale.detach().cpu().reshape(-1), exported[name + ".quantizer"].float().reshape(-1))):
                        raise ValueError(f"Non-down fixed per-tensor scale mismatch: {name}")
                if options.accept_mode == "w4a16":
                    wrapper.quantizer.bits = 16
                if wrapper.out_quantizer.bits != 16:
                    raise ValueError(f"Unexpected output quantization: {name}")
                snapshot[name] = {k: v.detach().cpu().clone() for k, v in wrapper.quantizer.named_buffers()}
            if len(wrappers) != 112 or down_count != 16:
                raise ValueError("Expected 112 A8 inputs, including 16 down")
            checkpoint = torch.load(args.load_qmodel_path, map_location="cpu", weights_only=False)
            quantizers = checkpoint["w_quantizers"]
            if len(quantizers) != 112:
                raise ValueError("Expected 112 W4 quantizers")
            for name, q in quantizers.items():
                actual = model.get_submodule(name).weight.detach().cpu()
                if (q.bits != 4 or not q.perchannel or not q.sym or q.weight_groupsize != -1
                        or tuple(q.scale.shape) != (actual.shape[0], 1)
                        or not torch.equal(actual, checkpoint["model"][name + ".weight"])):
                    raise ValueError(f"W4 per-channel checkpoint mismatch: {name}")
            del checkpoint
            source.update(quantized_linear_count=112,
                          non_down_int8_count=0 if options.accept_mode == "w4a16" else 96,
                          down_sp2_count=16 if scales is not None else 0,
                          activation_a16_count=112 if options.accept_mode == "w4a16" else (16 if scales is None else 0),
                          activation_input_bits={name: wrapper.quantizer.bits for name, wrapper in wrappers.items()},
                          weights_match_checkpoint=True,
                          weight_granularity="per output channel", activation_granularity="fixed per tensor",
                          high_precision="embedding, lm_head, norms; BF16 fake-quant GEMM")
        model.eval().requires_grad_(False)
        return model

    def validation_data(*unused, tokenizer=None, eval_mode=False, **kwargs):
        if not eval_mode:
            raise ValueError("Acceptance must not recalibrate or train")
        if saved_tokens is not None:
            if saved_tokens["metadata"]["window_length"] != tokenizer.model_max_length:
                raise ValueError("Saved evaluation window length differs from model_max_length")
            return SimpleNamespace(input_ids=saved_tokens["input_ids"])
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        return tokenizer("\n\n".join(data["text"]), return_tensors="pt")

    def evaluate(model, enc, device, args):
        start = time.monotonic()
        result = evaluate_full_validation(model, enc, device, args, original_evaluator,
                                          chunk_windows=options.accept_chunk_windows)
        for name, before in snapshot.items():
            after = dict(model.get_submodule(name).quantizer.named_buffers())
            if set(before) != set(after) or any(not torch.equal(v, after[k].detach().cpu()) for k, v in before.items()):
                raise RuntimeError(f"Activation quantizer changed: {name}")
        result.update(source, elapsed_seconds=time.monotonic() - start,
                      activation_scales_unchanged=True if snapshot else None)
        with (options.accept_output / "result.json").open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
        print("VALIDATION_ACCEPTANCE " + json.dumps(result), flush=True)
        return result["ppl"]

    ptq.ptq_model = prepare
    ptq.eval_utils.evaluator = evaluate
    ptq.data_utils.get_wikitext2 = validation_data
    try:
        ptq.train()
    finally:
        ptq.ptq_model = original_prepare
        ptq.eval_utils.evaluator = original_evaluator
        ptq.data_utils.get_wikitext2 = original_data


if __name__ == "__main__":
    main()
