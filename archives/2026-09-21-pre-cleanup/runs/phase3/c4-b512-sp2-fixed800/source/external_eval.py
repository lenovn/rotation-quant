import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import torch
from transformers import AutoConfig, LlamaTokenizerFast, set_seed

from eval_utils.modeling_llama import LlamaForCausalLM
from experiments.phase3.common import DATA_PATH, MODEL_PATH, PROJECT_ROOT, wrappers, write_json
from experiments.phase3.postprocess import load_static
from experiments.phase3.quantization import SP2Quantizer
from experiments.phase3.run import progress, source_record
from utils.eval_utils import evaluator


def load_c4_tokens(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    ids, metadata = saved["input_ids"], saved["metadata"]
    if (metadata["dataset"], metadata["subset"], metadata["split"]) != ("allenai/c4", "en", "validation"):
        raise ValueError("Expected fixed C4 English validation tokens")
    if Path(metadata["tokenizer_path"]).resolve() != MODEL_PATH.resolve():
        raise ValueError("C4 tokens use a different tokenizer")
    if metadata["add_bos_token"] or metadata["add_eos_token"]:
        raise ValueError("C4 token protocol adds BOS/EOS")
    if ids.dtype != torch.long or tuple(ids.shape) != (1, metadata["token_count"]):
        raise ValueError("C4 token tensor differs from its metadata")
    if metadata["window_length"] != 2048 or ids.numel() != metadata["windows"] * 2048:
        raise ValueError("Expected complete independent 2048-token C4 windows")
    if ids.numel() < 2048 or metadata["predicted_tokens"] != metadata["windows"] * 2047:
        raise ValueError("C4 target accounting differs from independent windows")
    return ids, metadata


def load_wikitext2_test_tokens():
    from datasets import Dataset

    data_path = DATA_PATH / "wikitext-test.arrow"
    dataset = Dataset.from_file(str(data_path))
    tokenizer = LlamaTokenizerFast.from_pretrained(str(MODEL_PATH), local_files_only=True,
        model_max_length=2048, padding_side="right", use_fast=True,
        add_eos_token=False, add_bos_token=False)
    ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    if ids.numel() < 2:
        raise ValueError("WikiText-2 test requires at least two tokens")
    full_windows, tail = divmod(ids.numel(), 2048)
    metadata = dict(dataset="Salesforce/wikitext", subset="wikitext-2-raw-v1", split="test",
        dataset_path=str(data_path), rows=len(dataset), tokenizer_path=str(MODEL_PATH),
        add_bos_token=False, add_eos_token=False, text_join="double-newline",
        token_count=ids.numel(), window_length=2048, full_windows=full_windows,
        tail_tokens=tail, windows=full_windows + int(tail >= 2),
        predicted_tokens=full_windows * 2047 + max(tail - 1, 0), unscored_tail_tokens=int(tail == 1),
        protocol="all test rows in original order; disjoint 2048-token windows including partial tail")
    return ids, metadata


def load_model(mode, package):
    if mode == "quantized":
        model, records = load_static(package)
        del records
    else:
        config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        original_tie = config.tie_word_embeddings
        config.tie_word_embeddings = False
        model = LlamaForCausalLM.from_pretrained(str(MODEL_PATH), config=config,
            torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
        if original_tie:
            model.lm_head.weight.data = model.model.embed_tokens.weight.detach().clone()
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    model.seqlen = 2048
    return model.cuda()


def quantizer_snapshot(model):
    return {name: {key: value.detach().cpu().clone() for key, value in wrapper.quantizer.state_dict().items()}
            for name, wrapper in wrappers(model).items()}


def evaluate_tokens(model, ids, chunk_windows):
    path = PROJECT_ROOT / "scripts/phase2/validation_acceptance.py"
    specification = importlib.util.spec_from_file_location("phase3_c4_acceptance", path)
    acceptance = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(acceptance)
    arguments = SimpleNamespace(eval_nsamples=None, bsz=1, capture_layer_io=False)
    return acceptance.evaluate_full_validation(model, SimpleNamespace(input_ids=ids), "cuda",
        arguments, evaluator, chunk_windows=chunk_windows)


@torch.no_grad()
def run(args):
    started = time.monotonic()
    reused_tokens = args.tokens is not None
    if reused_tokens:
        ids, metadata = load_c4_tokens(args.tokens)
        token_path = args.tokens
    else:
        ids, metadata = load_wikitext2_test_tokens()
        token_path = args.output / "input_tokens.pt"
        torch.save(dict(input_ids=ids, metadata=metadata), token_path)
    dataset_label = "c4" if reused_tokens else "wikitext2-test"
    write_json(args.output / "data.json", dict(input_token_path=str(token_path),
        input_metadata=metadata, reused_tokens=reused_tokens, calibration=False, training=False, selection=False))
    progress(args.output, "loading-fixed-model", mode=args.mode)
    model = load_model(args.mode, args.package)
    overlay_path = getattr(args, "down_int8_scales", None)
    overlay = None
    if overlay_path is not None:
        from experiments.phase3.uniform_baseline import apply_down_int8

        if args.mode != "quantized":
            raise ValueError("Only quantized mode accepts a down INT8 overlay")
        overlay = apply_down_int8(model, overlay_path, args.package)
    quantizers = wrappers(model)
    formats = Counter("sp2" if isinstance(wrapper.quantizer, SP2Quantizer) else "int8"
                      for wrapper in quantizers.values())
    expected_formats = {"int8": 112} if overlay is not None else {"int8": 96, "sp2": 16}
    if args.mode == "quantized" and (formats != expected_formats
            or any(wrapper.quantizer.bits != 8 or getattr(wrapper.quantizer, "observing", False)
                   or wrapper.online_full_had or wrapper.online_partial_had for wrapper in quantizers.values())):
        raise ValueError("Frozen candidate input quantization coverage differs")
    if args.mode == "bf16" and quantizers:
        raise ValueError("Original BF16 unexpectedly contains quantized linears")
    rotary_dtypes = {name: str(value.dtype) for name, value in model.named_buffers() if name.endswith("inv_freq")}
    if not rotary_dtypes or set(rotary_dtypes.values()) != {"torch.float32"}:
        raise ValueError("Original FP32 RoPE buffers were changed")
    before = quantizer_snapshot(model)
    torch.save(before, args.output / "quantizers_before.pt")
    write_json(args.output / "model.json", dict(mode=args.mode, package=str(args.package) if args.package else None,
        backbone_weight_bits=4 if quantizers else 16, activation_formats=dict(formats),
        rotary_buffer_dtypes=rotary_dtypes, kv_bits=16, use_cache=False, training=False, calibration=False,
        down_int8_overlay=overlay))
    torch.cuda.reset_peak_memory_stats()
    progress(args.output, dataset_label + "-fixed-evaluation", windows=metadata["windows"],
             chunk_windows=args.chunk_windows)
    result = evaluate_tokens(model, ids, args.chunk_windows)
    after = quantizer_snapshot(model)
    unchanged = before.keys() == after.keys() and all(before[name].keys() == after[name].keys()
        and all(torch.equal(value, after[name][key]) for key, value in record.items())
        for name, record in before.items())
    torch.save(after, args.output / "quantizers_after.pt")
    if (not unchanged or result["predicted_tokens"] != metadata["predicted_tokens"]
            or result["unscored_tail_tokens"] != metadata.get("unscored_tail_tokens", 0)):
        raise ValueError(f"Fixed-model state or {'C4' if reused_tokens else 'WikiText-2 test'} target accounting changed")
    result.update(dataset=metadata["dataset"], subset=metadata["subset"], split=metadata["split"], mode=args.mode,
        input_token_path=str(token_path), input_metadata=metadata, reused_tokens=reused_tokens,
        package=str(args.package) if args.package else None, chunk_windows=args.chunk_windows,
        down_int8_overlay=overlay, activation_formats=dict(formats),
        calibration=False, training=False, candidate_selection=False, activation_scales_unchanged=unchanged,
        elapsed_seconds=time.monotonic() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
        evaluation_precision="BF16 logits CE; per-token loss to FP32; log(float32 segment PPL), target-weighted NLL",
        acceptance_source=str(PROJECT_ROOT / "scripts/phase2/validation_acceptance.py"),
        evaluator_source=str(SOURCE_ROOT / "utils/eval_utils.py"),
        scope=("fixed 1024-window protocol when windows=1024; not all C4, native kernels or decode" if reused_tokens
               else "complete WikiText-2 test, including partial tail; frozen model; not native kernels or decode"))
    write_json(args.output / "result.json", result)
    print(("C4_RESULT " if reused_tokens else "WIKITEXT2_TEST_RESULT ") + json.dumps(result), flush=True)
    progress(args.output, "completed", mode=args.mode, ppl=result["ppl"], nll=result["nll"],
             predicted_tokens=result["predicted_tokens"], elapsed_seconds=result["elapsed_seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("bf16", "quantized"), required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--down-int8-scales", type=Path)
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--tokens", type=Path)
    data_source.add_argument("--wikitext2-test", action="store_true")
    parser.add_argument("--chunk-windows", type=int, default=128)
    args = parser.parse_args()
    if (args.mode == "quantized") != (args.package is not None) or args.chunk_windows < 1:
        parser.error("Only quantized mode requires a package; chunk-windows must be positive")
    if args.down_int8_scales is not None and args.mode != "quantized":
        parser.error("Only quantized mode accepts --down-int8-scales")
    args.tokens = args.tokens.resolve() if args.tokens else None
    args.package = args.package.resolve() if args.package else None
    args.down_int8_scales = args.down_int8_scales.resolve() if args.down_int8_scales else None
    args.output.mkdir(parents=True, exist_ok=False)
    source_directory = args.output / "source"
    source_directory.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source_directory / path.name)
    shutil.copy2(PROJECT_ROOT / "scripts/phase2/validation_acceptance.py", source_directory / "validation_acceptance.py")
    (source_directory / "tracked.diff").write_text(subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "diff", "HEAD"], text=True))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    try:
        write_json(args.output / "settings.json", dict(arguments={name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()}, source=source_record()))
        run(args)
    except BaseException as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    main()
