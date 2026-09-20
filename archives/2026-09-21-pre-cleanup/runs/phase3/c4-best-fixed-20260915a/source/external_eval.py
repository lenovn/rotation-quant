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
from transformers import AutoConfig, set_seed

from eval_utils.modeling_llama import LlamaForCausalLM
from experiments.phase3.common import MODEL_PATH, PROJECT_ROOT, wrappers, write_json
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
    ids, metadata = load_c4_tokens(args.tokens)
    write_json(args.output / "data.json", dict(input_token_path=str(args.tokens),
        input_metadata=metadata, reused_tokens=True, calibration=False, training=False, selection=False))
    progress(args.output, "loading-fixed-model", mode=args.mode)
    model = load_model(args.mode, args.package)
    quantizers = wrappers(model)
    formats = Counter("sp2" if isinstance(wrapper.quantizer, SP2Quantizer) else "int8"
                      for wrapper in quantizers.values())
    if args.mode == "quantized" and (formats != {"int8": 96, "sp2": 16}
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
        rotary_buffer_dtypes=rotary_dtypes, kv_bits=16, use_cache=False, training=False, calibration=False))
    torch.cuda.reset_peak_memory_stats()
    progress(args.output, "c4-full-fixed-subset", windows=metadata["windows"], chunk_windows=args.chunk_windows)
    result = evaluate_tokens(model, ids, args.chunk_windows)
    after = quantizer_snapshot(model)
    unchanged = before.keys() == after.keys() and all(before[name].keys() == after[name].keys()
        and all(torch.equal(value, after[name][key]) for key, value in record.items())
        for name, record in before.items())
    torch.save(after, args.output / "quantizers_after.pt")
    if not unchanged or result["predicted_tokens"] != metadata["predicted_tokens"] or result["unscored_tail_tokens"]:
        raise ValueError("Fixed-model state or C4 target accounting changed")
    result.update(dataset="allenai/c4", subset="en", split="validation", mode=args.mode,
        input_token_path=str(args.tokens), input_metadata=metadata, reused_tokens=True,
        package=str(args.package) if args.package else None, chunk_windows=args.chunk_windows,
        calibration=False, training=False, candidate_selection=False, activation_scales_unchanged=unchanged,
        elapsed_seconds=time.monotonic() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
        evaluation_precision="BF16 logits CE; per-token loss to FP32; log(float32 segment PPL), target-weighted NLL",
        acceptance_source=str(PROJECT_ROOT / "scripts/phase2/validation_acceptance.py"),
        evaluator_source=str(SOURCE_ROOT / "utils/eval_utils.py"),
        scope="fixed 1024-window protocol when windows=1024; not all C4, native kernels or decode")
    write_json(args.output / "result.json", result)
    print("C4_RESULT " + json.dumps(result), flush=True)
    progress(args.output, "completed", mode=args.mode, ppl=result["ppl"], nll=result["nll"],
             predicted_tokens=result["predicted_tokens"], elapsed_seconds=result["elapsed_seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("bf16", "quantized"), required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--chunk-windows", type=int, default=128)
    args = parser.parse_args()
    if (args.mode == "quantized") != (args.package is not None) or args.chunk_windows < 1:
        parser.error("Only quantized mode requires a package; chunk-windows must be positive")
    args.tokens = args.tokens.resolve()
    args.package = args.package.resolve() if args.package else None
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
