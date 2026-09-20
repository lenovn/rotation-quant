"""Native HF baseline audit; deliberately imports no rotation-quant helpers."""
import json
import math
import os
from pathlib import Path
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = ROOT / "runs/phase5/qwen3-1p7b"
MODEL = ROOT / "cache/models/qwen3-1.7b"


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


@torch.inference_mode()
def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    old = json.loads((BASE / "bf16-test-s42/result.json").read_text())
    arrow = Path(old["input_metadata"]["dataset_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True,
        model_max_length=2048, padding_side="right", use_fast=True,
        add_eos_token=False, add_bos_token=False)
    dataset = Dataset.from_file(str(arrow))
    ids = tokenizer("\n\n".join(dataset["text"]), add_special_tokens=False,
                    return_tensors="pt").input_ids
    comparisons = {}
    for name in ["bf16-test-s42", "sp2-qat400-test-s42", "sp2-qat400-test-s43", "sp2-qat400-test-s44"]:
        saved = torch.load(BASE / name / "input_tokens.pt", map_location="cpu", weights_only=True)
        comparisons[name] = torch.equal(ids, saved["input_ids"])
    assert all(comparisons.values()), comparisons
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).requires_grad_(False).eval().cuda()
    assert type(model).__module__.startswith("transformers.models.qwen3.")
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert all(type(module) is torch.nn.Linear for module in model.modules()
               if isinstance(module, torch.nn.Linear))
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    assert not any(name.startswith("experiments.phase3") for name in sys.modules)
    model.config.use_cache = False
    save("model.json", dict(model_class=type(model).__module__ + "." + type(model).__name__,
        model_path=str(MODEL), cached_revision=json.loads((BASE / "bf16-test-s42/settings.json").read_text())["source"]["model_cache_revision"],
        tokenizer_class=type(tokenizer).__name__, transformers=transformers.__version__, torch=torch.__version__,
        parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),
        tied_embeddings=True, linear_count=sum(isinstance(m, torch.nn.Linear) for m in model.modules()),
        forward_hooks=0, rotation_quant_helpers_imported=False, quantization=False,
        calibration=False, training=False, use_cache=False,
        rotary_buffers={n:str(v.dtype) for n,v in model.named_buffers() if n.endswith("inv_freq")},
        dataset=str(arrow), regenerated_test_ids_match=comparisons,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    torch.cuda.reset_peak_memory_stats()
    bf16_means, native_sums, counts = [], [], []
    for index, start in enumerate(range(0, ids.numel()-1, 2048)):
        batch = ids[:, start:start+2048].cuda()
        # HF performs its own causal shift and FP32 cross entropy for labels.
        outputs = model(input_ids=batch, labels=batch, use_cache=False, return_dict=True)
        targets = batch.numel()-1
        assert outputs.logits.dtype == torch.bfloat16
        bf16_loss = F.cross_entropy(outputs.logits[:, :-1].permute(0,2,1), batch[:,1:], reduction="none")
        bf16_means.append(bf16_loss.float().mean())
        native_sums.append(float(outputs.loss.item()) * targets)
        counts.append(targets)
        del outputs, bf16_loss, batch
        if (index+1) % 32 == 0:
            save("progress.json", dict(stage="native-full-test", windows=index+1, pid=os.getpid()))
            print(f"native test windows {index+1}", flush=True)
    means = torch.stack(bf16_means)
    full, tail = divmod(ids.numel(),2048)
    segments = []
    for first in range(0,full,128):
        last = min(first+128,full)
        ppl = means[first:last].mean().exp().item()
        segments.append(dict(start_token=first*2048,seqlen=2048,windows=last-first,
            predicted_tokens=(last-first)*2047,ppl=ppl,nll=math.log(ppl)))
    if tail>=2:
        ppl = means[-1].exp().item()
        segments.append(dict(start_token=full*2048,seqlen=tail,windows=1,
            predicted_tokens=tail-1,ppl=ppl,nll=math.log(ppl)))
    count = sum(counts)
    assert count == old["predicted_tokens"]
    nll = sum(x["nll"]*x["predicted_tokens"] for x in segments)/count
    native_nll = sum(native_sums)/count
    result = dict(token_count=ids.numel(),predicted_tokens=count,windows=len(counts),segments=segments,
        historical_bf16_ce_protocol=dict(ppl=math.exp(nll),nll=nll,
            reference_ppl=old["ppl"],difference_from_reference=math.exp(nll)-old["ppl"]),
        native_hf_fp32_loss_protocol=dict(ppl=math.exp(native_nll),nll=native_nll,
            note="Separate diagnostic protocol; does not replace historical results."),
        regenerated_test_ids_match=comparisons,
        elapsed_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        scope="Native unquantized HF checkpoint; one full test pass; no project model/evaluator helpers.")
    save("result.json", result)
    save("progress.json", dict(stage="completed",pid=os.getpid()))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save("failure.json",dict(type=type(error).__name__,message=str(error)))
        raise
