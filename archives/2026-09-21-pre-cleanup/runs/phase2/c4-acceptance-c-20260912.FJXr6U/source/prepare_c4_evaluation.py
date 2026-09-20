"""Prepare one shared C4-English validation token stream for fixed-model evaluation."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from datasets import load_dataset
from transformers import LlamaTokenizerFast


def prepare(output_dir, model_path, cache_dir, windows=1024, seqlen=2048, seed=42):
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    files = [f"https://huggingface.co/datasets/allenai/c4/resolve/main/"
             f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]
    # Explicit files prevent downloading or preparing C4's training split.
    data = load_dataset("json", data_files={"validation": files}, split="validation",
                        cache_dir=str(cache_dir))
    print(f"Loaded {len(data)} C4 English validation documents from 8 shards", flush=True)
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    tokenizer = LlamaTokenizerFast.from_pretrained(
        str(model_path), model_max_length=seqlen, padding_side="right", use_fast=True,
        add_eos_token=False, add_bos_token=False)
    target = windows * seqlen
    documents = []
    estimated_tokens = 0
    input_ids = None
    for start in range(0, len(order), 128):
        indices = order[start:start + 128]
        batch = data[indices]
        encoded = tokenizer(batch["text"], add_special_tokens=False, truncation=False)
        for index, text, url, tokens in zip(indices, batch["text"], batch["url"], encoded["input_ids"]):
            documents.append(dict(row_index=index, url=url, text=text))
            estimated_tokens += len(tokens)
        if estimated_tokens >= target + seqlen:
            # Tokenize the actual joined text once; boundaries can change BPE tokens.
            input_ids = tokenizer("\n\n".join(row["text"] for row in documents),
                                  add_special_tokens=False, truncation=False,
                                  return_tensors="pt").input_ids
            if input_ids.numel() >= target:
                break
    if input_ids is None or input_ids.numel() < target:
        raise ValueError("Not enough validation tokens for the requested evaluation")
    metadata = dict(
        dataset="allenai/c4", subset="en", split="validation", source_files=files,
        source_document_count=len(data), seed=seed,
        sampling="Python Random(seed) permutation of all validation document rows; "
                 "take document batches in that order, join with two newlines, retain token prefix",
        selected_document_count=len(documents), concatenated_token_count=input_ids.numel(),
        discarded_suffix_tokens=input_ids.numel() - target,
        token_count=target, window_length=seqlen, windows=windows,
        predicted_tokens=windows * (seqlen - 1), tokenizer_path=str(model_path.resolve()),
        tokenizer_class=type(tokenizer).__name__, add_bos_token=False, add_eos_token=False,
        text_separator="\n\n", purpose="external evaluation only; no calibration or selection",
    )
    with (output_dir / "documents.jsonl").open("x") as output:
        for row in documents:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "metadata.json").open("x") as output:
        json.dump(metadata, output, indent=2)
    with (output_dir / "input_tokens.pt").open("xb") as output:
        torch.save(dict(input_ids=input_ids[:, :target].clone(), metadata=metadata), output)
    print(f"Prepared {target} input tokens, {metadata['predicted_tokens']} targets "
          f"in {time.monotonic() - started:.1f}s: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=1024)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.output_dir, args.model_path, args.cache_dir, args.windows, args.seqlen, args.seed)
