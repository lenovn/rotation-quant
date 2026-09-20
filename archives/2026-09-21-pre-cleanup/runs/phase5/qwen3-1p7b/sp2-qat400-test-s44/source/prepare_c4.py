"""Retokenize the existing C4 documents without any new document sampling."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from experiments.phase3.architecture import tokenizer
from experiments.phase3.common import MODEL_PATH, PROJECT_ROOT, write_json


def prepare(output):
    source = PROJECT_ROOT / "runs/phase2/c4-acceptance-c-20260912.FJXr6U/data"
    original = json.loads((source / "metadata.json").read_text())
    documents = [json.loads(line) for line in (source / "documents.jsonl").open()]
    text = "\n\n".join(record["text"] for record in documents)
    ids = tokenizer(MODEL_PATH)(text, add_special_tokens=False, return_tensors="pt").input_ids
    before = ids.numel()
    ids = ids[:, :original["token_count"]].contiguous()
    full, tail = divmod(ids.numel(), 2048)
    metadata = dict(dataset="allenai/c4", subset="en", split="validation",
        tokenizer_path=str(MODEL_PATH), source_documents=str(source / "documents.jsonl"),
        source_metadata=str(source / "metadata.json"), documents=len(documents),
        document_row_indices=[record["row_index"] for record in documents],
        add_bos_token=False, add_eos_token=False, text_join="double-newline",
        tokens_before_truncation=before, prefix_token_limit=original["token_count"],
        token_count=ids.numel(), window_length=2048, full_windows=full,
        windows=full + int(tail >= 2), tail_tokens=tail,
        predicted_tokens=full * 2047 + max(tail - 1, 0), unscored_tail_tokens=int(tail == 1),
        protocol="same ordered raw documents; model-specific tokenizer; historical prefix limit; include scoreable tail")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(input_ids=ids, metadata=metadata), output / "input_tokens.pt")
    write_json(output / "metadata.json", metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != "document_row_indices"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
