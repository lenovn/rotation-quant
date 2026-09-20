"""Inspect raw C4 offsets; zero-width mappings require separate decode fallback.

This raw diagnostic must not replace the final C4_TEXT_COVERAGE report/JSON.
"""
import bisect
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, LlamaTokenizerFast

torch.set_num_threads(2)
root = Path('/home/dongpeiyan/projects/rotation-quant')
family = sys.argv[1]
source = root/'runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/documents.jsonl'
docs = [json.loads(line) for line in source.read_text().splitlines()]
text = '\n\n'.join(d['text'] for d in docs)
starts, ends = [], []
cursor = 0
for d in docs:
    starts.append(cursor)
    cursor += len(d['text'])
    ends.append(cursor)
    cursor += 2
assert cursor - 2 == len(text)
if family == 'llama':
    model = root/'cache/models/llama-3.2-1b-instruct'
    cls = LlamaTokenizerFast
    cached = source.parent/'input_tokens.pt'
else:
    assert family == 'qwen'
    model = root/'cache/models/qwen3-1.7b'
    cls = AutoTokenizer
    cached = root/'runs/phase5/qwen3-1p7b/c4-data/input_tokens.pt'
tokenizer = cls.from_pretrained(str(model), local_files_only=True, model_max_length=2048,
    padding_side='right', use_fast=True, add_bos_token=False, add_eos_token=False)
encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                     return_attention_mask=False, truncation=False, padding=False, verbose=False)
ids, offsets = encoding['input_ids'], encoding['offset_mapping']
saved = torch.load(cached, map_location='cpu', weights_only=True)
retained = saved['input_ids'].numel()
assert retained == 2097152
assert torch.equal(torch.tensor(ids[:retained], dtype=saved['input_ids'].dtype), saved['input_ids'].reshape(-1))
expected_full = 2150947 if family == 'llama' else 2191766
assert len(ids) == len(offsets) == expected_full

# Merge character spans instead of counting overlapping byte-token offsets twice.
merged = []
overlapping_adjacent = identical_adjacent = zero_spans = 0
previous = None
for start, end in offsets[:retained]:
    assert 0 <= start <= end <= len(text)
    if previous is not None:
        overlapping_adjacent += start < previous[1]
        identical_adjacent += (start, end) == previous
    previous = (start, end)
    if start == end:
        zero_spans += 1
        continue
    if merged and start <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], end)
    else:
        merged.append([start, end])
endpoint = merged[-1][1]
next_start, next_end = offsets[retained]
boundary_overlap = max(0, min(endpoint, next_end) - next_start)
# If the next discarded token touches the final covered character span, offsets
# alone cannot certify that the retained byte tokens encode that whole character.
certain_endpoint = min(endpoint, next_start) if boundary_overlap else endpoint
last = bisect.bisect_left(starts, endpoint) - 1
if last < 0:
    last = 0
complete_docs = sum(end <= certain_endpoint for end in ends)
span_docs = sum(end <= endpoint for end in ends)
doc_covered = sum(max(0, min(end, endpoint) - start) for start, end in zip(starts, ends))
certain_doc_covered = sum(max(0, min(end, certain_endpoint) - start) for start, end in zip(starts, ends))
context = []
for i in range(retained - 4, retained + 4):
    start, end = offsets[i]
    context.append(dict(token_index_zero_based=i, retained=i < retained, token_id=ids[i],
        tokenizer_piece=tokenizer.convert_ids_to_tokens(ids[i]), offset=[start,end],
        source_span=text[start:end], source_codepoints=[f'U+{ord(c):04X}' for c in text[start:end]],
        source_utf8_hex=text[start:end].encode('utf-8').hex()))
out = dict(family=family, python=sys.executable, transformers=transformers.__version__,
    tokenizers=__import__('tokenizers').__version__, tokenizer_class=type(tokenizer).__name__,
    tokenizer_path=str(model), source_documents=str(source), source_document_count=len(docs),
    cached_input_tokens=str(cached), cached_prefix_ids_exactly_equal=True,
    tokenization=dict(separator='\n\n', add_special_tokens=False, add_bos_token=False,
        add_eos_token=False, retained_tokens=retained, full_tokens=len(ids), discarded_tokens=len(ids)-retained),
    character_unit='Python Unicode code points, not UTF-8 bytes or grapheme clusters; zero-based half-open spans',
    total_source_document_characters=sum(len(d['text']) for d in docs),
    total_separator_characters=2*(len(docs)-1), total_joined_characters=len(text),
    retained_offset_union_character_count=sum(b-a for a,b in merged),
    retained_offset_union_intervals=merged, retained_offset_span_endpoint=endpoint,
    conservative_complete_character_prefix_endpoint=certain_endpoint,
    document_characters_in_offset_span=doc_covered,
    conservative_complete_document_characters=certain_doc_covered,
    complete_documents_conservative=complete_docs, complete_documents_by_offset_span=span_docs,
    last_touched_document=dict(ordinal_one_based=last+1, ordinal_zero_based=last,
        row_index=docs[last]['row_index'], global_character_span=[starts[last],ends[last]],
        total_characters=len(docs[last]['text']),
        covered_characters_by_offset_span=max(0,min(endpoint,ends[last])-starts[last]),
        conservative_complete_characters=max(0,min(certain_endpoint,ends[last])-starts[last])),
    boundary=dict(last_retained_offset=list(offsets[retained-1]), first_discarded_offset=list(offsets[retained]),
        overlap_character_count=boundary_overlap,
        endpoint_inside_separator=any(end < endpoint < start for end,start in zip(ends[:-1],starts[1:])),
        endpoint_at_document_end=endpoint in ends, endpoint_at_document_start=endpoint in starts,
        unicode_partial_character_possible=bool(boundary_overlap), token_neighborhood=context),
    retained_offsets=dict(adjacent_overlap_pairs=overlapping_adjacent, identical_adjacent_pairs=identical_adjacent,
        zero_length_spans=zero_spans),
    interpretation='Offset-span coverage is not proof of complete Unicode decoding if retained/discarded offsets overlap. These are input-prefix spans, not the subset of characters assigned a prediction loss.')
out['offset_mapping_reliable_for_coverage'] = zero_spans == 0
out['coverage_status'] = ('raw offsets usable' if zero_spans == 0 else
    'UNRESOLVED: zero-width offsets; derived raw span coverage is NOT accepted text coverage')
destination = root/f'runs/phase5/auditor/c4_text_coverage_{family}_raw_offsets_20260918.json'
destination.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(family=family,coverage_status=out['coverage_status'],
    boundary_overlap=boundary_overlap,coverage_intervals=len(merged)),ensure_ascii=False))
