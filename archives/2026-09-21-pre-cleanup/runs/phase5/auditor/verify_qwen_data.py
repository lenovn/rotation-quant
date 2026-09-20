"""Read-only reconstruction of Qwen C4 tokens from the inherited ordered texts."""
import json
from pathlib import Path
import random
import torch
from transformers import AutoTokenizer
root=Path('/home/dongpeiyan/projects/rotation-quant')
output=root/'runs/phase5/qwen3-1p7b/c4-data'
metadata=json.loads((output/'metadata.json').read_text())
saved=torch.load(output/'input_tokens.pt',map_location='cpu',weights_only=True)
assert saved['metadata']==metadata
source=Path(metadata['source_documents'])
documents=[json.loads(line) for line in source.open()]
indices=[row['row_index'] for row in documents]
assert len(documents)==metadata['documents']==4480
assert indices==metadata['document_row_indices']
original=json.loads(Path(metadata['source_metadata']).read_text())
order=list(range(original['source_document_count'])); random.Random(original['seed']).shuffle(order)
assert indices==order[:4480]
model=root/'cache/models/qwen3-1.7b'
assert Path(metadata['tokenizer_path']).resolve()==model.resolve()
revision=json.loads((model/'phase5_revision.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,use_fast=True,add_bos_token=False,add_eos_token=False)
assert type(tokenizer).__name__=='Qwen2TokenizerFast'
text='\n\n'.join(row['text'] for row in documents)
reconstructed=tokenizer(text,add_special_tokens=False,return_tensors='pt',truncation=False).input_ids
assert reconstructed.numel()==metadata['tokens_before_truncation']==2191766
limit=original['token_count']
assert limit==metadata['prefix_token_limit']==2097152
assert torch.equal(saved['input_ids'],reconstructed[:,:limit])
ids=saved['input_ids']; full,tail=divmod(ids.numel(),2048)
assert ids.dtype==torch.long and tuple(ids.shape)==(1,metadata['token_count'])
assert metadata['full_windows']==full==1024 and metadata['tail_tokens']==tail==0
assert metadata['windows']==full+int(tail>=2)
assert metadata['predicted_tokens']==full*2047+max(tail-1,0)==2096128
assert metadata['unscored_tail_tokens']==int(tail==1)==0
assert not metadata['add_bos_token'] and not metadata['add_eos_token']
assert (metadata['dataset'],metadata['subset'],metadata['split'])==('allenai/c4','en','validation')
print(json.dumps(dict(status='PASS',documents=4480,ordered_row_indices_match=True,random42_order_match=True,
 tokenizer_path=str(model),tokenizer_class=type(tokenizer).__name__,model_revision=revision,
 original_text_characters=len(text),tokens_before_truncation=reconstructed.numel(),retained_tokens=ids.numel(),
 discarded_suffix_tokens=reconstructed.numel()-ids.numel(),targets=metadata['predicted_tokens'],
 token_ids_exactly_reconstructed=True,tail_tokens=tail,source_documents=str(source),
 scope='CPU-only read-only token reconstruction; no model/PPL/GPU execution'),indent=2))
