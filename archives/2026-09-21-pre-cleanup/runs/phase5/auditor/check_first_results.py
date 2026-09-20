"""Read-only four-result Phase5 audit. No GPU or PPL execution."""
from pathlib import Path
import csv,json,math
import torch
r=Path('/home/dongpeiyan/projects/rotation-quant'); p5=r/'runs/phase5'
paths=['qwen3-1p7b/bf16-test-s42','qwen3-1p7b/bf16-validation-s42','qwen3-1p7b/bf16-c4-s42','llama32-1b/sp2-ptq-test-s42']
summary=list(csv.DictReader((p5/'summary.csv').open()))
assert len(summary)==17
for row in summary:
 d=json.loads(Path(row['evidence_path']).read_text())
 assert float(row['ppl'])==d['ppl'] and float(row['nll'])==d['nll']
 assert int(row['targets'])==d['predicted_tokens'] and int(row['input_tokens'])==d['token_count']
 assert int(row['unscored_tail_tokens'])==d['unscored_tail_tokens']
 if 'split' in d:assert row['split']==d['split'] and row['dataset']==d['dataset']
 if d.get('package') is not None:assert row['evaluation_package']==d['package']
results=[]
for rel in paths:
 p=p5/rel; d=json.loads((p/'result.json').read_text()); m=json.loads((p/'model.json').read_text())
 data=json.loads((p/'data.json').read_text()); settings=json.loads((p/'settings.json').read_text())
 progress=json.loads((p/'progress.json').read_text()); launch=json.loads((p5/(rel+'.launch.json')).read_text())
 assert progress['stage']=='completed' and not (p/'failure.json').exists()
 segments=d['segments'];total=sum(s['predicted_tokens'] for s in segments)
 nll=sum(s['nll']*s['predicted_tokens'] for s in segments)/total
 assert nll==d['nll'] and math.exp(nll)==d['ppl'] and total==d['predicted_tokens']
 assert all(s['nll']==math.log(s['ppl']) and s['predicted_tokens']==s['windows']*(s['seqlen']-1) for s in segments)
 assert d['activation_scales_unchanged'] and all(d[k] is False for k in ['calibration','training','candidate_selection'])
 assert m['kv_bits']==16 and m['use_cache'] is False and m['training'] is False and m['calibration'] is False
 assert set(m['rotary_buffer_dtypes'].values())=={'torch.float32'}
 assert m['package']==d['package'] and m['activation_formats']==d['activation_formats']
 saved=torch.load(d['input_token_path'],map_location='cpu',weights_only=True)
 assert saved['metadata']==d['input_metadata']==data['input_metadata']
 ids=saved['input_ids']; assert tuple(ids.shape)==(1,d['token_count']) and ids.dtype==torch.long
 metadata=d['input_metadata'];full,tail=divmod(ids.numel(),2048)
 assert d['predicted_tokens']==full*2047+max(tail-1,0) and d['unscored_tail_tokens']==int(tail==1)
 assert not metadata['add_bos_token'] and not metadata['add_eos_token']
 modeldir='qwen3-1.7b' if rel.startswith('qwen') else 'llama-3.2-1b-instruct'
 assert Path(metadata['tokenizer_path'])==r/'cache/models'/modeldir
 assert launch['child_environment']['PHASE5_MODEL_PATH']==metadata['tokenizer_path']
 before=torch.load(p/'quantizers_before.pt',map_location='cpu',weights_only=True)
 after=torch.load(p/'quantizers_after.pt',map_location='cpu',weights_only=True)
 assert before.keys()==after.keys()
 assert all(before[name].keys()==after[name].keys() and all(torch.equal(v,after[name][k]) for k,v in state.items()) for name,state in before.items())
 if rel.startswith('qwen'):
  assert before=={} and d['activation_formats']=={} and d['mode']=='bf16' and m['backbone_weight_bits']==16
 else:
  exact=r/'runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt'
  assert Path(d['package'])==exact and settings['arguments']['package']==str(exact)
  assert len(before)==112 and d['activation_formats']=={'int8':96,'sp2':16} and m['backbone_weight_bits']==4
  for old in ['wiki2-test-bf16-fixed-20260917a','wiki2-test-best-fixed-20260917a']:
   oldroot=r/'runs/phase3'/old
   oldtokens=torch.load(oldroot/'input_tokens.pt',map_location='cpu',weights_only=True)
   oldresult=json.loads((oldroot/'result.json').read_text())
   assert torch.equal(ids,oldtokens['input_ids']) and metadata==oldtokens['metadata']
   assert d['evaluation_precision']==oldresult['evaluation_precision']
   assert [[s[k] for k in ['start_token','seqlen','windows','predicted_tokens']] for s in segments]==[[s[k] for k in ['start_token','seqlen','windows','predicted_tokens']] for s in oldresult['segments']]
 results.append(dict(run=rel,ppl=d['ppl'],nll=d['nll'],input_tokens=d['token_count'],targets=d['predicted_tokens'],
   tail_tokens=tail,segments=len(segments),quantizers=len(before),package=d['package'],tokenizer_path=metadata['tokenizer_path'],
   exact_segment_recomputation=True,token_metadata_exact=True,all_quantizer_tensors_unchanged=True,
   use_cache=False,kv_bits=16,rotary_dtype='float32',python=launch['command'][0]))
print(json.dumps(dict(status='PASS',scope='read-only JSON and saved token/scale tensor audit; no GPU or PPL rerun',
 summary_rows=17,summary_all_values_match_raw=True,llama_ptq_tokens_match_bf16_and_qat_test_exactly=True,results=results),indent=2))
