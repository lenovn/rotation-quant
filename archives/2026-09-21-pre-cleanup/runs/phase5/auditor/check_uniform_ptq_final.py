from pathlib import Path
from collections import Counter
import csv,json,math
import torch
root=Path('/home/dongpeiyan/projects/rotation-quant'); p5=root/'runs/phase5'; r=p5/'llama32-1b'; stage=r/'uniform-down-readapt-s42'; package=stage/'static_w4a8.pt'
state=torch.load(package,map_location='cpu',weights_only=True,mmap=True)
parent=torch.load(r/'uniform-range-s42/static_w4a8.pt',map_location='cpu',weights_only=True,mmap=True)
settings=json.loads((stage/'settings.json').read_text())['arguments']; data=json.loads((stage/'data.json').read_text()); result=json.loads((stage/'result.json').read_text())
assert settings['mode']=='round' and settings['milestones']==[512,2048,8192] and settings['family']=='down'
assert settings['parent']==state['metadata']['parent']==result['parent']==str(r/'uniform-range-s42/static_w4a8.pt')
assert settings['reference_state']==str(root/'runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/state.pt')
assert data==json.loads((r/'uniform-down-round-s42/data.json').read_text())
assert data['fit_window_indices']==data['calibration_window_indices'][:24] and data['selection_window_indices']==data['calibration_window_indices'][24:]
assert data['selection_predicted_tokens']==16376
assert len(state['weights'])==len(state['activation'])==112 and set(a['format'] for a in state['activation'].values())=={'int8'}
for name,a in state['activation'].items():
 assert torch.equal(a['scale'],parent['activation'][name]['scale'])
 assert torch.equal(state['weights'][name]['scale'],parent['weights'][name]['scale'])
 if not name.endswith('down_proj'): assert torch.equal(state['weights'][name]['packed'],parent['weights'][name]['packed'])
modules=result['new_module_results'];assert len(modules)==16
score=result['initial_train_nll'];steps={}
for module in modules:
 assert module['previous_train_nll']==score
 score=module['selected_train_nll'];assert score==min(t['train_nll'] for t in module['trials'] if t.get('train_nll') is not None)
 trialsteps=[t['step'] for t in module['trials']];assert trialsteps[0]==0 and trialsteps[-1]<=8192
 steps[module['module']]=trialsteps
assert score==result['selected_train_nll']
assert (stage/'source/fixed_grid_rounding.py').read_bytes()==(r/'uniform-down-round-s42/source/fixed_grid_rounding.py').read_bytes()
source=(stage/'source/sequential_postprocess.py').read_text();assert 'model, _ = load_static(package)' in source
summary=list(csv.DictReader((p5/'summary.csv').open()));metrics=[];saved_before=[]
for directory,filename in [(stage,'validation.json'),(r/'uniform-ptq-test-s42','result.json'),(r/'uniform-ptq-c4-s42','result.json')]:
 d=json.loads((directory/filename).read_text());assert json.loads((directory/'progress.json').read_text())['stage']=='completed' and not(directory/'failure.json').exists()
 nll=sum(s['nll']*s['predicted_tokens'] for s in d['segments'])/d['predicted_tokens'];assert nll==d['nll'] and math.exp(nll)==d['ppl']
 assert sum(s['predicted_tokens'] for s in d['segments'])==d['predicted_tokens']
 rows=[row for row in summary if row['evidence_path']==str(directory/filename)];assert len(rows)==1
 row=rows[0];assert row['stage']=='uniform-ptq' and row['down_format']=='INT8' and row['qat_data']=='none' and row['evaluation_package']==str(package)
 for k in ['ppl','nll']:assert float(row[k])==d[k]
 assert int(row['targets'])==d['predicted_tokens'] and int(row['input_tokens'])==d['token_count']
 if filename=='result.json':
  assert d['package']==str(package) and d['activation_formats']=={'int8':112} and d['down_int8_overlay'] is None
  assert d['activation_scales_unchanged'] and not d['calibration'] and not d['training'] and not d['candidate_selection']
  model=json.loads((directory/'model.json').read_text());assert model['package']==str(package) and model['kv_bits']==16 and not model['use_cache'] and model['backbone_weight_bits']==4
  assert model['activation_formats']=={'int8':112} and set(model['rotary_buffer_dtypes'].values())=={'torch.float32'}
  before=torch.load(directory/'quantizers_before.pt',map_location='cpu',weights_only=True);after=torch.load(directory/'quantizers_after.pt',map_location='cpu',weights_only=True)
  assert before.keys()==after.keys() and len(before)==112
  for name,b in before.items():
   assert b.keys()==after[name].keys() and all(torch.equal(value,after[name][key]) for key,value in b.items())
   assert torch.equal(b['scale'],state['activation'][name]['scale'])
  saved_before.append(before)
  ids=torch.load(d['input_token_path'],map_location='cpu',weights_only=True);assert ids['metadata']==d['input_metadata']
  if d['dataset']=='Salesforce/wikitext':
   oldruns=['wiki2-test-bf16-fixed-20260917a','wiki2-test-best-fixed-20260917a']
  else:
   oldruns=['c4-bf16-fixed-20260915a','c4-best-fixed-20260915a','c4-ptq-parent-fixed-20260918a']
  for oldrun in oldruns:
   old=json.loads((root/'runs/phase3'/oldrun/'result.json').read_text()); oldids=torch.load(old['input_token_path'],map_location='cpu',weights_only=True)
   assert torch.equal(ids['input_ids'],oldids['input_ids']) and ids['metadata']==oldids['metadata']
   assert d['evaluation_precision']==old['evaluation_precision']
   assert [[s[k] for k in ['start_token','seqlen','windows','predicted_tokens']] for s in d['segments']]==[[s[k] for k in ['start_token','seqlen','windows','predicted_tokens']] for s in old['segments']]
 metrics.append(dict(run=directory.name,split=row['split'],dataset=row['dataset'],ppl=d['ppl'],nll=d['nll'],targets=d['predicted_tokens'],package=str(package)))
assert all(torch.equal(v,saved_before[1][name][k]) for name,d in saved_before[0].items() for k,v in d.items())
print(json.dumps(dict(status='PASS',scope='complete Uniform-PTQ seed42 only; QAT not covered; no GPU/PPL rerun',
 package=str(package),activation_formats={'int8':112},same_package_validation_test_c4=True,train_selection_nll_initial=result['initial_train_nll'],train_selection_nll_final=score,
 readapt_trial_steps=steps,reference=settings['reference_state'],train_only_24_8_selection=True,
 all_activation_scales_and_weight_scales_unchanged_in_readapt=True,non_down_weight_codes_unchanged=True,
 test_c4_before_after_equal_and_scales_equal_package=True,historical_tokens_and_metric_segments_match=True,summary_three_rows_match=True,results=metrics),indent=2))
