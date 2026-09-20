from pathlib import Path
from collections import Counter
import csv,json,math
import torch
root=Path('/home/dongpeiyan/projects/rotation-quant');r=root/'runs/phase5/llama32-1b'
names=['uniform-initial-s42','uniform-down-round-s42','uniform-range-s42']
packs=[torch.load(r/n/'static_w4a8.pt',map_location='cpu',weights_only=True,mmap=True) for n in names]
old=json.loads((root/'runs/phase3/route-b-adam-100-20260914a/data.json').read_text())
summary=list(csv.DictReader((r.parent/'summary.csv').open()))
report=[]
for idx,name in enumerate(names[1:],1):
 p=r/name;current=packs[idx];parent=packs[idx-1]
 d=json.loads((p/'data.json').read_text());settings=json.loads((p/'settings.json').read_text());out=json.loads((p/'result.json').read_text());v=json.loads((p/'validation.json').read_text())
 assert json.loads((p/'progress.json').read_text())['stage']=='completed' and not (p/'failure.json').exists()
 assert settings['arguments']['parent']==out['parent']==current['metadata']['parent']==str(r/names[idx-1]/'static_w4a8.pt')
 if idx==1:assert settings['arguments']['reference_state']==str(root/'runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/state.pt')
 else:assert settings['arguments']['reference_state'] is None
 assert d['calibration_window_indices']==old['calibration_window_indices']
 assert d['fit_window_indices']==d['calibration_window_indices'][:24] and d['selection_window_indices']==d['calibration_window_indices'][24:]
 assert d['calibration_length']==2048 and d['fit_rows']==24*2048 and d['heldout_rows']==8*2048 and d['selection_predicted_tokens']==8*2047
 assert len(current['weights'])==len(current['activation'])==112 and {a['format'] for a in current['activation'].values()}=={'int8'}
 modules=out['new_module_results'];assert len(modules)==16 and out['completed_targets']==[f'model.layers.{i}.mlp.down_proj' for i in range(16)]
 current_score=out['initial_train_nll']
 for row in modules:
  assert row['previous_train_nll']==current_score
  assert row['selected_train_nll']==min(t['train_nll'] for t in row['trials'] if t.get('train_nll') is not None)
  current_score=row['selected_train_nll']
 assert current_score==out['selected_train_nll']
 for key in parent['weights']:
  assert torch.equal(current['weights'][key]['scale'],parent['weights'][key]['scale'])
  if idx==2 or not key.endswith('down_proj'):
   assert torch.equal(current['weights'][key]['packed'],parent['weights'][key]['packed'])
  if idx==1 or not key.endswith('down_proj'):
   assert torch.equal(current['activation'][key]['scale'],parent['activation'][key]['scale'])
 if idx==1:
  assert settings['arguments']['mode']=='round' and settings['arguments']['milestones']==[512,2048,8192]
  for row in modules:
   steps=[t['step'] for t in row['trials']]
   assert steps in ([0,512,2048,8192],[0,40])
   if steps==[0,40]:assert row['module']=='model.layers.1.mlp.down_proj'
  assert modules[1]['selected_step']==40
  source=p/'source/fixed_grid_rounding.py'
  assert source.read_bytes()==(root/'scripts/phase2/fixed_grid_rounding.py').read_bytes()
  historical=root/'runs/phase3/seq-b100-down-round-20260914b/source/fixed_grid_rounding.py'
  assert source.read_bytes()==historical.read_bytes()
 else:
  assert settings['arguments']['mode']=='range' and settings['arguments']['alpha_factors']==[1,.875,.75,.5,.25,.125]
  for row in modules:
   assert [t['factor'] for t in row['trials']]==[1,.875,.75,.5,.25,.125]
   key=row['module'];torch.testing.assert_close(current['activation'][key]['scale'],parent['activation'][key]['scale']*row['selected_factor'],rtol=1e-7,atol=0)
 nll=sum(s['nll']*s['predicted_tokens'] for s in v['segments'])/v['predicted_tokens']
 assert nll==v['nll'] and math.exp(nll)==v['ppl'] and v['predicted_tokens']==252728
 rows=[row for row in summary if row['evidence_path']==str(p/'validation.json')];assert len(rows)==1
 row=rows[0];assert float(row['ppl'])==v['ppl'] and float(row['nll'])==v['nll'] and row['down_format']=='INT8' and row['split']=='validation'
 assert row['evaluation_package']==str(p/'static_w4a8.pt') and row['parent_package']==out['parent']
 report.append(dict(run=name,mode=out['mode'],initial_train_nll=out['initial_train_nll'],selected_train_nll=out['selected_train_nll'],ppl=v['ppl'],nll=v['nll'],actual_formats=dict(Counter(a['format'] for a in current['activation'].values())),
  changed_layers=sum((m.get('selected_step',0)!=0 if idx==1 else m['selected_factor']!=1) for m in modules),trial_counts=[len(m['trials']) for m in modules],summary_model=row['model'],parent=out['parent']))
print(json.dumps(dict(status='PASS',scope='completed intermediate packages only; no GPU or PPL rerun',results=report,
 round_layer1={'selected_step':40,'cause':'inherited no-improving-coordinate threshold snapshots current iterate then breaks','threshold':-1e-15,'snapshot_source_equal_historical':True},
 all_w4_scales_fixed=True,non_down_weight_codes_fixed=True,non_down_activation_scales_fixed=True,
 round_all_activation_scales_fixed=True,range_all_weight_codes_fixed=True),indent=2))
