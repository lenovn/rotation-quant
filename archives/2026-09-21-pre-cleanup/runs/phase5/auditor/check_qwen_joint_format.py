"""Bounded read-only CPU audit of actual Qwen Joint100 and matched INT8."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
import torch

torch.set_num_threads(2)
ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
P = ROOT/'runs/phase5'
Q = P/'qwen3-1p7b'
J = Q/'joint100-s42'
U = Q/'uniform-initial-s42'
CK = J/'checkpoint-0100'
def read(d, f): return json.loads((d/f).read_text())
def load(f): return torch.load(f, map_location='cpu', weights_only=True, mmap=True)
def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))

rows = [json.loads(s) for s in (J/'training.jsonl').read_text().splitlines()]
assert [x['step'] for x in rows] == list(range(1,101))
for row in rows:
    assert len(row['microbatch_losses']) == 8
    assert row['effective_targets'] == 16376 and row['train_tokens'] == 16384
    assert row['cumulative_train_tokens'] == row['step']*16384
    assert row['window_indices'] == list(range((row['step']-1)*8,row['step']*8))
    for group, n in [('R',29),('SA',168),('SW',196)]:
        assert row['updates'][group]['gradient_tensors'] == n
        assert row['updates'][group]['changed_elements'] > 0
    assert row['updates']['SP2']['gradient_tensors'] == 0
    assert row['updates']['SP2']['changed_elements'] == 0
settings = read(J,'settings.json')
assert settings['source']['seed'] == 42 and settings['arguments']['route'] == 'B'
assert settings['arguments']['accumulation'] == 8
assert read(J,'progress.json')['stage'] == 'completed' and not (J/'failure.json').exists()
data = read(J,'data.json')
assert data == read(Q/'init-s42','data.json')
indices = torch.randperm(data['train_windows']-8,generator=torch.Generator().manual_seed(42))[:32].tolist()
assert indices == data['calibration_window_indices']
initial = load(Q/'init-s42/initial.pt')
joint = load(CK/'state.pt')
assert initial['metadata']['seed'] == 42 and joint['metadata']['update_step'] == 100
rkeys = [n for n in joint['parameters'] if n.endswith('R1.weight') or n.endswith('R2.weight')]
assert len(rkeys) == 29
assert all(not torch.equal(joint['parameters'][k],initial['parameters'][k]) for k in rkeys)
sp = load(CK/'static_w4a8.pt')
un = load(U/'static_w4a8.pt')
assert dict(Counter(a['format'] for a in sp['activation'].values())) == {'int8':168,'sp2':28}
assert dict(Counter(a['format'] for a in un['activation'].values())) == {'int8':196}
assert sp['weights'].keys() == un['weights'].keys() and len(sp['weights']) == 196
for name, w in sp['weights'].items():
    for key,value in w.items():
        assert (bits_equal(value,un['weights'][name][key]) if torch.is_tensor(value)
                else value == un['weights'][name][key])
assert sp['high_precision'].keys() == un['high_precision'].keys()
nan_buffers = []
for name,value in sp['high_precision'].items():
    assert bits_equal(value,un['high_precision'][name]), name
    if value.is_floating_point() and torch.isnan(value).any(): nan_buffers.append(name)
qknorm = [n for n in sp['high_precision'] if 'q_norm' in n or 'k_norm' in n]
assert len(qknorm) == 56
for name,a in sp['activation'].items():
    if not name.endswith('down_proj'):
        assert torch.equal(a['scale'],un['activation'][name]['scale'])
assert sp['config'] == un['config']
assert un['metadata']['parent'] == str(CK/'static_w4a8.pt')
spcal = read(CK,'sp2_calibration.json')
incal = read(U,'range_search.json')
capture = read(U,'capture_metadata.json')
um = read(U,'data.json')['metadata']
assert um['seed'] == 42 and um['calibration_window_indices'] == indices
assert um['calibration_length'] == 128 and um['calibration_tokens'] == 4096 and um['split'] == 'train'
assert len(spcal) == len(incal) == len(capture) == 28
alpha_bounds = []
for name,sc in spcal.items():
    ic,cap = incal[name],capture[name]
    assert len(sc['candidates']) == len(ic['candidates']) == 50
    assert sc['sampled_rows'] == ic['sampled_rows'] == cap['sampled_rows'] == cap['historical_sampled_rows'] == 512
    assert sc['full_absmax'] == ic['full_absmax'] == cap['full_absmax'] == cap['historical_full_absmax']
    for rec in [sc,ic]:
        assert rec['selected'] == min(rec['candidates'],key=lambda x:x['output_mse'])
        coarse = torch.linspace(-16.,1.,33).tolist()
        assert [x['log2_ratio'] for x in rec['candidates'][:33]] == coarse
        best = min(range(33),key=lambda i:rec['candidates'][i]['output_mse'])
        refined = torch.linspace(coarse[max(0,best-1)],coarse[min(32,best+1)],17).tolist()
        assert [x['log2_ratio'] for x in rec['candidates'][33:]] == refined
    assert ic['historical_sp2_selected'] == sc['selected']
    assert torch.equal(un['activation'][name]['scale'],torch.tensor([ic['selected']['scale']],dtype=torch.float32))
    expected = float(torch.tensor([sc['selected']['alpha']/127],dtype=torch.float32).mul(127).item())
    assert sp['activation'][name]['alpha'] == expected
    alpha_bounds.append(dict(module=name,absmax=sc['full_absmax'],sp2_alpha=sc['selected']['alpha'],int8_alpha=ic['selected']['alpha']))
before,after = load(U/'quantizers_before.pt'),load(U/'quantizers_after.pt')
assert before.keys() == after.keys()
assert all(bits_equal(v,after[n][k]) for n,d in before.items() for k,v in d.items())
v = read(CK,'validation.json')
nll = sum(math.log(s['ppl'])*s['predicted_tokens'] for s in v['segments'])/v['predicted_tokens']
assert nll == v['nll'] and math.exp(nll) == v['ppl']
summary = list(csv.DictReader((P/'summary.csv').open()))
row, = [x for x in summary if x['evidence_path'] == str(CK/'validation.json')]
assert row['stage'] == 'B100-initial-SP2' and row['model'] == 'Qwen/Qwen3-1.7B'
assert float(row['ppl']) == v['ppl'] and float(row['nll']) == v['nll']
out = dict(status='PASS',scope='Joint100 plus matched initial INT8 package, not full PTQ or Uniform-QAT400; C4 pending',
    training=dict(steps=100,global_microbatches_per_update=8,input_token_visits=1638400,prediction_targets=1637600,
        seed=42,rotation_tensors=29,all_rotations_changed_from_initial=True,R_gradients_per_step=29,
        SA_gradients_per_step=168,SW_gradients_per_step=196,SP2_gradients_per_step=0,route='B'),
    matching=dict(weight_records_bitwise_equal=196,high_precision_tensors_bitwise_equal=len(sp['high_precision']),
        qk_norm_tensors_bitwise_equal=56,non_down_activation_scales_equal=168,
        sp2_formats={'int8':168,'sp2':28},int8_formats={'int8':196},config_equal=True,
        common_joint_parent=str(CK/'static_w4a8.pt'),identical_nan_placeholder_buffers=nan_buffers,
        rotation_equivalence='R is fused into identical frozen W and high_precision tensors; no independent R tensors in frozen package.'),
    calibration=dict(indices=indices,window_length=128,tokens=4096,sampled_rows_per_layer=512,layers=28,
        candidates_per_layer=50,coarse_log2_bounds=[-16,1],coarse_count=33,refinement_count=17,
        all_candidates_and_minima_checked=True,full_absmax_exactly_equal=True,parent_quantizer_states_unchanged=True,
        capture_evidence_limit='Same source capture algorithm/layout, indices, rows, bounds; arrays not saved, no elementwise capture equality claim.',layers_alpha=alpha_bounds),
    validation=dict(ppl=v['ppl'],nll=v['nll'],targets=v['predicted_tokens']))
(P/'auditor/QWEN_JOINT_FORMAT_20260918.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['status','training','validation']},indent=2))
