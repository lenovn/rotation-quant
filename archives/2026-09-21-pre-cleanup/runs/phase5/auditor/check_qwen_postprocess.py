"""Read-only CPU audit of completed Qwen intermediate postprocessing."""
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
def read(d, f): return json.loads((d/f).read_text())
def load(f): return torch.load(f,map_location='cpu',weights_only=True,mmap=True)
def same(a,b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))
def scale(a):
    return a['scale'] if a['format'] == 'int8' else torch.tensor([a['alpha']/127],dtype=torch.float32)

base = read(Q/'joint100-s42','data.json')
summary = list(csv.DictReader((P/'summary.csv').open()))
names = ['sp2-down-round-s42','sp2-range-s42','uniform-down-round-s42','uniform-range-s42']
parents = ['joint100-s42/checkpoint-0100','sp2-down-round-s42','uniform-initial-s42','uniform-down-round-s42']
reports = []
for name,parent_name in zip(names,parents):
    directory = Q/name
    parent_path = Q/parent_name/'static_w4a8.pt'
    current = load(directory/'static_w4a8.pt')
    parent = load(parent_path)
    data = read(directory,'data.json')
    settings = read(directory,'settings.json')
    args = settings['arguments']
    result = read(directory,'result.json')
    validation = read(directory,'validation.json')
    mode = 'range' if 'range' in name else 'round'
    formats = {'int8':168,'sp2':28} if name.startswith('sp2') else {'int8':196}
    assert read(directory,'progress.json')['stage'] == 'completed' and not (directory/'failure.json').exists()
    assert args['parent'] == result['parent'] == current['metadata']['parent'] == str(parent_path)
    assert args['mode'] == mode and args['family'] == 'down' and args['resume_prefix'] is None
    assert args['reference_state'] == str(Q/'joint100-s42/checkpoint-0100/state.pt')
    assert settings['source']['seed'] == data['seed'] == 42
    assert data['calibration_window_indices'] == base['calibration_window_indices']
    assert data['fit_window_indices'] == data['calibration_window_indices'][:24]
    assert data['selection_window_indices'] == data['calibration_window_indices'][24:]
    assert data['calibration_length'] == 2048 and data['fit_rows'] == 49152 and data['heldout_rows'] == 16384
    assert data['selection_predicted_tokens'] == 16376
    assert dict(Counter(a['format'] for a in current['activation'].values())) == formats
    assert len(current['weights']) == len(current['activation']) == 196
    assert current['config'] == parent['config']
    assert current['high_precision'].keys() == parent['high_precision'].keys()
    assert all(same(v,parent['high_precision'][k]) for k,v in current['high_precision'].items())
    assert len([k for k in current['high_precision'] if 'q_norm' in k or 'k_norm' in k]) == 56
    targets = [f'model.layers.{i}.mlp.down_proj' for i in range(28)]
    assert result['completed_targets'] == targets
    modules = result['new_module_results']
    assert [m['module'] for m in modules] == targets
    score = result['initial_train_nll']
    changed_weight_layers = []
    for key in parent['weights']:
        before,after = parent['weights'][key],current['weights'][key]
        assert before.keys() == after.keys()
        for k in before:
            if k == 'packed': continue
            assert same(before[k],after[k]) if torch.is_tensor(before[k]) else before[k] == after[k]
        changed = not same(before['packed'],after['packed'])
        if changed: changed_weight_layers.append(key)
        if mode == 'range' or not key.endswith('down_proj'): assert not changed
        olda,newa = parent['activation'][key],current['activation'][key]
        assert olda['format'] == newa['format']
        if mode == 'round' or not key.endswith('down_proj'):
            assert same(scale(olda),scale(newa))
    early = []
    changed_ranges = []
    for m in modules:
        assert m['previous_train_nll'] == score and m['trials'][0]['train_nll'] == score
        assert m['selected_train_nll'] == min(t['train_nll'] for t in m['trials'] if t.get('train_nll') is not None)
        score = m['selected_train_nll']
        key = m['module']
        if mode == 'round':
            assert args['milestones'] == [512,2048,8192]
            steps = [t['step'] for t in m['trials']]
            assert steps == [0,512,2048,8192] or (name == 'uniform-down-round-s42' and key == 'model.layers.2.mlp.down_proj' and steps == [0,14])
            assert m['trials'][0]['changed_codes'] == 0
            for t in m['trials'][1:]:
                if t['train_nll'] is None: assert t['heldout_mse'] >= m['trials'][0]['heldout_mse']
                else: assert t['heldout_mse'] < m['trials'][0]['heldout_mse']
            chosen = [t for t in m['trials'] if t['step'] == m['selected_step']]
            assert len(chosen) == 1 and chosen[0]['train_nll'] == score
            assert (key in changed_weight_layers) == (m['selected_step'] != 0)
            if steps != [0,512,2048,8192]: early.append(dict(module=key,steps=steps,selected_step=m['selected_step']))
        else:
            factors = [1.,.875,.75,.5,.25,.125]
            assert args['alpha_factors'] == [t['factor'] for t in m['trials']] == factors
            assert m['trials'][0]['reused_prefix'] is True
            chosen, = [t for t in m['trials'] if t['factor'] == m['selected_factor']]
            assert chosen['train_nll'] == score
            oldscale = scale(parent['activation'][key])
            expected = oldscale if m['selected_factor'] == 1 else torch.tensor([chosen['alpha']/127],dtype=torch.float32)
            assert chosen['alpha'] == float(oldscale)*127*m['selected_factor']
            # SP2 serializes scale*127 as alpha then cold-loads alpha/127.
            if current['activation'][key]['format'] == 'sp2':
                assert current['activation'][key]['alpha'] == float((expected*127).item())
            else: assert same(current['activation'][key]['scale'],expected)
            if m['selected_factor'] != 1: changed_ranges.append(dict(module=key,factor=m['selected_factor']))
    assert score == result['selected_train_nll']
    assert sum(s['predicted_tokens'] for s in validation['segments']) == validation['predicted_tokens'] == 262208
    nll = sum(math.log(s['ppl'])*s['predicted_tokens'] for s in validation['segments'])/validation['predicted_tokens']
    assert nll == validation['nll'] and math.exp(nll) == validation['ppl']
    row, = [x for x in summary if x['evidence_path'] == str(directory/'validation.json')]
    assert row['model'] == 'Qwen/Qwen3-1.7B' and row['seed'] == '42' and row['split'] == 'validation'
    assert row['down_format'] == ('SP2' if name.startswith('sp2') else 'INT8')
    assert row['evaluation_package'] == str(directory/'static_w4a8.pt') and row['parent_package'] == str(parent_path)
    assert float(row['ppl']) == validation['ppl'] and float(row['nll']) == validation['nll']
    assert int(row['targets']) == 262208 and int(row['input_tokens']) == 262337
    helper = directory/'source/fixed_grid_rounding.py'
    assert helper.read_bytes() == (ROOT/'runs/phase3/seq-b100-down-round-20260914b/source/fixed_grid_rounding.py').read_bytes()
    reports.append(dict(run=name,parent=str(parent_path),mode=mode,formats=formats,
        initial_train_nll=result['initial_train_nll'],selected_train_nll=score,
        validation_ppl=validation['ppl'],validation_nll=validation['nll'],summary_stage=row['stage'],
        targets_covered=28,total_trial_records=sum(len(m['trials']) for m in modules),
        changed_weight_layers=changed_weight_layers,changed_ranges=changed_ranges,early_convergence=early,
        high_precision_tensors_bitwise_unchanged=len(current['high_precision']),qk_norm_tensors_bitwise_unchanged=56))
out = dict(status='PASS',scope='four completed intermediate stages only; readaptation and final PTQ acceptance pending',
    common_seed=42,common_calibration_indices=base['calibration_window_indices'],train_windows=32,fit_windows=24,
    selection_windows=8,window_length=2048,selection_targets=16376,
    candidate_selection_train_only=True,parent_candidate_always_present=True,
    all_w4_scales_and_non_down_weight_codes_unchanged=True,non_down_activation_scales_unchanged=True,
    round_only_down_codes_change=True,range_only_down_input_scales_change=True,
    results=reports)
(P/'auditor/QWEN_POSTPROCESS_20260918.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
