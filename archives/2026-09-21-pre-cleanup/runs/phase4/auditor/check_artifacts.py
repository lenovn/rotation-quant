"""Read-only CPU tensor/source comparison; no model construction or forward."""
import json
import math
import subprocess
from pathlib import Path
import torch

torch.set_num_threads(2)
root = Path('/home/dongpeiyan/projects/rotation-quant')
phase = root / 'runs/phase4'
src = root / 'worktrees/SpinQuant-phase4-w4'
old = root / 'worktrees/SpinQuant-phase3-joint'
target = 'model.layers.8.mlp.down_proj'
def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
def equal(a, b):
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and bool(torch.all((a == b) | (torch.isnan(a) & torch.isnan(b))))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    return a == b
parent = load(root / 'runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt')
inheritance = json.loads((phase / 'inheritance.json').read_text())
tracked = subprocess.check_output(['git','-C',str(old),'diff','--name-only'], text=True).splitlines()
source = {name: (old/name).read_bytes() == (src/name).read_bytes()
          for name in tracked + inheritance['untracked_files']}
assert all(source.values())
assert (phase/'inheritance.diff').read_bytes() == subprocess.check_output(['git','-C',str(old),'diff','--binary'])
result = {'inherited_source_equal': source, 'phase3_diff_equals_inheritance': True, 'packages': {}}
for label, directory, recpath in [
    ('A1', phase/'a-layer8-final-20260915a/A1', 'updates.pt'),
    ('A2', phase/'a-layer8-final-20260915a/A2', 'updates.pt'),
    ('B', phase/'b-guided-g1-layer8-20260915a', 'candidate_records.pt')]:
    state = load(directory/'static_w4a8.pt')
    assert len(state['weights']) == 112
    assert equal(state['activation'], parent['activation'])
    assert equal(state['high_precision'], parent['high_precision'])
    changed_q, changed_sw = [], []
    for name, record in state['weights'].items():
        assert record['scale'].shape == (record['shape'][0], 1)
        assert bool(torch.isfinite(record['scale']).all() and (record['scale'] > 0).all())
        assert record['packed'].dtype == torch.uint8
        assert record['packed'].numel()*2 == math.prod(record['shape'])
        if not equal(record['packed'], parent['weights'][name]['packed']): changed_q.append(name)
        if not equal(record['scale'], parent['weights'][name]['scale']): changed_sw.append(name)
        if name != target: assert equal(record, parent['weights'][name])
    assert changed_q == [target]
    assert changed_sw == ([target] if label == 'A2' else [])
    updates = load(directory/recpath)
    if label == 'B':
        selected = json.loads((directory/'selection.json').read_text())['selected_cycle']
        record = updates[selected-1]
    else: record = updates[target]
    assert equal(record, state['weights'][target])
    validation = json.loads((directory/'validation.json').read_text())
    assert validation['token_count'] == 252852 and validation['predicted_tokens'] == 252728
    assert [(s['windows'],s['seqlen'],s['predicted_tokens']) for s in validation['segments']] == [(123,2048,251781),(1,948,947)]
    nll = sum(s['nll']*s['predicted_tokens'] for s in validation['segments'])/252728
    assert nll == validation['nll'] and math.exp(nll) == validation['ppl']
    result['packages'][label] = dict(changed_q=changed_q,changed_sw=changed_sw,
        activation_equal=True,high_precision_equal=True,export_matches_selected_record=True,
        int8_inputs=sum(x['format']=='int8' for x in state['activation'].values()),
        sp2_inputs=sum(x['format']=='sp2' for x in state['activation'].values()),
        nll=nll,ppl=validation['ppl'],metadata=state['metadata'])
    del state, updates
source_snapshots = {}
for run in ['a-sw-code-matched-20260915a','a-layer8-final-20260915a','b-guided-g1-layer8-20260915a']:
    for path in (phase/run/'source').glob('*.py'):
        matched = path.read_bytes() == (src/'experiments/phase4'/path.name).read_bytes()
        source_snapshots[run+'/'+path.name] = matched
        if run != 'a-sw-code-matched-20260915a': assert matched
saliency = load(phase/'b-guided-g1-layer8-20260915a/saliency.pt').reshape(32,2048)
assert torch.isfinite(saliency).all() and (saliency >= 0).all()
assert (saliency[:,-1] == 0).all() and (saliency[:,:-1] > 0).all()
result.update(status='PASS',run_source_matches_live=source_snapshots,
              first_a_source_difference='Later addition of diagnostic --evaluate-best-rejected export, manually reviewed; candidate generation unchanged',
              saliency_zero_exactly_each_window_final_position=True,
              model_forwards=0,gpu_used=False)
print(json.dumps(result,indent=2))
