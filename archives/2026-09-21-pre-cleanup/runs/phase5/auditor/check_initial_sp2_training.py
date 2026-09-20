"""Read-only evidence audit; no model creation or GPU calls."""
import ast
import json
import math
from collections import Counter
from pathlib import Path

import torch

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
A = ROOT / 'runs/phase5/llama32-1b/initial-sp2-qat400-s42'
B = A.with_name(A.name + '-r25')
H = ROOT / 'runs/phase3/distill-b100-refined-ref-adam1e5-400-20260914a'
def read(d, name):
    return json.loads((d / name).read_text())
def rows(d):
    return [json.loads(s) for s in (d / 'training.jsonl').read_text().splitlines()]

a, b, h = rows(A), rows(B), rows(H)
combined = a + b
assert [r['step'] for r in a] == list(range(1, 26))
assert [r['step'] for r in b] == list(range(26, 401))
assert [r['step'] for r in h] == list(range(1, 401))
assert read(A, 'data.json') == read(B, 'data.json') == read(H, 'data.json')
assert read(A, 'parameter_coverage.json') == read(B, 'parameter_coverage.json') == read(H, 'parameter_coverage.json')
for r, historical in zip(combined, h):
    step = r['step']
    assert len(r['microbatches']) == 8
    # data_windows returns windows[:-8]; four held-out probe windows never train.
    expected = [(800 + (step - 1) * 8 + i) % 1180 for i in range(8)]
    assert [v['window'] for v in r['microbatches']] == expected
    assert [v['window'] for v in historical['microbatches']] == expected
    assert all(v['predicted_tokens'] == 2047 for v in r['microbatches'])
    assert r['cumulative_train_tokens'] == step * 8 * 2048
    assert r['learning_rates']['W'] == historical['learning_rates']['W']
    assert r['gradient_tensors'] == historical['gradient_tensors'] == dict(W=112, SA=96, SW=112, SP2=16)

settings = [read(d, 'settings.json')['arguments'] for d in (A, B, H)]
matched_keys = ['reference_state', 'steps', 'schedule_steps', 'accumulation', 'warmup', 'weight_lr',
                'weight_optimizer', 'offload_teacher_body', 'momentum', 'relative_scale_lr',
                'temperature', 'ce_weight', 'data_start', 'resume_every', 'checkpoints']
assert all(settings[0][k] == settings[1][k] == settings[2][k] for k in matched_keys)
assert settings[1]['resume'] == str(A / 'resume.pt')
assert settings[0]['target_ppl'] is None and settings[1]['target_ppl'] is None
assert settings[0]['validation_steps'] == settings[1]['validation_steps'] == [400]
for name in ['distill.py', 'common.py', 'postprocess.py', 'quantization.py', 'run.py', 'architecture.py']:
    assert (A / 'source' / name).read_bytes() == (B / 'source' / name).read_bytes()
def definitions(d, name):
    return {n.name: ast.dump(n) for n in ast.parse((d/'source'/name).read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
old, new = definitions(H, 'distill.py'), definitions(B, 'distill.py')
identical_functions = ['TrainableQuantLinear', 'cell_reference_initialization', 'reference_for_parent',
                       'parameter_groups', 'chunk_objective', 'make_optimizers']
assert all(old[k] == new[k] for k in identical_functions)
assert definitions(H, 'run.py')['schedule'] == definitions(B, 'run.py')['schedule']

resumes = {}
for d, expected_step in [(A,25),(B,400),(H,400)]:
    state = torch.load(d/'resume.pt', map_location='cpu', weights_only=False, mmap=True)
    assert state['metadata']['step'] == expected_step
    assert len(state['parameters']) == 336
    optimizer_steps = []
    for opt, count in zip(state['optimizers'], [112,224]):
        assert len(opt['state']) == count
        steps = sorted({int(s['step'].item()) for s in opt['state'].values()})
        assert steps == [expected_step]
        optimizer_steps.append(steps)
    resumes[d.name] = dict(metadata=state['metadata'], parameter_tensors=len(state['parameters']),
                          optimizer_step_values=optimizer_steps, rng_keys=[k for k in state if 'rng' in k])
    del state

result = read(B, 'result.json')
assert result['completed_steps'] == 400 and result['target_reached'] is False
validation = read(B, 'checkpoint-0400/validation.json')
nll = sum(math.log(s['ppl']) * s['predicted_tokens'] for s in validation['segments']) / validation['predicted_tokens']
assert nll == validation['nll'] and math.exp(nll) == validation['ppl']
assert validation['ppl'] == result['checkpoints'][-1]['validation_ppl']
package = torch.load(B/'checkpoint-0400/static_w4a8.pt', map_location='cpu', weights_only=True, mmap=True)
formats = dict(Counter(s['format'] for s in package['activation'].values()))
assert formats == {'int8':96,'sp2':16}
assert len(package['weights']) == 112
output = dict(status='PASS', scope='training evidence only; no external acceptance claim',
              paths=dict(first=str(A), resumed=str(B), historical=str(H)),
              continuous_steps=[1,400], partition_steps=[25,375], optimizer_updates=400,
              microbatches=3200, effective_input_token_visits=6553600,
              effective_prediction_target_visits=6550400, matched_settings=matched_keys,
              historical_exact_window_order=True, historical_exact_weight_lr_schedule=True,
              identical_source_functions=identical_functions, resumes=resumes,
              package_formats=formats, validation=validation,
              operational_differences=['parent package', 'migration/resume and GPU assignment',
                  'full validation only at 400, historical at 100/200/400',
                  'target_ppl disabled, historical threshold reached only at final step 400'],
              budget_note='Effective committed update budget; interrupted next-step work before migration was discarded and is not included.')
destination = ROOT/'runs/phase5/auditor/initial_sp2_training_checks_20260918.json'
destination.write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps({k: output[k] for k in ['status','partition_steps','optimizer_updates','effective_input_token_visits','effective_prediction_target_visits','package_formats']}))
