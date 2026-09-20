from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import torch

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
SOURCE = ROOT / 'worktrees/SpinQuant-phase3-joint'
RUNS = ROOT / 'runs/phase3'
OUTPUT = Path(__file__).resolve().parent
PREVIOUS = RUNS / 'auditor/first-validation-20260914'


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def same_bytes(left, right):
    return left.read_bytes() == right.read_bytes()


def log_events(path):
    events = []
    for line in path.read_text().splitlines():
        if line.startswith('{'):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def main():
    torch.set_num_threads(2)
    evidence = OUTPUT / 'evidence'
    evidence.mkdir(exist_ok=True)
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    for filename, arguments in [('head.txt', ['rev-parse', 'HEAD']),
                                ('branch.txt', ['branch', '--show-current']),
                                ('status.txt', ['status', '--short']),
                                ('dirty.patch', ['diff', '--binary'])]:
        content = subprocess.check_output(['git', '-C', str(SOURCE), *arguments], env=environment)
        (evidence / filename).write_bytes(content)
    source_checks = dict(
        head_matches_first_audit=same_bytes(evidence / 'head.txt', PREVIOUS / 'evidence/source.head'),
        branch_matches_first_audit=same_bytes(evidence / 'branch.txt', PREVIOUS / 'evidence/source.branch'),
        tracked_dirty_matches_first_audit=same_bytes(evidence / 'dirty.patch', PREVIOUS / 'evidence/source.dirty.patch'),
        tracked_dirty_matches_handoff=same_bytes(evidence / 'dirty.patch', RUNS / 'handoff-20260914.NByJvb/inherited_source.patch'),
        acceptance_unchanged=same_bytes(ROOT / 'scripts/phase2/validation_acceptance.py', PREVIOUS / 'evidence/validation_acceptance.py'))
    for relative in ['utils/eval_utils.py', 'eval_utils/modeling_llama.py', 'utils/quant_utils.py',
                     'train_utils/quant_linear.py', 'train_utils/modeling_llama_quant.py',
                     'train_utils/optimizer.py']:
        source_checks[relative] = same_bytes(SOURCE / relative, PREVIOUS / 'evidence/source_snapshot' / relative)
    baseline = read_json(PREVIOUS / 'bf16.result.json')
    prior_audit = read_json(PREVIOUS / 'records_audit.json')
    prior_files = read_json(PREVIOUS / 'evidence/file_metadata.json')
    reused_model_checks = {}
    for relative in ['model.safetensors', 'tokenizer.json']:
        path = ROOT / 'cache/models/llama-3.2-1b-instruct' / relative
        saved = prior_files[str(path)]
        reused_model_checks[relative] = path.stat().st_size == saved['size'] and path.stat().st_mtime_ns == saved['mtime_ns']
    for filename in ['.mv', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        reused_model_checks[filename] = same_bytes(ROOT / 'cache/models/llama-3.2-1b-instruct' / filename,
                                                   PREVIOUS / 'evidence/model_metadata' / filename)
    initial_path = RUNS / 'common-init-20260914a/initial.pt'
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)['parameters']
    common_data = prior_audit['data']
    trajectories = {}
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(),
                   role='same independent experimental auditor; read-only evidence audit',
                   scope=dict(gpu_evaluations=0, training_updates=0, recalibrations=0,
                              pytest_runs=0, model_forwards=0, algorithm_searches=0),
                   source_checks=source_checks, model_metadata_checks=reused_model_checks,
                   reused_evidence=dict(directory=str(PREVIOUS), baseline=baseline,
                                        model_and_tokenizer='reuse first-audit source evidence; small metadata and file size/mtime rechecked, no model-weight rescan',
                                        data='reuse first-audit tokenizer/calibration proof; each run data.json rechecked, no retokenization',
                                        initial='first-audit 241 equal step0 parameters reused'), routes={})
    for route in 'ABC':
        name = f'route-{route.lower()}-adam-100-20260914a'
        directory = RUNS / name
        destination = evidence / name
        destination.mkdir(exist_ok=True)
        for filename in ['settings.json', 'data.json', 'progress.json', 'results.json', 'training.jsonl']:
            shutil.copy2(directory / filename, destination / filename)
        for suffix in ['.launch.json', '.log']:
            shutil.copy2(RUNS / (name + suffix), destination / (name + suffix))
        shutil.copytree(directory / 'source', destination / 'source', dirs_exist_ok=True)
        for step in [10, 50, 100]:
            checkpoint = directory / f'checkpoint-{step:04d}'
            target = destination / checkpoint.name
            target.mkdir(exist_ok=True)
            for path in checkpoint.glob('*.json'):
                shutil.copy2(path, target / path.name)
        configuration = read_json(destination / 'settings.json')
        arguments = configuration['arguments']
        launch = read_json(destination / (name + '.launch.json'))
        progress = read_json(destination / 'progress.json')
        local_training = read_rows(destination / 'training.jsonl')
        training = list(local_training)
        if route == 'C':
            parent = Path(arguments['resume'])
            training = read_rows(parent / 'training.jsonl') + training
            for filename in ['training.jsonl', 'settings.json']:
                shutil.copy2(parent / filename, destination / ('parent-' + filename))
        trajectories[route] = training
        events = log_events(destination / (name + '.log'))
        checkpoint = directory / 'checkpoint-0100'
        old_result = read_json(directory / 'checkpoint-0010/validation.json')
        result = read_json(checkpoint / 'validation.json')
        frozen = torch.load(checkpoint / 'static_w4a8.pt', map_location='cpu', weights_only=True, mmap=True)
        state = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=True)
        resume = torch.load(directory / 'resume.pt', map_location='cpu', weights_only=True)
        suffix_shapes = dict(q_proj=(2048, 2048), k_proj=(512, 2048), v_proj=(512, 2048),
                             o_proj=(2048, 2048), gate_proj=(8192, 2048), up_proj=(8192, 2048), down_proj=(2048, 8192))
        expected = {f'model.layers.{layer}.{family}.{suffix}': shape
                    for layer in range(16) for suffix, shape in suffix_shapes.items()
                    for family in ['self_attn' if suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'mlp']}
        rates = {group.get('parameter_name', group['name']): group['initial_lr']
                 for optimizer in resume['optimizers'] for group in optimizer['param_groups']}
        cpu_mean_differences = {key: dict(saved_initial_lr=rate,
            cpu_reference=arguments['relative_scale_lr'] * float(initial[key].mean()))
            for key, rate in rates.items() if key != 'R'
            and rate != arguments['relative_scale_lr'] * float(initial[key].mean())}
        adam_steps = {}
        for group in resume['optimizers'][1]['param_groups']:
            saved_state = resume['optimizers'][1]['state'].get(group['params'][0], {})
            adam_steps[group['parameter_name']] = int(saved_state.get('step', 0))
        learning_rate_errors = []
        for row in training:
            update_index = row['step'] - 1
            factor = ((update_index + 1) / 10 if update_index < 10
                      else 0.5 * (1 + math.cos(math.pi * ((update_index - 10) / 90))))
            for key, rate in rates.items():
                if row['learning_rates'].get(key) != rate * factor:
                    learning_rate_errors.append(dict(step=row['step'], parameter=key))
        completion = [event for event in events if event.get('stage') == 'completed']
        checkpoint_completion = [event for event in events if event.get('stage') == 'checkpoint-completed' and event.get('step') == 100]
        validation_events = [event for event in events if event.get('stage') == 'full-validation' and event.get('step') == 100]
        checks = dict(
            recorded_source=configuration['source']['source_root'] == str(SOURCE),
            recorded_head=configuration['source']['head'] == (evidence / 'head.txt').read_text().strip(),
            original_settings_unchanged=configuration == read_json(PREVIOUS / 'evidence' / name / 'settings.json'),
            data_matches_previous=read_json(destination / 'data.json') == common_data,
            exact_steps=[row['step'] for row in training] == list(range(1, 101)),
            local_steps=[row['step'] for row in local_training] == list(range(3 if route == 'C' else 1, 101)),
            exact_windows=[index for row in training for index in row['window_indices']] == list(range(800)),
            each_update_tokens=all(row['train_tokens'] == 16384 and row['effective_targets'] == 16376 for row in training),
            cumulative_tokens=all(row['cumulative_train_tokens'] == row['step'] * 16384 for row in training),
            schedule=arguments['steps'] == arguments['schedule_steps'] == 100 and arguments['warmup'] == 10,
            learning_rates_match=not learning_rate_errors,
            saved_adam_update_counts=all(count == (50 if route == 'A' else 100)
                                        if '.module.quantizer.scale' in key else
                                        count == (100 if route == 'C' else 0)
                                        if '.mlp.down_proj.quantizer.scale' in key else count == 100
                                        for key, count in adam_steps.items()),
            completed=progress['stage'] == 'completed' and progress['steps'] == 100,
            pid_chain=launch['pid'] == configuration['pid'] == progress['pid'],
            completion_events=len(completion) == len(checkpoint_completion) == len(validation_events) == 1,
            completed_event_matches=len(completion) == 1 and completion[0] == progress,
            validation_event_matches=len(checkpoint_completion) == 1 and checkpoint_completion[0]['ppl'] == result['ppl'] and checkpoint_completion[0]['nll'] == result['nll'],
            no_failure=not (directory / 'failure.json').exists(),
            state_step=state['metadata']['update_step'] == 100 and state['metadata']['training_tokens'] == 1638400,
            resume_step=resume['metadata']['update_step'] == 100,
            resume_parameters_match=all(torch.equal(value, resume['parameters'][key]) for key, value in state['parameters'].items()),
            full_tokens=result['token_count'] == 252852 and result['predicted_tokens'] == 252728 and result['unscored_tail_tokens'] == 0,
            segments=[(row['start_token'], row['seqlen'], row['windows'], row['predicted_tokens']) for row in result['segments']] == [(0, 2048, 123, 251781), (251904, 948, 1, 947)],
            segment_nll=all(row['nll'] == math.log(row['ppl']) for row in result['segments']),
            aggregate=result['nll'] == sum(row['nll'] * row['predicted_tokens'] for row in result['segments']) / 252728 and result['ppl'] == math.exp(result['nll']),
            same_10_100_protocol=all(result[key] == old_result[key] for key in ['evaluation_precision', 'acceptance_source', 'evaluator_source', 'format', 'source_root', 'scale_initialization', 'token_count', 'predicted_tokens']),
            weight_coverage=set(frozen['weights']) == set(expected),
            activation_coverage=set(frozen['activation']) == set(expected),
            activation_formats=Counter(record['format'] for record in frozen['activation'].values()) == {'int8': 96, 'sp2': 16},
            frozen_route_step=frozen['metadata']['route'] == route and frozen['metadata']['step'] == 100,
            weight_shapes=all(tuple(record['shape']) == expected[name] for name, record in frozen['weights'].items()),
            packed_layout=all(record['packed'].dtype == torch.uint8 and record['packed'].numel() * 2 == math.prod(record['shape']) for record in frozen['weights'].values()),
            sw_shapes_and_values=all(record['scale'].dtype == torch.float32 and tuple(record['scale'].shape) == (record['shape'][0], 1)
                                     and bool(torch.isfinite(record['scale']).all()) and bool((record['scale'] > 0).all()) for record in frozen['weights'].values()),
            saved_learned_sw=all(torch.equal(record['scale'], state['parameters'][name + '.module.quantizer.scale']) for name, record in frozen['weights'].items()),
            saved_learned_sa=all(record['format'] == 'int8' and record['scale'].dtype == torch.float32 and tuple(record['scale'].shape) == (1,)
                                and bool(torch.isfinite(record['scale']).all()) and bool((record['scale'] > 0).all())
                                and torch.equal(record['scale'], state['parameters'][name + '.quantizer.scale'])
                                for name, record in frozen['activation'].items() if not name.endswith('down_proj')),
            positive_sp2=all(record['format'] == 'sp2' and math.isfinite(record['alpha']) and record['alpha'] > 0
                             for name, record in frozen['activation'].items() if name.endswith('down_proj')))
        for filename in ['run.py', 'common.py', 'quantization.py']:
            checks['source_snapshot_' + filename] = same_bytes(destination / 'source' / filename, PREVIOUS / 'evidence' / name / 'source' / filename)
        for filename in ['common.py', 'quantization.py']:
            checks['current_source_' + filename] = same_bytes(destination / 'source' / filename, SOURCE / 'experiments/phase3' / filename)
        reload = read_json(checkpoint / 'reload_check.json')
        checks['primary_reload_probe_exact'] = reload['exact'] and reload['before'] == reload['after']
        switches = [event for event in events if event.get('stage') == 'A-switch-to-W4']
        if route == 'A':
            checks['switch_once'] = len(switches) == 1 and switches[0]['step'] == 50
            checks['first50_sw_inactive'] = all(row['updates']['SW']['gradient_tensors'] == 0 and row['updates']['SW']['changed_elements'] == 0 for row in training[:50])
            checks['last50_sw_active'] = all(row['updates']['SW']['gradient_tensors'] == 112 and row['updates']['SW']['changed_elements'] > 0 for row in training[50:])
        else:
            checks['all100_sw_active'] = all(row['updates']['SW']['gradient_tensors'] == 112 and row['updates']['SW']['changed_elements'] > 0 for row in training)
            checks['no_a_switch_event'] = not switches
        if route == 'C':
            checks['all100_sp2_active'] = all(row['updates']['SP2']['gradient_tensors'] == 16 and row['updates']['SP2']['changed_elements'] > 0 for row in training)
            checks['saved_learned_sp2'] = all(record['alpha'] == float(state['parameters'][name + '.quantizer.scale'] * 127) for name, record in frozen['activation'].items() if name.endswith('down_proj'))
            checks['no_export_calibration'] = not (checkpoint / 'sp2_calibration.json').exists()
        else:
            checks['all100_training_down_a16'] = all(row['updates']['SP2']['gradient_tensors'] == 0 and row['updates']['SP2']['changed_elements'] == 0 for row in training)
            calibration = read_json(checkpoint / 'sp2_calibration.json')
            checks['sp2_calibration_coverage'] = set(calibration) == {name for name in expected if name.endswith('down_proj')}
            checks['sp2_selected_minimum'] = all(len(record['candidates']) == 50 and record['selected']['output_mse'] == min(candidate['output_mse'] for candidate in record['candidates']) for record in calibration.values())
            checks['sp2_selected_saved'] = all(frozen['activation'][name]['alpha'] == float(torch.tensor([record['selected']['alpha'] / 127], dtype=torch.float32) * 127) for name, record in calibration.items())
        summary['routes'][route] = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
            command=launch['shell_command'], progress=progress, validation=result, validation10=old_result,
            delta_10_to_100=dict(nll=result['nll'] - old_result['nll'], ppl=result['ppl'] - old_result['ppl']),
            training_tokens=sum(row['train_tokens'] for row in training), training_targets=sum(row['effective_targets'] for row in training),
            local_record_count=len(local_training), combined_record_count=len(training), lr_errors=learning_rate_errors,
            saved_optimizer_initial_lrs=rates, cpu_mean_rounding_differences=cpu_mean_differences,
            saved_adam_update_steps=adam_steps,
            switch_events=switches, selected_update_evidence=[row for row in training if row['step'] in [1, 2, 3, 10, 50, 51, 100]],
            frozen_metadata=frozen['metadata'], frozen_bytes=(checkpoint / 'static_w4a8.pt').stat().st_size,
            frozen_weight_shapes={name: list(record['shape']) for name, record in frozen['weights'].items()},
            activation_summary={name: dict(format=record['format'], value=float(record.get('alpha', record.get('scale')))) for name, record in frozen['activation'].items()})
        print(route, summary['routes'][route]['status'], 'failed:', [key for key, value in checks.items() if not value], flush=True)
        del frozen, state, resume
    summary['paired_checks'] = dict(
        all_100_recorded_lrs_equal=all(trajectories['A'][index]['learning_rates'] == trajectories['B'][index]['learning_rates'] == trajectories['C'][index]['learning_rates'] for index in range(100)),
        all_800_windows_equal=all(trajectories['A'][index]['window_indices'] == trajectories['B'][index]['window_indices'] == trajectories['C'][index]['window_indices'] for index in range(100)))
    summary['ranking'] = sorted(summary['routes'], key=lambda route: summary['routes'][route]['validation']['ppl'])
    best = summary['routes'][summary['ranking'][0]]['validation']
    summary['target'] = dict(status='PASS' if best['ppl'] <= baseline['ppl'] + 1 else 'FAIL',
                            scope='three-route100 stage, not entire goal', reused_bf16=baseline['ppl'], threshold=baseline['ppl'] + 1,
                            delta_ppl=best['ppl'] - baseline['ppl'], delta_nll=best['nll'] - baseline['nll'], gap=best['ppl'] - baseline['ppl'] - 1)
    summary['not_tested'] = ['new independent GPU reproduction of 100-step packages', 'CPU pytest', 'initialization groups final results',
                             'ongoing discrete W4/SP2/D postprocessing', 'final winner package', 'NPU/decode/KV8']
    summary['state'] = 'PAUSED after read-only three-route100 evidence audit'
    write_json(OUTPUT / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ['paired_checks', 'ranking', 'target', 'scope']}, indent=2))


if __name__ == '__main__':
    main()
