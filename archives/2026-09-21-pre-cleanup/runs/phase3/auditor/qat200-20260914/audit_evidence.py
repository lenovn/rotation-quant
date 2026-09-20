from collections import Counter
from datetime import datetime, timezone
import difflib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import torch


ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
RUNS = ROOT / 'runs/phase3'
SOURCE = ROOT / 'worktrees/SpinQuant-phase3-joint'
OUTPUT = Path(__file__).resolve().parent
PRIOR = RUNS / 'auditor/first-validation-20260914'
CENTER = 'distill-b100-d-adam2e5-200-20260914a'
REFERENCE = 'distill-b100-d-adam2e5-ref-200-20260914a'
OTHER = ['distill-b100-d-sgd01-100-20260914a',
         'distill-b100-d-sgd0001-100-20260914a',
         'distill-b100-d-adam2e5-continue800-20260914a']
PARENT = RUNS / 'post-b100-d-20260914a/static_w4a8.pt'


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def events(path):
    found = []
    for number, line in enumerate(path.read_bytes().decode().split('\n'), 1):
        if line.startswith('{'):
            try:
                found.append(dict(line=number, value=json.loads(line)))
            except json.JSONDecodeError:
                pass
    return found


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_cpu(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def same_bytes(left, right):
    return left.read_bytes() == right.read_bytes()


def tensor_equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))


def protocol(result):
    segments = result['segments']
    return dict(tokens=result['token_count'] == 252852 and result['predicted_tokens'] == 252728,
                tail=result['unscored_tail_tokens'] == 0,
                windows=[(record['start_token'], record['seqlen'], record['windows'], record['predicted_tokens'])
                         for record in segments] == [(0, 2048, 123, 251781), (251904, 948, 1, 947)],
                segment_log=all(record['nll'] == math.log(record['ppl']) for record in segments),
                weighted_nll=result['nll'] == sum(record['nll'] * record['predicted_tokens'] for record in segments) / 252728,
                final_exp=result['ppl'] == math.exp(result['nll']),
                evaluator=result['evaluator_source'] == str(SOURCE / 'utils/eval_utils.py'),
                precision=result['evaluation_precision'] == 'existing evaluator: BF16 logits CE, per-token loss to FP32; log(float32 PPL), weighted by predicted tokens')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This audit requires explicitly hidden GPUs')
    torch.set_num_threads(2)
    evidence = OUTPUT / 'evidence'
    evidence.mkdir(exist_ok=True)
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(),
                   role='same independent experiment auditor; CPU evidence only',
                   scope=dict(gpu_runs=0, model_forwards=0, training_updates=0,
                              recalibrations=0, pytest_runs=0, algorithm_searches=0),
                   reused_audits=[str(PRIOR / 'REPORT.md'), str(RUNS / 'auditor/three-routes-100-20260914/REPORT.md')],
                   runs={}, source={}, files={})
    for filename, arguments in [('head.txt', ['rev-parse', 'HEAD']), ('branch.txt', ['branch', '--show-current']),
                                ('status.txt', ['status', '--short']), ('dirty.patch', ['diff', '--binary'])]:
        (evidence / filename).write_bytes(subprocess.check_output(['git', '-C', str(SOURCE), *arguments],
            env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')))
    summary['source']['head'] = (evidence / 'head.txt').read_text().strip()
    summary['source']['prior_head_equal'] = same_bytes(evidence / 'head.txt', PRIOR / 'evidence/source.head')
    summary['source']['prior_dirty_equal'] = same_bytes(evidence / 'dirty.patch', PRIOR / 'evidence/source.dirty.patch')
    summary['source']['acceptance_unchanged'] = same_bytes(ROOT / 'scripts/phase2/validation_acceptance.py', PRIOR / 'evidence/validation_acceptance.py')
    summary['source']['reused_dependencies_unchanged'] = {
        relative: same_bytes(SOURCE / relative, PRIOR / 'evidence/source_snapshot' / relative)
        for relative in ['utils/eval_utils.py', 'eval_utils/modeling_llama.py', 'utils/quant_utils.py',
                         'train_utils/quant_linear.py', 'train_utils/modeling_llama_quant.py', 'train_utils/optimizer.py']}
    old_metadata = read_json(PRIOR / 'evidence/file_metadata.json')
    summary['model_reuse_checks'] = {}
    for filename in ['model.safetensors', 'tokenizer.json']:
        path = ROOT / 'cache/models/llama-3.2-1b-instruct' / filename
        previous = old_metadata[str(path)]
        summary['model_reuse_checks'][filename] = (path.stat().st_size == previous['size'] and path.stat().st_mtime_ns == previous['mtime_ns'])
    for filename in ['.mv', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        summary['model_reuse_checks'][filename] = same_bytes(ROOT / 'cache/models/llama-3.2-1b-instruct' / filename, PRIOR / 'evidence/model_metadata' / filename)
    prior_data = read_json(PRIOR / 'records_audit.json')['data']
    training = {}
    for name in [CENTER, REFERENCE, *OTHER]:
        directory = RUNS / name
        destination = evidence / name
        destination.mkdir(exist_ok=True)
        for path in [*directory.glob('*.json'), *directory.glob('checkpoint-*/*.json'),
                     RUNS / (name + '.launch.json'), RUNS / (name + '.log')]:
            relative = path.relative_to(directory) if path.is_relative_to(directory) else Path(path.name)
            target = destination / relative
            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(path, target)
        shutil.copytree(directory / 'source', destination / 'source', dirs_exist_ok=True)
        for path in [directory / 'training.jsonl', directory / 'resume.pt', *directory.glob('checkpoint-*/*.pt')]:
            stat = path.stat()
            summary['files'][str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        training[name] = rows(directory / 'training.jsonl')
        configuration = read_json(directory / 'settings.json')
        arguments = configuration['arguments']
        launch = read_json(RUNS / (name + '.launch.json'))
        recorded_events = events(RUNS / (name + '.log'))
        write_json(destination / 'events.json', recorded_events)
        resume = load_cpu(directory / 'resume.pt')
        rate_groups = [group for optimizer in resume['optimizers'] for group in optimizer['param_groups']]
        initial_rates = {group.get('parameter_name', group['name']): group['initial_lr'] for group in rate_groups}
        lr_errors = []
        window_errors = []
        for row in training[name]:
            update = row['step'] - 1
            total = arguments['schedule_steps'] or arguments['steps']
            multiplier = (update + 1) / arguments['warmup'] if update < arguments['warmup'] else 0.5 * (1 + math.cos(math.pi * ((update - arguments['warmup']) / (total - arguments['warmup']))))
            if row['learning_rates'] != {key: value * multiplier for key, value in initial_rates.items()}:
                lr_errors.append(row['step'])
            expected_windows = [(arguments['data_start'] + update * arguments['accumulation'] + microbatch) % 1180 for microbatch in range(8)]
            if [microbatch['window'] for microbatch in row['microbatches']] != expected_windows:
                window_errors.append(row['step'])
        run_summary = dict(settings=configuration, launch=launch,
            source_snapshot_equal_current={path.name: same_bytes(path, SOURCE / 'experiments/phase3' / path.name) for path in (directory / 'source').glob('*.py')},
            dirty_equal_current=same_bytes(directory / 'source/tracked.diff', evidence / 'dirty.patch'),
            data_reuse_equal=read_json(directory / 'data.json') == prior_data,
            recorded_source_and_head=configuration['source']['source_root'] == str(SOURCE) and configuration['source']['head'] == summary['source']['head'],
            launch_pid_match=launch['pid'] == configuration['pid'],
            training_steps=[row['step'] for row in training[name]], resume_metadata=resume['metadata'],
            learning_rate_errors=lr_errors, window_errors=window_errors, initial_rates=initial_rates,
            tokens_this_directory=len(training[name]) * 8 * 2048, targets_this_directory=len(training[name]) * 8 * 2047,
            all_microbatch_targets=all(len(row['microbatches']) == 8 and all(micro['predicted_tokens'] == 2047 for micro in row['microbatches']) for row in training[name]),
            all_cumulative_tokens=all(row['cumulative_train_tokens'] == row['step'] * 8 * 2048 for row in training[name]),
            all_gradient_coverage=all(row['gradient_tensors'] == dict(W=112, SA=96, SW=112, SP2=16) for row in training[name]),
            all_finite_training=all(math.isfinite(row[key]) for row in training[name] for key in ['objective', 'ce', 'kl', 'gradient_norm_before_clip']),
            master_tensors_sampled_every_step=all(len(row['sampled_master_weight_changes']) == 112 for row in training[name]),
            sampled_changed_master_tensors_range=[min(sum(value > 0 for value in row['sampled_master_weight_changes'].values()) for row in training[name]), max(sum(value > 0 for value in row['sampled_master_weight_changes'].values()) for row in training[name])],
            optimizer_saved_state_counts=[len(optimizer['state']) for optimizer in resume['optimizers']],
            optimizer_saved_steps=[dict(Counter(int(state['step']) for state in optimizer['state'].values() if 'step' in state)) for optimizer in resume['optimizers']],
            resume_parameter_count=len(resume['parameters']),
            resume_parameters_fp32=all(parameter.dtype == torch.float32 for parameter in resume['parameters'].values()),
            progress=read_json(directory / 'progress.json'),
            validation={str(path.relative_to(directory)): dict(result=read_json(path), checks=protocol(read_json(path))) for path in directory.glob('checkpoint-*/validation.json')})
        summary['runs'][name] = run_summary
        del resume
    center_rows, reference_rows = training[CENTER], training[REFERENCE]
    summary['pair'] = dict(exact_200_steps=all([row['step'] for row in training[name]] == list(range(1, 201)) for name in [CENTER, REFERENCE]),
        initial_probe_equal=read_json(RUNS / CENTER / 'initial_probe.json') == read_json(RUNS / REFERENCE / 'initial_probe.json'),
        initial_rates_equal=summary['runs'][CENTER]['initial_rates'] == summary['runs'][REFERENCE]['initial_rates'],
        learning_rates_equal_every_update=all(left['learning_rates'] == right['learning_rates'] for left, right in zip(center_rows, reference_rows)),
        learning_rate_values_compared=sum(len(row['learning_rates']) for row in center_rows),
        windows_equal_every_update=all([micro['window'] for micro in left['microbatches']] == [micro['window'] for micro in right['microbatches']] for left, right in zip(center_rows, reference_rows)),
        first_update_microbatch_losses_equal=center_rows[0]['microbatches'] == reference_rows[0]['microbatches'],
        first_update_gradient_norms=[center_rows[0]['gradient_norm_before_clip'], reference_rows[0]['gradient_norm_before_clip']],
        source_equal={path.name: same_bytes(path, RUNS / REFERENCE / 'source' / path.name) for path in (RUNS / CENTER / 'source').iterdir()},
        unique_windows=len({micro['window'] for row in center_rows for micro in row['microbatches']}),
        first_windows=[micro['window'] for micro in center_rows[0]['microbatches']],
        last_windows=[micro['window'] for micro in center_rows[-1]['microbatches']])
    (evidence / 'center-to-reference.distill.diff').write_text(''.join(difflib.unified_diff(
        (RUNS / CENTER / 'source/distill.py').read_text().splitlines(True),
        (RUNS / REFERENCE / 'source/distill.py').read_text().splitlines(True), fromfile=CENTER + '/source/distill.py', tofile=REFERENCE + '/source/distill.py')))
    print('Completed command/source/trajectory checks; now CPU packed tensors', flush=True)
    parent = load_cpu(PARENT)
    shapes = dict(q_proj=(2048, 2048), k_proj=(512, 2048), v_proj=(512, 2048), o_proj=(2048, 2048),
                  gate_proj=(8192, 2048), up_proj=(8192, 2048), down_proj=(2048, 8192))
    expected = {f'model.layers.{layer}.{family}.{suffix}': shape for layer in range(16) for suffix, shape in shapes.items()
                for family in ['self_attn' if suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'mlp']}
    for name in [CENTER, REFERENCE]:
        package = load_cpu(RUNS / name / 'checkpoint-0200/static_w4a8.pt')
        resume = load_cpu(RUNS / name / 'resume.pt')
        changes = {}
        master_mismatches = {}
        center_initial_mismatches = {}
        for key, record in package['weights'].items():
            packed = record['packed'].reshape(-1)
            parent_packed = parent['weights'][key]['packed'].reshape(-1)
            changed = 0
            for start in range(0, packed.numel(), 1024 * 1024):
                final_chunk = packed[start:start + 1024 * 1024]
                parent_chunk = parent_packed[start:start + 1024 * 1024]
                changed += int(((final_chunk & 15) != (parent_chunk & 15)).sum())
                changed += int(((final_chunk >> 4) != (parent_chunk >> 4)).sum())
            changes[key] = dict(changed_codes=changed, total_codes=math.prod(record['shape']))
            master = resume['parameters'][key + '.module.weight']
            master_errors = 0
            center_errors = 0
            columns = record['shape'][1]
            for start in range(0, record['shape'][0], 128):
                stop = min(start + 128, record['shape'][0])
                chunk = packed[start * columns // 2:stop * columns // 2]
                codes = torch.stack((chunk & 15, chunk >> 4), dim=1).reshape(stop - start, columns).to(torch.int16) - 8
                master_codes = (master[start:stop] / record['scale'][start:stop]).round().clamp(-8, 7).to(torch.int16)
                master_errors += int((master_codes != codes).sum())
                if name == CENTER:
                    parent_chunk = parent_packed[start * columns // 2:stop * columns // 2]
                    parent_codes = torch.stack((parent_chunk & 15, parent_chunk >> 4), dim=1).reshape(stop - start, columns).to(torch.int16) - 8
                    parent_scale = parent['weights'][key]['scale'][start:stop]
                    recovered = ((parent_codes.float() * parent_scale).to(torch.bfloat16).float() / parent_scale).round().clamp(-8, 7).to(torch.int16)
                    center_errors += int((recovered != parent_codes).sum())
            master_mismatches[key] = master_errors
            if name == CENTER:
                center_initial_mismatches[key] = center_errors
        expected_parameters = {key + suffix for key in expected for suffix in ['.module.weight', '.module.quantizer.scale', '.quantizer.scale']}
        pack_checks = dict(weight_coverage=set(package['weights']) == set(expected),
            activation_coverage=set(package['activation']) == set(expected),
            resume_parameter_coverage=set(resume['parameters']) == expected_parameters,
            weight_shapes=all(tuple(record['shape']) == expected[key] for key, record in package['weights'].items()),
            packed_layout=all(record['packed'].dtype == torch.uint8 and record['packed'].numel() * 2 == math.prod(record['shape']) for record in package['weights'].values()),
            positive_fp32_sw=all(record['scale'].dtype == torch.float32 and tuple(record['scale'].shape) == (record['shape'][0], 1) and bool(torch.isfinite(record['scale']).all()) and bool((record['scale'] > 0).all()) for record in package['weights'].values()),
            sw_matches_resume=all(torch.equal(record['scale'], resume['parameters'][key + '.module.quantizer.scale']) for key, record in package['weights'].items()),
            activation_formats=Counter(record['format'] for record in package['activation'].values()) == dict(int8=96, sp2=16),
            sa_matches_resume=all(record['scale'].dtype == torch.float32 and tuple(record['scale'].shape) == (1,) and bool(torch.isfinite(record['scale']).all()) and bool((record['scale'] > 0).all()) and torch.equal(record['scale'], resume['parameters'][key + '.quantizer.scale']) for key, record in package['activation'].items() if not key.endswith('down_proj')),
            sp2_matches_resume=all(math.isfinite(record['alpha']) and record['alpha'] > 0 and record['alpha'] == float(resume['parameters'][key + '.quantizer.scale'] * 127) for key, record in package['activation'].items() if key.endswith('down_proj')),
            high_precision_keys_unchanged=set(package['high_precision']) == set(parent['high_precision']),
            high_precision_bytes_unchanged=all(tensor_equal(value, parent['high_precision'][key]) for key, value in package['high_precision'].items()),
            exported_codes_match_final_fp32_masters=not any(master_mismatches.values()),
            recorded_changes_exact=changes == read_json(RUNS / name / 'checkpoint-0200/code_changes.json'))
        summary['runs'][name]['packed'] = dict(checks=pack_checks, metadata=package['metadata'],
            total_codes=sum(record['total_codes'] for record in changes.values()),
            changed_codes=sum(record['changed_codes'] for record in changes.values()),
            master_export_mismatches=master_mismatches, center_initial_requantization_mismatches=center_initial_mismatches,
            high_precision_tensors=len(package['high_precision']))
        write_json(evidence / name / 'independent_cpu_code_changes.json', changes)
        print(name, 'packed checks', pack_checks, flush=True)
        del package, resume
    summary['parent'] = dict(metadata=parent['metadata'],
        result=read_json(PARENT.parent / 'result.json'),
        reference_initialization=read_json(RUNS / REFERENCE / 'master_initialization.json'))
    reference_state = load_cpu(Path(parent['metadata']['reference_state']))
    summary['parent']['reference_state_metadata'] = reference_state['metadata']
    del reference_state
    b100 = load_cpu(Path(parent['metadata']['parent']))
    summary['parent']['changed_packed_weight_names'] = [key for key in expected if not torch.equal(parent['weights'][key]['packed'], b100['weights'][key]['packed'])]
    summary['parent']['changed_sw_names'] = [key for key in expected if not torch.equal(parent['weights'][key]['scale'], b100['weights'][key]['scale'])]
    summary['parent']['changed_activation_names'] = [key for key in expected if (parent['activation'][key]['alpha'] != b100['activation'][key]['alpha'] if key.endswith('down_proj') else not torch.equal(parent['activation'][key]['scale'], b100['activation'][key]['scale']))]
    summary['parent']['high_precision_unchanged'] = set(parent['high_precision']) == set(b100['high_precision']) and all(tensor_equal(value, b100['high_precision'][key]) for key, value in parent['high_precision'].items())
    del parent, b100
    archive = read_json(RUNS / REFERENCE / 'recovery_evaluation.json')
    reference_events = events(RUNS / (REFERENCE + '.log'))
    recovery_events = {record['value']['event']: record for record in reference_events if record['value'].get('event', '').startswith('recovery-evaluation-')}
    recovery = archive['recovery']
    summary['recovery'] = dict(archive_matches_log=all(recovery[key] == recovery_events['recovery-evaluation-' + suffix]['value'] for key, suffix in [('preflight', 'preflight'), ('start', 'start'), ('completed', 'completed')]),
        full_result_matches_log=recovery['completed']['result'] == read_json(RUNS / REFERENCE / 'checkpoint-0200/validation.json'),
        data_matches=recovery['completed']['data'] == prior_data,
        pid_chain=recovery['start']['pid'] == recovery['completed']['pid'] == 1493471,
        command_matches=recovery['preflight']['command'][-4:] == recovery['start']['command'],
        snapshot_comparison=recovery['start']['source_snapshot_equal'],
        log_lines={key: record['line'] for key, record in recovery_events.items()},
        original_driver=archive['original_driver'],
        original_full_validation_steps=[record['value']['step'] for record in reference_events if record['value'].get('stage') == 'full-validation'],
        preserved_checkpoint_ledger_steps=[record['step'] for record in read_json(RUNS / REFERENCE / 'checkpoints.json')])
    summary['artifact_metadata_stable'] = all(Path(path).stat().st_size == metadata['size'] and Path(path).stat().st_mtime_ns == metadata['mtime_ns'] for path, metadata in summary['files'].items())
    summary['cuda_runtime_initialized'] = torch.cuda.is_initialized()
    write_json(OUTPUT / 'summary.json', summary)
    print('CPU audit saved:', OUTPUT / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
