from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
RUNS = ROOT / 'runs/phase3'
SOURCE = ROOT / 'worktrees/SpinQuant-phase3-joint'
OUTPUT = Path(__file__).resolve().parent
FIRST = RUNS / 'auditor/first-validation-20260914'
PREVIOUS = RUNS / 'auditor/qat200-20260914'
BEST = 'distill-b100-refined-ref-adam1e5-400-20260914a'
CONTROL = 'distill-b100-refined-ref-adam2e5-400-20260914a'
sys.path.insert(0, str(PREVIOUS))

import torch
from audit_evidence import events, load_cpu, protocol, read_json, same_bytes, tensor_equal, write_json


def metadata(path):
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Evidence scan is CPU-only')
    torch.set_num_threads(2)
    evidence = OUTPUT / 'evidence'
    evidence.mkdir(exist_ok=True)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), role='independent final experiment audit',
                  runs={}, source={}, reused_evidence=[str(FIRST / 'REPORT.md'), str(PREVIOUS / 'REPORT.md')], files={})
    for filename, arguments in [('head.txt', ['rev-parse', 'HEAD']), ('branch.txt', ['branch', '--show-current']),
                                ('status.txt', ['status', '--short']), ('dirty.patch', ['diff', '--binary'])]:
        (evidence / filename).write_bytes(subprocess.check_output(['git', '-C', str(SOURCE), *arguments],
            env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')))
    report['source']['head'] = (evidence / 'head.txt').read_text().strip()
    report['source']['prior_dirty_equal'] = same_bytes(evidence / 'dirty.patch', FIRST / 'evidence/source.dirty.patch')
    report['source']['acceptance_unchanged'] = same_bytes(ROOT / 'scripts/phase2/validation_acceptance.py', FIRST / 'evidence/validation_acceptance.py')
    report['source']['dependencies_unchanged'] = {relative: same_bytes(SOURCE / relative, FIRST / 'evidence/source_snapshot' / relative)
        for relative in ['utils/eval_utils.py', 'eval_utils/modeling_llama.py', 'utils/quant_utils.py', 'train_utils/quant_linear.py', 'train_utils/modeling_llama_quant.py']}
    old_files = read_json(FIRST / 'evidence/file_metadata.json')
    report['model_reuse_checks'] = {filename: metadata(ROOT / 'cache/models/llama-3.2-1b-instruct' / filename) == old_files[str(ROOT / 'cache/models/llama-3.2-1b-instruct' / filename)]
        for filename in ['model.safetensors', 'tokenizer.json']}
    for filename in ['.mv', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        report['model_reuse_checks'][filename] = same_bytes(ROOT / 'cache/models/llama-3.2-1b-instruct' / filename, FIRST / 'evidence/model_metadata' / filename)
    old_data = read_json(FIRST / 'records_audit.json')['data']
    trajectories = {}
    for name in [BEST, CONTROL]:
        directory = RUNS / name
        destination = evidence / name
        destination.mkdir(exist_ok=True)
        for path in [*directory.glob('*.json'), *directory.glob('checkpoint-*/*.json')]:
            target = destination / path.relative_to(directory)
            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(path, target)
        shutil.copytree(directory / 'source', destination / 'source', dirs_exist_ok=True)
        for suffix in ['.launch.json', '.log']:
            shutil.copy2(RUNS / (name + suffix), destination / (name + suffix))
        recorded_events = events(RUNS / (name + '.log'))
        write_json(destination / 'events.json', recorded_events)
        training = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines()]
        trajectories[name] = training
        config = read_json(directory / 'settings.json')
        args = config['arguments']
        launch = read_json(RUNS / (name + '.launch.json'))
        resume = load_cpu(directory / 'resume.pt')
        package_path = directory / 'checkpoint-0400/static_w4a8.pt'
        package = load_cpu(package_path)
        for path in [directory / 'training.jsonl', directory / 'resume.pt', package_path, Path(args['reference_state']), Path(args['parent'])]:
            report['files'][str(path)] = metadata(path)
        rates = {group.get('parameter_name', group['name']): group['initial_lr'] for optimizer in resume['optimizers'] for group in optimizer['param_groups']}
        lr_errors = []
        window_errors = []
        for row in training:
            update = row['step'] - 1
            factor = (update + 1) / 10 if update < 10 else 0.5 * (1 + math.cos(math.pi * ((update - 10) / 390)))
            if row['learning_rates'] != {key: value * factor for key, value in rates.items()}:
                lr_errors.append(row['step'])
            if [micro['window'] for micro in row['microbatches']] != [(800 + update * 8 + micro) % 1180 for micro in range(8)]:
                window_errors.append(row['step'])
        completion = [record['value'] for record in recorded_events if record['value'].get('stage') == 'completed']
        progress = read_json(directory / 'progress.json')
        validation = read_json(directory / 'checkpoint-0400/validation.json')
        result = read_json(directory / 'result.json')
        checks = dict(source_root=config['source']['source_root'] == str(SOURCE), source_head=config['source']['head'] == report['source']['head'],
            dirty_matches=same_bytes(directory / 'source/tracked.diff', evidence / 'dirty.patch'),
            distill_matches_previous_audited=same_bytes(directory / 'source/distill.py', PREVIOUS / 'evidence/distill-b100-d-adam2e5-ref-200-20260914a/source/distill.py'),
            data_equal_prior=read_json(directory / 'data.json') == old_data,
            fresh_run=args['resume'] is None,
            exact_steps=[row['step'] for row in training] == list(range(1, 401)),
            all_microbatch_targets=all(len(row['microbatches']) == 8 and all(micro['predicted_tokens'] == 2047 for micro in row['microbatches']) for row in training),
            all_cumulative_tokens=all(row['cumulative_train_tokens'] == row['step'] * 16384 for row in training),
            gradient_coverage=all(row['gradient_tensors'] == dict(W=112, SA=96, SW=112, SP2=16) for row in training),
            sampled_all_masters_change_each_step=all(len(row['sampled_master_weight_changes']) == 112 and all(value > 0 for value in row['sampled_master_weight_changes'].values()) for row in training),
            finite_losses=all(math.isfinite(row[key]) for row in training for key in ['objective', 'ce', 'kl', 'gradient_norm_before_clip']),
            lr_schedule=not lr_errors, window_sequence=not window_errors,
            resume_step=resume['metadata']['step'] == 400,
            adam_state_steps=all(len(optimizer['state']) == count and all(int(state['step']) == 400 for state in optimizer['state'].values()) for optimizer, count in zip(resume['optimizers'], [112, 224])),
            resume_fp32=len(resume['parameters']) == 336 and all(parameter.dtype == torch.float32 for parameter in resume['parameters'].values()),
            completed=progress['stage'] == 'completed' and progress['steps'] == result['completed_steps'] == 400,
            completion_event=completion == [progress], pid_chain=launch['pid'] == config['pid'] == progress['pid'],
            final_result_matches=result['checkpoints'][-1]['validation_ppl'] == validation['ppl'] and result['checkpoints'][-1]['validation_nll'] == validation['nll'],
            no_failure=not (directory / 'failure.json').exists(),
            full_protocol=all(protocol(validation).values()),
            final_sw_matches_resume=all(torch.equal(record['scale'], resume['parameters'][key + '.module.quantizer.scale']) for key, record in package['weights'].items()),
            final_sa_matches_resume=all(torch.equal(record['scale'], resume['parameters'][key + '.quantizer.scale']) for key, record in package['activation'].items() if not key.endswith('down_proj')),
            final_sp2_matches_resume=all(record['alpha'] == float(resume['parameters'][key + '.quantizer.scale'] * 127) for key, record in package['activation'].items() if key.endswith('down_proj')))
        record = dict(checks=checks, arguments=args, launch=launch, initial_rates=rates,
            source_snapshot_equal_current={path.name: same_bytes(path, SOURCE / 'experiments/phase3' / path.name) for path in (directory / 'source').glob('*.py')},
            progress=progress, resume_metadata=resume['metadata'], package_metadata=package['metadata'], parameter_coverage=read_json(directory / 'parameter_coverage.json'),
            validation=validation, full_validation_steps=[row['value']['step'] for row in recorded_events if row['value'].get('stage') == 'full-validation'],
            train_tokens=400 * 8 * 2048, train_targets=400 * 8 * 2047, lr_errors=lr_errors, window_errors=window_errors,
            first_windows=[micro['window'] for micro in training[0]['microbatches']], last_windows=[micro['window'] for micro in training[-1]['microbatches']],
            unique_windows=len({micro['window'] for row in training for micro in row['microbatches']}),
            initial_probe=read_json(directory / 'initial_probe.json'), master_initialization=read_json(directory / 'master_initialization.json'))
        report['runs'][name] = record
        del resume, package
    best_rows, control_rows = trajectories[BEST], trajectories[CONTROL]
    report['pair'] = dict(same_windows=all([micro['window'] for micro in left['microbatches']] == [micro['window'] for micro in right['microbatches']] for left, right in zip(best_rows, control_rows)),
        same_scale_LRs=all({key: value for key, value in left['learning_rates'].items() if key != 'W'} == {key: value for key, value in right['learning_rates'].items() if key != 'W'} for left, right in zip(best_rows, control_rows)),
        weight_LR_exact_half=all(left['learning_rates']['W'] == right['learning_rates']['W'] / 2 for left, right in zip(best_rows, control_rows)),
        scale_LR_pairs_checked=400 * 224, weight_LR_pairs_checked=400,
        same_initial_probe=report['runs'][BEST]['initial_probe'] == report['runs'][CONTROL]['initial_probe'],
        same_master_initialization=report['runs'][BEST]['master_initialization'] == report['runs'][CONTROL]['master_initialization'],
        same_arguments_except_weightLR_output={key: value for key, value in report['runs'][BEST]['arguments'].items() if key not in ['weight_lr', 'output']} == {key: value for key, value in report['runs'][CONTROL]['arguments'].items() if key not in ['weight_lr', 'output']},
        same_source_snapshots={path.name: same_bytes(path, RUNS / CONTROL / 'source' / path.name) for path in (RUNS / BEST / 'source').iterdir()},
        first_microbatch_losses_equal=best_rows[0]['microbatches'] == control_rows[0]['microbatches'],
        identical_gradients_or_learned_scales_claimed=False)
    print('400-step trajectories and LR pairing audited; reading final packed tensors', flush=True)
    package = load_cpu(RUNS / BEST / 'checkpoint-0400/static_w4a8.pt')
    parent_path = Path(report['runs'][BEST]['arguments']['parent'])
    parent = load_cpu(parent_path)
    resume = load_cpu(RUNS / BEST / 'resume.pt')
    shapes = dict(q_proj=(2048, 2048), k_proj=(512, 2048), v_proj=(512, 2048), o_proj=(2048, 2048), gate_proj=(8192, 2048), up_proj=(8192, 2048), down_proj=(2048, 8192))
    expected = {f'model.layers.{layer}.{family}.{suffix}': shape for layer in range(16) for suffix, shape in shapes.items()
                for family in ['self_attn' if suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'mlp']}
    changes = {}
    master_errors = {}
    for key, record in package['weights'].items():
        packed = record['packed'].reshape(-1)
        previous = parent['weights'][key]['packed'].reshape(-1)
        changed = 0
        for start in range(0, packed.numel(), 1024 * 1024):
            current_chunk, old_chunk = packed[start:start + 1024 * 1024], previous[start:start + 1024 * 1024]
            changed += int(((current_chunk & 15) != (old_chunk & 15)).sum()) + int(((current_chunk >> 4) != (old_chunk >> 4)).sum())
        changes[key] = dict(changed_codes=changed, total_codes=math.prod(record['shape']))
        errors = 0
        columns = record['shape'][1]
        for start in range(0, record['shape'][0], 128):
            stop = min(start + 128, record['shape'][0])
            chunk = packed[start * columns // 2:stop * columns // 2]
            codes = torch.stack((chunk & 15, chunk >> 4), dim=1).reshape(stop - start, columns).to(torch.int16) - 8
            master_codes = (resume['parameters'][key + '.module.weight'][start:stop] / record['scale'][start:stop]).round().clamp(-8, 7).to(torch.int16)
            errors += int((codes != master_codes).sum())
        master_errors[key] = errors
    report['packed'] = dict(checks=dict(weight_coverage=set(package['weights']) == set(expected),
        activation_coverage=set(package['activation']) == set(expected),
        shape_and_layout=all(tuple(record['shape']) == expected[key] and record['packed'].dtype == torch.uint8 and record['packed'].numel() * 2 == math.prod(record['shape']) for key, record in package['weights'].items()),
        sw_format=all(record['scale'].dtype == torch.float32 and tuple(record['scale'].shape) == (record['shape'][0], 1) and bool(torch.isfinite(record['scale']).all()) and bool((record['scale'] > 0).all()) for record in package['weights'].values()),
        activation_formats=Counter(record['format'] for record in package['activation'].values()) == dict(int8=96, sp2=16),
        HP_keys=set(package['high_precision']) == set(parent['high_precision']),
        HP_bytes_unchanged=all(tensor_equal(value, parent['high_precision'][key]) for key, value in package['high_precision'].items()),
        all_codes_match_final_masters=not any(master_errors.values()),
        code_changes_match=changes == read_json(RUNS / BEST / 'checkpoint-0400/code_changes.json')),
        changed_codes=sum(record['changed_codes'] for record in changes.values()), total_codes=sum(record['total_codes'] for record in changes.values()), master_errors=master_errors)
    write_json(evidence / 'independent_cpu_code_changes.json', changes)
    del package, parent, resume
    report['parent_chain'] = []
    while True:
        frozen = load_cpu(parent_path)
        package_metadata = frozen['metadata']
        del frozen
        directory = parent_path.parent
        checkpoint = directory.name.startswith('checkpoint-')
        run_directory = directory.parent if checkpoint else directory
        destination = evidence / run_directory.name
        destination.mkdir(exist_ok=True)
        for filename in ['settings.json', 'data.json', 'result.json', 'initial_train_selection.json']:
            if (run_directory / filename).exists():
                shutil.copy2(run_directory / filename, destination / filename)
        if (run_directory / 'source').exists() and not checkpoint:
            shutil.copytree(run_directory / 'source', destination / 'source', dirs_exist_ok=True)
        if (RUNS / (run_directory.name + '.launch.json')).exists():
            shutil.copy2(RUNS / (run_directory.name + '.launch.json'), destination / 'launch.json')
        measured = read_json(directory / 'validation.json')
        shutil.copy2(directory / 'validation.json', destination / 'validation.json')
        report['parent_chain'].append(dict(path=str(parent_path), metadata=package_metadata, validation=measured,
            protocol=protocol(measured), settings=read_json(run_directory / 'settings.json')))
        if 'parent' not in package_metadata:
            break
        parent_path = Path(package_metadata['parent'])
    reference = load_cpu(Path(report['runs'][BEST]['arguments']['reference_state']))
    report['reference_state_metadata'] = reference['metadata']
    del reference
    report['files_stable_during_audit'] = all(metadata(Path(path)) == value for path, value in report['files'].items())
    report['evidence_scan_cuda_initialized'] = torch.cuda.is_initialized()
    shutil.copy2(RUNS / 'STATUS.md', evidence / 'phase3-STATUS.md')
    shutil.copy2(RUNS / 'handoff-20260914.NByJvb/HANDOFF.md', evidence / 'HANDOFF.md')
    write_json(OUTPUT / 'records.json', report)
    print(json.dumps(dict(runs={key: value['checks'] for key, value in report['runs'].items()}, pair=report['pair'], packed=report['packed']['checks']), indent=2), flush=True)


if __name__ == '__main__':
    main()
