from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
SOURCE = ROOT / 'worktrees/SpinQuant-phase3-joint'
OUTPUT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/phase3'
sys.path.insert(0, str(SOURCE))

import torch

torch.set_num_threads(4)


def read_json(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parameters(path):
    return torch.load(path, map_location='cpu', weights_only=True)['parameters']


def audit():
    initial = parameters(RUNS / 'common-init-20260914a/initial.pt')
    common_data = read_json(RUNS / 'common-init-20260914a/data.json')
    report = {}
    trajectories = {}
    settings = {}
    for name in ['route-a-adam-100-20260914a', 'route-b-adam-100-20260914a',
                 'route-c-adam-100-20260914a', 'route-c-sgd-10-20260914a']:
        directory = RUNS / name
        checkpoint = directory / 'checkpoint-0010'
        configuration = read_json(directory / 'settings.json')
        settings[name] = configuration
        arguments = configuration['arguments']
        source = configuration['source']
        launch = read_json(RUNS / (name + '.launch.json'))
        result = read_json(checkpoint / 'validation.json')
        training = records(directory / 'training.jsonl')
        origin = directory
        if arguments['resume']:
            origin = Path(arguments['resume'])
            training = records(origin / 'training.jsonl') + training
        training = [row for row in training if row['step'] <= 10]
        trajectories[name] = training
        initial_actual = parameters(origin / 'checkpoint-0000/state.pt')
        state = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=True)
        package = torch.load(checkpoint / 'static_w4a8.pt', map_location='cpu', weights_only=True)
        original_log = (RUNS / (name + '.log')).read_text()
        events = []
        for line in original_log.splitlines():
            if line.startswith('{'):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        completed = [row for row in events if row.get('stage') == 'checkpoint-completed'
                     and row.get('step') == 10]
        checks = dict(
            source_root=source['source_root'] == str(SOURCE),
            revision=source['head'] == '24918316ed594848d4de797c356b120f2a4ee0f3',
            launch_pid=launch['pid'] == configuration['pid'],
            data_metadata=read_json(directory / 'data.json') == common_data,
            common_initial_keys=set(initial_actual) == set(initial),
            common_initial_parameters=all(torch.equal(initial[key], initial_actual[key]) for key in initial),
            exact_update_sequence=[row['step'] for row in training] == list(range(1, 11)),
            exact_train_windows=[index for row in training for index in row['window_indices']] == list(range(80)),
            train_tokens=all(row['train_tokens'] == 16384 and row['effective_targets'] == 16376 for row in training),
            total_train_tokens=training[-1]['cumulative_train_tokens'] == 163840,
            total_schedule=arguments['schedule_steps'] == 100,
            full_validation=result['token_count'] == 252852 and result['predicted_tokens'] == 252728,
            segments=[(row['seqlen'], row['windows'], row['predicted_tokens']) for row in result['segments']]
                     == [(2048, 123, 251781), (948, 1, 947)],
            weighted_nll=result['nll'] == sum(row['nll'] * row['predicted_tokens'] for row in result['segments']) / 252728,
            ppl=math.exp(result['nll']) == result['ppl'],
            complete_log=len(completed) == 1 and completed[0]['ppl'] == result['ppl'],
            no_failure_file=not (directory / 'failure.json').exists(),
            package_metadata=package['metadata']['step'] == 10 and package['metadata']['route'] == arguments['route'],
            weights_count=len(package['weights']) == 112,
            activation_formats=Counter(row['format'] for row in package['activation'].values()) == {'int8': 96, 'sp2': 16},
            packed_weight_shapes=all(record['packed'].dtype == torch.uint8
                                    and record['packed'].numel() * 2 == math.prod(record['shape'])
                                    and record['scale'].shape == (record['shape'][0], 1)
                                    and bool(torch.isfinite(record['scale']).all() and (record['scale'] > 0).all())
                                    for record in package['weights'].values()),
            non_down_sa_saved=all(torch.equal(record['scale'], state['parameters'][module + '.quantizer.scale'])
                                  for module, record in package['activation'].items() if record['format'] == 'int8'),
            runtime_source_common_matches=(directory / 'source/common.py').read_bytes() == (SOURCE / 'experiments/phase3/common.py').read_bytes(),
            runtime_source_quantization_matches=(directory / 'source/quantization.py').read_bytes() == (SOURCE / 'experiments/phase3/quantization.py').read_bytes(),
        )
        if arguments['route'] != 'A':
            checks['learned_sw_saved'] = all(torch.equal(record['scale'], state['parameters'][module + '.module.quantizer.scale'])
                                              for module, record in package['weights'].items())
        else:
            checks['a_sw_untrained_before_switch'] = all(torch.equal(state['parameters'][key], initial[key])
                                                         for key in initial if '.module.quantizer.scale' in key)
        if arguments['route'] == 'C':
            checks['learned_sp2_saved'] = all(record['alpha'] == float(state['parameters'][module + '.quantizer.scale'] * 127)
                                              for module, record in package['activation'].items() if record['format'] == 'sp2')
            checks['no_export_recalibration'] = not (checkpoint / 'sp2_calibration.json').exists()
        else:
            calibration = read_json(checkpoint / 'sp2_calibration.json')
            checks['train_only_sp2_export'] = len(calibration) == 16
            checks['sp2_selects_output_mse'] = all(record['selected']['output_mse'] == min(row['output_mse'] for row in record['candidates'])
                                                  and len(record['candidates']) == 50 for record in calibration.values())
            checks['sp2_selected_range_saved'] = all(package['activation'][module]['alpha'] ==
                float(torch.tensor([record['selected']['alpha'] / 127], dtype=torch.float32) * 127)
                for module, record in calibration.items())
        gradients = {group: dict(gradient_tensors=[row['updates'][group]['gradient_tensors'] for row in training],
                                 changed_elements=[row['updates'][group]['changed_elements'] for row in training])
                     for group in ('R', 'SA', 'SW', 'SP2')}
        report[name] = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                            validation=result, command=launch['shell_command'], pid=launch['pid'], gpu=launch['gpu'],
                            resume_parent=str(origin) if origin != directory else None,
                            checkpoint_state_metadata=state['metadata'], gradients=gradients,
                            cumulative_training_targets=sum(row['effective_targets'] for row in training),
                            completed_event=completed, package_bytes=(checkpoint / 'static_w4a8.pt').stat().st_size)
        del package, state, initial_actual
    first, second, third = list(trajectories)[:3]
    pairs = {}
    for left, right in [(first, second), (second, third)]:
        settings_left = settings[left]['arguments']
        settings_right = settings[right]['arguments']
        differences = {key: [settings_left.get(key), settings_right.get(key)]
                       for key in set(settings_left) | set(settings_right)
                       if settings_left.get(key) != settings_right.get(key)}
        shared_rates = all(row_left['learning_rates'] == row_right['learning_rates']
                           for row_left, row_right in zip(trajectories[left], trajectories[right]))
        pairs[left + ' vs ' + right] = dict(argument_differences=differences,
                                            all_recorded_learning_rates_equal=shared_rates,
                                            identical_sample_sequence=all(row_left['window_indices'] == row_right['window_indices']
                                                for row_left, row_right in zip(trajectories[left], trajectories[right])))
    report['matched_pairs'] = pairs
    report['data'] = common_data
    report['history_bf16'] = read_json(ROOT / 'runs/phase2/validation-acceptance-c-20260909.9WVDue/w16a16/result.json')
    (OUTPUT / 'records_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {'status': result['status'], 'failed_checks': [key for key, value in result['checks'].items() if not value]}
                      for name, result in report.items() if 'checks' in result}, indent=2))
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    audit()
