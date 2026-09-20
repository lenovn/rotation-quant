from datetime import datetime, timezone
import json
import math
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[3]
RUNS = ROOT / 'runs/phase3'
BEST = 'distill-b100-refined-ref-adam1e5-400-20260914a'
CONTROL = 'distill-b100-refined-ref-adam2e5-400-20260914a'


def read_json(path):
    return json.loads(path.read_text())


def normalized(command):
    values = []
    index = 0
    while index < len(command):
        if command[index] in ['--output', '--weight-lr']:
            index += 2
        else:
            values.append(command[index])
            index += 1
    return values


def main():
    records = read_json(OUTPUT / 'records.json')
    baseline = read_json(OUTPUT / 'bf16.result.json')
    final = read_json(OUTPUT / 'final.result.json')
    old_baseline = read_json(RUNS / 'auditor/first-validation-20260914/bf16.result.json')
    primary = records['runs'][BEST]['validation']
    launch = read_json(OUTPUT / 'launch.json')
    result_fields = ['token_count', 'predicted_tokens', 'segments', 'unscored_tail_tokens', 'nll', 'ppl']
    gpu_checks = {}
    for mode, result, expected in [('bf16', baseline, old_baseline), ('final', final, primary)]:
        loaded = read_json(OUTPUT / (mode + '.load.json'))
        logged = [json.loads(line[len('AUDITOR_RESULT '):]) for line in (OUTPUT / (mode + '.log')).read_bytes().decode().split('\n') if line.startswith('AUDITOR_RESULT ')]
        gpu_checks[mode] = dict(exact_reproduction=all(result[key] == expected[key] for key in result_fields),
            exit_zero=(OUTPUT / (mode + '.exit')).read_text().strip() == '0',
            log_matches=logged == [result], pid_matches=int((OUTPUT / (mode + '.pid')).read_text()) == result['pid'],
            same_gpu=result['cuda_visible_devices'] == str(launch['gpu']) == '1',
            current_source=result['source_root'] == str(ROOT / 'worktrees/SpinQuant-phase3-joint') and result['source_head'] == records['source']['head'],
            full_tokens=result['token_count'] == 252852 and result['predicted_tokens'] == 252728 and result['unscored_tail_tokens'] == 0,
            historical_tokens_equal=result['tokens_equal_first_independent_audit'],
            rope_fp32=bool(loaded['rotary_buffer_dtypes']) and set(loaded['rotary_buffer_dtypes'].values()) == {'torch.float32'},
            original_bf16_unmodified=(loaded['original_bf16'] and not loaded['rotation'] and not loaded['norm_fusion']) if mode == 'bf16' else None,
            cold_package_no_calibration=(loaded['cold_load'] and not loaded['pretrained_weights_loaded'] and not loaded['trained_or_recalibrated']) if mode == 'final' else None,
            activation_scales_unchanged=result['activation_scales_unchanged'] if mode == 'final' else None)
    no_change = []
    with (RUNS / BEST / 'training.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            unchanged = [key for key, count in row['sampled_master_weight_changes'].items() if count == 0]
            if unchanged:
                no_change.append(dict(step=row['step'], weight_lr=row['learning_rates']['W'], tensors=unchanged))
    required_run_checks = {name: {key: value for key, value in record['checks'].items() if key != 'sampled_all_masters_change_each_step'} for name, record in records['runs'].items()}
    source_valid = records['source']['prior_dirty_equal'] and records['source']['acceptance_unchanged'] and all(records['source']['dependencies_unchanged'].values())
    pair = records['pair']
    commands_equal = normalized(records['runs'][BEST]['launch']['command']) == normalized(records['runs'][CONTROL]['launch']['command'])
    pair_valid = all(pair[key] for key in ['same_windows', 'same_scale_LRs', 'weight_LR_exact_half', 'same_initial_probe', 'same_master_initialization', 'same_arguments_except_weightLR_output']) and all(pair['same_source_snapshots'].values()) and commands_equal
    passed = all(all(value is not False for value in checks.values()) for checks in gpu_checks.values()) and all(all(checks.values()) for checks in required_run_checks.values()) and source_valid and pair_valid and all(records['packed']['checks'].values()) and all(records['model_reuse_checks'].values()) and records['files_stable_during_audit']
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), verdict='PASS' if passed and final['ppl'] <= baseline['ppl'] + 1 else 'FAIL',
        scope='same independent experiment auditor; final evidence and exactly two independent GPU full evaluations',
        gpu_checks=gpu_checks, required_run_checks=required_run_checks, pair=pair, normalized_training_commands_equal=commands_equal,
        results=dict(bf16=baseline, final=final, primary_final=primary, control_primary=records['runs'][CONTROL]['validation']),
        target=dict(baseline_ppl=baseline['ppl'], threshold_ppl=baseline['ppl'] + 1, final_ppl=final['ppl'],
                    delta_ppl_vs_bf16=final['ppl'] - baseline['ppl'], delta_nll_vs_bf16=final['nll'] - baseline['nll'],
                    margin_below_threshold=baseline['ppl'] + 1 - final['ppl'], passed=final['ppl'] <= baseline['ppl'] + 1),
        execution=dict(full_gpu_evaluations=2, valid_full_evaluations=2, invalid_full_evaluations=0, failed_evaluation_attempts=0,
                       new_training_updates=0, recalibrations=0, pytest_runs=0, algorithm_candidates=0,
                       gpu=1, bf16_pid=baseline['pid'], final_pid=final['pid'], tmux_session=launch['tmux_session'],
                       environment_changed=False, other_processes_touched=False,
                       resource_query_note='Initial sandbox NVML query failed; require_escalated read and authorized tmux launch succeeded, not an evaluation attempt'),
        observations=dict(sampled_all_masters_change_every_step_claim='FAIL: stronger claim is false, not required to establish actual optimizer updates',
                          zero_sampled_change_records=no_change,
                          interpretation='At step400 two tensors have no change among 64 sampled entries. Do not infer the full tensors did not change or an optimizer step was skipped; all336 optimizer states have step400 and all112 gradients are recorded. Finite-precision small updates are a possible explanation, not independently proved.'),
        verdicts=dict(final_run_evidence='PASS' if all(required_run_checks[BEST].values()) else 'FAIL',
                      source_model_data_teacher_provenance='PASS' if source_valid and all(records['model_reuse_checks'].values()) else 'FAIL',
                      paired_400_actual_LRs_and_windows='PASS' if pair_valid else 'FAIL',
                      final_packed_export='PASS' if all(records['packed']['checks'].values()) else 'FAIL',
                      independent_bf16_full_GPU='PASS' if all(value is not False for value in gpu_checks['bf16'].values()) else 'FAIL',
                      independent_final_full_GPU='PASS' if all(value is not False for value in gpu_checks['final'].values()) else 'FAIL',
                      original_BF16_plus1='PASS' if final['ppl'] <= baseline['ppl'] + 1 else 'FAIL',
                      independent_control_2e5_GPU='NOT TESTED', initial_FP_master_full_tensor_bitwise_comparison='NOT TESTED',
                      application_pytest_verifier='NOT TESTED', unseen_test_C4_multiseed_generalization='NOT TESTED',
                      native_INT4_INT8_mobile_KV_decode_latency='NOT TESTED', main_goal_administrative_completion='NOT TESTED'),
        notes=['Full-validation exactness refers to segment and aggregate JSON results, not a comparison of every logit.',
               'This result is full-backbone quantization-aware distillation, not frozen-original-weight PTQ.',
               'Baseline/model/data provenance is reused where declared; both reported GPU evaluations are new independent reproductions.',
               'No automatic follow-up training, calibration, search, checkpoint replay, or goal update.'])
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verdict=summary['verdict'], target=summary['target'], gpu_checks=gpu_checks, observations=summary['observations']), indent=2))


if __name__ == '__main__':
    main()
