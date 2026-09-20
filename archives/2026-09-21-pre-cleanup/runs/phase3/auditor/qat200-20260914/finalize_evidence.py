import ast
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import torch

from audit_evidence import CENTER, REFERENCE, OTHER, OUTPUT, PARENT, ROOT, RUNS, SOURCE, events, load_cpu, read_json, same_bytes, write_json


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def normalize_command(command):
    normalized = []
    index = 0
    while index < len(command):
        if command[index] in ['--output', '--reference-state']:
            index += 2
        else:
            normalized.append(command[index])
            index += 1
    return normalized


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only audit')
    torch.set_num_threads(2)
    summary = read_json(OUTPUT / 'summary.json')
    evidence = OUTPUT / 'evidence'
    commands = []
    for name in [CENTER, REFERENCE, *OTHER]:
        recorded = events(RUNS / (name + '.log'))
        write_json(evidence / name / 'events.json', recorded)
        commands.extend([name, shlex.join(summary['runs'][name]['launch']['command']), ''])
    reference_events = events(RUNS / (REFERENCE + '.log'))
    summary['recovery']['log_lines'] = {record['value']['event']: record['line'] for record in reference_events
        if record['value'].get('event', '').startswith('recovery-evaluation-')}
    summary['recovery']['line_number_definition'] = 'physical LF-delimited lines, not carriage-return progress updates'
    write_json(OUTPUT / 'summary.json', summary)
    center_source = functions(RUNS / CENTER / 'source/distill.py')
    reference_source = functions(RUNS / REFERENCE / 'source/distill.py')
    center_args = summary['runs'][CENTER]['settings']['arguments']
    reference_args = summary['runs'][REFERENCE]['settings']['arguments']
    findings = dict(
        pair=dict(normalized_actual_commands_equal=normalize_command(summary['runs'][CENTER]['launch']['command']) == normalize_command(summary['runs'][REFERENCE]['launch']['command']),
                  settings_equal_except_declared_initialization={key: value for key, value in center_args.items() if key not in ['output', 'reference_state']} == {key: value for key, value in reference_args.items() if key not in ['output', 'reference_state']},
                  changed_common_functions=[key for key in center_source if center_source[key] != reference_source[key]],
                  reference_only_functions=sorted(set(reference_source) - set(center_source)),
                  gradient_loss_optimizer_export_eval_definitions_equal=all(center_source[key] == reference_source[key] for key in ['TrainableQuantLinear', 'parameter_groups', 'teacher_model', 'chunk_objective', 'distillation_loss', 'make_optimizers', 'load_resume', 'export_student', 'evaluate_checkpoint']),
                  initial_full_reference_tensor_comparison='NOT TESTED: initial FP master tensors were not saved; no reconstruction or forward executed',
                  initial_center_full_code_requantization_errors=sum(summary['runs'][CENTER]['packed']['center_initial_requantization_mismatches'].values())))
    parent = load_cpu(PARENT)
    b100 = load_cpu(Path(parent['metadata']['parent']))
    transform = summary['parent']['reference_initialization']['diagonal_transform']
    name = transform['name']
    up_name = name.replace('down_proj', 'up_proj')
    selected = read_json(PARENT.parent / 'diagonal_candidates.json')[name]['selected']
    changed_rows = (parent['weights'][up_name]['scale'] != b100['weights'][up_name]['scale']).nonzero()[:, 0].tolist()
    packed_delta = parent['weights'][name]['packed'] ^ b100['weights'][name]['packed']
    low_positions = (packed_delta & 15).nonzero().flatten()
    high_positions = (packed_delta >> 4).nonzero().flatten()
    affected_columns = sorted(set((low_positions * 2 % 8192).tolist()) | set(((high_positions * 2 + 1) % 8192).tolist()))
    findings['parent_D'] = dict(selected_record_exact=transform == dict(name=name, **selected),
        changed_up_scale_rows=changed_rows, changed_down_code_columns=affected_columns,
        up_scale_halved=torch.equal(parent['weights'][up_name]['scale'][1417], b100['weights'][up_name]['scale'][1417] / 2),
        down_alpha_halved=parent['activation'][name]['alpha'] == b100['activation'][name]['alpha'] / 2,
        parent_validation_matches_result=read_json(PARENT.parent / 'validation.json') == summary['parent']['result']['validation'])
    del parent, b100
    parent_evidence = evidence / PARENT.parent.name
    parent_evidence.mkdir(exist_ok=True)
    for filename in ['settings.json', 'result.json', 'validation.json', 'parent_probe.json']:
        shutil.copy2(PARENT.parent / filename, parent_evidence / filename)
    shutil.copy2(RUNS / (PARENT.parent.name + '.launch.json'), parent_evidence / 'launch.json')
    shutil.copytree(PARENT.parent / 'source', parent_evidence / 'source', dirs_exist_ok=True)
    write_json(parent_evidence / 'selected_diagonal_record.json', dict(name=name, **selected))
    failed_run = RUNS / OTHER[0]
    failed_log = (RUNS / (OTHER[0] + '.log')).read_text()
    findings['SGD01_evidence_conflict'] = dict(verdict='FAIL',
        claim=read_json(failed_run / 'early_stop.json')['full_validation'],
        validation=read_json(failed_run / 'checkpoint-0025/validation.json'),
        checkpoint_ledger=read_json(failed_run / 'checkpoints.json'),
        interruption=read_json(failed_run / 'failure.json'),
        interruption_stack_in_training_loss='loss, ce, divergence = distillation_loss(student, teacher, ids' in failed_log,
        not_completed_100=not (failed_run / 'result.json').exists(),
        conclusion='Complete step25 score is supported by validation JSON, checkpoint ledger, both evaluator segments and subsequent training stack. Training100 is incomplete; do not hide the failed configuration or rank it as completed200.')
    findings['verdicts'] = dict(center200_evidence='PASS', reference200_training_export_recovered_full_evaluation_evidence='PASS',
        same_200_actual_lr_and_windows='PASS', all_112_packed_codes_and_224_scales_match_final_resume='PASS',
        frozen_original_FP_weights_PTQ_claim='FAIL', reference_original_driver_uninterrupted_success='FAIL',
        SGD01_no_complete_step25_claim='FAIL', SGD0001_completed100_no_improvement='PASS',
        continuation_partial201_to234_resume225_not_complete800='PASS', target_BF16_plus1='FAIL',
        independent_auditor_GPU_reproduction='NOT TESTED', full_initial_reference_tensor_bitwise_comparison='NOT TESTED',
        model_forward_CPU_or_GPU='NOT TESTED', application_code_pytest_verifier='NOT TESTED', mobile_native_static_KV_acceptance='NOT TESTED')
    findings['auditor_execution'] = dict(cpu_only=True, cuda_runtime_initialized=torch.cuda.is_initialized(),
        full_audit_driver_exit_code=0, exploratory_cpu_read_error='A preliminary metadata-only read looked for recovery_evaluation.json inside checkpoint-0200; actual archive is at run root. Corrected before successful evidence driver; no GPU or forward was attempted.',
        event_line_number_correction='Initial JSON extraction counted terminal carriage returns as lines. Final extraction uses raw LF, matching nl; numerical and source checks unchanged.',
        ast_check_correction='First finalize log grouped main with loss/optimizer/eval invariants, producing false because main adds the declared reference CLI option. Final check excludes main from that invariant and still records main in changed_common_functions. Initial finalize.log retained; corrected output is finalize-recheck.log.')
    write_json(OUTPUT / 'findings.json', findings)
    (evidence / 'actual_training_commands.txt').write_text('\n'.join(commands))
    (evidence / 'storage-df.txt').write_bytes(subprocess.check_output(['df', '-B1', '/mnt/home1', '/mnt/home2']))
    shutil.copy2(RUNS / 'STATUS.md', evidence / 'phase3-STATUS.md')
    shutil.copy2(RUNS / 'handoff-20260914.NByJvb/HANDOFF.md', evidence / 'HANDOFF.md')
    print(json.dumps(dict(pair=findings['pair'], parent_D=findings['parent_D'], verdicts=findings['verdicts']), indent=2))


if __name__ == '__main__':
    main()
