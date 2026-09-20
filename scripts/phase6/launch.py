"""Start an isolated Phase6 training process in tmux; preserve existing runs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--model', default='llama-3.2-1b-instruct')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--arm', default='joint')
    p.add_argument('--initial')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    output = ROOT / 'runs/phase6' / a.name
    launch = output.with_suffix('.launch.json')
    if a.resume:
        launch = output.parent / (a.name + '-resume-' + str(a.steps) + '.launch.json')
        if not (output / 'resume.pt').exists() or launch.exists():
            raise FileExistsError('Missing resume checkpoint or already launched this extension')
    elif output.exists() or launch.exists():
        raise FileExistsError(str(output))
    query = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    total, used, utilization = map(int, query.split(','))
    if used / total >= .9:
        raise RuntimeError('GPU already exceeds allowed memory usage')
    env = dict(CUDA_VISIBLE_DEVICES=str(a.gpu), PHASE5_MODEL_PATH=str(ROOT/'cache/models'/a.model),
               PHASE5_SEED='42', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               HF_HOME=str(ROOT/'cache/huggingface'))
    env['PHASE6_MEMORY_FRACTION'] = str(min(.85, .88 - used / total))
    if float(env['PHASE6_MEMORY_FRACTION']) <= 0:
        raise RuntimeError('No usable memory remains below the total-card limit and runtime margin')
    command = [str(ROOT/'runs/phase5/env/bin/python'), '-u', str(ROOT/'scripts/phase6/joint.py'),
               '--output', str(output), '--steps', str(a.steps), '--arm', a.arm]
    if a.initial:
        command += ['--initial', a.initial]
    if a.resume:
        command += ['--resume']
    session = 'phase6-' + a.name + ('-resume-' + str(a.steps) if a.resume else '')
    log = output.parent / (session[len('phase6-'):] + '.log')
    shell = 'exec env ' + ' '.join(shlex.quote(k+'='+v) for k,v in env.items()) + ' ' + shlex.join(command) + ' > ' + shlex.quote(str(log)) + ' 2>&1'
    child = subprocess.run(['tmux', '-L', 'rotation-quant-phase6', 'new-session', '-d', '-s', session, '-c', str(ROOT), '-P', '-F', '#{pane_pid}', shell], check=True, capture_output=True, text=True)
    record = dict(pid=int(child.stdout.strip()), session=session, socket='rotation-quant-phase6', command=command, environment=env,
                  output=str(output), log=str(log), gpu_at_launch=dict(total_mib=total, used_mib=used, utilization=utilization))
    launch.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
