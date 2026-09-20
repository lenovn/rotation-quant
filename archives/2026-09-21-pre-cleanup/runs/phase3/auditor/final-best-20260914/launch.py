from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
OUTPUT = Path(__file__).resolve().parent
SOCKET = 'rotation-quant-phase3'
SESSION = 'auditor-final-best-20260914'

environment = dict(CUDA_VISIBLE_DEVICES='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
    HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', GIT_OPTIONAL_LOCKS='0',
    HF_HOME=str(ROOT / 'cache/huggingface'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
    PYTHONPATH=str(ROOT / 'worktrees/SpinQuant-phase3-joint'), TMPDIR=str(OUTPUT / 'tmp'))
(OUTPUT / 'tmp').mkdir(exist_ok=True)
snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], text=True)
command = ['env', *[key + '=' + value for key, value in environment.items()], 'bash', str(OUTPUT / 'run_serial.sh')]
shell_command = 'exec ' + shlex.join(command) + ' > ' + shlex.quote(str(OUTPUT / 'serial.log')) + ' 2>&1'
record = dict(created_at=datetime.now(timezone.utc).isoformat(), gpu=1, gpu_snapshot=snapshot,
    compute_processes=processes, backend='tmux', tmux_socket=SOCKET, tmux_session=SESSION,
    command=command, shell_command=shell_command, environment=environment,
    expected_evaluations=['original_bf16', 'final_saved_checkpoint0400'], training=False, recalibration=False,
    attach_command='tmux -L ' + SOCKET + ' attach -t ' + SESSION)
with (OUTPUT / 'launch.json').open('x') as handle:
    json.dump(record, handle, indent=2)
    handle.write('\n')
subprocess.run(['tmux', '-L', SOCKET, 'new-session', '-d', '-s', SESSION, '-c', str(ROOT), shell_command], check=True)
pane = subprocess.check_output(['tmux', '-L', SOCKET, 'list-panes', '-t', SESSION, '-F', '#{pane_pid}'], text=True).strip()
(OUTPUT / 'tmux-pane.pid').write_text(pane + '\n')
print(json.dumps(dict(session=SESSION, socket=SOCKET, gpu=1, pane_pid=pane, snapshot=snapshot)))
