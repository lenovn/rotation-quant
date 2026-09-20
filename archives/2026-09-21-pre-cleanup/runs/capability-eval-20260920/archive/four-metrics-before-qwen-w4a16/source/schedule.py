"""One independent process per model/task; preserve completed jobs and failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/capability-eval-20260920'
PYTHON=ROOT/'runs/phase5/env/bin/python'
TASKS=['boolq','piqa','social_iqa','hellaswag','winogrande','arc_easy','arc_challenge','openbookqa','mmlu']

def worker(model, method, gpu):
    if method == 'w4a16' and not all((RUN / model / method / ('ppl-' + kind) / 'result.json').exists() for kind in ['test','c4']):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1')
        with (RUN / 'w4a16-ppl.log').open('a') as log:
            subprocess.run([str(PYTHON), '-u', str(ROOT / 'scripts/capability_eval/ppl_w4a16.py')], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    for task in TASKS:
        output=RUN/model/method/task
        if (output/'results.json').exists():
            continue
        output.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
          HF_HOME=str(ROOT/'cache/huggingface'),HF_DATASETS_TRUST_REMOTE_CODE='1',
          OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
          NLTK_DATA=str(RUN/'nltk_data'),HF_HUB_DOWNLOAD_TIMEOUT='120',HF_HUB_ETAG_TIMEOUT='60')
        env.pop('HF_HUB_OFFLINE',None);env.pop('HF_DATASETS_OFFLINE',None)
        command=[str(PYTHON),'-u',str(ROOT/'scripts/capability_eval/run.py'),
          '--model',model,'--method',method,'--task',task,'--batch-size','8' if task in ('gsm8k_cot','ifeval') else '4','--output',str(output)]
        attempt=len(list(output.glob('launch*.json')))+1
        while True:
            try:
                log_file = (output/f'run-{attempt}.log').open('x')
                break
            except FileExistsError:
                attempt += 1
        with log_file as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            record=dict(command=command,pid=proc.pid,gpu=gpu,started=time.time(),output=str(output))
            with (output/f'launch-{attempt}.json').open('x') as record_file:
                record_file.write(json.dumps(record,indent=2)+'\n')
            print(json.dumps(record),flush=True)
            code=proc.wait()
        print(json.dumps(dict(model=model,method=method,task=task,exit_code=code)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='+',required=True,help='model:method:gpu');p.add_argument('--tasks',nargs='+',choices=TASKS);a=p.parse_args()
    if a.tasks:TASKS=a.tasks
    jobs=[x.split(':') for x in a.jobs]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(lambda x:worker(*x),jobs))
