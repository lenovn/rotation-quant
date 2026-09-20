import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/capability-eval-20260920'
import yaml
paths=sorted((RUN/'deps/lm_eval/tasks/mmlu/default').glob('mmlu_*.yaml'))
def one(p):
 cfg=yaml.safe_load(p.read_text()); name=cfg['dataset_name']
 try:
  data=load_dataset('hails/mmlu_no_train',name)
  result=dict(config=name,splits={s:len(v) for s,v in data.items()})
 except Exception as e: result=dict(config=name,error=repr(e))
 print(json.dumps(result),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,paths))
