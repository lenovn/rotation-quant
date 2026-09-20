from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/capability-eval-20260920/data_preparation'
OUT.mkdir(exist_ok=True)
TASKS=[('boolq','super_glue','boolq'),('piqa','piqa',None),('social_iqa','social_i_qa',None),('hellaswag','hellaswag',None),('winogrande','winogrande','winogrande_xl'),('arc_easy','allenai/ai2_arc','ARC-Easy'),('arc_challenge','allenai/ai2_arc','ARC-Challenge'),('openbookqa','openbookqa','main'),('gsm8k_cot','gsm8k','main'),('ifeval','google/IFEval',None)]
def one(item):
 task,path,name=item
 try:
  d=load_dataset(path,name,trust_remote_code=True)
  r=dict(task=task,path=path,name=name,splits={s:len(v) for s,v in d.items()},status='ready')
 except Exception as e:
  r=dict(task=task,path=path,name=name,status='failed',error=repr(e))
 (OUT/(task+'.json')).write_text(json.dumps(r,indent=2)+'\n'); print(r,flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,TASKS))
