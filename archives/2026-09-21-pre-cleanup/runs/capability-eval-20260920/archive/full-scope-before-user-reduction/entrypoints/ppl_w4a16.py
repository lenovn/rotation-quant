import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/capability-eval-20260920'
sys.path.insert(0,str(ROOT/'worktrees/SpinQuant-multimodel'))
os.environ['PHASE5_MODEL_PATH']=str(ROOT/'cache/models/llama-3.2-1b-instruct')
import torch
from w4a16 import load_w4a16
from experiments.phase3.external_eval import load_wikitext2_test_tokens,load_c4_tokens,evaluate_tokens
cfg=json.loads((RUN/'models.json').read_text())['llama']['w4a16']
reused=json.loads((RUN/'ppl_reuse.json').read_text())
c4=next(x for x in reused if x['model_key']=='llama' and x['method']=='bf16' and x['dataset']=='allenai/c4')
original=json.loads(Path(c4['evidence_path']).read_text())
tokens=Path(original['input_token_path'])
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,loading=load_w4a16(cfg['model_path'],cfg['package']);model.seqlen=2048;model.config.use_cache=False
for kind in ['test','c4']:
 out=RUN/'llama/w4a16'/('ppl-'+kind);out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():continue
 started=time.time()
 ids,meta=load_wikitext2_test_tokens() if kind=='test' else load_c4_tokens(tokens)
 result=evaluate_tokens(model,ids,128)
 assert result['predicted_tokens']==meta['predicted_tokens']
 result.update(input_metadata=meta,package=cfg['package'],model_path=cfg['model_path'],loading=loading,
               calibration=False,training=False,elapsed_seconds=time.time()-started)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(kind,result['ppl'],flush=True)
