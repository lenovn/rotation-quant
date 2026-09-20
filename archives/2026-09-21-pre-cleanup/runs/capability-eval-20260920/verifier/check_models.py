import os, sys, json, traceback
from pathlib import Path
from collections import Counter
ROOT=Path('/home/dongpeiyan/projects/rotation-quant')
RUN=ROOT/'runs/capability-eval-20260920'
sys.path[:0]=[str(RUN/'deps'),str(ROOT/'worktrees/SpinQuant-multimodel')]
kind=sys.argv[1]
record=json.loads((RUN/'models.json').read_text())[kind]['firon']
os.environ['PHASE5_MODEL_PATH']=record['model_path']
import torch
from transformers import DynamicCache
from experiments.phase3.postprocess import load_static
from experiments.phase3.architecture import tokenizer
from experiments.phase3.common import wrappers
from experiments.phase3.external_eval import quantizer_snapshot
torch.set_num_threads(4)
report={'model':kind,'selected':record,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
try:
    model,_=load_static(record['package'])
    tok=tokenizer(record['model_path'])
    tok.pad_token_id=tok.eos_token_id
    prompt=tok('The capital of France is',return_tensors='pt').to('cuda')
    before=quantizer_snapshot(model)
    calls={name:Counter() for name in wrappers(model)}
    stage=['score']
    for name,w in wrappers(model).items():
        def hook(q,args,name=name):
            calls[name][stage[0]+('_decode' if args[0].shape[-2]==1 else '_prefill')]+=1
            assert q.bits==8 and not getattr(q,'observing',False)
        w.quantizer.register_forward_pre_hook(hook)
    with torch.no_grad():
        logits=model(**prompt,use_cache=False).logits
        report['score_finite']=bool(torch.isfinite(logits).all())
        stage[0]='generate'
        output=model.generate(**prompt,past_key_values=DynamicCache(),use_cache=True,
            min_new_tokens=2,max_new_tokens=2,do_sample=False,pad_token_id=tok.eos_token_id)
    after=quantizer_snapshot(model)
    report['scales_unchanged']=before.keys()==after.keys() and all(
        before[n].keys()==after[n].keys() and all(torch.equal(v,after[n][k]) for k,v in before[n].items()) for n in before)
    report['calls']=calls
    report['num_quantizers']=len(calls)
    report['output_tokens']=output.tolist()
    report['pass']=report['score_finite'] and report['scales_unchanged'] and all(
        all(c[k]>=1 for k in ('score_prefill','generate_prefill','generate_decode')) for c in calls.values())
except BaseException:
    report['pass']=False
    report['error']=traceback.format_exc()
(RUN/'verifier'/f'{kind}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='calls'},indent=2))
if not report['pass']:sys.exit(1)
