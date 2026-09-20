import os,sys,json,traceback
from pathlib import Path
from collections import Counter
ROOT=Path('/home/dongpeiyan/projects/rotation-quant'); RUN=ROOT/'runs/capability-eval-20260920'
sys.path[:0]=[str(RUN/'deps'),str(ROOT/'worktrees/SpinQuant-multimodel')]
record=json.loads((RUN/'models.json').read_text())['llama']['firon']
os.environ['PHASE5_MODEL_PATH']=record['model_path']
import torch
from experiments.phase3.postprocess import load_static
from experiments.phase3.architecture import tokenizer
from experiments.phase3.common import wrappers
from experiments.phase3.external_eval import quantizer_snapshot
from lm_eval.models.huggingface import HFLM
torch.set_num_threads(4)
report={'selected':record,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
try:
    model,_=load_static(record['package'])
    tok=tokenizer(record['model_path']); tok.pad_token_id=tok.eos_token_id
    lm=HFLM(pretrained=model,tokenizer=tok,backend='causal',batch_size=2,max_length=8192,add_bos_token=False)
    ids,mask=lm.tok_batch_encode(['The capital of France is','In a short answer, please tell me what the capital of France is.'])
    report.update(pad_id=tok.pad_token_id,unk_id=tok.unk_token_id,embedding_rows=model.get_input_embeddings().weight.shape[0],input_ids=ids.tolist(),attention_mask=mask.tolist())
    assert tok.pad_token_id < report['embedding_rows'] and int(ids.max()) < report['embedding_rows']
    assert (mask==0).any() and mask.sum(1)[0]!=mask.sum(1)[1]
    before=quantizer_snapshot(model);counts={n:Counter() for n in wrappers(model)}
    for n,w in wrappers(model).items():
        def hook(q,args,n=n):
            counts[n]['decode' if args[0].shape[-2]==1 else 'prefill']+=1
            assert q.bits==8 and not getattr(q,'observing',False)
        w.quantizer.register_forward_pre_hook(hook)
    with torch.no_grad():
        output=lm._model_generate(ids.cuda(),ids.shape[1]+2,stop=[],attention_mask=mask.cuda(),min_new_tokens=2,do_sample=False)
    after=quantizer_snapshot(model)
    report['scales_unchanged']=before.keys()==after.keys() and all(before[n].keys()==after[n].keys() and all(torch.equal(v,after[n][k]) for k,v in before[n].items()) for n in before)
    report['counts']=counts;report['output_tokens']=output.tolist()
    report['pass']=report['scales_unchanged'] and len(counts)==112 and all(c['prefill'] and c['decode'] for c in counts.values()) and output.shape[1]==ids.shape[1]+2
except BaseException:
    report['pass']=False;report['error']=traceback.format_exc()
(RUN/'verifier/llama_padding.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='counts'},indent=2))
if not report['pass']:sys.exit(1)
