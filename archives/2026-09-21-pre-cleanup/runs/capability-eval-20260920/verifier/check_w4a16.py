import os,sys,json,traceback
from pathlib import Path
ROOT=Path('/home/dongpeiyan/projects/rotation-quant'); RUN=ROOT/'runs/capability-eval-20260920'
sys.path[:0]=[str(RUN/'deps'),str(ROOT/'worktrees/SpinQuant-multimodel'),str(ROOT/'scripts/capability_eval')]
import torch
from transformers import DynamicCache
from w4a16 import load_w4a16
from experiments.phase3.architecture import tokenizer
torch.set_num_threads(4)
record=json.loads((RUN/'models.json').read_text())['llama']['w4a16']
report={'selected':record,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
try:
    model,info=load_w4a16(record['model_path'],record['package'])
    report['loading']=info
    state=torch.load(record['package'],map_location='cpu',weights_only=False)
    expected={f'model.layers.{i}.{s}.module' for i in range(model.config.num_hidden_layers) for s in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj')}
    report['exact_w4_names']=set(state['w_quantizers'])==expected
    report['all_state_tensors_equal']=all(torch.equal(v.cpu(),state['model'].get(n.rpartition('.')[0]+'.module.'+n.rpartition('.')[2],state['model'].get(n))) for n,v in model.state_dict().items())
    report['model_state_tensor_count']=len(model.state_dict())
    report['all_linear_count']=sum(isinstance(m,torch.nn.Linear) for m in model.modules())
    report['activation_wrappers']=sum(type(m).__name__=='ActQuantWrapper' for m in model.modules())
    del state
    tok=tokenizer(record['model_path']); tok.pad_token_id=tok.eos_token_id
    inputs=tok('The capital of France is',return_tensors='pt').to('cuda')
    with torch.no_grad():
        report['finite_logits']=bool(torch.isfinite(model(**inputs,use_cache=False).logits).all())
        outputs=model.generate(**inputs,past_key_values=DynamicCache(),use_cache=True,min_new_tokens=2,max_new_tokens=2,do_sample=False,pad_token_id=tok.eos_token_id)
    report['output_tokens']=outputs.tolist()
    report['pass']=report['exact_w4_names'] and report['all_state_tensors_equal'] and report['finite_logits'] and report['activation_wrappers']==0
except BaseException:
    report['pass']=False; report['error']=traceback.format_exc()
(RUN/'verifier/w4a16.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['pass']:sys.exit(1)
