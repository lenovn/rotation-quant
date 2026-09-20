import json,os,sys,gc
from pathlib import Path
ROOT=Path('/home/dongpeiyan/projects/rotation-quant');RUN=ROOT/'runs/capability-eval-20260920'
OUT=RUN/'qwen/gptq_w4a16_package';MODEL=ROOT/'cache/models/qwen3-1.7b'
sys.path[:0]=[str(ROOT/'scripts/capability_eval'),str(ROOT/'worktrees/SpinQuant-multimodel')]
os.environ['PHASE5_MODEL_PATH']=str(MODEL)
import torch
from experiments.phase3.architecture import load_model
from w4a16 import load_w4a16
torch.set_num_threads(4)
package=OUT/'w4_gptq_model.pt'
state=torch.load(package,map_location='cpu',weights_only=False)
expected={f'model.layers.{i}.{s}' for i in range(28) for s in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj')}
weights={}
for name,q in state['w_quantizers'].items():
 w=state['model'][name+'.weight'];scale=q.scale
 codes=(w.float()/scale).round()
 weights[name]=bool(q.bits==4 and q.sym and q.perchannel and q.weight_groupsize==-1 and scale.shape==(w.shape[0],1) and torch.isfinite(scale).all() and (scale>0).all() and (codes>=-8).all() and (codes<=7).all() and torch.equal((codes*scale).to(w),w))
original=load_model(MODEL,untie=True);original_state=original.state_dict()
high={n:torch.equal(t,original_state[n]) for n,t in state['model'].items() if n not in {name+'.weight' for name in expected}}
del original,original_state;gc.collect()
loaded,loading=load_w4a16(MODEL,package,device='cpu')
restored={n:torch.equal(v,state['model'][n]) for n,v in loaded.state_dict().items()}
report=dict(exact_coverage=set(state['w_quantizers'])==expected,weight_grid=weights,high_precision_vs_original=high,loader_exact=restored,
 no_activation_wrappers=not any(type(m).__name__=='ActQuantWrapper' for m in loaded.modules()),no_rotation_parameters=not any('.R1' in n or '.R2' in n for n in loaded.state_dict()),
 settings=state['settings'],calibration=json.loads((OUT/'calibration.json').read_text()),loading=loading)
report['pass']=report['exact_coverage'] and all(weights.values()) and all(high.values()) and all(restored.values()) and report['no_activation_wrappers'] and report['no_rotation_parameters']
(RUN/'verifier/qwen_gptq_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=report['pass'],weights=len(weights),high_precision=len(high),restored=len(restored))))
