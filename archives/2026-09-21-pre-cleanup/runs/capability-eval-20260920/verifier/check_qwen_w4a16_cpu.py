import os,sys,json
from pathlib import Path
ROOT=Path('/home/dongpeiyan/projects/rotation-quant'); RUN=ROOT/'runs/capability-eval-20260920'
selected=json.loads((RUN/'models.json').read_text())['qwen']['w4a16']
os.environ['PHASE5_MODEL_PATH']=selected['model_path']
sys.path[:0]=[str(ROOT/'scripts/capability_eval'),str(ROOT/'worktrees/SpinQuant-multimodel')]
import torch
from w4a16 import load_w4a16
from experiments.phase3.quantization import unpack_int4
from experiments.phase3.common import wrappers
torch.set_num_threads(4)
model,loading=load_w4a16(selected['model_path'],selected['package'],device='cpu')
state=torch.load(selected['package'],map_location='cpu',weights_only=True)
checks={}
for name,record in state['weights'].items():
 weight=model.get_submodule(name).weight
 expected=(unpack_int4(record['packed'],record['shape']).float()*record['scale']).to(weight)
 checks[name]=torch.equal(weight,expected)
current=model.state_dict()
def exact_with_nan(a,b):
 return torch.equal(a,b) or (a.shape==b.shape and a.dtype==b.dtype and bool(torch.all((a==b)|(torch.isnan(a)&torch.isnan(b)))))
hp={name:exact_with_nan(current[name],v) for name,v in state['high_precision'].items()}
expected_names={f'model.layers.{i}.{s}' for i in range(model.config.num_hidden_layers) for s in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')}
report=dict(selected=selected,loading=loading,weight_checks=checks,high_precision_checks=hp,
 exact_coverage=set(checks)==expected_names,backbone_activation_wrappers=len(wrappers(model)),device=str(next(model.parameters()).device))
report['pass']=all(checks.values()) and all(hp.values()) and report['exact_coverage'] and report['backbone_activation_wrappers']==0
(RUN/'verifier/qwen_w4a16_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=report['pass'],weights=len(checks),high_precision=len(hp),device=report['device'])))
