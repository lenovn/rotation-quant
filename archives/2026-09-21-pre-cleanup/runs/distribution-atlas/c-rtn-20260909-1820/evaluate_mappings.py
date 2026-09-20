"""Fixed C R/SA test PPL for weight-MSE fitted INT4 and SP2."""
import json
import sys
from pathlib import Path
import torch

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'worktrees/SpinQuant-distribution-experiment'))
sys.path.insert(0, str(ROOT/'repos/SpinQuant'))
import ptq
from analyze_weights import project
from experiments.distribution_atlas import collect as atlas

original = ptq.eval_utils.evaluator

@torch.no_grad()
def evaluate(model, testenc, device, args):
    results = {'c_rtn':float(original(model,testenc,device,args))}
    (OUT/'ppl_mappings.json').write_text(json.dumps(results,indent=2))
    # This reconstructs BF16 from the original model and C-R; never requantize RTN.
    bf = atlas._load_rotated_model(ROOT/'cache/models/llama-3.2-1b-instruct',Path(args.optimized_rotation_path))
    sources = {name:module.weight.detach().cpu() for name,module in atlas._weight_sources(bf)}
    del bf
    torch.cuda.empty_cache()
    mapping = torch.load(OUT/'mapping_channels.pt',map_location='cpu',weights_only=True)
    for kind in ['int4','sp2']:
        modules = dict(model.named_modules())
        for detail in mapping:
            name = detail['source_name']
            w = sources[name].to(device).float()
            alpha = detail[kind+'_alpha'].to(device)[:,None]
            q = project(w,alpha,kind).bfloat16()
            wrapper = modules[name]
            assert wrapper.weight.data_ptr() == wrapper.module.weight.data_ptr()
            wrapper.module.weight.data.copy_(q.to(wrapper.module.weight.device))
        results[kind+'_weight_mse'] = float(original(model,testenc,device,args))
        (OUT/'ppl_mappings.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results),flush=True)
    return results['sp2_weight_mse']

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32=False
    ptq.eval_utils.evaluator=evaluate
    try:
        ptq.train()
    finally:
        ptq.eval_utils.evaluator=original
