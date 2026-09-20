"""Atlas of the actual C RTN W4/non-down static A8/down A16 model."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
OUT = Path(__file__).resolve().parent / 'rtn'
sys.path.insert(0, str(ROOT/'worktrees/SpinQuant-distribution-experiment'))
sys.path.insert(0, str(ROOT/'repos/SpinQuant'))
import ptq
from experiments.distribution_atlas import collect as atlas
from utils.quant_phase import QuantPhase

original_evaluator = ptq.eval_utils.evaluator

@torch.no_grad()
def evaluate(model, testenc, device, args):
    ppl = float(original_evaluator(model, testenc, device, args))
    print(f'C RTN reference PPL: {ppl}', flush=True)
    model.to(device).eval()
    weights, activations = [], []
    for i, layer in enumerate(model.model.layers):
        for suffix in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                       'mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
            owner, attr = suffix.split('.')
            wrapper = getattr(getattr(layer, owner), attr)
            weights.append((f'model.layers.{i}.{suffix}', wrapper.module))
        for label, wrapper in [('self_attn.qkv_input',layer.self_attn.q_proj),
                               ('self_attn.o_proj_input',layer.self_attn.o_proj),
                               ('mlp.gate_up_input',layer.mlp.gate_proj),
                               ('mlp.down_proj_input',layer.mlp.down_proj)]:
            activations.append((f'model.layers.{i}.{label}',wrapper))
    tokenizer = atlas.LlamaTokenizerFast.from_pretrained(str(ROOT/'cache/models/llama-3.2-1b-instruct'),
        model_max_length=2048,padding_side='right',use_fast=True,add_eos_token=False,add_bos_token=False)
    windows = atlas._load_wikitext_windows(tokenizer)
    def run_windows(model, windows, device):
        for i, ids in enumerate(windows):
            model.model(ids.to(device),use_cache=False,quant_phase=QuantPhase.PREFILL)
            if (i+1)%32==0:
                print(f'Atlas windows {i+1}/{len(windows)}', flush=True)
    original_run = atlas._run_windows
    atlas._run_windows = run_windows
    try:
        w_records = atlas._collect_weight_records(weights)
        a_records = atlas._collect_activation_records(model,activations,windows,device)
    finally:
        atlas._run_windows = original_run
    atlas._validate_records(w_records,112,'weight')
    atlas._validate_records(a_records,64,'activation')
    OUT.mkdir(exist_ok=True)
    for filename, data in [('weight_records.json',w_records),('activation_records.json',a_records),
                           ('runtime.json',dict(ppl=ppl,weights=args.load_qmodel_path,
                            rotation=args.optimized_rotation_path,scales=args.static_scale_path,
                            precision='C RTN W4/static non-down A8/down A16/KV16',
                            activation_hook='before each input quantizer, actual quantized upstream trajectory',
                            windows=128,seqlen=2048,seed=42,split='train'))]:
        (OUT/filename).write_text(json.dumps(data,indent=2,allow_nan=False))
    print('RTN ATLAS COMPLETE',flush=True)
    return ppl

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32=False
    ptq.eval_utils.evaluator=evaluate
    try:
        ptq.train()
    finally:
        ptq.eval_utils.evaluator=original_evaluator
