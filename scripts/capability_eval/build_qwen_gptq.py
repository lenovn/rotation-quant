"""One-shot train-only GPTQ of original Qwen BF16, using the project's solver."""
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/capability-eval-20260920'
SOURCE = ROOT / 'worktrees/SpinQuant-multimodel'
MODEL = ROOT / 'cache/models/qwen3-1.7b'
OUT = RUN / 'qwen/gptq_w4a16_package'
sys.path.insert(0, str(SOURCE))
os.environ['PHASE5_MODEL_PATH'] = str(MODEL)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from datasets import Dataset
from transformers import set_seed
from experiments.phase3.architecture import load_model, tokenizer
from experiments.phase3.common import DATA_PATH
from eval_utils.gptq_utils import GPTQ
from utils.quant_utils import WeightQuantizer

GROUPS = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
          ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    package = OUT / 'w4_gptq_model.pt'
    if package.exists():
        raise FileExistsError('Preserve completed GPTQ package')
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    settings = dict(model_path=str(MODEL), source=str(SOURCE), seed=42,
        nsamples=128, seqlen=2048, w_bits=4, activation_bits=16,
        perchannel=True, sym=True, groupsize=-1, mse_clip=False,
        blocksize=128, percdamp=0.01, actorder=False, static_groups=False,
        rotation=False, distillation=False, training=False,
        solver='eval_utils.gptq_utils.GPTQ.fasterquant', sequential_groups=GROUPS,
        high_precision='embedding, lm_head, all norms including Q/K norms',
        model_revision=json.loads((MODEL / 'phase5_revision.json').read_text()))
    save('settings.json', settings)
    data = Dataset.from_file(str(DATA_PATH / 'wikitext-train.arrow'))
    ids = tokenizer(MODEL)('\n\n'.join(data['text']), return_tensors='pt',
                           add_special_tokens=False).input_ids
    rng = random.Random(42)
    starts = [rng.randint(0, ids.shape[1] - 2048 - 1) for _ in range(128)]
    save('calibration.json', dict(dataset='Salesforce/wikitext', subset='wikitext-2-raw-v1',
        split='train', input_path=str(DATA_PATH / 'wikitext-train.arrow'),
        tokenizer_path=str(MODEL), text_join='double-newline', add_special_tokens=False,
        train_tokens=ids.numel(), starts=starts, window_length=2048, nsamples=128,
        calibration_tokens=128*2048, seed=42))
    model = load_model(MODEL, untie=True).eval().cuda().requires_grad_(False)
    model.config.use_cache = False
    layers = model.model.layers
    names = [f'model.layers.{i}.{name}' for i in range(len(layers)) for group in GROUPS for name in group]
    weight_keys = {name + '.weight' for name in names}
    high_before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in weight_keys}
    inps = torch.empty((128, 2048, model.config.hidden_size), dtype=torch.bfloat16, device='cuda')
    captured = dict(count=0, kwargs=None)
    class CapturedFirstLayer(Exception):
        pass
    def capture(module, args, kwargs):
        inps[captured['count']].copy_(args[0][0])
        captured['count'] += 1
        captured['kwargs'] = kwargs
        raise CapturedFirstLayer
    handle = layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for start in starts:
            try:
                model.model(ids[:, start:start+2048].cuda(), use_cache=False)
            except CapturedFirstLayer:
                pass
    finally:
        handle.remove()
    if captured['count'] != 128 or 'position_embeddings' not in captured['kwargs']:
        raise RuntimeError('Qwen first-layer capture incomplete')
    kwargs = captured['kwargs']
    save('capture.json', dict(samples=captured['count'], kwargs=list(kwargs),
        use_cache=kwargs.get('use_cache'), input_shape=list(inps.shape)))
    outs = torch.empty_like(inps)
    quantizers = {}
    for index, layer in enumerate(layers):
        for group in GROUPS:
            solvers = {}
            handles = []
            for name in group:
                solver = GPTQ(layer.get_submodule(name))
                solver.quantizer = WeightQuantizer()
                solver.quantizer.configure(4, perchannel=True, sym=True, mse=False, weight_groupsize=-1)
                solvers[name] = solver
                def collect(module, inp, out, solver=solver):
                    solver.add_batch(inp[0].detach(), out.detach())
                handles.append(layer.get_submodule(name).register_forward_hook(collect))
            try:
                for j in range(128):
                    layer(inps[j:j+1], **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            for name, solver in solvers.items():
                solver.fasterquant(blocksize=128, percdamp=0.01, groupsize=-1,
                    actorder=False, static_groups=False, export_to_et=False)
                if not torch.isfinite(layer.get_submodule(name).weight).all():
                    raise RuntimeError('Non-finite GPTQ weight: ' + name)
                quantizers[f'model.layers.{index}.{name}'] = solver.quantizer.cpu()
                solver.free()
            del solvers
        for j in range(128):
            outs[j].copy_(layer(inps[j:j+1], **kwargs)[0][0])
        inps, outs = outs, inps
        save('progress.json', dict(completed_layers=index+1, total_layers=len(layers), elapsed_seconds=time.time()-started))
        print(f'GPTQ layer {index+1}/{len(layers)} complete', flush=True)
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    if not all(torch.equal(value, state[key]) for key,value in high_before.items()):
        raise RuntimeError('High-precision tensors changed')
    if set(quantizers) != set(names):
        raise RuntimeError('GPTQ weight coverage incomplete')
    torch.save(dict(model=state, w_quantizers=quantizers, settings=settings), package)
    save('result.json', dict(package=str(package), w4_linears=len(quantizers),
        high_precision_unchanged=True, high_precision_tensors=len(high_before),
        elapsed_seconds=time.time()-started))
    print('SAVED ' + str(package), flush=True)

if __name__ == '__main__':
    main()
