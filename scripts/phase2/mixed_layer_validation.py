"""One block restored to fixed-R BF16 in the original frozen C + SP2 model."""
import json
import math
import time
from types import SimpleNamespace

import torch

from down_d_search import ROOT, FrozenCodebookQuantizer, dump, evaluate_full_validation


@torch.no_grad()
def run_mixed_layer(model, fp, scales, tokenizer, c, options, budget):
    from datasets import load_dataset
    from utils.eval_utils import evaluator
    prefix = f'model.layers.{options.mixed_layer}.'
    alpha_path = ROOT/'runs/phase2/down-codebooks-c-20260909.6YRIty/results/down_scales.json'
    alphas = json.loads(alpha_path.read_text())['sp2']
    checkpoint = torch.load(c/'rtn/w4_rtn_model.pt', map_location='cpu', mmap=True, weights_only=False)
    restored, unchanged = [], []
    try:
        for name in fp:
            wrapper = model.get_submodule(name)
            restore = name.startswith(prefix)
            expected = fp[name] if restore else checkpoint['model'][name+'.module.weight']
            wrapper.module.weight.data = expected.clone()
            if name.endswith('down_proj'):
                wrapper.quantizer = FrozenCodebookQuantizer('sp2', alphas[name])
            else:
                wrapper.quantizer.load_scale(scales['activation'][name+'.quantizer'])
            wrapper.quantizer.bits = 16 if restore else 8
            wrapper.out_quantizer.bits = 16
            if wrapper.online_full_had or wrapper.online_partial_had:
                raise RuntimeError('Unexpected online transform')
            (restored if restore else unchanged).append(name)
            if not torch.equal(wrapper.module.weight.cpu(), expected):
                raise RuntimeError('Restoration mismatch')
        assert len(restored) == 7 and len(unchanged) == 105
        # Verify high-precision boundary tensors against the original frozen C too.
        state = model.state_dict()
        boundary = []
        backbone_keys = {name+suffix for name in fp
                         for suffix in ('.weight', '.module.weight', '.bias', '.module.bias')}
        for name, value in checkpoint['model'].items():
            if name in state and (name.endswith('.weight') or name.endswith('.bias')) and name not in backbone_keys:
                if not torch.equal(state[name].cpu(), value.cpu()):
                    raise RuntimeError(f'High-precision boundary mismatch: {name}')
                boundary.append(name)
        before = {name: {k: v.cpu().clone() for k,v in model.get_submodule(name).quantizer.named_buffers()}
                  for name in fp}
        dump(options.output/'mixed_settings.json', dict(layer=options.mixed_layer,
             restored_bf16=restored, unchanged_w4=unchanged, down_alpha_path=str(alpha_path),
             weight_path=str(c/'rtn/w4_rtn_model.pt'), rotation_path=str(c/'rotation/R.bin'),
             scale_path=str(c/'rotation/quant_scales.pt'), recalibrated=False, requantized=False,
             boundary_tensors_checked=boundary, activation_a16_count=7, non_down_int8_count=90,
             down_sp2_count=15, reference='Original C + SP2 frozen alpha, historical PPL 17.64239997588965'))
        data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
        enc = tokenizer('\n\n'.join(data['text']), return_tensors='pt')
        assert enc.input_ids.numel() == 252852
        budget.check('mixed_layer/full_validation')
        result = evaluate_full_validation(model, enc, 'cuda',
                 SimpleNamespace(eval_nsamples=None, bsz=1, capture_layer_io=False), evaluator)
        budget.evaluations = 1
        if not math.isfinite(result['nll']):
            raise RuntimeError('Non-finite mixed precision NLL')
        for name in fp:
            wrapper = model.get_submodule(name)
            expected = fp[name] if name in restored else checkpoint['model'][name+'.module.weight']
            if not torch.equal(wrapper.module.weight.cpu(), expected):
                raise RuntimeError(f'Weight changed: {name}')
            after = dict(wrapper.quantizer.named_buffers())
            if any(not torch.equal(v, after[k].cpu()) for k,v in before[name].items()):
                raise RuntimeError(f'Scale changed: {name}')
            assert wrapper.quantizer.bits == (16 if name in restored else 8)
        result.update(layer=options.mixed_layer, restored_weights=7, remaining_w4=105,
             split='validation', use_cache=False, kv_bits=16, frozen_checks_pass=True,
             historical_base_ppl=17.64239997588965, historical_base_nll=2.8703050943792365,
             delta_ppl=result['ppl']-17.64239997588965,
             delta_nll=result['nll']-2.8703050943792365,
             bf16_plus_one_gap=result['ppl']-14.634657725643203)
        dump(options.output/'mixed_result.json', result)
        print('MIXED_RESULT '+json.dumps(result), flush=True)
        budget.check('mixed_layer_completed')
    finally:
        dump(options.output/'budget.json', dict(gpu_seconds=time.monotonic()-budget.started,
             prior_gpu_seconds=budget.prior, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
             peak_process_mib=budget.peak_process_mib, full_validation_evaluations=budget.evaluations))
