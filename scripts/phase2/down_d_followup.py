"""Train-only bounded strength probes after the nine frozen validation calls."""
import time
from types import SimpleNamespace

import torch

from down_d_search import (FrozenCodebookQuantizer, dump, search_model, set_weights,
                           unpack)


@torch.no_grad()
def train_diagnostics(model, payload, windows, directory, budget):
    from utils.eval_utils import evaluator
    set_weights(model, payload['weights'])
    for key, scale in payload['non_down_sa'].items():
        model.get_submodule(key).load_scale(scale)
    for name, alpha in payload['down_alphas'].items():
        model.get_submodule(name).quantizer = FrozenCodebookQuantizer('int8', alpha)
    enc = SimpleNamespace(input_ids=torch.cat(windows, dim=1))
    results = {}
    for mode in ('all_a16', 'down_a16', 'full_a8'):
        for name in payload['weights']:
            model.get_submodule(name).quantizer.bits = (
                16 if mode == 'all_a16' or (mode == 'down_a16' and name.endswith('down_proj')) else 8)
        statistics, handles = {}, []
        if mode == 'full_a8':
            for name, alpha in payload['down_alphas'].items():
                stats = dict(elements=0, original_zero=0, quantized_zero=0, outside_fullrange=0)
                statistics[name] = stats
                def observe(module, args, stats=stats, alpha=alpha):
                    x = args[0]
                    stats['elements'] += x.numel()
                    stats['original_zero'] += int((x == 0).sum())
                    stats['quantized_zero'] += int(((x.float() / (alpha / 127)).round() == 0).sum())
                    stats['outside_fullrange'] += int((x.abs() > alpha).sum())
                handles.append(model.get_submodule(name).register_forward_pre_hook(observe))
        before = {name: {k: v.cpu().clone() for k, v in model.get_submodule(name).quantizer.named_buffers()}
                  for name in payload['weights']}
        budget.check(f'{directory.name}/{mode}/train')
        try:
            args = SimpleNamespace(eval_nsamples=32, bsz=1, capture_layer_io=False)
            # The existing evaluator scores each 2048 window independently.
            ppl = float(evaluator(model, enc, 'cuda', args))
        finally:
            for handle in handles:
                handle.remove()
        import math
        if not math.isfinite(ppl):
            raise RuntimeError('Non-finite train PPL')
        for name, row in payload['weights'].items():
            expected = (unpack(row['packed'], row['shape']).float() * row['scale']).to(torch.bfloat16)
            if not torch.equal(expected, model.get_submodule(name).module.weight.cpu()):
                raise RuntimeError('Train diagnostic changed frozen weights')
            after = dict(model.get_submodule(name).quantizer.named_buffers())
            if any(not torch.equal(v, after[k].cpu()) for k, v in before[name].items()):
                raise RuntimeError('Train diagnostic changed frozen SA')
        for stats in statistics.values():
            stats['new_zero_fraction'] = (stats['quantized_zero'] - stats['original_zero']) / stats['elements']
            if stats['elements'] != 65536 * 8192 or stats['outside_fullrange']:
                raise RuntimeError(f'Train range/coverage differs: {stats}')
        results[mode] = dict(ppl=ppl, nll=math.log(ppl), split='train', windows=32,
                             predicted_tokens=32*2047, frozen_unchanged=True,
                             down_statistics=statistics)
        dump(directory / f'train_{mode}.json', results[mode])
        print(f'TRAIN {directory.name} {mode}: PPL={ppl:.9f}, NLL={math.log(ppl):.9f}', flush=True)
        budget.check()
    return results


@torch.no_grad()
def run_followup(model, fp, scales, initial, metadata, rows, windows, options, budget):
    results = {}
    dump(options.output/'followup_settings.json', dict(parent=str(options.followup_parent),
         hypothesis='Layer MSE strength selection can differ from train end-to-end NLL preference',
         controls=['identity_recomputed', 'search_recomputed'], new_strengths=[.25, 1.],
         selection_split='train', additional_full_validation=0))
    try:
        for arm in ('identity_recomputed', 'search_recomputed', 'fixed_t025', 'fixed_t100'):
            directory = options.output / arm
            directory.mkdir()
            if arm in ('identity_recomputed', 'search_recomputed'):
                source = options.followup_parent / arm / 'packed_model.pt'
                payload = torch.load(source, map_location='cpu', weights_only=True)
                dump(directory/'source.json', dict(frozen_payload=str(source), recalibrated=False))
            else:
                strength = .25 if arm == 'fixed_t025' else 1.
                payload = search_model(model, fp, scales, initial, metadata, rows,
                                       'search_recomputed', directory, budget, strengths=(strength,))
            results[arm] = train_diagnostics(model, payload, windows, directory, budget)
            dump(options.output/'train_results.json', results)
            del payload
        budget.check('followup_completed')
    finally:
        dump(options.output/'budget.json', dict(gpu_seconds=time.monotonic()-budget.started,
             prior_gpu_seconds=budget.prior, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
             peak_process_mib=budget.peak_process_mib, full_validation_evaluations=0))
