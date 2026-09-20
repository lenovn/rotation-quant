"""Bounded, gradient-free down-channel search with offline paired fusion."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'repos/SpinQuant'))
from down_codebooks import FrozenCodebookQuantizer, quantize
from down_codebook_experiment import text_windows
from validation_acceptance import evaluate_full_validation

STRENGTHS = (0., .25, .5, .75, 1.)


def dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def direction(activation, weight):
    a = activation.double().clamp_min(1e-12)
    w = weight.double().abs().amax(dim=0).clamp_min(1e-12)
    u = .5 * ((a.log() - a.log().mean()) - (w.log() - w.log().mean()))
    if float(u.max() - u.min()) < 1e-12:
        return torch.zeros_like(activation, dtype=torch.float32)
    bound = math.log(4.)
    lo, hi = u.min() - bound, u.max() + bound
    for _ in range(70):
        center = (lo + hi) / 2
        if (u - center).clamp(-bound, bound).mean() > 0:
            lo = center
        else:
            hi = center
    return (u - (lo + hi) / 2).clamp(-bound, bound).float()


def fuse_pair(up, down, d):
    if d.ndim != 1 or d.numel() != up.shape[0] or down.shape[1] != d.numel():
        raise ValueError('D shape mismatch')
    if not torch.isfinite(d).all() or not (d > 0).all():
        raise ValueError('D must be finite and positive')
    return up / d[:, None], down * d[None, :]


def weight_quant(weight, scale=None):
    w = weight.float()
    if scale is None:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 7
    else:
        scale = scale.to(w.device).float().clone()
    codes = (w / scale).round().clamp(-8, 7).to(torch.int8)
    return (codes.float() * scale).to(weight.dtype), codes, scale


def pack(codes):
    if not ((codes >= -8) & (codes <= 7)).all() or codes.numel() % 2:
        raise ValueError('Invalid INT4 codes')
    q = (codes.reshape(-1).to(torch.int16) + 8).to(torch.uint8)
    return q[::2] | (q[1::2] << 4)


def unpack(packed, shape):
    q = torch.stack((packed & 15, packed >> 4), dim=1).reshape(shape)
    return (q.to(torch.int16) - 8).to(torch.int8)


def static_input(x, scale):
    return ((x.float() / scale).round().clamp(-128, 127) * scale).to(x.dtype)


def mlp(h, gate, up, down, gate_sa=None, up_sa=None, alpha=None):
    hg = h if gate_sa is None else static_input(h, gate_sa)
    hu = h if up_sa is None else static_input(h, up_sa)
    z = F.silu(F.linear(hg, gate)) * F.linear(hu, up)
    if alpha is not None:
        z = quantize(z, alpha, 'int8')
    return F.linear(z, down)


class Budget:
    def __init__(self, directory, prior_seconds=0.):
        self.directory = directory
        self.started = time.monotonic()
        self.prior = prior_seconds
        self.evaluations = 0
        self.stage = 'initializing'
        self.last_poll = 0.
        self.peak_process_mib = 0

    def check(self, stage=None, **details):
        if stage:
            self.stage = stage
        elapsed = self.prior + time.monotonic() - self.started
        peak = torch.cuda.max_memory_reserved() / 2**30
        if time.monotonic() - self.last_poll > 5:
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            for line in query.stdout.splitlines():
                pid, memory = line.split(',')
                if int(pid.strip()) == os.getpid():
                    self.peak_process_mib = max(self.peak_process_mib, int(memory.strip()))
            self.last_poll = time.monotonic()
        dump(self.directory / 'progress.json', dict(pid=os.getpid(), stage=self.stage,
             gpu_seconds=elapsed, gpu_budget_seconds=None, peak_reserved_gib=peak,
             full_validation_evaluations=self.evaluations, **details))
        if peak > 12 or self.peak_process_mib > 12288:
            raise RuntimeError(f'Budget exceeded: {elapsed:.1f}s, {peak:.2f}GiB')


def capture_initial(model, windows, budget):
    """Use the model's own positional/mask preparation, stopping before layer 0."""
    captured, metadata = [], {}
    class Captured(Exception):
        pass
    def hook(module, inputs, kwargs):
        captured.append(inputs[0].detach().cpu())
        metadata.update({k: v.detach().cpu() if torch.is_tensor(v) else v
                         for k, v in kwargs.items()})
        raise Captured()
    model.model.embed_tokens.cuda()
    handle = model.model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for ids in windows:
            try:
                model.model(ids.cuda(), use_cache=False)
            except Captured:
                pass
            budget.check()
    finally:
        handle.remove()
        model.model.embed_tokens.cpu()
    metadata.pop('past_key_value', None)
    return captured, metadata


def layer_kwargs(metadata):
    from utils.quant_phase import QuantPhase
    result = {k: v.cuda() if torch.is_tensor(v) else v for k, v in metadata.items()}
    result.update(use_cache=False, quant_phase=QuantPhase.PREFILL)
    return result


def set_weights(model, records):
    for name, row in records.items():
        q = unpack(row['packed'], row['shape'])
        w = (q.float() * row['scale']).to(torch.bfloat16)
        model.get_submodule(name).module.weight.data = w


def record_weight(q, codes, scale):
    packed = pack(codes.cpu())
    if not torch.equal(unpack(packed, tuple(codes.shape)), codes.cpu()):
        raise RuntimeError('INT4 pack roundtrip failed')
    return dict(packed=packed, shape=tuple(codes.shape), scale=scale.cpu())


@torch.no_grad()
def search_model(model, fp, scales, initial, metadata, selected_rows, arm, directory, budget,
                 layer_limit=16, strengths=STRENGTHS, c_weights=None):
    retained = arm == 'identity_learned'
    search = arm == 'search_recomputed'
    hidden = list(initial)
    records, down_alphas, ds, choices = {}, {}, {}, []
    for index, layer in enumerate(model.model.layers):
        if index >= layer_limit:
            break
        budget.check(f'{arm}/layer{index}/start')
        prefix = f'model.layers.{index}'
        layer.cuda()
        for local, wrapper in layer.named_modules():
            if not hasattr(wrapper, 'module') or not hasattr(wrapper, 'quantizer'):
                continue
            name = prefix + '.' + local
            scale = scales['weight'][name + '.module.quantizer'] if retained else None
            q, codes, sw = weight_quant(fp[name].cuda(), scale)
            if retained and c_weights is not None and not torch.equal(q.cpu(), c_weights[name+'.module.weight']):
                raise RuntimeError(f'Reconstructed C weight mismatch: {name}')
            wrapper.module.weight.data = q
            records[name] = record_weight(q, codes, sw)
            if local.endswith('down_proj'):
                wrapper.quantizer.bits = 16
            else:
                wrapper.quantizer.bits = 8
        # Attention uses the actual W4/non-down SA; stop at normalized MLP input.
        inputs = []
        class Captured(Exception):
            pass
        def capture(module, args):
            inputs.append(args[0].detach().cpu())
            raise Captured()
        handle = layer.mlp.register_forward_pre_hook(capture)
        kwargs = layer_kwargs(metadata)
        try:
            for h in hidden:
                try:
                    layer(h.cuda(), **kwargs)
                except Captured:
                    pass
                budget.check()
        finally:
            handle.remove()
        names = [prefix + '.mlp.' + p for p in ('gate_proj', 'up_proj', 'down_proj')]
        fg, fu, fd = [fp[n].cuda() for n in names]
        qg = layer.mlp.gate_proj.module.weight
        gsa = scales['activation'][names[0] + '.quantizer'].cuda()
        usa = scales['activation'][names[1] + '.quantizer'].cuda()
        # Same token rows for every candidate; reference excludes quantization.
        sample = torch.cat([h[0, rows] for h, rows in zip(inputs, selected_rows)]).cuda()
        reference = mlp(sample, fg, fu, fd)
        a = torch.zeros(fu.shape[0], device='cuda', dtype=torch.float32)
        base_up = layer.mlp.up_proj.module.weight
        for h in inputs:
            h = h.cuda()
            z = F.silu(F.linear(static_input(h, gsa), qg)) * F.linear(static_input(h, usa), base_up)
            a = torch.maximum(a, z.float().abs().reshape(-1, z.shape[-1]).amax(dim=0))
            budget.check()
        v = direction(a, fd) if search else torch.zeros_like(a)
        trials, best = [], None
        for strength in (strengths if search else (0.,)):
            d = (v * strength).exp()
            if strength == 0:
                up, down = fu, fd
            else:
                up, down = fuse_pair(fu.float(), fd.float(), d)
                up, down = up.to(fu.dtype), down.to(fd.dtype)
            qu, cu, su = weight_quant(up, scales['weight'][names[1] + '.module.quantizer'] if retained else None)
            qd, cd, sd = weight_quant(down, scales['weight'][names[2] + '.module.quantizer'] if retained else None)
            bound = 0.
            for h in inputs:
                h = h.cuda()
                z = F.silu(F.linear(static_input(h, gsa), qg)) * F.linear(static_input(h, usa), qu)
                bound = max(bound, float(z.abs().max()))
                if not torch.isfinite(z).all():
                    raise RuntimeError('Non-finite calibration activation')
                budget.check()
            alpha = bound if bound else 127.  # Zero-only row: safe unit step.
            output = mlp(sample, qg, qu, qd, gsa, usa, alpha)
            error = output.float() - reference.float()
            mse = float(error.square().mean())
            if not math.isfinite(mse):
                raise RuntimeError('Non-finite MLP MSE')
            trial = dict(t=strength, mse=mse, full_absmax=bound, alpha=alpha,
                         calibration_token_rows=sum(h.shape[1] for h in inputs),
                         mse_token_rows=sample.shape[0], d_min=float(d.min()),
                         d_max=float(d.max()), log_d_mean=float(d.double().log().mean()))
            trials.append(trial)
            if best is None or mse < best['mse']:
                best = dict(trial, up=qu.cpu(), down=qd.cpu(), d=d.cpu(),
                            up_record=record_weight(qu, cu, su), down_record=record_weight(qd, cd, sd))
        layer.mlp.up_proj.module.weight.data = best['up'].cuda()
        layer.mlp.down_proj.module.weight.data = best['down'].cuda()
        layer.mlp.down_proj.quantizer = FrozenCodebookQuantizer('int8', best['alpha']).cuda()
        down_alphas[names[2]] = best['alpha']
        ds[names[2]] = best['d']
        records[names[1]], records[names[2]] = best['up_record'], best['down_record']
        layer_result = dict(layer=index, chosen_t=best['t'], trials=trials,
                            direction_nonconstant=bool(v.max() - v.min() > 1e-8))
        choices.append(layer_result)
        dump(directory / f'layer_{index:02d}.json', layer_result)
        print(f'{arm} layer {index}: t={best["t"]}, mse={best["mse"]:.8g}, alpha={best["alpha"]:.8g}', flush=True)
        # Propagate the selected, frozen candidate across ALL calibration tokens.
        observed = {'rows': 0, 'outside_fullrange': 0}
        def check_range(module, args):
            x = args[0]
            observed['rows'] += x.numel() // x.shape[-1]
            observed['outside_fullrange'] += int((x.abs() > best['alpha']).sum())
        handle = layer.mlp.down_proj.register_forward_pre_hook(check_range)
        try:
            propagated = []
            for h in hidden:
                propagated.append(layer(h.cuda(), **kwargs)[0].cpu())
                budget.check()
            hidden = propagated
        finally:
            handle.remove()
        if observed['outside_fullrange']:
            raise RuntimeError(f'Frozen candidate exceeds calibrated range: {observed}')
        layer_result['propagation_check'] = observed
        dump(directory / f'layer_{index:02d}.json', layer_result)
        layer.cpu()
        del fg, fu, fd, qg, qu, qd, cu, cd, su, sd, inputs, sample, reference, output, error, best, h, z
        torch.cuda.empty_cache()
        budget.check(f'{arm}/layer{index}/complete', chosen_t=choices[-1]['chosen_t'])
    payload = dict(weights=records, down_alphas=down_alphas, D=ds,
                   non_down_sa={k: v.clone() for k, v in scales['activation'].items()},
                   sw_rule='C learned' if retained else 'all 112 current effective row absmax/7',
                   choices=choices)
    torch.save(payload, directory / 'packed_model.pt')
    loaded = torch.load(directory / 'packed_model.pt', map_location='cpu', weights_only=True)
    for name, row in records.items():
        for key in ('packed', 'scale'):
            if not torch.equal(row[key], loaded['weights'][name][key]):
                raise RuntimeError('Frozen checkpoint reload differs')
    if loaded['down_alphas'] != down_alphas:
        raise RuntimeError('SA reload differs')
    for key in ds:
        if not torch.equal(ds[key], loaded['D'][key]):
            raise RuntimeError('D reload differs')
    for key, value in scales['activation'].items():
        if not torch.equal(value, loaded['non_down_sa'][key]):
            raise RuntimeError('Non-down SA reload differs')
        model.get_submodule(key).load_scale(loaded['non_down_sa'][key])
    set_weights(model, loaded['weights'])
    for name, alpha in loaded['down_alphas'].items():
        model.get_submodule(name).quantizer = FrozenCodebookQuantizer('int8', alpha)
    dump(directory / 'search_summary.json', dict(arm=arm, choices=choices,
         frozen_reload_pass=True, weight_count=len(records), down_count=len(down_alphas)))
    return loaded


@torch.no_grad()
def diagnose(model, payload, enc, directory, budget):
    from utils.eval_utils import evaluator
    results = {}
    for mode in ('all_a16', 'down_a16', 'full_a8'):
        budget.check(f'{directory.name}/{mode}/validation')
        for name, module in model.named_modules():
            if name in payload['weights']:
                module.quantizer.bits = 16 if mode == 'all_a16' or (mode == 'down_a16' and name.endswith('down_proj')) else 8
        before = {n: {k: v.detach().cpu().clone() for k, v in m.quantizer.named_buffers()}
                  for n, m in model.named_modules() if n in payload['weights']}
        args = SimpleNamespace(eval_nsamples=None, bsz=1, capture_layer_io=False)
        result = evaluate_full_validation(model, enc, 'cuda', args, evaluator)
        budget.evaluations += 1
        if budget.evaluations > 9:
            raise RuntimeError('Nine-validation limit exceeded')
        for name, values in before.items():
            after = dict(model.get_submodule(name).quantizer.named_buffers())
            if any(not torch.equal(v, after[k].cpu()) for k, v in values.items()):
                raise RuntimeError('Frozen activation buffers changed')
        for name, row in payload['weights'].items():
            expected = (unpack(row['packed'], row['shape']).float() * row['scale']).to(torch.bfloat16)
            if not torch.equal(expected, model.get_submodule(name).module.weight.cpu()):
                raise RuntimeError('Frozen weight changed in evaluation')
        result.update(mode=mode, frozen_buffers_unchanged=True, frozen_weights_unchanged=True,
                      split='validation', seed=42, window_length=2048, use_cache=False, kv_bits=16)
        results[mode] = result
        dump(directory / f'{mode}.json', result)
        print(f'RESULT {directory.name} {mode}: PPL={result["ppl"]:.10f} NLL={result["nll"]:.10f}', flush=True)
        budget.check()
    return results


@torch.no_grad()
def run(options):
    from transformers import AutoConfig, LlamaTokenizerFast
    from datasets import load_dataset
    from eval_utils.modeling_llama import LlamaForCausalLM
    from eval_utils.rotation_utils import rotate_model
    from utils.fuse_norm_utils import fuse_layer_norms
    from utils.quant_utils import add_actquant, RotationStaticActQuantizer
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Leave 1 GiB for CUDA context/library memory outside the torch allocator.
    torch.cuda.set_per_process_memory_fraction(11 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    budget = Budget(options.output, options.prior_gpu_seconds)
    budget.check('loading_model')
    model_path = ROOT / 'cache/models/llama-3.2-1b-instruct'
    c = ROOT / 'runs/phase2/learned-sw-c-20260909.ByFYAM/C'
    settings = json.loads((ROOT / 'runs/phase2/down-codebooks-c-20260909.6YRIty/results/settings.json').read_text())
    dump(options.output / 'settings.json', dict(model_path=str(model_path), c_path=str(c),
         seed=42, train_windows=32, seqlen=2048, mse_rows_per_window=64,
         strengths=STRENGTHS, d_bounds=[.25,4], gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
         command=sys.argv, hypothesis='Channel balance improves current W4/full-range A8 without online transforms',
         reference='Same BF16 normalized MLP input and fixed-R BF16 weights; no quantizers',
         calibration_window_indices=settings['calibration_window_indices']))
    config = AutoConfig.from_pretrained(model_path)
    config.tie_word_embeddings = False
    model = LlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model.lm_head.weight.data = model.model.embed_tokens.weight.detach().clone()
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    model.seqlen = 2048
    fuse_layer_norms(model)
    rotate_model(model, SimpleNamespace(rotate_mode='hadamard', optimized_rotation_path=str(c/'rotation/R.bin'), rotation_components='r1_r2'))
    # ptq.train normally moves the entire model to CUDA before rotation.
    # Here weights are staged per layer, but model-level RoPE must match inputs.
    model.model.rotary_emb.cuda()
    add_actquant(model)
    scales = torch.load(c/'rotation/quant_scales.pt', map_location='cpu', weights_only=True)
    fp = {}
    for name, module in model.named_modules():
        if name.startswith('model.layers.') and hasattr(module, 'module') and hasattr(module, 'quantizer'):
            fp[name] = module.module.weight.detach().cpu().clone()
            q = RotationStaticActQuantizer()
            if not name.endswith('down_proj'):
                q.load_scale(scales['activation'][name+'.quantizer'])
            else:
                q.bits = 16
            module.quantizer = q
    assert len(fp) == 112 and len(scales['activation']) == 96
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path, model_max_length=2048,
                padding_side='right', use_fast=True, add_eos_token=False, add_bos_token=False)
    if options.precision_loop:
        from precision_loop import run_loop
        run_loop(model, fp, scales, tokenizer, c, options, budget)
        return
    if options.mixed_layer is not None:
        from mixed_layer_validation import run_mixed_layer
        run_mixed_layer(model, fp, scales, tokenizer, c, options, budget)
        return
    windows, indices = text_windows(tokenizer, 'train', 32, 2048, 42)
    assert indices == settings['calibration_window_indices']
    g = torch.Generator().manual_seed(42)
    rows = [torch.randperm(2048, generator=g)[:64] for _ in windows]
    initial, metadata = capture_initial(model, windows, budget)
    if options.followup_parent is not None:
        from down_d_followup import run_followup
        run_followup(model, fp, scales, initial, metadata, rows, windows, options, budget)
        return
    smoke = options.output / 'smoke'
    smoke.mkdir()
    search_model(model, fp, scales, initial[:2], metadata, rows[:2],
                 'search_recomputed', smoke, budget, layer_limit=1, strengths=(0.,1.))
    dump(smoke/'verdict.json', dict(status='PASS', windows=2, strengths=[0.,1.]))
    data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
    enc = tokenizer('\n\n'.join(data['text']), return_tensors='pt')
    assert enc.input_ids.numel() == 252852
    all_results = {}
    c_checkpoint = torch.load(c/'rtn/w4_rtn_model.pt', map_location='cpu', mmap=True, weights_only=False)
    try:
        for arm in ('identity_learned', 'identity_recomputed', 'search_recomputed'):
            directory = options.output / arm
            directory.mkdir()
            payload = search_model(model, fp, scales, initial, metadata, rows, arm, directory, budget,
                                   c_weights=c_checkpoint['model'])
            all_results[arm] = diagnose(model, payload, enc, directory, budget)
            dump(options.output / 'results.json', all_results)
            del payload
        budget.check('completed')
    finally:
        dump(options.output / 'budget.json', dict(gpu_seconds=time.monotonic()-budget.started,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
             peak_process_mib=budget.peak_process_mib, full_validation_evaluations=budget.evaluations))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-gpu-seconds', type=float, default=0.)
    parser.add_argument('--followup-parent', type=Path)
    parser.add_argument('--mixed-layer', type=int, choices=range(16))
    parser.add_argument('--precision-loop', choices=('fixed_code_sw', 'c_integrity', 'attribution', 'rounding', 'rounding_downs', 'rounding_mlp', 'rounding_coverage', 'rounding_downs_nll', 'sparse_d_sp2', 'sparse_d_rms_sp2', 'residual_diagnostics', 'sp2_alpha_expand', 'sp2_alpha_boundary', 'neighbor_rounding', 'rounding_continuation', 'neighbor_downs_nll', 'neighbor_mlp', 'external_c4', 'non_down_sensitivity', 'non_down_rounding'))
    parser.add_argument('--loop-parent', type=Path)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    if (options.output/'settings.json').exists():
        raise RuntimeError('Refusing to repeat an existing run')
    started = time.monotonic()
    try:
        run(options)
    except Exception as exc:
        dump(options.output/'failure.json', dict(error=repr(exc), pid=os.getpid()))
        raise
    finally:
        if not (options.output/'budget.json').exists():
            dump(options.output/'budget.json', dict(gpu_seconds=time.monotonic()-started,
                 prior_gpu_seconds=options.prior_gpu_seconds, initialization_failed=True))


if __name__ == '__main__':
    main()
