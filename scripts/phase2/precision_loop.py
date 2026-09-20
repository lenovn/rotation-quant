"""Measured precision experiments on the frozen original C + SP2 background."""
import json
import math
import time
from types import SimpleNamespace

import torch

from down_d_search import ROOT, FrozenCodebookQuantizer, dump, evaluate_full_validation


def configure(model, fp, scales, checkpoint, alphas, restore_w=(), restore_a=(), replacements=None):
    expected = {}
    replacements = replacements or {}
    for name in fp:
        wrapper = model.get_submodule(name)
        w = replacements.get(name, fp[name] if name in restore_w else checkpoint['model'][name+'.module.weight'])
        expected[name] = w
        wrapper.module.weight.data = w.clone()
        if name.endswith('down_proj'):
            wrapper.quantizer = FrozenCodebookQuantizer('sp2', alphas[name])
        else:
            wrapper.quantizer.load_scale(scales['activation'][name+'.quantizer'])
        wrapper.quantizer.bits = 16 if name in restore_a else 8
        assert wrapper.out_quantizer.bits == 16
        assert not wrapper.online_full_had and not wrapper.online_partial_had
    backbone_keys = {name+suffix for name in fp for suffix in ('.weight','.module.weight','.bias','.module.bias')}
    state = model.state_dict()
    for name, w in checkpoint['model'].items():
        if name in state and name not in backbone_keys and (name.endswith('.weight') or name.endswith('.bias')):
            if not torch.equal(state[name].cpu(), w.cpu()):
                raise RuntimeError(f'Boundary mismatch: {name}')
    return expected


@torch.no_grad()
def measure(model, expected, enc, directory, budget, restore_w=(), restore_a=()):
    before = {name: {k:v.cpu().clone() for k,v in model.get_submodule(name).quantizer.named_buffers()}
              for name in expected}
    budget.check(directory.name+'/validation')
    result = evaluate_full_validation(model, enc, 'cuda',
             SimpleNamespace(eval_nsamples=None, bsz=1, capture_layer_io=False), __import__('utils.eval_utils',fromlist=['evaluator']).evaluator)
    budget.evaluations += 1
    for name,w in expected.items():
        wrapper = model.get_submodule(name)
        assert torch.equal(wrapper.module.weight.cpu(),w.cpu()), name
        assert wrapper.quantizer.bits == (16 if name in restore_a else 8), name
        after = dict(wrapper.quantizer.named_buffers())
        assert all(torch.equal(v,after[k].cpu()) for k,v in before[name].items()),name
    assert math.isfinite(result['nll']) and result['predicted_tokens']==252728
    result.update(restored_w=list(restore_w), restored_a=list(restore_a), frozen_checks_pass=True,
                  delta_ppl=result['ppl']-17.64239997588965, delta_nll=result['nll']-2.8703050943792365,
                  split='validation',kv_bits=16,use_cache=False,reference_ppl=17.64239997588965)
    dump(directory/'result.json',result)
    print('LOOP_RESULT '+directory.name+' '+json.dumps(result),flush=True)
    budget.check(directory.name+'/complete')
    return result


@torch.no_grad()
def run_loop(model, fp, scales, tokenizer, c, options, budget):
    from datasets import load_dataset
    checkpoint = torch.load(c/'rtn/w4_rtn_model.pt',map_location='cpu',mmap=True,weights_only=False)
    alpha_path=ROOT/'runs/phase2/down-codebooks-c-20260909.6YRIty/results/down_scales.json'
    alphas=json.loads(alpha_path.read_text())['sp2']
    data=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='validation')
    enc=tokenizer('\n\n'.join(data['text']),return_tensors='pt')
    assert enc.input_ids.numel()==252852
    results={}
    try:
        if options.precision_loop=='c_integrity':
            # Isolate the C rotation/fusion numerical path from all W/A quantization.
            names=list(fp)
            assert len(names)==112
            directory=options.output/'c_w16a16';directory.mkdir()
            dump(options.output/'loop_settings.json',dict(hypothesis='Check C rotation/fusion with all backbone W/A16',
                 rotation=str(c/'rotation/R.bin'),restore_w=names,restore_a=names,
                 recalibration=False,selection=False,reference='Original unrotated BF16 matched validation'))
            expected=configure(model,fp,scales,checkpoint,alphas,names,names)
            results['c_w16a16']=measure(model,expected,enc,directory,budget,names,names)
            dump(options.output/'loop_results.json',results)
        elif options.precision_loop=='attribution':
            layer=[n for n in fp if n.startswith('model.layers.1.')]
            down=['model.layers.1.mlp.down_proj']
            assert len(layer)==7
            cases=[('layer1_w16',layer,[]),('layer1_a16',[],layer),('layer1_down_w16a16',down,down)]
            dump(options.output/'loop_settings.json',dict(hypothesis='Disentangle layer1 weight and activation sensitivity',
                 cases=cases,base=str(c/'rtn/w4_rtn_model.pt'),alphas=str(alpha_path),recalibration=False))
            for case,rw,ra in cases:
                directory=options.output/case;directory.mkdir()
                expected=configure(model,fp,scales,checkpoint,alphas,rw,ra)
                results[case]=measure(model,expected,enc,directory,budget,rw,ra)
                dump(options.output/'loop_results.json',results)
        elif options.precision_loop in ('rounding','rounding_mlp','rounding_coverage'):
            results=run_rounding(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        elif options.precision_loop in ('sparse_d_sp2','sparse_d_rms_sp2'):
            from sparse_d_sp2 import run_sparse
            results=run_sparse(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        elif options.precision_loop in ('sp2_alpha_expand','sp2_alpha_boundary'):
            from sp2_alpha_loop import run_alpha
            results=run_alpha(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        elif options.precision_loop in ('neighbor_rounding','rounding_continuation','neighbor_mlp'):
            from neighbor_rounding_loop import run_neighbor
            results=run_neighbor(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        elif options.precision_loop=='external_c4':
            from frozen_c4_loop import run_c4
            results=run_c4(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        elif options.precision_loop=='non_down_sensitivity':
            from non_down_sensitivity import run_sensitivity
            results=run_sensitivity(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        elif options.precision_loop=='fixed_code_sw':
            from fixed_code_sw import run_sw
            results=run_sw(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        elif options.precision_loop=='non_down_rounding':
            from non_down_rounding import run_non_down
            results=run_non_down(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        elif options.precision_loop=='residual_diagnostics':
            if options.loop_parent is None:
                raise ValueError('An explicit frozen rounding parent is required')
            from down_d_search import unpack
            parent=options.loop_parent
            records=torch.load(parent/'rounding_weights.pt',map_location='cpu',weights_only=True)
            downs=[n for n in fp if n.endswith('down_proj')]
            others=[n for n in fp if n not in downs]
            assert set(downs)<=set(records)<=set(fp) and len(downs)==16 and len(others)==96
            parent_alpha=parent/'down_scales.json'
            if parent_alpha.exists():
                alphas=json.loads(parent_alpha.read_text())['sp2']
            assert set(alphas)==set(downs)
            replacements={n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
                          for n,r in records.items()}
            for n,r in records.items():
                assert torch.equal(r['scale'],scales['weight'][n+'.module.quantizer'])
            cases=[('all_a16',[],list(fp)),('down_a16',[],downs),
                   ('down_w16',downs,[]),('non_down_w16',others,[])]
            parent_results=json.loads((parent/'loop_results.json').read_text())
            parent_key='non_down_rounded' if 'non_down_rounded' in parent_results else 'rounded_downs'
            results['full_a8']=dict(parent_results[parent_key],reused_frozen_parent=True)
            assert results['full_a8']['frozen_checks_pass']
            assert not results['full_a8']['restored_w'] and not results['full_a8']['restored_a']
            dump(options.output/'loop_settings.json',dict(hypothesis='Locate residual error after accepted weight and alpha optimization',
                 parent=str(parent),cases=cases,recalibration=False,selection=False,
                 note='High-precision conditions are diagnostics only; shared parent integer weights and scales'))
            for case,rw,ra in cases:
                directory=options.output/case;directory.mkdir()
                active={n:w for n,w in replacements.items() if n not in rw}
                expected=configure(model,fp,scales,checkpoint,alphas,rw,ra,replacements=active)
                results[case]=measure(model,expected,enc,directory,budget,rw,ra)
                results[case].update(parent_ppl=results['full_a8']['ppl'],
                     delta_parent_ppl=results[case]['ppl']-results['full_a8']['ppl'],
                     delta_parent_nll=results[case]['nll']-results['full_a8']['nll'])
                dump(directory/'result.json',results[case])
                dump(options.output/'loop_results.json',results)
        else:
            from rounding_family import run_family
            results=run_family(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget)
        budget.check('precision_loop_completed')
    finally:
        dump(options.output/'budget.json',dict(gpu_seconds=time.monotonic()-budget.started,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,peak_process_mib=budget.peak_process_mib,
             full_validation_evaluations=budget.evaluations))


@torch.no_grad()
def run_rounding(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    from down_d_search import capture_initial,layer_kwargs,pack,unpack
    from down_codebook_experiment import text_windows
    from fixed_grid_rounding import coordinate_round
    name='model.layers.1.mlp.down_proj'
    configure(model,fp,scales,checkpoint,alphas)
    windows,indices=text_windows(tokenizer,'train',32,2048,42)
    settings=json.loads((ROOT/'runs/phase2/down-codebooks-c-20260909.6YRIty/results/settings.json').read_text())
    assert indices==settings['calibration_window_indices']
    hidden,metadata=capture_initial(model,windows,budget)
    g=torch.Generator().manual_seed(42)
    rows=[torch.randperm(2048,generator=g)[:128] for _ in windows]
    fits,heldouts=[],[]
    reference_inputs=[]
    mlp_reference=options.precision_loop=='rounding_mlp'
    full_coverage=options.precision_loop=='rounding_coverage'
    collected=[0]
    def collect_mlp(module,args):
        r=rows[len(reference_inputs)].to(args[0].device)
        reference_inputs.append(args[0][0,r].cpu())
    target=model.get_submodule(name)
    def collect(module,args,output):
        if full_coverage:
            (fits if collected[0]<24 else heldouts).append(output[0].cpu())
            collected[0]+=1
            return
        i=len(fits);r=rows[i].to(output.device)
        fits.append(output[0,r[:64]].cpu())
        heldouts.append(output[0,r[64:]].cpu())
    handle=target.quantizer.register_forward_hook(collect)
    reference_handle=model.model.layers[1].mlp.register_forward_pre_hook(collect_mlp) if mlp_reference else None
    try:
        for i,layer in enumerate(model.model.layers):
            if i>1:break
            layer.cuda();propagated=[]
            for h in hidden:
                propagated.append(layer(h.cuda(),**layer_kwargs(metadata))[0].cpu())
                budget.check(f'rounding/capture/layer{i}')
            hidden=propagated;layer.cpu()
    finally:
        handle.remove()
        if reference_handle is not None:reference_handle.remove()
    assert (len(fits),len(heldouts))==((24,8) if full_coverage else (32,32))
    x=torch.cat(fits).cuda();hold=torch.cat(heldouts).cuda()
    w=fp[name].cuda();s=scales['weight'][name+'.module.quantizer'].cuda()
    yfit,yhold=None,None
    if mlp_reference:
        from down_d_search import mlp
        assert len(reference_inputs)==32
        fg=fp['model.layers.1.mlp.gate_proj'].cuda()
        fu=fp['model.layers.1.mlp.up_proj'].cuda()
        reference_outputs=[mlp(h.cuda(),fg,fu,w).cpu() for h in reference_inputs]
        yfit=torch.cat([y[:64] for y in reference_outputs]).cuda()
        yhold=torch.cat([y[64:] for y in reference_outputs]).cuda()
        del fg,fu,reference_outputs,reference_inputs
    original=(w.float()/s).round().clamp(-8,7).to(torch.int8)
    assert torch.equal((original.float()*s).to(w.dtype).cpu(),checkpoint['model'][name+'.module.weight'])
    milestones=(512,2048,8192) if full_coverage else (32,128,512)
    dump(options.output/'loop_settings.json',dict(hypothesis='Fixed-SW rounding recovers sensitive down weight accuracy',
         target=name,calibration_window_indices=indices,fit_rows=x.shape[0],heldout_rows=hold.shape[0],
         fit_window_indices=indices[:24] if full_coverage else indices,
         heldout_window_indices=indices[24:] if full_coverage else indices,
         heldout=('Disjoint full train windows24/8; no validation selection' if full_coverage else
                  'Disjoint rows within same train windows, not validation or independent documents'),
         milestones=[0,*milestones],scales_frozen=True,activation='actual frozen SP2 output',
         objective=('Same-input same-R BF16 MLP reference; candidate down sees actual W4/A8 gate/up and SP2 input'
                    if mlp_reference else '||(Wq-Wfp) X_SP2||^2; weight-only local reconstruction'),recalibration=False))
    trials=coordinate_round(w,s,x,hold,milestones=milestones,progress=lambda step:budget.check('rounding/coordinate',step=step),
                            target_fit=yfit,target_heldout=yhold)
    best=min(trials,key=lambda t:t['heldout_mse'])
    summary=[{k:v for k,v in t.items() if k!='codes'} for t in trials]
    dump(options.output/'rounding_trials.json',dict(trials=summary,selected_step=best['step']))
    for t in summary:print('ROUND_TRIAL '+json.dumps(t),flush=True)
    q=best['codes'];packed=pack(q)
    assert torch.equal(unpack(packed,q.shape),q)
    payload=dict(target=name,packed=packed,shape=tuple(q.shape),scale=s.cpu(),selected_step=best['step'])
    torch.save(payload,options.output/'rounding_weights.pt')
    loaded=torch.load(options.output/'rounding_weights.pt',map_location='cpu',weights_only=True)
    restored=unpack(loaded['packed'],loaded['shape'])
    assert torch.equal(restored,q) and torch.equal(loaded['scale'],s.cpu())
    replacement=(restored.float()*loaded['scale']).to(torch.bfloat16)
    del x,hold,w,s,hidden,fits,heldouts,trials,yfit,yhold
    torch.cuda.empty_cache()
    if best['step']==0:
        result=dict(status='no_local_gain',reference_ppl=17.64239997588965)
    else:
        directory=options.output/'rounded_down1';directory.mkdir()
        expected=configure(model,fp,scales,checkpoint,alphas,replacements={name:replacement})
        result=measure(model,expected,enc,directory,budget)
    results={'rounded_down1':result}
    dump(options.output/'loop_results.json',results)
    return results
