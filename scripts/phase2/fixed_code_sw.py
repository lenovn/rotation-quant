"""Bounded SW expansion at frozen integer codes, R, D and activation scales."""
import json
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from down_d_search import capture_initial, layer_kwargs, unpack, dump
from down_codebook_experiment import text_windows
from precision_loop import configure, measure


def bounded_scale_fit(numerator, denominator, initial):
    """Constrained row least squares; unobserved rows retain the initial scale."""
    base=initial.reshape(-1).float()
    assert numerator.shape==denominator.shape==base.shape
    assert torch.isfinite(numerator).all() and torch.isfinite(denominator).all()
    assert torch.isfinite(base).all() and (base>0).all() and (denominator>=0).all()
    fitted=torch.where(denominator>1e-20,numerator/denominator.clamp_min(1e-20),base)
    return torch.minimum(torch.maximum(fitted,base),base*1.125).reshape(initial.shape)


def decode(records):
    return {n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
            for n,r in records.items()}


@torch.no_grad()
def run_sw(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    from utils.eval_utils import evaluator
    parent=options.loop_parent
    payload=torch.load(parent/'selected_d/packed_model.pt',map_location='cpu',weights_only=True)
    records=payload['weights']
    downs=[n for n in fp if n.endswith('down_proj')]
    assert len(downs)==16 and set(downs)<=set(records)<=set(fp)
    d=payload['D']
    assert d.shape==(fp['model.layers.1.mlp.down_proj'].shape[1],)
    assert torch.isfinite(d).all() and (d>0).all()
    alphas=json.loads((parent/'down_scales.json').read_text())['sp2']
    assert set(alphas)==set(downs)
    parent_result=json.loads((parent/'loop_results.json').read_text())['selected_d']
    parent_nll=json.loads((parent/'selection_summary.json').read_text())['final_train_nll']
    configure(model,fp,scales,checkpoint,alphas,replacements=decode(records))
    windows,indices=text_windows(tokenizer,'train',32,2048,42)
    assert indices==json.loads((parent/'loop_settings.json').read_text())['calibration_window_indices']
    hidden,metadata=capture_initial(model,windows,budget)
    fitted={}
    fit_stats={}
    for index,layer in enumerate(model.model.layers):
        name=f'model.layers.{index}.mlp.down_proj'
        codes=unpack(records[name]['packed'],records[name]['shape']).float().cuda()
        reference=fp[name]
        if index==1:reference=(reference.float()*d[None,:]).to(torch.bfloat16)
        reference=reference.float().cuda()
        numerator=torch.zeros(codes.shape[0],dtype=torch.float64,device='cuda')
        denominator=torch.zeros_like(numerator)
        count=[0]
        def collect(module,inputs,output):
            if count[0]<24:
                x=output.reshape(-1,output.shape[-1]).float()
                qy=F.linear(x,codes)
                target=F.linear(x,reference)
                numerator.add_((qy.double()*target.double()).sum(0))
                denominator.add_(qy.double().square().sum(0))
            count[0]+=1
        handle=layer.mlp.down_proj.quantizer.register_forward_hook(collect)
        layer.cuda();propagated=[]
        try:
            for h in hidden:
                propagated.append(layer(h.cuda(),**layer_kwargs(metadata))[0].cpu())
                budget.check('fixed_sw/capture',layer=index,window=count[0])
        finally:handle.remove()
        assert count[0]==32
        initial=records[name]['scale']
        fitted[name]=bounded_scale_fit(numerator.cpu(),denominator.cpu(),initial).float()
        ratio=fitted[name]/initial
        fit_stats[name]=dict(expanded_rows=int((ratio>1).sum()),ratio_min=float(ratio.min()),ratio_max=float(ratio.max()))
        hidden=propagated;layer.cpu()
        del codes,reference,numerator,denominator
        torch.cuda.empty_cache()
    del hidden
    strengths=(0.,.25,.5,.75,1.)
    dump(options.output/'loop_settings.json',dict(parent=str(parent),hypothesis='Expand SW at fixed optimized integer codes',
         strengths=strengths,ratio_bounds=[1,1.125],targets=downs,fit_rows=49152,heldout_rows=16384,
         calibration_window_indices=indices,reference='D-fused FP down on identical actual parent SP2 input',
         fitting_path='One frozen parent pass; each full candidate NLL uses its actual forward',
         frozen='R/D/all integer codes/96 SA/16 alpha/non-down SW',gradients=False,fit_stats=fit_stats))
    held=SimpleNamespace(input_ids=torch.cat(windows[24:],dim=1))
    args=SimpleNamespace(eval_nsamples=8,bsz=1,capture_layer_io=False)
    trials=[dict(strength=0.,train_nll=parent_nll,reused_parent=True)]
    best_nll=parent_nll;best_strength=0.;best_records=records
    for strength in strengths[1:]:
        candidate=dict(records)
        for name in downs:
            initial=records[name]['scale']
            scale=initial+strength*(fitted[name]-initial)
            assert (scale>=initial).all() and (scale<=initial*1.125+1e-12).all()
            candidate[name]=dict(records[name],scale=scale)
        configure(model,fp,scales,checkpoint,alphas,replacements=decode(candidate))
        budget.check('fixed_sw/train_nll',strength=strength)
        nll=math.log(evaluator(model,held,'cuda',args))
        if not math.isfinite(nll):raise RuntimeError('Non-finite SW candidate NLL')
        row=dict(strength=strength,train_nll=nll)
        trials.append(row);dump(options.output/'trials.json',trials)
        print('FIXED_SW '+json.dumps(row),flush=True)
        if nll<best_nll:best_nll=nll;best_strength=strength;best_records=candidate
    output=dict(payload,weights=best_records)
    torch.save(output,options.output/'packed_model.pt')
    loaded=torch.load(options.output/'packed_model.pt',map_location='cpu',weights_only=True)
    assert torch.equal(loaded['D'],d) and set(loaded['weights'])==set(records)
    for name,r in loaded['weights'].items():
        assert torch.equal(r['packed'],records[name]['packed']) and r['shape']==records[name]['shape']
        assert torch.equal(r['scale'],best_records[name]['scale'])
        if name not in downs:assert torch.equal(r['scale'],records[name]['scale'])
    dump(options.output/'down_scales.json',dict(sp2=alphas))
    dump(options.output/'selection_summary.json',dict(final_train_nll=best_nll,selected_strength=best_strength,trials=trials))
    if best_strength:
        expected=configure(model,fp,scales,checkpoint,alphas,replacements=decode(loaded['weights']))
        directory=options.output/'fixed_code_sw';directory.mkdir()
        result=measure(model,expected,enc,directory,budget)
    else:result=dict(parent_result,reused_frozen_parent=True)
    result.update(parent_ppl=parent_result['ppl'],delta_parent_ppl=result['ppl']-parent_result['ppl'],
                  delta_parent_nll=result['nll']-parent_result['nll'])
    results={'fixed_code_sw':result};dump(options.output/'loop_results.json',results)
    return results
