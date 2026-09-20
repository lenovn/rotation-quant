"""Finite static SP2 range expansion on frozen parent integer weights."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from down_d_search import ROOT, FrozenCodebookQuantizer, unpack, dump
from down_codebook_experiment import text_windows
from precision_loop import configure, measure


@torch.no_grad()
def run_alpha(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    from utils.eval_utils import evaluator
    parent=options.loop_parent
    if parent is None:raise ValueError('Explicit frozen rounding parent required')
    boundary=options.precision_loop=='sp2_alpha_boundary'
    parent_settings=json.loads((parent/'loop_settings.json').read_text())
    weight_parent=Path(parent_settings.get('weight_parent',parent_settings['parent'])) if boundary else parent
    if boundary:alphas=json.loads((parent/'down_scales.json').read_text())['sp2']
    records=torch.load(weight_parent/'rounding_weights.pt',map_location='cpu',weights_only=True)
    names=[n for n in fp if n.endswith('down_proj')]
    assert set(records)==set(names) and len(names)==16
    replacements={n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
                  for n,r in records.items()}
    for n,r in records.items():assert torch.equal(r['scale'],scales['weight'][n+'.module.quantizer'])
    expected=configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
    windows,indices=text_windows(tokenizer,'train',32,2048,42)
    settings=json.loads((weight_parent/'loop_settings.json').read_text())
    assert indices==settings['calibration_window_indices']
    assert settings['heldout']=='24/8 disjoint full train windows'
    last=json.loads((parent/'layer_15.json').read_text())
    key='factor' if boundary else 'step'
    best_nll=next(t['train_nll'] for t in last['trials'] if t[key]==last['selected_'+key])
    parent_result=json.loads((parent/'loop_results.json').read_text())['expanded_sp2' if boundary else 'rounded_downs']
    heldout=SimpleNamespace(input_ids=torch.cat(windows[24:],dim=1))
    args=SimpleNamespace(eval_nsamples=8,bsz=1,capture_layer_io=False)
    factors=(1.,2.,4.,8.,16.) if boundary else (1.,1.125,1.25,1.5,2.)
    searched=set(names)
    if boundary:
        assert indices==parent_settings['calibration_window_indices']
        searched={json.loads(p.read_text())['name'] for p in parent.glob('layer_*.json')
                  if json.loads(p.read_text())['selected_factor']==2.}
        if not searched:raise ValueError('No boundary optimum to expand')
    selected=dict(alphas)
    dump(options.output/'loop_settings.json',dict(hypothesis='Task NLL can improve static SP2 range after weight rounding',
         parent=str(parent),weight_parent=str(weight_parent),factors=factors,calibration_window_indices=indices,
         searched_layers=sorted(searched),boundary_followup=boundary,
         selection_windows=indices[24:],predicted_train_tokens=8*2047,
         weights='All parent codes and112SW frozen',non_down_sa='All96 frozen',
         down_range='Only expansion relative to original SP2; never threshold shrink',
         selection='Greedy actual next-token train NLL, previous accepted prefix retained',
         original_train_nll=best_nll,teacher=False,gradients=False))
    for index,name in enumerate(names):
        if name not in searched:continue
        original=float(alphas[name])
        trials=[dict(factor=1.,alpha=original,train_nll=best_nll,reused_prefix=True)]
        best=trials[0]
        for factor in factors[1:]:
            alpha=original*factor
            assert alpha>=original and math.isfinite(alpha)
            model.get_submodule(name).quantizer=FrozenCodebookQuantizer('sp2',alpha)
            budget.check('alpha/train_nll',layer=index,factor=factor)
            score=math.log(evaluator(model,heldout,'cuda',args))
            if not math.isfinite(score):raise RuntimeError('Non-finite alpha train NLL')
            trial=dict(factor=factor,alpha=alpha,train_nll=score)
            trials.append(trial)
            if score<best['train_nll']:best=trial
        selected[name]=best['alpha']
        best_nll=best['train_nll']
        model.get_submodule(name).quantizer=FrozenCodebookQuantizer('sp2',best['alpha'])
        row=dict(layer=index,name=name,selected_factor=best['factor'],selected_alpha=best['alpha'],trials=trials)
        dump(options.output/f'layer_{index:02d}.json',row)
        print('ALPHA_LAYER '+json.dumps(row),flush=True)
    dump(options.output/'down_scales.json',dict(sp2=selected))
    dump(options.output/'selection_summary.json',dict(final_train_nll=best_nll,search_layers=len(searched)))
    loaded=json.loads((options.output/'down_scales.json').read_text())['sp2']
    assert loaded==selected
    assert all(loaded[n]>=float(alphas[n]) for n in names)
    if loaded==alphas:
        result=dict(parent_result,reused_frozen_parent=True)
    else:
        expected=configure(model,fp,scales,checkpoint,loaded,replacements=replacements)
        directory=options.output/'expanded_sp2';directory.mkdir()
        result=measure(model,expected,enc,directory,budget)
    result.update(parent_ppl=parent_result['ppl'],parent_nll=parent_result['nll'],
         delta_parent_ppl=result['ppl']-parent_result['ppl'],delta_parent_nll=result['nll']-parent_result['nll'])
    results=dict(expanded_sp2=result)
    dump(options.output/'loop_results.json',results)
    return results
