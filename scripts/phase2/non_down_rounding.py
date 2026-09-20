"""Finite non-down W4 code optimization selected by train-only sensitivity."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from down_d_search import capture_initial,layer_kwargs,pack,unpack,dump
from down_codebook_experiment import text_windows
from fixed_grid_rounding import coordinate_round
from precision_loop import configure,measure


@torch.no_grad()
def run_non_down(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    from utils.eval_utils import evaluator
    ranking=json.loads((options.loop_parent/'non_down_sensitivity.json').read_text())
    parent=Path(ranking['parent'])
    chosen=set(ranking['selected_weights'])
    names=[n for n in fp if n in chosen]
    assert 0<len(names)<=6 and all(not n.endswith('down_proj') for n in names)
    records=torch.load(parent/'rounding_weights.pt',map_location='cpu',weights_only=True)
    downs={n for n in fp if n.endswith('down_proj')}
    assert downs<=set(records)<=set(fp)
    replacements={n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
                  for n,r in records.items()}
    for n,r in records.items():assert torch.equal(r['scale'],scales['weight'][n+'.module.quantizer'])
    alphas=json.loads((parent/'down_scales.json').read_text())['sp2']
    configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
    windows,indices=text_windows(tokenizer,'train',32,2048,42)
    assert indices==ranking['calibration_window_indices']
    initial_hidden,metadata=capture_initial(model,windows,budget)
    best_nll=ranking['parent_train_nll']
    parent_results=json.loads((parent/'loop_results.json').read_text())
    assert len(parent_results)==1
    parent_result=next(iter(parent_results.values()))
    heldout=SimpleNamespace(input_ids=torch.cat(windows[24:],dim=1))
    args=SimpleNamespace(eval_nsamples=8,bsz=1,capture_layer_io=False)
    dump(options.output/'loop_settings.json',dict(parent=str(parent),ranking=str(options.loop_parent),
         targets=names,calibration_window_indices=indices,fit_rows=49152,heldout_rows=16384,
         milestones=[0,512,2048,8192],reference='Wfp applied to same actual static A8 module input',
         weights='Parent identity plus neighbor codes in original INT4 grid',
         sw='All112 originalC unchanged',activation='All96SA/16SP2alpha unchanged',
         selection='Heldout MSE prefilter then actual train NLL',gradients=False))
    any_changed=False
    details=options.output/'matrices';details.mkdir()
    for name in names:
        target_index=int(name.split('.')[2])
        fits,holds=[],[]
        collected=[0]
        class Captured(Exception):pass
        def collect(module,inputs,output):
            (fits if collected[0]<24 else holds).append(output[0].cpu())
            collected[0]+=1
            raise Captured()
        handle=model.get_submodule(name).quantizer.register_forward_hook(collect)
        hidden=initial_hidden
        try:
            for index,layer in enumerate(model.model.layers[:target_index+1]):
                layer.cuda();propagated=[]
                for h in hidden:
                    try:propagated.append(layer(h.cuda(),**layer_kwargs(metadata))[0].cpu())
                    except Captured:pass
                    budget.check('non_down/capture',target=name,layer=index)
                hidden=propagated;layer.cpu()
        finally:handle.remove()
        assert len(fits)==24 and len(holds)==8
        x=torch.cat(fits).cuda();hold=torch.cat(holds).cuda()
        w=fp[name].cuda();s=scales['weight'][name+'.module.quantizer'].cuda()
        if name in records:
            initial=unpack(records[name]['packed'],records[name]['shape'])
        else:
            initial=(w.float()/s).round().clamp(-8,7).to(torch.int8).cpu()
            assert torch.equal((initial.float()*s.cpu()).to(torch.bfloat16),checkpoint['model'][name+'.module.weight'])
        trials=coordinate_round(w,s,x,hold,milestones=(512,2048,8192),initial_codes=initial,
             neighbor_search=True,progress=lambda step:budget.check('non_down/round',target=name,step=step))
        assert torch.equal(trials[0]['codes'],initial)
        best=trials[0];best['train_nll']=best_nll
        del x,hold,fits,holds,w
        for trial in trials[1:]:
            if trial['heldout_mse']>=trials[0]['heldout_mse']:
                trial['train_nll']=None
                continue
            model.get_submodule(name).module.weight.data=(trial['codes'].float()*s.cpu()).to(torch.bfloat16)
            budget.check('non_down/train_nll',target=name,step=trial['step'])
            score=math.log(evaluator(model,heldout,'cuda',args))
            if not math.isfinite(score):raise RuntimeError('Non-finite non-down train NLL')
            trial['train_nll']=score
            if score<best['train_nll']:best=trial
        best_nll=best['train_nll']
        any_changed|=not torch.equal(best['codes'],initial)
        replacements[name]=(best['codes'].float()*s.cpu()).to(torch.bfloat16)
        model.get_submodule(name).module.weight.data=replacements[name].clone()
        records[name]=dict(packed=pack(best['codes']),shape=tuple(best['codes'].shape),scale=s.cpu())
        row=dict(name=name,selected_step=best['step'],trials=[{k:v for k,v in t.items() if k!='codes'} for t in trials])
        dump(details/(name+'.json'),row)
        print('NON_DOWN_ROUNDING '+json.dumps(row),flush=True)
        del trials,s,hidden
        torch.cuda.empty_cache()
    torch.save(records,options.output/'rounding_weights.pt')
    loaded=torch.load(options.output/'rounding_weights.pt',map_location='cpu',weights_only=True)
    for n,r in loaded.items():
        q=(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
        assert torch.equal(q,replacements[n])
        assert torch.equal(r['scale'],scales['weight'][n+'.module.quantizer'])
    dump(options.output/'down_scales.json',dict(sp2=alphas))
    dump(options.output/'selection_summary.json',dict(final_train_nll=best_nll))
    if any_changed:
        expected=configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
        directory=options.output/'non_down_rounded';directory.mkdir()
        result=measure(model,expected,enc,directory,budget)
    else:
        result=dict(parent_result,reused_frozen_parent=True)
    result.update(parent_ppl=parent_result['ppl'],delta_parent_ppl=result['ppl']-parent_result['ppl'],
                  delta_parent_nll=result['nll']-parent_result['nll'])
    results={'non_down_rounded':result}
    dump(options.output/'loop_results.json',results)
    return results
