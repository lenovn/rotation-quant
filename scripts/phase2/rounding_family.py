"""Sequential down-family rounding on actual selected upstream quantized outputs."""
import json
import math
from types import SimpleNamespace
import torch
from down_d_search import ROOT, capture_initial, layer_kwargs, pack, unpack, dump
from down_codebook_experiment import text_windows
from fixed_grid_rounding import coordinate_round
from precision_loop import configure,measure


@torch.no_grad()
def run_family(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    neighbor=options.precision_loop=='neighbor_downs_nll'
    parent_records,parent_result=None,None
    if neighbor:
        if options.loop_parent is None:raise ValueError('Explicit neighbor parent required')
        parent_records=torch.load(options.loop_parent/'rounding_weights.pt',map_location='cpu',weights_only=True)
        assert set(parent_records)=={n for n in fp if n.endswith('down_proj')}
        alphas=json.loads((options.loop_parent/'down_scales.json').read_text())['sp2']
        parent_weights={n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
                        for n,r in parent_records.items()}
        parent_results=json.loads((options.loop_parent/'loop_results.json').read_text())
        assert len(parent_results)==1
        parent_result=next(iter(parent_results.values()))
        configure(model,fp,scales,checkpoint,alphas,replacements=parent_weights)
    else:
        configure(model,fp,scales,checkpoint,alphas)
    windows,indices=text_windows(tokenizer,'train',32,2048,42)
    settings=json.loads((ROOT/'runs/phase2/down-codebooks-c-20260909.6YRIty/results/settings.json').read_text())
    assert indices==settings['calibration_window_indices']
    hidden,metadata=capture_initial(model,windows,budget)
    g=torch.Generator().manual_seed(42)
    rows=[torch.randperm(2048,generator=g)[:128] for _ in windows]
    replacements,records,summaries={},{},[]
    nll_selection=options.precision_loop in ('rounding_downs_nll','neighbor_downs_nll')
    milestones=(512,2048,8192) if nll_selection else (32,128,512)
    train_scores=0
    best_nll=None
    if nll_selection:
        from utils.eval_utils import evaluator
        heldout_enc=SimpleNamespace(input_ids=torch.cat(windows[24:],dim=1))
        eval_args=SimpleNamespace(eval_nsamples=8,bsz=1,capture_layer_io=False)
        best_nll=math.log(evaluator(model,heldout_enc,'cuda',eval_args))
        train_scores=1
        dump(options.output/'original_train_nll.json',dict(nll=best_nll,ppl=math.exp(best_nll),
             windows=indices[24:],predicted_tokens=8*2047))
    dump(options.output/'loop_settings.json',dict(hypothesis='Sequential fixed-SW down rounding retains aggregate W4A8 benefit',
         neighbor_search=neighbor,parent=str(options.loop_parent) if neighbor else None,
         calibration_window_indices=indices,fit_rows=49152 if nll_selection else 2048,
         heldout_rows=16384 if nll_selection else 2048,milestones=[0,*milestones],
         upstream='Selected preceding W4+frozen A8 blocks',reference='Wfp applied to same actual SP2 input',
         sw='Original C learned; fixed all112',recalibration=False,
         heldout='24/8 disjoint full train windows' if nll_selection else 'disjoint rows in same train windows',
         selection='Local heldout MSE prefilter then true heldout-train NLL vs current prefix' if nll_selection else 'heldout MSE'))
    kwargs=layer_kwargs(metadata)
    for index,layer in enumerate(model.model.layers):
        name=f'model.layers.{index}.mlp.down_proj'
        layer.cuda();fits,holds=[],[]
        collected=[0]
        class Captured(Exception):pass
        def collect(module,args,output):
            if nll_selection:
                (fits if collected[0]<24 else holds).append(output[0].cpu())
                collected[0]+=1
                raise Captured()
            r=rows[len(fits)].to(output.device)
            fits.append(output[0,r[:64]].cpu());holds.append(output[0,r[64:]].cpu())
            raise Captured()
        handle=layer.mlp.down_proj.quantizer.register_forward_hook(collect)
        try:
            for h in hidden:
                try:layer(h.cuda(),**kwargs)
                except Captured:pass
                budget.check(f'family/layer{index}/capture')
        finally:handle.remove()
        assert (len(fits),len(holds))==((24,8) if nll_selection else (32,32))
        x=torch.cat(fits).cuda();hold=torch.cat(holds).cuda()
        w=fp[name].cuda();s=scales['weight'][name+'.module.quantizer'].cuda()
        original=(w.float()/s).round().clamp(-8,7).to(torch.int8)
        assert torch.equal((original.float()*s).to(w.dtype).cpu(),checkpoint['model'][name+'.module.weight'])
        initial=None
        if neighbor:
            r=parent_records[name]
            assert torch.equal(r['scale'],s.cpu())
            initial=unpack(r['packed'],r['shape'])
        trials=coordinate_round(w,s,x,hold,milestones=milestones,
                                initial_codes=initial,neighbor_search=neighbor,
                                progress=lambda step:budget.check(f'family/layer{index}/round',step=step))
        if neighbor:assert torch.equal(trials[0]['codes'],initial)
        best=min(trials,key=lambda t:t['heldout_mse'])
        if nll_selection:
            best=trials[0]
            best['train_nll']=best_nll
            for trial in trials[1:]:
                if trial['heldout_mse']>=trials[0]['heldout_mse']:
                    trial['train_nll']=None
                    continue
                layer.mlp.down_proj.module.weight.data=(trial['codes'].float()*s.cpu()).to(torch.bfloat16)
                budget.check(f'family/layer{index}/train_nll',candidate_step=trial['step'],train_scores=train_scores)
                score=math.log(evaluator(model,heldout_enc,'cuda',eval_args))
                train_scores+=1
                if not math.isfinite(score):raise RuntimeError('Non-finite train NLL')
                trial['train_nll']=score
                if score<best['train_nll']:best=trial
            best_nll=best['train_nll']
        summaries.append(dict(layer=index,selected_step=best['step'],
                         trials=[{k:v for k,v in t.items() if k!='codes'} for t in trials]))
        dump(options.output/f'layer_{index:02d}.json',summaries[-1])
        codes=best['codes'];p=pack(codes);assert torch.equal(unpack(p,codes.shape),codes)
        records[name]=dict(packed=p,shape=tuple(codes.shape),scale=s.cpu())
        replacements[name]=(codes.float()*s.cpu()).to(torch.bfloat16)
        layer.cuda()
        layer.mlp.down_proj.module.weight.data=replacements[name].cuda()
        propagated=[]
        for h in hidden:
            propagated.append(layer(h.cuda(),**kwargs)[0].cpu())
            budget.check(f'family/layer{index}/propagate')
        hidden=propagated;layer.cpu()
        print('FAMILY_LAYER '+json.dumps(summaries[-1]),flush=True)
        del x,hold,w,s,trials,fits,holds,h,original
        torch.cuda.empty_cache()
    torch.save(records,options.output/'rounding_weights.pt')
    loaded=torch.load(options.output/'rounding_weights.pt',map_location='cpu',weights_only=True)
    for name,record in loaded.items():
        w=(unpack(record['packed'],record['shape']).float()*record['scale']).to(torch.bfloat16)
        assert torch.equal(w,replacements[name])
        assert torch.equal(record['scale'],scales['weight'][name+'.module.quantizer'])
        replacements[name]=w
    directory=options.output/'rounded_downs';directory.mkdir()
    expected=configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
    unchanged=neighbor and all(torch.equal(r['packed'],parent_records[n]['packed']) for n,r in loaded.items())
    result=dict(parent_result,reused_frozen_parent=True) if unchanged else measure(model,expected,enc,directory,budget)
    results={'rounded_downs':result}
    if neighbor:
        dump(options.output/'down_scales.json',dict(sp2=alphas))
        dump(options.output/'selection_summary.json',dict(final_train_nll=best_nll))
        result.update(parent_ppl=parent_result['ppl'],delta_parent_ppl=result['ppl']-parent_result['ppl'],
                      delta_parent_nll=result['nll']-parent_result['nll'])
    dump(options.output/'loop_results.json',results)
    return results
