"""Bounded neighbor-code search from the frozen best W4A8 parent."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from down_d_search import ROOT,capture_initial,layer_kwargs,pack,unpack,dump,mlp
from down_codebook_experiment import text_windows
from fixed_grid_rounding import coordinate_round
from precision_loop import configure,measure


@torch.no_grad()
def run_neighbor(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    from utils.eval_utils import evaluator
    parent=options.loop_parent
    if parent is None:raise ValueError('Explicit frozen alpha parent required')
    mlp_reference=options.precision_loop=='neighbor_mlp'
    neighbor=options.precision_loop!='rounding_continuation'
    case='neighbor_mlp_down1' if mlp_reference else ('neighbor_down1' if neighbor else 'floorceil_down1')
    parent_settings=json.loads((parent/'loop_settings.json').read_text())
    weight_parent=parent if (parent/'rounding_weights.pt').exists() else Path(parent_settings['weight_parent'])
    records=torch.load(weight_parent/'rounding_weights.pt',map_location='cpu',weights_only=True)
    names=[n for n in fp if n.endswith('down_proj')]
    assert set(records)==set(names)
    replacements={n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
                  for n,r in records.items()}
    for n,r in records.items():assert torch.equal(r['scale'],scales['weight'][n+'.module.quantizer'])
    alphas=json.loads((parent/'down_scales.json').read_text())['sp2']
    configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
    parent_results=json.loads((parent/'loop_results.json').read_text())
    assert len(parent_results)==1
    parent_result=next(iter(parent_results.values()))
    parent_nll=json.loads((parent/'selection_summary.json').read_text())['final_train_nll']
    windows,indices=text_windows(tokenizer,'train',32,2048,42)
    assert indices==parent_settings['calibration_window_indices']
    hidden,metadata=capture_initial(model,windows,budget)
    name='model.layers.1.mlp.down_proj'
    fits,holds=[],[]
    collected=[0]
    class Captured(Exception):pass
    def collect(module,args,output):
        (fits if collected[0]<24 else holds).append(output[0].cpu())
        collected[0]+=1
        raise Captured()
    handle=model.get_submodule(name).quantizer.register_forward_hook(collect)
    reference_inputs=[]
    def collect_reference(module,args):reference_inputs.append(args[0].cpu())
    reference_handle=model.model.layers[1].mlp.register_forward_pre_hook(collect_reference) if mlp_reference else None
    try:
        for index,layer in enumerate(model.model.layers[:2]):
            layer.cuda();propagated=[]
            for h in hidden:
                try:propagated.append(layer(h.cuda(),**layer_kwargs(metadata))[0].cpu())
                except Captured:pass
                budget.check(f'neighbor/capture/layer{index}')
            hidden=propagated;layer.cpu()
    finally:
        handle.remove()
        if reference_handle is not None:reference_handle.remove()
    assert len(fits)==24 and len(holds)==8
    x=torch.cat(fits).cuda();hold=torch.cat(holds).cuda()
    initial=unpack(records[name]['packed'],records[name]['shape'])
    s=records[name]['scale'].cuda()
    target_fit,target_hold=None,None
    if mlp_reference:
        assert len(reference_inputs)==32
        fg=fp['model.layers.1.mlp.gate_proj'].cuda()
        fu=fp['model.layers.1.mlp.up_proj'].cuda()
        fd=fp[name].cuda()
        outputs=[mlp(h.cuda(),fg,fu,fd).reshape(-1,fd.shape[0]).cpu() for h in reference_inputs]
        target_fit=torch.cat(outputs[:24]).cuda();target_hold=torch.cat(outputs[24:]).cuda()
        del fg,fu,fd,outputs,reference_inputs
    dump(options.output/'loop_settings.json',dict(hypothesis='Floor/ceil restriction limits compensation on a fixed W4 grid',
         parent=str(parent),weight_parent=str(weight_parent),target=name,
         calibration_window_indices=indices,fit_rows=49152,heldout_rows=16384,
         steps=[0,512,2048,8192],initial_codes='Exact parent',code_domain=[-8,7],
         neighbor_search=neighbor,
         sw='Original112 C learned, unchanged',activation='Parent SP2 alpha and96SA unchanged',
         reference='Same-input same-R quantization-before BF16 MLP' if mlp_reference else 'Original same-R down weight acting on actual parent W4/SP2 input',
         selection='Heldout MSE prefilter and true8window train NLL',gradient_training=False))
    trials=coordinate_round(fp[name].cuda(),s,x,hold,milestones=(512,2048,8192),
         initial_codes=initial,neighbor_search=neighbor,
         target_fit=target_fit,target_heldout=target_hold,
         progress=lambda step:budget.check('neighbor/round',step=step))
    assert torch.equal(trials[0]['codes'],initial)
    del x,hold,fits,holds,target_fit,target_hold
    best=trials[0];best['train_nll']=parent_nll
    args=SimpleNamespace(eval_nsamples=8,bsz=1,capture_layer_io=False)
    heldout=SimpleNamespace(input_ids=torch.cat(windows[24:],dim=1))
    for trial in trials[1:]:
        if trial['heldout_mse']>=trials[0]['heldout_mse']:
            trial['train_nll']=None
            continue
        candidate=dict(replacements)
        candidate[name]=(trial['codes'].float()*s.cpu()).to(torch.bfloat16)
        configure(model,fp,scales,checkpoint,alphas,replacements=candidate)
        budget.check('neighbor/train_nll',step=trial['step'])
        score=math.log(evaluator(model,heldout,'cuda',args))
        if not math.isfinite(score):raise RuntimeError('Non-finite neighbor train NLL')
        trial['train_nll']=score
        if score<best['train_nll']:best=trial
    dump(options.output/'trials.json',dict(selected_step=best['step'],
         trials=[{k:v for k,v in t.items() if k!='codes'} for t in trials]))
    selected=dict(records)
    selected[name]=dict(packed=pack(best['codes']),shape=tuple(best['codes'].shape),scale=s.cpu())
    torch.save(selected,options.output/'rounding_weights.pt')
    loaded=torch.load(options.output/'rounding_weights.pt',map_location='cpu',weights_only=True)
    for n,r in loaded.items():
        assert torch.equal(r['scale'],records[n]['scale'])
        expected_codes=best['codes'] if n==name else unpack(records[n]['packed'],records[n]['shape'])
        assert torch.equal(unpack(r['packed'],r['shape']),expected_codes)
        replacements[n]=(expected_codes.float()*r['scale']).to(torch.bfloat16)
    dump(options.output/'down_scales.json',dict(sp2=alphas))
    dump(options.output/'selection_summary.json',dict(final_train_nll=best['train_nll']))
    if best['step']==0:
        result=dict(parent_result,reused_frozen_parent=True)
    else:
        expected=configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
        directory=options.output/case;directory.mkdir()
        result=measure(model,expected,enc,directory,budget)
    result.update(parent_ppl=parent_result['ppl'],delta_parent_ppl=result['ppl']-parent_result['ppl'],
                  delta_parent_nll=result['nll']-parent_result['nll'])
    results={case:result}
    dump(options.output/'loop_results.json',results)
    return results
