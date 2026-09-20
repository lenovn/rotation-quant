"""One-channel offline paired D, with fixed SP2 alpha and explicit SW control."""
import json
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from down_d_search import (ROOT,STRENGTHS,capture_initial,layer_kwargs,static_input,
                           direction,fuse_pair,weight_quant,pack,unpack,dump)
from down_codebook_experiment import text_windows
from down_codebooks import quantize
from fixed_grid_rounding import coordinate_round
from precision_loop import configure,measure


def decode(records):
    return {n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
            for n,r in records.items()}


@torch.no_grad()
def run_sparse(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    from utils.eval_utils import evaluator
    parent=options.loop_parent
    rms_mode=options.precision_loop=='sparse_d_rms_sp2'
    if parent is None:raise ValueError('Sparse D requires an explicit frozen parent')
    parent_records=torch.load(parent/'rounding_weights.pt',map_location='cpu',weights_only=True)
    downs={n for n in fp if n.endswith('down_proj')}
    assert downs<=set(parent_records)<=set(fp)
    if rms_mode:
        alphas=json.loads((parent/'down_scales.json').read_text())['sp2']
        assert set(alphas)==downs
    parent_weights=decode(parent_records)
    configure(model,fp,scales,checkpoint,alphas,replacements=parent_weights)
    windows,indices=text_windows(tokenizer,'train',32,2048,42)
    parent_settings=json.loads((parent/'loop_settings.json').read_text())
    assert indices==parent_settings['calibration_window_indices']
    if rms_mode:
        assert parent_settings['fit_rows']==49152 and parent_settings['heldout_rows']==16384
        parent_nll=json.loads((parent/'selection_summary.json').read_text())['final_train_nll']
        parent_results=json.loads((parent/'loop_results.json').read_text())
        assert len(parent_results)==1
        parent_result=next(iter(parent_results.values()))
    else:
        assert parent_settings['heldout']=='24/8 disjoint full train windows'
        last=json.loads((parent/'layer_15.json').read_text())
        parent_nll=next(t['train_nll'] for t in last['trials'] if t['step']==last['selected_step'])
        parent_result=json.loads((parent/'loop_results.json').read_text())['rounded_downs']
    hidden,metadata=capture_initial(model,windows,budget)
    kwargs=layer_kwargs(metadata)
    inputs=[]
    class Captured(Exception):pass
    def collect(module,args):inputs.append(args[0].cpu());raise Captured()
    handle=model.model.layers[1].mlp.register_forward_pre_hook(collect)
    try:
        for index,layer in enumerate(model.model.layers[:2]):
            layer.cuda();propagated=[]
            for h in hidden:
                try:propagated.append(layer(h.cuda(),**kwargs)[0].cpu())
                except Captured:pass
                budget.check(f'sparse/capture/layer{index}')
            hidden=propagated;layer.cpu()
    finally:handle.remove()
    assert len(inputs)==32
    up='model.layers.1.mlp.up_proj';down='model.layers.1.mlp.down_proj';gate='model.layers.1.mlp.gate_proj'
    fg=fp[gate].cuda();fu=fp[up].cuda();fd=fp[down].cuda()
    qg=parent_weights.get(gate,checkpoint['model'][gate+'.module.weight']).cuda()
    original_up=parent_weights.get(up,checkpoint['model'][up+'.module.weight']).cuda()
    gsa=scales['activation'][gate+'.quantizer'].cuda();usa=scales['activation'][up+'.quantizer'].cuda()
    a=torch.zeros(fu.shape[0],device='cuda')
    for h in inputs[:24]:
        z=F.silu(F.linear(static_input(h.cuda(),gsa),qg))*F.linear(static_input(h.cuda(),usa),original_up)
        a+=(z.float().square() if rms_mode else z.float().abs()).sum((0,1))
    a/=24*2048
    if rms_mode:a=a.sqrt()
    wcol=fd.float().abs().amax(0).clamp_min(1e-12)
    channel=int((a/wcol).argmax())
    v=torch.zeros_like(a);v[channel]=direction(a,fd)[channel].clamp_min(0);v-=v.mean()
    assert float(v.max()-v.min())>1e-8
    dump(options.output/'loop_settings.json',dict(hypothesis='Sparse paired scaling protects salient down W4 channel under SP2',
         parent=str(parent),channel=channel,activation_stat=('RMS' if rms_mode else 'mean absolute')+' actual pre-SP2 input, first24train windows',
         channel_ranking='a_j / max_i abs(Wdown_ij)',strengths=list(STRENGTHS),bounds=[.25,4],
         sw_up='C SW / D',sw_down='max(C SW, transformed BF16 row absmax / 7)',
         alpha='Frozen parent SP2 alpha unchanged; no threshold shrink',fit_rows=49152,heldout_rows=16384,
         neighbor_search=rms_mode,selection='actual train NLL across finite steps' if rms_mode else 'local heldout MSE then train NLL across D',
         calibration_window_indices=indices,reference='Weight-local after actual W4 gate/up and SP2 input',
         source='AWQ-inspired selective scaling, not a reproduction of full AWQ'))
    eval_args=SimpleNamespace(eval_nsamples=8,bsz=1,capture_layer_io=False)
    held_enc=SimpleNamespace(input_ids=torch.cat(windows[24:],dim=1))
    candidates=[]
    for strength in STRENGTHS:
        d=(strength*v).exp()
        assert float(d.min())>=.25 and float(d.max())<=4.000001
        assert abs(float(d.double().log().mean()))<1e-6
        uf,df=fuse_pair(fu.float(),fd.float(),d)
        # FP64 same-input equivalence before BF16 effective-weight rounding.
        h=inputs[0][:,:2].cuda().double()
        ref=F.linear(F.silu(F.linear(h,fg.double()))*F.linear(h,fu.double()),fd.double())
        actual=F.linear(F.silu(F.linear(h,fg.double()))*F.linear(h,fu.double()/d.double()[:,None]),fd.double()*d.double()[None,:])
        assert torch.allclose(ref,actual,atol=1e-10,rtol=1e-10)
        uf=uf.to(torch.bfloat16);df=df.to(torch.bfloat16)
        su=scales['weight'][up+'.module.quantizer'].cuda()/d[:,None]
        sd=torch.maximum(scales['weight'][down+'.module.quantizer'].cuda(),df.float().abs().amax(1,keepdim=True).clamp_min(1e-5)/7)
        qu,cu,su=weight_quant(uf,su)
        fits,holds=[],[]
        for index,h in enumerate(inputs):
            z=F.silu(F.linear(static_input(h.cuda(),gsa),qg))*F.linear(static_input(h.cuda(),usa),qu)
            qz=quantize(z,alphas[down],'sp2').reshape(-1,z.shape[-1]).cpu()
            (fits if index<24 else holds).append(qz)
        x=torch.cat(fits).cuda();hold=torch.cat(holds).cuda()
        trials=coordinate_round(df,sd,x,hold,milestones=(512,2048,8192),
                  neighbor_search=rms_mode,
                  progress=lambda step:budget.check('sparse/round',strength=strength,step=step))
        if rms_mode:
            # Each D starts from its independently fused FP weight. Never reuse
            # codes optimized under a different D or quantize that candidate again.
            for trial in trials:
                trial_records=dict(parent_records)
                trial_records[up]=dict(packed=pack(cu.cpu()),shape=tuple(cu.shape),scale=su.cpu())
                trial_records[down]=dict(packed=pack(trial['codes']),shape=tuple(trial['codes'].shape),scale=sd.cpu())
                configure(model,fp,scales,checkpoint,alphas,replacements=decode(trial_records))
                budget.check('sparse/train_nll',strength=strength,step=trial['step'])
                trial['train_nll']=math.log(evaluator(model,held_enc,'cuda',eval_args))
                if not math.isfinite(trial['train_nll']):raise RuntimeError('Non-finite D candidate NLL')
        chosen=min(trials,key=lambda t:t['heldout_mse'])
        if rms_mode:chosen=min(trials,key=lambda t:t['train_nll'])
        records=dict(parent_records)
        records[up]=dict(packed=pack(cu.cpu()),shape=tuple(cu.shape),scale=su.cpu())
        records[down]=dict(packed=pack(chosen['codes']),shape=tuple(chosen['codes'].shape),scale=sd.cpu())
        weights=decode(records)
        same_parent=torch.equal(weights[up],original_up.cpu()) and torch.equal(weights[down],parent_weights[down])
        if same_parent:
            score=parent_nll
        elif rms_mode:
            score=chosen['train_nll']
        else:
            configure(model,fp,scales,checkpoint,alphas,replacements=weights)
            score=math.log(evaluator(model,held_enc,'cuda',eval_args))
        if not math.isfinite(score):raise RuntimeError('Non-finite D train NLL')
        row=dict(strength=strength,channel=channel,d_min=float(d.min()),d_max=float(d.max()),
             mean_log_d=float(d.double().log().mean()),selected_round_step=chosen['step'],train_nll=score,
             same_parent=same_parent,fp64_equivalence=True,up_sw_transport=True,
             down_sw_expanded_rows=int((sd.cpu()>scales['weight'][down+'.module.quantizer']).sum()),
             trials=[{k:val for k,val in t.items() if k!='codes'} for t in trials])
        dump(options.output/f'd_{strength:.2f}.json',row)
        print('SPARSE_D '+json.dumps(row),flush=True)
        candidates.append(dict(row=row,records=records,d=d.cpu()))
        del x,hold,fits,holds,trials,weights,uf,df,qu,cu,su,sd,z,qz
        torch.cuda.empty_cache()
    best=min(candidates,key=lambda c:c['row']['train_nll'])
    dump(options.output/'down_scales.json',dict(sp2=alphas))
    dump(options.output/'selection_summary.json',dict(final_train_nll=best['row']['train_nll'],selected_strength=best['row']['strength'],parent_train_nll=parent_nll))
    results={}
    for label,candidate in [('identity_rule',candidates[0]),('selected_d',best)]:
        if candidate['row']['same_parent']:
            results[label]=dict(parent_result,reused_frozen_parent=True)
            continue
        if label=='selected_d' and best is candidates[0]:
            results[label]=dict(results['identity_rule'],reused_identity=True)
            continue
        directory=options.output/label;directory.mkdir()
        payload=dict(weights=candidate['records'],D=candidate['d'],base_checkpoint=str(ROOT/'runs/phase2/learned-sw-c-20260909.ByFYAM/C/rtn/w4_rtn_model.pt'))
        torch.save(payload,directory/'packed_model.pt')
        loaded=torch.load(directory/'packed_model.pt',map_location='cpu',weights_only=True)
        assert torch.equal(loaded['D'],candidate['d'])
        for name,row in candidate['records'].items():
            assert torch.equal(row['packed'],loaded['weights'][name]['packed'])
            assert torch.equal(row['scale'],loaded['weights'][name]['scale'])
        expected=configure(model,fp,scales,checkpoint,alphas,replacements=decode(loaded['weights']))
        results[label]=measure(model,expected,enc,directory,budget)
        dump(options.output/'loop_results.json',results)
    dump(options.output/'loop_results.json',results)
    return results
