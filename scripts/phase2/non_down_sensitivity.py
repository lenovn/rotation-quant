"""Train-only non-down W16 restoration diagnostics on a frozen W4A8 parent."""
import json
import math
from types import SimpleNamespace
import torch
from down_d_search import unpack,dump
from down_codebook_experiment import text_windows
from precision_loop import configure


@torch.no_grad()
def run_sensitivity(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    from utils.eval_utils import evaluator
    parent=options.loop_parent
    if parent is None:raise ValueError('Explicit frozen parent required')
    records=torch.load(parent/'rounding_weights.pt',map_location='cpu',weights_only=True)
    downs={n for n in fp if n.endswith('down_proj')}
    assert downs<=set(records)<=set(fp)
    replacements={n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
                  for n,r in records.items()}
    for n,r in records.items():assert torch.equal(r['scale'],scales['weight'][n+'.module.quantizer'])
    alphas=json.loads((parent/'down_scales.json').read_text())['sp2']
    configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
    windows,indices=text_windows(tokenizer,'train',32,2048,42)
    parent_settings=json.loads((parent/'loop_settings.json').read_text())
    assert indices==parent_settings['calibration_window_indices']
    base=json.loads((parent/'selection_summary.json').read_text())['final_train_nll']
    heldout=SimpleNamespace(input_ids=torch.cat(windows[24:],dim=1))
    args=SimpleNamespace(eval_nsamples=8,bsz=1,capture_layer_io=False)
    summary=dict(parent=str(parent),calibration_window_indices=indices,selection_windows=indices[24:],
         predicted_tokens=8*2047,split='train',parent_train_nll=base,
         activation='All96SA/16SP2 alpha unchanged, allA8',
         purpose='Conditional localization only; restored W16 is not a deployment candidate',cases=[])
    dump(options.output/'loop_settings.json',dict(parent=str(parent),mode='train-only W16 sensitivity',
         calibration_window_indices=indices,groups_per_layer=['gate_up','qkv','o_proj'],
         maximum_selected_groups=2,teacher=False,gradients=False,recalibration=False))
    families=[('gate_up',('mlp.gate_proj','mlp.up_proj')),
              ('qkv',('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj')),
              ('o_proj',('self_attn.o_proj',))]
    for index in range(16):
        for family,suffixes in families:
            restored=[f'model.layers.{index}.{suffix}' for suffix in suffixes]
            assert all(n in fp and n not in downs for n in restored)
            active={n:w for n,w in replacements.items() if n not in restored}
            expected=configure(model,fp,scales,checkpoint,alphas,restore_w=restored,replacements=active)
            budget.check('non_down_sensitivity/train_nll',layer=index,family=family)
            score=math.log(evaluator(model,heldout,'cuda',args))
            if not math.isfinite(score):raise RuntimeError('Non-finite sensitivity NLL')
            for n in restored:assert torch.equal(model.get_submodule(n).module.weight.cpu(),fp[n])
            assert all(model.get_submodule(n).quantizer.bits==8 for n in fp)
            row=dict(layer=index,family=family,restored_weights=restored,
                     train_nll=score,delta_train_nll=score-base)
            summary['cases'].append(row)
            dump(options.output/'non_down_sensitivity.json',summary)
            print('NON_DOWN_SENSITIVITY '+json.dumps(row),flush=True)
    summary['selected_groups']=sorted([c for c in summary['cases'] if c['train_nll']<base],
                                      key=lambda c:c['train_nll'])[:2]
    summary['selected_weights']=[n for c in summary['selected_groups'] for n in c['restored_weights']]
    expected=configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
    assert all(torch.equal(model.get_submodule(n).module.weight.cpu(),w.cpu()) for n,w in expected.items())
    dump(options.output/'non_down_sensitivity.json',summary)
    budget.check('non_down_sensitivity/completed')
    return {'non_down_sensitivity':summary}
