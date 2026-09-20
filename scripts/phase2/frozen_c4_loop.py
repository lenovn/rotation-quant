"""External fixed-token C4 check; no calibration or candidate selection."""
import json
from types import SimpleNamespace
import torch
from down_d_search import ROOT,unpack,dump,evaluate_full_validation
from precision_loop import configure


@torch.no_grad()
def run_c4(model,fp,scales,checkpoint,alphas,tokenizer,enc,options,budget):
    from utils.eval_utils import evaluator
    parent=options.loop_parent
    if parent is None:raise ValueError('Explicit frozen candidate required')
    records=torch.load(parent/'rounding_weights.pt',map_location='cpu',weights_only=True)
    assert set(records)=={n for n in fp if n.endswith('down_proj')}
    replacements={n:(unpack(r['packed'],r['shape']).float()*r['scale']).to(torch.bfloat16)
                  for n,r in records.items()}
    for n,r in records.items():assert torch.equal(r['scale'],scales['weight'][n+'.module.quantizer'])
    alphas=json.loads((parent/'down_scales.json').read_text())['sp2']
    expected=configure(model,fp,scales,checkpoint,alphas,replacements=replacements)
    original=ROOT/'runs/phase2/c4-acceptance-c-20260912.FJXr6U'
    token_path=original/'data/input_tokens.pt'
    saved=torch.load(token_path,map_location='cpu',weights_only=True)
    metadata=saved['metadata']
    old=json.loads((original/'w4a8/result.json').read_text())
    bf16=json.loads((original/'w16a16/result.json').read_text())
    assert metadata==old['input_metadata']==bf16['input_metadata']
    assert metadata['dataset']=='allenai/c4' and metadata['split']=='validation'
    assert metadata['tokenizer_path']==str(ROOT/'cache/models/llama-3.2-1b-instruct')
    ids=saved['input_ids']
    assert ids.shape==(1,2097152) and model.seqlen==2048
    before={n:{k:v.cpu().clone() for k,v in model.get_submodule(n).quantizer.named_buffers()} for n in fp}
    dump(options.output/'loop_settings.json',dict(parent=str(parent),input_token_path=str(token_path),
         input_metadata=metadata,chunk_windows=128,calibration=False,selection=False,
         comparison='Historical same-token BF16 and C+SP2; no repeated baseline run'))
    args=SimpleNamespace(eval_nsamples=None,bsz=1,capture_layer_io=False)
    budget.check('external_c4/evaluation')
    result=evaluate_full_validation(model,SimpleNamespace(input_ids=ids),'cuda',args,evaluator,chunk_windows=128)
    for n,w in expected.items():
        wrapper=model.get_submodule(n)
        assert torch.equal(wrapper.module.weight.cpu(),w.cpu()) and wrapper.quantizer.bits==8
        after=dict(wrapper.quantizer.named_buffers())
        assert all(torch.equal(v,after[k].cpu()) for k,v in before[n].items())
    assert result['predicted_tokens']==old['predicted_tokens']==bf16['predicted_tokens']==2096128
    assert [(s['start_token'],s['seqlen'],s['windows']) for s in result['segments']]==[
        (s['start_token'],s['seqlen'],s['windows']) for s in old['segments']]
    result.update(dataset='allenai/c4',subset='en',split='validation',input_token_path=str(token_path),
         input_metadata=metadata,kv_bits=16,use_cache=False,frozen_checks_pass=True,
         original_c_ppl=old['ppl'],original_c_nll=old['nll'],bf16_ppl=bf16['ppl'],
         delta_c_ppl=result['ppl']-old['ppl'],delta_c_nll=result['nll']-old['nll'],
         delta_bf16_ppl=result['ppl']-bf16['ppl'],external_evaluations=1)
    dump(options.output/'c4_result.json',result)
    print('C4_RESULT '+json.dumps(result),flush=True)
    budget.check('external_c4/completed')
    return {'c4':result}
