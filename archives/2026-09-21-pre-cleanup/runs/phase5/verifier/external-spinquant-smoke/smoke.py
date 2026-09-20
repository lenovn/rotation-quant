"""Bounded, independent local SpinQuant learned-R/GPTQ dynamic-A8 run check."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
SOURCE = ROOT / 'repos/SpinQuant'
OUT = ROOT / 'runs/phase5/verifier/external-spinquant-smoke'
MODEL = ROOT / 'cache/models/llama-3.2-1b-instruct'
ARROW = ROOT / 'cache/huggingface/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow'
sys.path.insert(0, str(SOURCE))
import datasets
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizerFast
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as TrainingModel
from eval_utils.modeling_llama import LlamaForCausalLM as EvaluationModel
from eval_utils import main as ptq_main
from train_utils.optimizer import SGDG
from utils import data_utils, quant_utils
from utils.hadamard_utils import random_hadamard_matrix
from utils.process_args import parser_gen

torch.set_num_threads(4)
transformers.set_seed(42)
started = time.time()
resume = os.environ.get('EXTERNAL_SMOKE_RESUME_PTQ') == '1'
report = dict(status='RUNNING', pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
    source=str(SOURCE), model=str(MODEL), torch=torch.__version__, transformers=transformers.__version__, checks={},
    candidate='Local adapted SpinQuant learned-R + GPTQ W4/group32, dynamic asymmetric A8, KV16',
    smoke_only=True, rotation_updates=1, rotation_token_visits=2048, gptq_nsamples=1,
    scope='Local callable core functions; not CLI Trainer/DDP orchestration, not official full reproduction or formal PPL')

if resume:
    report = json.loads((OUT/'result.json').read_text())
    report['status']='RUNNING'
    report['pid']=os.getpid()
    report['checks'].pop('failure',None)


def record(name, value):
    report['checks'][name] = value
    report['elapsed_seconds'] = time.time()-started
    (OUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(name, json.dumps(value), flush=True)


def wrappers(model):
    return {n: m for n,m in model.named_modules() if n.startswith('model.layers.') and isinstance(m, quant_utils.ActQuantWrapper)}


class Rotation(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.weight = torch.nn.Parameter(matrix.float())


def main():
    report['head'] = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'], text=True).strip()
    report['git_status'] = subprocess.check_output(['git','-C',str(SOURCE),'status','--short'],text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
    (OUT / 'source.diff').write_bytes(subprocess.check_output(['git','-C',str(SOURCE),'diff','--no-ext-diff']))
    sys.argv = ['external-smoke','--seed','42','--rotate','--w_bits','16','--a_bits','8','--a_asym',
        '--w_groupsize','32','--w_clip','--k_bits','16','--v_bits','16','--nsamples','1']
    args, _ = parser_gen()
    args.optimized_rotation_path = None
    args.save_qmodel_path = None
    args.load_qmodel_path = None
    (OUT / 'settings.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    if resume:
        windows=torch.load(OUT/'input_windows.pt',weights_only=True)
        ids=windows['rotation_window']
        loader=[(windows['gptq_window'],windows['gptq_window'].clone())]
        config=AutoConfig.from_pretrained(MODEL,local_files_only=True)
        tied=config.tie_word_embeddings
        config.tie_word_embeddings=False
    else:
        dataset = datasets.Dataset.from_file(str(ARROW))
        tokenizer = LlamaTokenizerFast.from_pretrained(MODEL, local_files_only=True,
            model_max_length=2048,padding_side='right',use_fast=True,add_eos_token=False,add_bos_token=False)
        training_data = data_utils.CustomJsonDataset(dataset,tokenizer,block_size=2048)
        ids = torch.tensor(training_data[0]['input_ids'],dtype=torch.long).reshape(1,-1)
        assert ids.shape == (1,2048)
        gptq_tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=False)
        # Only dataset acquisition is supplied from the same existing Arrow cache;
        # get_wikitext2 retains its own join/tokenizer/random-window sampling code.
        original_loader = data_utils.datasets.load_dataset
        data_utils.datasets.load_dataset = lambda *a, **k: {'train': dataset}
        try:
            loader = data_utils.get_wikitext2(nsamples=1,seed=42,seqlen=2048,tokenizer=gptq_tokenizer)
        finally:
            data_utils.datasets.load_dataset = original_loader
        torch.save({'rotation_window':ids,'gptq_window':loader[0][0]},OUT/'input_windows.pt')
        record('data',dict(split='train',arrow=str(ARROW),rotation_window_index=0,
            rotation_protocol='native CustomJsonDataset row tokenization without BOS/EOS, concatenation, first 2048-token window',
            gptq_protocol='native get_wikitext2 double-newline raw train join, default tokenizer special tokens, seed42 random continuous window',
            rotation_targets=2047,gptq_window_tokens=2048,training_pool_windows=len(training_data)))
        config = AutoConfig.from_pretrained(MODEL,local_files_only=True)
        tied = config.tie_word_embeddings
        config.tie_word_embeddings=False
        model = TrainingModel.from_pretrained(MODEL,config=config,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa')
        if tied:
            model.lm_head.weight.data=model.model.embed_tokens.weight.detach().clone()
        model=prepare_model(args,model)
        model.requires_grad_(False)
        model.R1=Rotation(random_hadamard_matrix(config.hidden_size,'cpu'))
        for layer in model.model.layers:
            layer.self_attn.R2=Rotation(random_hadamard_matrix(config.hidden_size//config.num_attention_heads,'cpu'))
        rotations=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        assert len(rotations)==17 and all(name=='R1.weight' or name.endswith('R2.weight') for name,p in rotations)
        before={name:p.detach().clone() for name,p in rotations}
        model.cuda().train()
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=SGDG([{'params':[p for name,p in rotations],'lr':1.5,'stiefel':True}],lr=1.5)
        loss=model(input_ids=ids.cuda(),labels=ids.cuda(),use_cache=False).loss
        loss.backward()
        grads={name:dict(finite=bool(p.grad is not None and torch.isfinite(p.grad).all()),max_abs=float(p.grad.abs().max())) for name,p in rotations}
        assert all(v['finite'] and v['max_abs']>0 for v in grads.values())
        norm=torch.nn.utils.clip_grad_norm_([p for name,p in rotations],1.0,error_if_nonfinite=True)
        optimizer.step()
        changes={name:float((p.detach().cpu()-before[name]).abs().max()) for name,p in rotations}
        assert all(v>0 for v in changes.values())
        rotation_state={name.removesuffix('.weight'):p.detach().cpu() for name,p in rotations}
        torch.save(rotation_state,OUT/'R.one_update.bin')
        record('rotation_update',dict(loss=float(loss),gradient_norm_before_clip=float(norm),learned_parameters=len(rotations),
            gradients=grads,max_parameter_changes=changes,online_down_hadamard=sum(w.online_full_had for w in wrappers(model).values()),
            W=16,A=8,A_asymmetric=True,KV=16,lr=1.5,updates=1))
        del model,optimizer,rotations,loss,before,training_data
        gc.collect(); torch.cuda.empty_cache()
    model=EvaluationModel.from_pretrained(MODEL,config=config,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa')
    if tied:
        model.lm_head.weight.data=model.model.embed_tokens.weight.detach().clone()
    model.cuda()  # Same caller-side placement as ptq.py before ptq_model.
    args.w_bits=4
    args.optimized_rotation_path=str(OUT/'R.one_update.bin')
    (OUT/'ptq_settings.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    original_gptq=ptq_main.gptq_utils.gptq_fwrd
    original_data=data_utils.get_wikitext2
    def supplied_data(**kwargs):
        assert kwargs['nsamples']==1 and kwargs['seqlen']==2048 and kwargs['seed']==42 and not kwargs['eval_mode']
        return loader
    def measured_gptq(model,dataloader,dev,args):
        selected=wrappers(model)
        assert len(selected)==112
        assert all(w.quantizer.bits==8 and not w.quantizer.sym and not w.quantizer.static_enabled for w in selected.values())
        record('gptq_entry',dict(backbone_dynamic_A8=len(selected),down_online_hadamard=sum(w.online_full_had for w in selected.values()),
            o_proj_groupsizes=sorted(set(w.quantizer.groupsize for n,w in selected.items() if n.endswith('o_proj'))),
            order='input dynamic A8 configured BEFORE GPTQ, local adaptation differs from official current order'))
        quantizers=original_gptq(model,dataloader,dev,args)
        assert len(quantizers)==112 and all(q.bits==4 for q in quantizers.values())
        record('gptq_complete',dict(quantized_backbone_matrices=len(quantizers),w_bits=4,groupsize=args.w_groupsize,
            w_clip=args.w_clip,nsamples=args.nsamples,act_order=args.act_order,percdamp=args.percdamp,
            all_quantized_weights_finite=all(bool(torch.isfinite(w.module.weight).all()) for w in wrappers(model).values())))
        return quantizers
    data_utils.get_wikitext2=supplied_data
    ptq_main.gptq_utils.gptq_fwrd=measured_gptq
    try:
        model=ptq_main.ptq_model(args,model,SimpleNamespace(input_model=str(MODEL)))
    finally:
        data_utils.get_wikitext2=original_data
        ptq_main.gptq_utils.gptq_fwrd=original_gptq
    model.cuda().eval()
    model.config.use_cache=False
    calls={n:0 for n in wrappers(model)}
    handles=[]
    for name,w in wrappers(model).items():
        def count(module,inputs,name=name):
            calls[name]+=1
        handles.append(w.register_forward_pre_hook(count))
    with torch.no_grad():
        output=model(input_ids=ids.cuda(),labels=ids.cuda(),use_cache=False)
    for handle in handles:
        handle.remove()
    assert torch.isfinite(output.logits).all() and torch.isfinite(output.loss)
    assert len(calls)==112 and all(value==1 for value in calls.values())
    record('finite_quantized_forward',dict(input_tokens=2048,targets=2047,nll=float(output.loss),
        finite_logits=True,backbone_dynamic_quantizer_calls=calls,KV=16,use_cache=False,
        embedding_head_precision='BF16, explicitly untied',online_down_hadamard=16,
        qk_postrope_quantization_wrappers=0))
    report['status']='PASS'
    report['max_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    record('conclusion', 'Local adapted callable learned-R + GPTQ/group32 + dynamic asymmetric A8/KV16 path runnable on existing Llama; smoke only, no formal baseline score')


try:
    main()
except Exception as error:
    report['status']='FAIL'
    record('failure',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    raise
