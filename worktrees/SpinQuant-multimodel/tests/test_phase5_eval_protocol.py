"""Independent CPU checks for new split routing and inherited metric arithmetic."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.phase3 import architecture, external_eval


@pytest.mark.parametrize('split,tail', [('test', 3), ('validation', 1), ('test', 0)])
def test_split_loading_no_special_tokens_and_tail_metadata(monkeypatch, tmp_path, split, tail):
    from datasets import Dataset
    reads=[]
    class Rows:
        def __len__(self): return 3
        def __getitem__(self, key):
            assert key=='text'; return ['alpha', '', 'beta']
    def read(path):
        reads.append(Path(path).name)
        assert Path(path)==tmp_path/f'wikitext-{split}.arrow'
        return Rows()
    ids=torch.arange(2048+tail).unsqueeze(0)
    class Tokenizer:
        def __call__(self,text,**kwargs):
            assert text=='alpha\n\n\n\nbeta'
            assert kwargs==dict(return_tensors='pt',add_special_tokens=False)
            return SimpleNamespace(input_ids=ids)
    monkeypatch.setattr(external_eval,'DATA_PATH',tmp_path)
    monkeypatch.setattr(Dataset,'from_file',staticmethod(read))
    monkeypatch.setattr(external_eval,'model_tokenizer',lambda path:Tokenizer())
    # Unspecified argument must remain official test.
    actual,meta=external_eval.load_wikitext2_test_tokens() if split=='test' else external_eval.load_wikitext2_test_tokens(split)
    assert actual is ids and reads==[f'wikitext-{split}.arrow'] and meta['split']==split
    assert meta['predicted_tokens']==2047+max(tail-1,0)
    assert meta['unscored_tail_tokens']==int(tail==1)
    assert meta['windows']==1+int(tail>=2) and meta['tail_tokens']==tail
    with pytest.raises(ValueError):external_eval.load_wikitext2_test_tokens('train')


@pytest.mark.parametrize('requested', [False, True])
def test_external_driver_routes_validation_flag_without_changing_default(monkeypatch,tmp_path,requested):
    calls=[]
    class StopAfterData(Exception):pass
    def load(split='test'):
        calls.append(split)
        return torch.tensor([[1,2,3]]),dict(split=split)
    def stop(*args):raise StopAfterData()
    monkeypatch.setattr(external_eval,'load_wikitext2_test_tokens',load)
    monkeypatch.setattr(external_eval,'load_model',stop)
    monkeypatch.setattr(external_eval,'progress',lambda *a,**k:None)
    args=SimpleNamespace(tokens=None,output=tmp_path,mode='bf16',package=None)
    if requested:args.wikitext2_validation=True
    with pytest.raises(StopAfterData):external_eval.run(args)
    expected='validation' if requested else 'test'
    assert calls==[expected]
    saved=torch.load(tmp_path/'input_tokens.pt',weights_only=True)
    assert saved['metadata']['split']==expected


def acceptance():
    path=external_eval.PROJECT_ROOT/'scripts/phase2/validation_acceptance.py'
    spec=importlib.util.spec_from_file_location('phase5_independent_acceptance',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('tail', [1, 3])
def test_qwen_official_tiny_forward_bf16_ce_fp32_ppl_and_weighted_tail(tail):
    torch.manual_seed(47)
    config=Qwen3Config(vocab_size=19,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=1,head_dim=4,max_position_embeddings=32,
                       tie_word_embeddings=True,attention_dropout=0.)
    config._attn_implementation='eager'
    model=Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    model.seqlen=5
    ids=(torch.arange(10+tail)%19).unsqueeze(0)
    # Independent oracle deliberately preserves BF16 CE then FP32 exp per segment.
    scores=[]
    with torch.no_grad():
        for start,length,windows in [(0,5,2)]+([(10,tail,1)] if tail>=2 else []):
            per_window=[]
            for row in ids[:,start:start+length*windows].reshape(-1,length):
                logits=model.lm_head(model.model(row.unsqueeze(0),use_cache=False,return_dict=True)[0])
                assert logits.dtype==torch.bfloat16
                losses=F.cross_entropy(logits[:,:-1].transpose(1,2),row[1:].unsqueeze(0),reduction='none')
                assert losses.dtype==torch.bfloat16
                per_window.append(losses.float().mean(dim=1))
            scores.append(torch.exp(torch.cat(per_window).mean()).item())
    measured=acceptance().evaluate_full_validation(model,SimpleNamespace(input_ids=ids),'cpu',
                SimpleNamespace(eval_nsamples=None,bsz=1,capture_layer_io=False),architecture.qwen_evaluator,
                chunk_windows=2)
    assert [segment['ppl'] for segment in measured['segments']]==scores
    targets=[8]+([tail-1] if tail>=2 else [])
    expected_nll=sum(math.log(ppl)*count for ppl,count in zip(scores,targets))/sum(targets)
    assert measured['nll']==expected_nll and measured['ppl']==math.exp(expected_nll)
    assert measured['predicted_tokens']==sum(targets)
    assert measured['unscored_tail_tokens']==int(tail==1) and model.seqlen==5
    assert architecture.evaluator_for(model) is architecture.qwen_evaluator
