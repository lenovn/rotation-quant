import importlib.util
from pathlib import Path
import torch
import pytest

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
spec = importlib.util.spec_from_file_location('phase6', ROOT / 'scripts/phase6/down_precision.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from utils.quant_utils import ActQuantWrapper, RotationStaticActQuantizer
from experiments.phase3.quantization import SP2Quantizer


def model():
    m = torch.nn.Module()
    m.model = torch.nn.Module()
    m.model.layers = torch.nn.ModuleList([torch.nn.Module(),torch.nn.Module()])
    for layer in m.model.layers:
        layer.mlp = torch.nn.Module()
        for name in ['down_proj','up_proj']:
            w = ActQuantWrapper(torch.nn.Linear(3,3,bias=False))
            w.module.weight.data.copy_(torch.eye(3))
            w.quantizer = SP2Quantizer(0.01) if name=='down_proj' else RotationStaticActQuantizer()
            if name=='up_proj': w.quantizer.load_scale(torch.tensor([0.1]))
            layer.mlp.add_module(name,w)
    return m

@pytest.mark.parametrize('bits', [8,16])
def test_integer_grid_and_clipping(bits):
    alpha=2.0
    q=p.StaticSignedQuantizer(bits,alpha)
    hi=(1<<(bits-1))-1
    step=alpha/hi
    x=torch.tensor([-4.,-2.,-0.5*step,0.,0.5*step,1.5*step,2.,4.])
    out=q(x)
    expected=torch.tensor([-(hi+1)*step,-2.,0.,0.,0.,2*step,2.,2.])
    torch.testing.assert_close(out,expected,atol=1e-7,rtol=1e-6)
    assert q.counts[0]==8 and q.counts[1]==2 and q.calls==1
    y=q(torch.tensor([0.12345,0.333,3.],dtype=torch.bfloat16))
    assert y.dtype==torch.bfloat16 and q.scale.dtype==torch.float32

@pytest.mark.parametrize('precision', ['float','int8','int16'])
def test_real_wrapper_path_isolation(precision):
    m=model(); before={k:v.clone() for k,v in m.state_dict().items()}
    bounds={'model.layers.0.mlp.down_proj':1.,'model.layers.1.mlp.down_proj':2.}
    handles,qs=p.install_down_format(m,precision,bounds)
    x=torch.tensor([[0.123456,0.76543,3.]])
    for idx in range(2):
        w=m.model.layers[idx].mlp.down_proj
        out=w(x)
        if precision=='float':
            assert torch.equal(out,x)
        else:
            q=qs[f'model.layers.{idx}.mlp.down_proj']
            assert q.calls==1 and q.counts[0]==3
            expected=(x/q.scale).round().clamp(q.qmin,q.qmax)*q.scale
            assert torch.equal(out,expected) and not torch.equal(out,x)
            assert w.module.phase6_input_quantizer is q
    for k,v in before.items(): torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0,equal_nan=True,msg=k)
    assert m.model.layers[0].mlp.up_proj.quantizer.bits==8
    for h in handles:h.remove()
    for layer in m.model.layers: assert torch.equal(layer.mlp.down_proj(x),x)


def test_registered_scale_follows_layer_device():
    m=model()
    _,qs=p.install_down_format(m,'int16',{'model.layers.0.mlp.down_proj':1.,'model.layers.1.mlp.down_proj':2.})
    m.model.layers[0].to('meta')
    assert qs['model.layers.0.mlp.down_proj'].scale.device.type=='meta'
    assert qs['model.layers.0.mlp.down_proj'].counts.device.type=='meta'
    assert qs['model.layers.1.mlp.down_proj'].scale.device.type=='cpu'


def test_historical_bounds_and_tokens():
    assert len(p.load_bounds())==16
    ids, windows=p.validation_windows()
    assert ids.shape==(1,252852) and len(windows)==124 and windows[-1].shape==(1,948)
    assert sum(w.numel()-1 for w in windows)==252728
    from datasets import Dataset
    tokenizer=p.LlamaTokenizerFast.from_pretrained(str(p.MODEL_PATH),local_files_only=True,model_max_length=2048,padding_side='right',use_fast=True,add_eos_token=False,add_bos_token=False)
    data=Dataset.from_file(str(p.DATA_PATH / 'wikitext-validation.arrow'))
    old=tokenizer('\n\n'.join(data['text']),return_tensors='pt').input_ids
    assert torch.equal(old,ids)
