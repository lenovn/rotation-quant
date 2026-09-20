import importlib.util, math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
P=Path('/home/dongpeiyan/projects/rotation-quant')
s=importlib.util.spec_from_file_location('joint',P/'scripts/phase6/joint.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
from utils.quant_utils import ActQuantWrapper,RotationStaticActQuantizer

class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([torch.nn.Module()]);layer=self.model.layers[0];layer.mlp=torch.nn.Module()
  for name in ['down_proj','up_proj']:
   w=ActQuantWrapper(torch.nn.Linear(2,2,bias=False));w.module.weight.data.copy_(torch.eye(2));w.quantizer=RotationStaticActQuantizer();w.quantizer.load_scale(torch.tensor([1.]));layer.mlp.add_module(name,w)
  self.config=SimpleNamespace(to_dict=lambda:{'model_type':'synthetic'},use_cache=False)
 def cuda(self):return self

@pytest.mark.parametrize('bits',[8,16])
def test_lsq_forward_backward(bits):
 q=j.SignedActivation(bits,0.01);hi=2**(bits-1)-1
 x=torch.tensor([0.013,0.046,(hi+10)*.01,-(hi+10)*.01],requires_grad=True)
 out=q(x);out.sum().backward()
 t=x.detach()/.01;terms=torch.where(t< -hi-1,-hi-1,torch.where(t>hi,hi,t.round()-t))
 assert q.scale.grad.dtype==torch.float32
 torch.testing.assert_close(q.scale.grad,terms.sum().reshape(1)/math.sqrt(4*hi))
 assert torch.equal(x.grad,torch.tensor([1.,1.,0.,0.]))
 assert torch.equal(out,(t.round().clamp(-hi-1,hi)*.01))
 bf=torch.tensor([0.013],dtype=torch.bfloat16,requires_grad=True);q.zero_grad();q(bf).sum().backward();assert q.scale.grad.dtype==torch.float32 and torch.isfinite(q.scale.grad).all()

def test_down_hook_once_and_fixed_scale():
 m=Model();j.install(m)
 w=m.model.layers[0].mlp.down_proj
 torch.testing.assert_close(w.quantizer.scale,torch.tensor([127/32767]))
 w.quantizer.scale.requires_grad_(False)
 x=torch.tensor([[.12,.31]],requires_grad=True);before=w.quantizer.scale.clone()
 y=w(x);assert w.quantizer.calls==1 and not torch.equal(x,y)
 y.sum().backward();assert x.grad is not None and w.quantizer.scale.grad is None
 assert torch.equal(before,w.quantizer.scale)
 j.install(m,{n:w.quantizer.scale for n,w in j.c.wrappers(m).items()})
 w(x);assert w.quantizer.calls==1

def test_export_load_and_coverage(tmp_path,monkeypatch):
 m=Model();j.install(m,learnable=False)
 records={}
 for n,w in j.c.wrappers(m).items():
  codes=torch.tensor([[1,0],[0,1]],dtype=torch.int8)
  from experiments.phase3.quantization import pack_int4
  records[n]={'packed':pack_int4(codes),'shape':(2,2),'scale':torch.ones(2,1)}
 p=tmp_path/'m.pt';j.save_package(m,records,p,{'arm':'joint'})
 monkeypatch.setattr(j.c,'load_model',lambda *a,**kw:Model());monkeypatch.setattr(j.c,'add_actquant',lambda m:None)
 restored,state=j.load_package(p)
 x=torch.tensor([[.123,.734]])
 for name,w in j.c.wrappers(restored).items():
  assert not w.quantizer.scale.requires_grad
  assert torch.equal(w(x),m.get_submodule(name)(x))
 del state['activation'][next(iter(state['activation']))];torch.save(state,p)
 with pytest.raises(ValueError,match='coverage'):j.load_package(p)

def test_parameter_coverage(tmp_path):
 m=Model();m.R1=torch.nn.Linear(2,2,bias=False)
 for w in j.c.wrappers(m).values():
  w.module.quantizer=torch.nn.Module();w.module.quantizer.scale=torch.nn.Parameter(torch.ones(2,1))
 j.install(m);p=tmp_path/'p.pt';j.save_parameters(m,p);state=j.load_parameters(m,p)
 del state['parameters'][next(iter(state['parameters']))];torch.save(state,p)
 with pytest.raises(ValueError,match='coverage'):j.load_parameters(m,p)

def test_data_800_and_512_schedule(tmp_path,monkeypatch):
 windows=torch.arange(900*2048).reshape(900,2048)
 monkeypatch.setattr(j.c,'data_windows',lambda:(windows,[],['probe'],['validation'],{}))
 w,cal,probe,val=j.data(tmp_path)
 assert torch.equal(w,windows[:800]) and len(cal)==32 and all(x.shape==(1,2048) for x in cal)
 indices=torch.randperm(800,generator=torch.Generator().manual_seed(42))[:32]
 assert all(torch.equal(x,w[i:i+1]) for x,i in zip(cal,indices))
 assert j.schedule(0,512,10)==.1 and j.schedule(9,512,10)==1 and j.schedule(512,512,10)==0
 assert j.schedule(100,512,10)>j.schedule(256,512,10)>j.schedule(511,512,10)>0

def test_checkpoint_restores_device_before_probe(tmp_path,monkeypatch):
 class Fake:
  on_cuda=True
  def cuda(self):self.on_cuda=True;return self
 frozen=Fake();observed=[]
 monkeypatch.setattr(j,'save_parameters',lambda *a,**k:None)
 monkeypatch.setattr(j,'save_package',lambda *a,**k:None)
 monkeypatch.setattr(j,'freeze',lambda *a,**k:(frozen,{}))
 def validation(m,windows):m.on_cuda=False;return {'ppl':2.,'nll':math.log(2.)}
 monkeypatch.setattr(j.c,'full_validation',validation)
 def backbone(m,ids):
  assert m.on_cuda,'probe called while frozen decoder layers are on CPU'
  observed.append(True);return torch.ones(1)
 monkeypatch.setattr(j.c,'backbone',backbone)
 monkeypatch.setattr(j,'load_package',lambda *a:(Fake(),{}))
 monkeypatch.setattr(torch.Tensor,'cuda',lambda x,*a,**k:x)
 args=SimpleNamespace(output=tmp_path,arm='joint',initial=None)
 j.checkpoint_eval(None,args,0,[],[],[torch.zeros(1,32)])
 assert len(observed)==2

@pytest.mark.parametrize('bad',['shape','nan','zero'])
def test_parameter_value_checks(bad):
 m=Model();m.R1=torch.nn.Linear(2,2,bias=False)
 for w in j.c.wrappers(m).values():
  w.module.quantizer=torch.nn.Module();w.module.quantizer.scale=torch.nn.Parameter(torch.ones(2,1))
 j.install(m);values={n:p.detach().clone() for n,p in j.parameters(m).items()}
 name=next(n for n in values if n.endswith('.scale'))
 if bad=='shape':values[name]=torch.ones(13)
 elif bad=='nan':values[name].fill_(float('nan'))
 else:values[name].zero_()
 with pytest.raises(ValueError):j.load_parameter_values(m,values)


def test_checkpoint_complete_reused(tmp_path,monkeypatch):
 import json
 directory=tmp_path/'checkpoint-0512';directory.mkdir()
 (directory/'validation.json').write_text(json.dumps({'ppl':2.,'nll':math.log(2.)}))
 monkeypatch.setattr(j,'freeze',lambda *a,**k:pytest.fail('completed checkpoint was rerun'))
 result=j.checkpoint_eval(None,SimpleNamespace(output=tmp_path),512,[],[],[])
 assert result['step']==512 and result['ppl']==2.
