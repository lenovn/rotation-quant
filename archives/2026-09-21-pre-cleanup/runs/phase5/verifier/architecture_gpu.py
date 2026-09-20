"""Independent short-input actual-checkpoint verification; use assigned CUDA device."""
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
sys.path.insert(0, str(ROOT / 'worktrees/SpinQuant-multimodel'))
os.environ['PHASE5_MODEL_PATH'] = str(ROOT / 'cache/models/qwen3-1.7b')
import torch
from experiments.phase3 import architecture, common, postprocess, distill
from utils.fuse_norm_utils import fuse_layer_norms

torch.set_num_threads(4)
OUT = ROOT / 'runs/phase5/verifier'
resume = os.environ.get('PHASE5_VERIFY_RESUME') == '1'
report = {'device': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch': torch.__version__, 'checks': {}}

if resume:
    report = json.loads((OUT / 'architecture_gpu.json').read_text())


def record(name, result):
    report['checks'][name] = result
    (OUT / 'architecture_gpu.json').write_text(json.dumps(report, indent=2) + '\n')
    print(name, json.dumps(result), flush=True)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def output(model):
    return model.lm_head(common.backbone(model, ids)).float().cpu()


def difference(a, b):
    d = (a-b).float()
    return {'max_abs': d.abs().max().item(), 'rms': d.square().mean().sqrt().item(),
            'relative_rms': (d.square().mean() / b.float().square().mean()).sqrt().item()}


tokenizer = architecture.tokenizer(common.MODEL_PATH)
ids = tokenizer('The history of science is a story of careful observation and repeated experiments. A small model can represent useful patterns in language.', return_tensors='pt', add_special_tokens=False).input_ids[:, :32].cuda()
report['input_ids'] = ids.cpu().tolist()
if not resume:
    model = architecture.load_model(common.MODEL_PATH).cuda().eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    raw = output(model)
    norms = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.q_norm.' in n or '.k_norm.' in n}
    del model
    cleanup()
    model = architecture.load_model(common.MODEL_PATH, training=True, untie=True).cuda().eval()
    adapted = output(model)
    torch.testing.assert_close(adapted, raw, rtol=0, atol=0)
    record('native_vs_adapted_unquantized', difference(adapted, raw))
    with torch.no_grad():
        embedding = model.model.embed_tokens.weight.detach().clone()
        w = embedding.double()
        model.model.embed_tokens.weight.copy_((w-w.mean(-1, keepdim=True)).bfloat16())
        del w
    centered = output(model)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(embedding)
    del embedding
    fuse_layer_norms(model)
    fused = output(model)
    record('centering_effect_separate', difference(centered, raw))
    record('norm_fusion_after_same_centering', difference(fused, centered))
    for n, p in model.named_parameters():
        if n in norms:
            torch.testing.assert_close(p.cpu(), norms[n], rtol=0, atol=0)
    del model
    cleanup()
model = common.build_training_model().cuda().eval()
if resume:
    norms = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.q_norm.' in n or '.k_norm.' in n}
else:
    for wrapper in common.wrappers(model).values():
        wrapper.quantizer.bits = 16
        wrapper.module.quantizer.bits = 16
    rotated = output(model)
    rotation_error = difference(rotated, fused)
    record('rotation_same_preprocessing_unquantized', rotation_error)
    # Report float error rather than assuming BF16 multi-layer rotation is bitwise exact.
    assert rotation_error['relative_rms'] < .1
for wrapper in common.wrappers(model).values():
    wrapper.module.quantizer.bits = 4
common.initialize_scales(model, [ids])
groups = common.learned_parameters(model)
coverage = {key: len(values) for key, values in groups.items()}
assert coverage == dict(R=29, SA=168, SW=196, SP2=28)
common.training_mode(model, 'B', 0, 0)
model.train()
loss = common.token_nll(model, ids)
loss.backward()
gradient = {name: {'finite': bool(p.grad is not None and torch.isfinite(p.grad).all()),
                   'max_abs': float(p.grad.abs().max()) if p.grad is not None else None}
            for name, p in groups['R']}
assert all(v['finite'] and v['max_abs'] > 0 for v in gradient.values())
assert all(not p.requires_grad for name, p in model.named_parameters() if name in norms)
record('real_W4_A8_checkpoint_backward', {'loss': float(loss), 'coverage': coverage, 'rotation_gradients': gradient})
model.zero_grad(set_to_none=True)
model.eval()
frozen, records = common.frozen_model(model)
for name, wrapper in common.wrappers(frozen).items():
    if name.endswith('down_proj'):
        wrapper.quantizer.bits = 8
del model, groups, loss
cleanup()
expected = output(frozen)
path = OUT / 'architecture_short_input_static.pt'
common.save_frozen(frozen, records, path, {'purpose': 'short-input architecture verification; not calibrated formal package'})
del frozen, records
cleanup()
model, records = postprocess.load_static(path)
scales = {name: wrapper.quantizer.scale.detach().clone() for name, wrapper in common.wrappers(model).items()}
actual = output(model)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
for name, wrapper in common.wrappers(model).items():
    torch.testing.assert_close(wrapper.quantizer.scale, scales[name], rtol=0, atol=0)
for name, p in model.named_parameters():
    if name in norms:
        torch.testing.assert_close(p.cpu(), norms[name], rtol=0, atol=0)
assert not model.config.tie_word_embeddings
assert model.lm_head.module.weight is not model.model.embed_tokens.weight
record('static_export_cold_load', {'difference': difference(actual, expected), 'coverage': len(records), 'qparams_unchanged': True, 'qk_norms_preserved': True})
del model, records, scales
cleanup()
# Actual tied teacher body movement across GPU/CPU, with a tiny student to bound memory.
from transformers import Qwen3Config, Qwen3ForCausalLM
teacher = distill.teacher_model().cuda()
student = Qwen3ForCausalLM(Qwen3Config(vocab_size=teacher.config.vocab_size, hidden_size=32,
    intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    head_dim=8, tie_word_embeddings=False)).bfloat16().cuda().eval()
values = distill.distillation_loss(student, teacher, ids, offload_teacher_body=True)
values[0].backward()
assert all(torch.isfinite(v) for v in values)
assert teacher.lm_head.weight is teacher.model.embed_tokens.weight
assert teacher.lm_head.weight.device.type == 'cuda'
assert next(teacher.model.layers[0].parameters()).device.type == 'cpu'
record('tied_teacher_gpu_offload', {'loss': float(values[0]), 'tied': True, 'body_on_cpu': True, 'head_on_cuda': True})
report['status'] = 'PASS'
report['max_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
(OUT / 'architecture_gpu.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS', flush=True)
