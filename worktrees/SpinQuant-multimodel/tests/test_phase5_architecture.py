"""Independent, bounded Phase5 architecture checks (tiny CPU models only)."""
import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM

from experiments.phase3 import architecture, common, distill, postprocess
from experiments.phase3.quantization import SP2Quantizer
from utils.fuse_norm_utils import fuse_layer_norms

torch.set_num_threads(2)


def tiny_config(layers=2):
    return Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, tie_word_embeddings=True,
        attention_dropout=0.0, rms_norm_eps=1e-6)


@pytest.fixture
def ids():
    return torch.tensor([[1, 4, 8, 3, 7, 2, 9, 5]])


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(123)
    model = Qwen3ForCausalLM(tiny_config()).bfloat16().eval()
    # Nonunit norm weights expose accidental norm fusion or dropped state.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('norm.weight'):
                parameter.copy_(torch.linspace(.8, 1.2, parameter.numel()))
    path = tmp_path / 'tiny-qwen'
    model.save_pretrained(path)
    monkeypatch.setattr(common, 'MODEL_PATH', path)
    monkeypatch.setattr(postprocess, 'MODEL_PATH', path)
    monkeypatch.setattr(distill, 'MODEL_PATH', path)
    return path


def logits(model, ids):
    return model.lm_head(common.backbone(model, ids)).float()


def bypass(model):
    for wrapper in common.wrappers(model).values():
        wrapper.quantizer.bits = 16
        if hasattr(wrapper.module, 'quantizer'):
            wrapper.module.quantizer.bits = 16


def test_native_adaptation_and_explicit_untie(checkpoint, ids):
    native = Qwen3ForCausalLM.from_pretrained(checkpoint,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
    adapted = architecture.load_model(checkpoint, training=True, untie=True).eval()
    assert native.lm_head.weight is native.model.embed_tokens.weight
    assert adapted.lm_head.weight is not adapted.model.embed_tokens.weight
    assert not adapted.config.tie_word_embeddings
    with torch.no_grad():
        torch.testing.assert_close(logits(adapted, ids), native(ids).logits.float(), rtol=0, atol=0)
    adapted.tie_weights()
    assert adapted.lm_head.weight is not adapted.model.embed_tokens.weight


def test_centering_is_separate_and_qk_norms_preserved(checkpoint, ids):
    model = architecture.load_model(checkpoint, training=True, untie=True).eval()
    saved = {name: p.clone() for name, p in model.named_parameters() if '.q_norm.' in name or '.k_norm.' in name}
    before = model.model.embed_tokens.weight.detach().clone()
    eps = [(layer.self_attn.q_norm.variance_epsilon, layer.self_attn.k_norm.variance_epsilon) for layer in model.model.layers]
    with torch.no_grad():
        raw = logits(model, ids)
        model.model.embed_tokens.weight.copy_((before.double() - before.double().mean(-1, keepdim=True)).bfloat16())
        centered = logits(model, ids)
        model.model.embed_tokens.weight.copy_(before)
        fuse_layer_norms(model)
        fused = logits(model, ids)
    # Centering is an actual preprocessing change, never swallowed in R tolerance.
    assert (raw - centered).abs().max() > 1e-4
    torch.testing.assert_close(fused, centered, rtol=.035, atol=.012)
    for name, value in saved.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    assert eps == [(layer.self_attn.q_norm.variance_epsilon, layer.self_attn.k_norm.variance_epsilon) for layer in model.model.layers]
    training = common.build_training_model()
    assert all(not p.requires_grad for name, p in training.named_parameters() if '.q_norm.' in name or '.k_norm.' in name)


def test_unquantized_rotation_gqa_and_checkpoint_gradients(checkpoint, ids):
    reference = architecture.load_model(checkpoint, training=True, untie=True).eval()
    fuse_layer_norms(reference)
    rotated = common.build_training_model()
    bypass(rotated)
    rotated.eval()
    with torch.no_grad():
        torch.testing.assert_close(logits(rotated, ids), logits(reference, ids), rtol=.05, atol=.014)
    # Compare gradients with vs without checkpoint recompute, including all R2s.
    gradients = []
    for enabled in (False, True):
        rotated.zero_grad(set_to_none=True)
        if enabled:
            rotated.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            rotated.gradient_checkpointing_disable()
        rotated.train()
        common.token_nll(rotated, ids).backward()
        gradients.append({name: p.grad.clone() for name, p in rotated.named_parameters() if name == 'R1.weight' or name.endswith('R2.weight')})
    assert len(gradients[0]) == 3
    for name in gradients[0]:
        assert torch.isfinite(gradients[0][name]).all()
        assert gradients[0][name].abs().max() > 0
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)


def test_static_export_cold_load_qparams_and_qk_norms(checkpoint, ids, tmp_path):
    training = common.build_training_model()
    common.initialize_scales(training, [ids])
    frozen, records = common.frozen_model(training)
    for name, wrapper in common.wrappers(frozen).items():
        if name.endswith('down_proj'):
            wrapper.quantizer.bits = 8
    path = tmp_path / 'static.pt'
    common.save_frozen(frozen, records, path, {'test': 'independent tiny'})
    loaded, loaded_records = postprocess.load_static(path, device='cpu')
    assert len(records) == len(loaded_records) == 14
    assert not loaded.config.tie_word_embeddings
    assert loaded.lm_head.module.weight is not loaded.model.embed_tokens.weight
    before = {name: value.clone() for name, value in loaded.state_dict().items()}
    with torch.no_grad():
        torch.testing.assert_close(logits(loaded, ids), logits(frozen, ids), rtol=0, atol=0)
        logits(loaded, ids.flip(-1))
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, equal_nan=True, msg=name)
    for name, value in frozen.state_dict().items():
        if '.q_norm.' in name or '.k_norm.' in name:
            torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    loaded.tie_weights()
    assert loaded.lm_head.module.weight is not loaded.model.embed_tokens.weight
    distill.prepare_student(loaded, loaded_records)
    assert all(not p.requires_grad for name, p in loaded.named_parameters() if '.q_norm.' in name or '.k_norm.' in name)


def test_28_layer_coverage_without_fullsize_forward(tmp_path, monkeypatch):
    path = tmp_path / '28layer-tiny'
    Qwen3ForCausalLM(tiny_config(layers=28)).bfloat16().save_pretrained(path)
    monkeypatch.setattr(common, 'MODEL_PATH', path)
    model = common.build_training_model()
    groups = common.learned_parameters(model)
    assert {name: len(values) for name, values in groups.items()} == dict(R=29, SA=168, SW=196, SP2=28)
    assert len(common.wrappers(model)) == 196
    assert all(not wrapper.online_full_had and not wrapper.online_partial_had for wrapper in common.wrappers(model).values())


def test_native_teacher_tied_and_frozen(checkpoint, ids):
    teacher = distill.teacher_model()
    assert teacher.lm_head.weight is teacher.model.embed_tokens.weight
    assert teacher.config.tie_word_embeddings
    assert not any(p.requires_grad for p in teacher.parameters())
    student = architecture.load_model(checkpoint, untie=True)
    regular = distill.distillation_loss(student, teacher, ids)
    offload = distill.distillation_loss(student, teacher, ids, offload_teacher_body=True)
    for left, right in zip(regular, offload):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert teacher.lm_head.weight is teacher.model.embed_tokens.weight


def test_llama_loader_narrow_regression(tmp_path, monkeypatch, ids):
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, tie_word_embeddings=False)
    path = tmp_path / 'tiny-llama'
    LlamaForCausalLM(config).bfloat16().save_pretrained(path)
    monkeypatch.setattr(common, 'MODEL_PATH', path)
    native = architecture.load_model(path).eval()
    model = common.build_training_model().eval()
    bypass(model)
    fuse_layer_norms(native)
    with torch.no_grad():
        torch.testing.assert_close(logits(model, ids), logits(native, ids), rtol=.05, atol=.014)
