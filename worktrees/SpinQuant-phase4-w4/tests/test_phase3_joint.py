import copy
import inspect
import json
import math
from pathlib import Path
import random
import shlex
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from eval_utils.modeling_llama import LlamaForCausalLM as EvaluationModel
from experiments.phase3 import common, quantization, run
from train_utils.optimizer import SGDG
from utils.quant_utils import ChannelScaleQuantize, LSQQuantize


@pytest.fixture
def tiny_loader(monkeypatch):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    config.head_dim = 4

    def read_config(*args, **kwargs):
        return copy.deepcopy(config)

    def read_model(model_class, *args, **kwargs):
        configuration = kwargs["config"]
        configuration._attn_implementation = kwargs["attn_implementation"]
        return model_class(configuration).to(dtype=kwargs["torch_dtype"])

    monkeypatch.setattr(common.AutoConfig, "from_pretrained", read_config)
    monkeypatch.setattr(common.TrainingModel, "from_pretrained", classmethod(read_model))
    monkeypatch.setattr(EvaluationModel, "from_pretrained", classmethod(read_model))
    return config


@pytest.fixture
def calibration():
    return [torch.tensor([[1, 4, 8, 3, 7, 2, 9, 5]]), torch.tensor([[6, 2, 5, 1, 8, 4]])]


@pytest.fixture
def initialized_model(tiny_loader, calibration):
    model = common.build_training_model()
    common.initialize_scales(model, calibration)
    return model


def oracle_sp2(inputs, alpha, levels):
    normalized = (inputs.float().abs() / alpha).clamp(max=1)
    distance = (normalized.double().unsqueeze(-1) - levels.double()).abs()
    indices = distance.argmin(dim=-1)
    return (levels[indices] * inputs.float().sign() * alpha).to(inputs.dtype)


def cloned_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_state_unchanged(model, before):
    after = model.state_dict()
    assert set(after) == set(before)
    for name, value in before.items():
        torch.testing.assert_close(after[name], value, rtol=0, atol=0, equal_nan=True, msg=name)


def test_sp2_codebook():
    levels = quantization.sp2_levels()
    expected = sorted({left + right
                       for left in [0.0] + [math.ldexp(1.0, -power) for power in range(1, 16)]
                       for right in [0.0] + [math.ldexp(1.0, -power) for power in range(1, 8)]})
    assert levels.tolist() == expected
    assert levels[0] == 0 and levels[-1] == 1
    assert torch.all(levels[1:] > levels[:-1])
    assert 2 * levels.numel() - 1 <= 256


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sp2_exact_forward_all_levels_ties_and_saturation(dtype):
    levels = quantization.sp2_levels()
    midpoints = (levels[:-1] + levels[1:]) / 2
    normalized = torch.cat((levels, midpoints, torch.tensor([1.25, 2.0])))
    inputs = torch.cat((-normalized.flip(0), normalized)).mul(127).to(dtype)
    scale = torch.tensor([1.0], requires_grad=True)
    expected = oracle_sp2(inputs, scale.detach() * 127, levels)
    actual = quantization.SP2ScaleSTE.apply(inputs, scale, levels)
    assert torch.equal(actual, expected)
    frozen = quantization.SP2Quantizer(127.0)
    assert torch.equal(actual, frozen(inputs))
    if dtype == torch.float32:
        assert torch.equal(quantization.sp2_project(midpoints, 1.0, levels), levels[:-1])


def test_sp2_gradients_match_explicit_surrogate_and_update():
    inputs = torch.tensor([-3.0, -2.54, -0.731, -0.021, 0.0, 0.179, 1.33, 2.54, 4.0], requires_grad=True)
    quantizer = quantization.SP2Quantizer(2.54, learnable=True)
    upstream = torch.tensor([1.0, -2.0, 3.0, 0.5, 2.0, -1.0, 4.0, 1.0, 2.0])
    normalized = inputs.detach().float() / quantizer.alpha.detach()
    projected = oracle_sp2(normalized, torch.ones(1), quantizer.levels)
    inside = normalized.abs() <= 1
    expected_scale = (upstream * 127 * (projected - normalized * inside)).sum()
    expected_scale /= math.sqrt(inputs.numel() * 127)
    before = quantizer.scale.detach().clone()
    identity = quantizer.scale
    optimizer = torch.optim.SGD(quantizer.parameters(), lr=1e-5)
    (quantizer(inputs) * upstream).sum().backward()
    torch.testing.assert_close(inputs.grad, upstream * inside, rtol=0, atol=0)
    torch.testing.assert_close(quantizer.scale.grad, expected_scale.reshape(1))
    assert quantizer.scale.grad.abs().item() > 0
    optimizer.step()
    assert quantizer.scale is identity and quantizer.scale.dtype == torch.float32
    assert torch.isfinite(quantizer.scale).all() and (quantizer.scale > 0).all()
    assert not torch.equal(quantizer.scale, before)
    frozen = quantization.SP2Quantizer(1.0)
    frozen.scale.copy_(quantizer.scale.detach())
    assert torch.equal(quantizer(inputs.detach()), frozen(inputs.detach()))
    quantizer.eval()
    assert torch.equal(quantizer(inputs.detach()), frozen(inputs.detach()))


@pytest.mark.parametrize("alpha", [0.0, -1.0, math.inf, math.nan])
def test_sp2_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        quantization.SP2Quantizer(alpha, learnable=True)


def test_int4_ties_bounds_zero_row_and_pack_layout():
    weights = torch.tensor([[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -8.5, 7.5], [0.0] * 8])
    scales = torch.tensor([[1.0], [1e-5]])
    expected = torch.tensor([[0, 2, 2, 0, -2, -2, -8, 7], [0] * 8], dtype=torch.int8)
    assert torch.equal(quantization.int4_codes(weights, scales), expected)
    codes = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
    packed = quantization.pack_int4(codes)
    assert packed.dtype == torch.uint8
    assert packed.tolist() == [16, 50, 84, 118, 152, 186, 220, 254]
    assert torch.equal(quantization.unpack_int4(packed, codes.shape), codes)
    assert torch.equal(quantization.unpack_int4(quantization.pack_int4(codes.T), codes.T.shape), codes.T)


@pytest.mark.parametrize("scale", [torch.ones(2), torch.tensor([[0.0], [1.0]]),
                                  torch.tensor([[-1.0], [1.0]]), torch.tensor([[math.nan], [1.0]])])
def test_int4_rejects_invalid_scales(scale):
    with pytest.raises(ValueError):
        quantization.int4_codes(torch.ones(2, 4), scale)


@pytest.mark.parametrize("codes", [torch.tensor([-8, 8]), torch.tensor([-9, 7]), torch.tensor([1])])
def test_pack_rejects_invalid_bounds_or_odd_length(codes):
    with pytest.raises(ValueError):
        quantization.pack_int4(codes)


def test_lsq_weight_and_activation_gradients_are_distinct():
    inputs = torch.tensor([[-9.0, -8.0, -1.5, 0.5, 7.0, 9.0]], requires_grad=True)
    scale = torch.ones(1, 1, requires_grad=True)
    upstream = torch.tensor([[1.0, 2.0, 3.0, -4.0, 5.0, 6.0]])
    (ChannelScaleQuantize.apply(inputs, scale, torch.tensor(7)) * upstream).sum().backward()
    torch.testing.assert_close(inputs.grad, upstream, rtol=0, atol=0)
    terms = torch.tensor([[-8.0, 0.0, -0.5, -0.5, 0.0, 7.0]])
    torch.testing.assert_close(scale.grad, (upstream * terms).sum().reshape(1, 1) / math.sqrt(6 * 7))
    activations = torch.tensor([-129.0, -128.0, -0.5, 0.5, 127.0, 129.0], requires_grad=True)
    activation_scale = torch.ones(1, requires_grad=True)
    LSQQuantize.apply(activations, activation_scale, torch.tensor(127)).sum().backward()
    assert torch.equal(activations.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
    torch.testing.assert_close(activation_scale.grad, torch.tensor([-1.0 / math.sqrt(6 * 127)]))


@pytest.mark.parametrize("route,step,weight_enabled,down_bits", [
    ("A", 0, False, 16), ("A", 49, False, 16), ("A", 50, True, 16),
    ("B", 0, True, 16), ("C", 0, True, 8),
])
def test_training_mode_actual_quantization(initialized_model, route, step, weight_enabled, down_bits):
    model = initialized_model
    groups = common.learned_parameters(model)
    assert {name: len(values) for name, values in groups.items()} == dict(R=17, SA=96, SW=112, SP2=16)
    parameter_ids = {name: id(parameter) for values in groups.values() for name, parameter in values}
    assert common.training_mode(model, route, step, 50) is weight_enabled
    for name, wrapper in common.wrappers(model).items():
        weight_quantizer = wrapper.module.quantizer
        values = weight_quantizer.scale.detach().expand_as(wrapper.weight) * 0.375
        if weight_enabled:
            assert not torch.equal(weight_quantizer.quantize(values), values)
        else:
            assert weight_quantizer.quantize(values) is values
        assert weight_quantizer.scale.requires_grad is weight_enabled
        if name.endswith("down_proj"):
            assert wrapper.quantizer.bits == down_bits
            assert wrapper.quantizer.scale.requires_grad is (route == "C")
            inputs = torch.full((1, 2, wrapper.module.in_features), float(wrapper.quantizer.alpha) * 0.37)
            seen = []
            handle = wrapper.quantizer.register_forward_hook(lambda module, args, output: seen.append(output))
            wrapper(inputs.to(wrapper.weight.dtype), model.R1.weight, transpose=True)
            handle.remove()
            assert len(seen) == (1 if route == "C" else 0)
        else:
            assert wrapper.quantizer.bits == 8 and wrapper.quantizer.scale.requires_grad
    assert parameter_ids == {name: id(parameter) for values in common.learned_parameters(model).values()
                             for name, parameter in values}


def test_schedule_update_index_and_matched_trajectory():
    values = [run.schedule(step, 100, 10) for step in range(100)]
    assert values[:10] == [(step + 1) / 10 for step in range(10)]
    assert values[10] == 1
    assert all(0 < value <= 1 for value in values)
    assert all(left >= right for left, right in zip(values[10:], values[11:]))
    assert run.schedule(100, 100, 10) == 0
    assert run.schedule(0, 100, 0) == 1
    assert run.schedule(9, 100, 0) != run.schedule(9, 10, 0)


def test_checkpoint_roundtrip_preserves_parameter_identity(initialized_model, tmp_path):
    model = initialized_model
    path = tmp_path / "state.pt"
    saved = common.save_state(model, path, update_step=10)
    parameters = dict(model.named_parameters())
    original_ids = {name: id(parameters[name]) for name in saved["parameters"]}
    with torch.no_grad():
        for name in saved["parameters"]:
            parameters[name].add_(0.001)
    loaded = common.load_state(model, path)
    assert loaded["metadata"]["update_step"] == 10
    for name, value in saved["parameters"].items():
        assert id(parameters[name]) == original_ids[name]
        assert torch.equal(parameters[name], value)
        assert parameters[name].dtype == torch.float32


@pytest.mark.parametrize("corruption", ["missing", "shape", "zero", "negative", "nan"])
def test_load_state_rejects_bad_checkpoint(initialized_model, tmp_path, corruption):
    model = initialized_model
    path = tmp_path / "state.pt"
    saved = common.save_state(model, path)
    name = "model.layers.0.self_attn.q_proj.module.quantizer.scale"
    if corruption == "missing":
        saved["parameters"].pop(name)
    elif corruption == "shape":
        saved["parameters"][name] = torch.ones(1)
    else:
        saved["parameters"][name].fill_({"zero": 0, "negative": -1, "nan": math.nan}[corruption])
    torch.save(saved, path)
    with pytest.raises(ValueError):
        common.load_state(model, path)


def test_build_from_checkpoint_rejects_wrong_sw_shape(initialized_model, tmp_path):
    path = tmp_path / "state.pt"
    saved = common.save_state(initialized_model, path)
    saved["parameters"]["model.layers.0.self_attn.q_proj.module.quantizer.scale"] = torch.ones(1)
    torch.save(saved, path)
    with pytest.raises(ValueError):
        common.build_training_model(path)


def test_rotated_weight_gqa_matches_independent_block_oracle(initialized_model):
    generator = torch.Generator().manual_seed(81)
    rotation1 = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64, generator=generator))[0]
    rotation2 = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64, generator=generator))[0]
    block_kv = torch.block_diag(rotation2)
    block_query = torch.block_diag(rotation2, rotation2)
    for name, wrapper, unused_rotation, transpose in common.rotated_linears(initialized_model):
        if not name.startswith("model.layers.0."):
            continue
        linear = copy.deepcopy(wrapper.module).double()
        original = linear.weight.detach().clone()
        if name.endswith("v_proj"):
            expected = block_kv.T @ original @ rotation1
            actual = linear.rotated_weight(rotation1, rotation2, transpose)
        elif name.endswith("o_proj"):
            expected = rotation1.T @ original @ block_query
            actual = linear.rotated_weight(rotation1, rotation2, transpose)
        else:
            expected = rotation1.T @ original if transpose else original @ rotation1
            actual = linear.rotated_weight(rotation1, None, transpose)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        assert torch.equal(original, linear.weight)


@pytest.mark.parametrize("optimizer_method", ["sgd", "adam"])
def test_training_fp_weights_frozen_and_joint_checkpoint_update(initialized_model, calibration, optimizer_method):
    model = initialized_model
    common.training_mode(model, "C", 0, 50)
    groups = common.learned_parameters(model)
    learned_names = {name for values in groups.values() for name, parameter in values}
    original_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                           if name not in learned_names}
    assert model.lm_head.module.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if name in original_parameters)
    before = {name: parameter.detach().clone() for values in groups.values() for name, parameter in values}
    model.train()
    model.gradient_checkpointing_disable()
    plain_loss = common.token_nll(model, calibration[0], chunk_size=3)
    plain_loss.backward()
    plain_gradients = {name: parameter.grad.detach().clone()
                       for values in groups.values() for name, parameter in values}
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    loss = common.token_nll(model, calibration[0], chunk_size=3)
    loss.backward()
    torch.testing.assert_close(loss, plain_loss, rtol=0, atol=0)
    for group, values in groups.items():
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for name, parameter in values), group
        assert sum(float(parameter.grad.abs().sum()) for name, parameter in values) > 0, group
        for name, parameter in values:
            torch.testing.assert_close(parameter.grad, plain_gradients[name], rtol=1e-5, atol=1e-6)
    optimizers = run.make_optimizers(groups, optimizer_arguments(optimizer_method))
    for optimizer in optimizers:
        optimizer.step()
    for group, values in groups.items():
        assert any(not torch.equal(before[name], parameter) for name, parameter in values), group
        assert all(parameter.dtype == torch.float32 and torch.isfinite(parameter).all()
                   for name, parameter in values), group
        if group != "R":
            assert all((parameter > 0).all() for name, parameter in values), group
    for name, parameter in model.named_parameters():
        if name in original_parameters:
            assert parameter.grad is None
            assert torch.equal(parameter, original_parameters[name]), name


def test_route_a_sw_reset_only_at_current_rotation(initialized_model):
    model = initialized_model
    before = cloned_state(model)
    identities = {name: id(wrapper.module.quantizer.scale) for name, wrapper in common.wrappers(model).items()}
    with torch.no_grad():
        model.R1.weight.copy_(torch.eye(8))
    common.reset_weight_scales(model)
    for name, wrapper, rotation2, transpose in common.rotated_linears(model):
        expected = wrapper.module.rotated_weight(model.R1.weight, rotation2, transpose).float()
        expected = expected.abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 7
        assert torch.equal(wrapper.module.quantizer.scale, expected), name
        assert id(wrapper.module.quantizer.scale) == identities[name]
    for name, value in model.state_dict().items():
        if name != "R1.weight" and not name.endswith(".module.quantizer.scale"):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, equal_nan=True, msg=name)


def test_frozen_eval_class_codes_and_identity_head_forward(initialized_model, calibration):
    model = initialized_model
    common.training_mode(model, "C", 0, 50)
    with torch.no_grad():
        model.R1.weight.copy_(torch.eye(8))
    original = cloned_state(model)
    frozen, records = common.frozen_model(model)
    assert isinstance(frozen, EvaluationModel)
    assert not hasattr(frozen, "R1")
    assert all(not hasattr(layer.self_attn, "R2") for layer in frozen.model.layers)
    assert not frozen.training and not frozen.config.use_cache
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    assert len(records) == 112
    assert len(common.wrappers(frozen)) == 112
    assert_state_unchanged(model, original)
    for name, wrapper, rotation2, transpose in common.rotated_linears(model):
        destination = frozen.get_submodule(name)
        weight = wrapper.module.rotated_weight(model.R1.weight, rotation2, transpose)
        expected_codes = (weight.float() / wrapper.module.quantizer.scale.detach()).round().clamp(-8, 7).to(torch.int8)
        record = records[name]
        assert torch.equal(quantization.unpack_int4(record["packed"], record["shape"]), expected_codes)
        assert torch.equal(record["scale"], wrapper.module.quantizer.scale)
        assert torch.equal(destination.weight, (expected_codes.float() * record["scale"]).to(torch.bfloat16))
        assert not destination.online_full_had and not destination.online_partial_had
        assert not hasattr(destination.module, "quantizer")
        assert destination.quantizer.bits == 8 and destination.out_quantizer.bits == 16
        assert destination.weight.data_ptr() != wrapper.weight.data_ptr()
    model.eval()
    with torch.no_grad():
        assert torch.equal(common.backbone(model, calibration[0]), common.backbone(frozen, calibration[0]))
        assert common.token_nll(model, calibration[0]) == common.token_nll(frozen, calibration[0])


def test_frozen_calibration_isolated_to_sixteen_downs(initialized_model, calibration):
    model = initialized_model
    common.training_mode(model, "B", 0, 50)
    original = cloned_state(model)
    frozen, records = common.frozen_model(model)
    before = cloned_state(frozen)
    result = common.calibrate_sp2(frozen, calibration)
    assert len(result) == 16
    assert_state_unchanged(model, original)
    for name, value in frozen.state_dict().items():
        if ".down_proj.quantizer." not in name:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, equal_nan=True, msg=name)
    for name, wrapper in common.wrappers(frozen).items():
        assert not wrapper._forward_pre_hooks
        assert wrapper.quantizer.bits == 8
        if name.endswith("down_proj"):
            selected = result[name]["selected"]
            assert selected["output_mse"] == min(candidate["output_mse"] for candidate in result[name]["candidates"])
            assert math.isfinite(float(wrapper.quantizer.alpha)) and float(wrapper.quantizer.alpha) > 0
            assert not isinstance(wrapper.quantizer.scale, torch.nn.Parameter)


def test_frozen_cold_reload_recovers_weights_scales_and_output(initialized_model, calibration, tmp_path):
    common.training_mode(initialized_model, "C", 0, 50)
    frozen, records = common.frozen_model(initialized_model)
    path = tmp_path / "frozen.pt"
    common.save_frozen(frozen, records, path, dict(route="C"))
    before = cloned_state(frozen)
    expected = common.evaluate(frozen, calibration)
    cold = copy.deepcopy(frozen)
    with torch.no_grad():
        for parameter in cold.parameters():
            parameter.fill_(0.25)
        for wrapper in common.wrappers(cold).values():
            wrapper.quantizer.scale.fill_(0.25)
    common.reload_frozen(cold, path)
    assert_state_unchanged(cold, before)
    assert common.evaluate(cold, calibration) == expected
    assert all(not parameter.requires_grad for parameter in cold.parameters())


@pytest.mark.parametrize("corruption", ["sw_negative", "sw_shape", "sw_nan", "sa_negative", "missing_head", "format"])
def test_reload_frozen_rejects_malformed_package(initialized_model, tmp_path, corruption):
    common.training_mode(initialized_model, "C", 0, 50)
    frozen, records = common.frozen_model(initialized_model)
    path = tmp_path / "frozen.pt"
    common.save_frozen(frozen, records, path, {})
    state = torch.load(path, map_location="cpu", weights_only=True)
    name = "model.layers.0.self_attn.q_proj"
    if corruption == "sw_negative":
        state["weights"][name]["scale"].fill_(-1)
    elif corruption == "sw_shape":
        state["weights"][name]["scale"] = torch.ones(1)
    elif corruption == "sw_nan":
        state["weights"][name]["scale"].fill_(math.nan)
    elif corruption == "sa_negative":
        state["activation"][name]["scale"].fill_(-1)
    elif corruption == "missing_head":
        state["high_precision"] = {key: value for key, value in state["high_precision"].items()
                                   if not key.startswith("lm_head.")}
    else:
        state["activation"][name]["format"] = "not_int8"
    torch.save(state, path)
    with pytest.raises(ValueError):
        common.reload_frozen(frozen, path)


def test_token_nll_fp32_chunking_and_tail_weighted_probe(initialized_model, calibration):
    model = initialized_model
    common.training_mode(model, "C", 0, 50)
    with torch.no_grad():
        hidden = common.backbone(model, calibration[0])
        logits = model.lm_head(hidden).float()
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                     calibration[0][:, 1:].reshape(-1), reduction="mean")
        for chunk_size in (1, 3, 128):
            torch.testing.assert_close(common.token_nll(model, calibration[0], chunk_size), expected)
    model.train()
    before = cloned_state(model)
    result = common.evaluate(model, calibration)
    assert model.training
    assert result["predicted_tokens"] == 12 and result["token_count"] == 14
    torch.testing.assert_close(torch.tensor(result["nll"]),
                               torch.tensor(sum(row["nll"] * row["predicted_tokens"] for row in result["windows"]) / 12))
    assert result["ppl"] == math.exp(result["nll"])
    assert_state_unchanged(model, before)


def test_full_validation_current_evaluator_wiring_and_full_tail_accounting(monkeypatch):
    from utils import eval_utils

    calls = []
    original_sys_path = list(sys.path)

    def evaluator(model, encoding, device, arguments):
        calls.append((model.seqlen, encoding.input_ids.numel(), device, arguments.bsz, arguments.eval_nsamples))
        return math.exp(2.0 if model.seqlen == 2048 else 3.0)

    monkeypatch.setattr(eval_utils, "evaluator", evaluator)
    model = torch.nn.Linear(1, 1)
    tokens = torch.zeros(1, 252852, dtype=torch.long)
    windows = [tokens[:, start:start + 2048] for start in range(0, tokens.numel(), 2048)]
    result = common.full_validation(model, windows)
    assert calls == [(2048, 251904, "cuda", 1, None), (948, 948, "cuda", 1, None)]
    assert result["token_count"] == 252852 and result["predicted_tokens"] == 252728
    assert result["unscored_tail_tokens"] == 0
    assert result["nll"] == (2.0 * (123 * 2047) + 3.0 * 947) / 252728
    assert model.seqlen == 2048
    assert sys.path == original_sys_path
    assert Path(eval_utils.__file__).resolve() == common.SOURCE_ROOT / "utils/eval_utils.py"
    assert result["evaluator_source"] == str(common.SOURCE_ROOT / "utils/eval_utils.py")


def test_existing_evaluator_actual_bf16_ce_on_cpu(initialized_model, calibration, monkeypatch):
    from utils import eval_utils

    common.training_mode(initialized_model, "C", 0, 50)
    frozen, records = common.frozen_model(initialized_model)
    tokens = calibration[0]
    frozen.seqlen = tokens.shape[1]
    with torch.no_grad():
        logits = frozen.lm_head(common.backbone(frozen, tokens))
        assert logits.dtype == torch.bfloat16
        token_losses = torch.nn.functional.cross_entropy(logits[:, :-1].permute(0, 2, 1),
                                                         tokens[:, 1:], reduction="none")
        assert token_losses.dtype == torch.bfloat16
        expected = torch.exp(token_losses.float().mean()).item()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = eval_utils.evaluator(frozen, SimpleNamespace(input_ids=tokens), "cpu",
                                  SimpleNamespace(eval_nsamples=None, bsz=1, capture_layer_io=False))
    assert result == expected


@pytest.mark.parametrize("suffix", ["self_attn.q_proj", "mlp.down_proj"])
def test_build_from_checkpoint_rejects_wrong_activation_shape(initialized_model, tmp_path, suffix):
    path = tmp_path / "state.pt"
    saved = common.save_state(initialized_model, path)
    saved["parameters"][f"model.layers.0.{suffix}.quantizer.scale"] = torch.ones(1, 1)
    torch.save(saved, path)
    with pytest.raises(ValueError):
        common.build_training_model(path)


def optimizer_arguments(method="adam", relative_scale_lr=0.001):
    return SimpleNamespace(scale_optimizer=method, r_lr=1e-4, sa_lr=1e-4, sw_lr=1e-4,
                           sp2_lr=1e-4, relative_scale_lr=relative_scale_lr)


@pytest.mark.parametrize("method", ["sgd", "adam"])
def test_make_optimizers_identity_rates_and_no_model_mutation(initialized_model, method):
    model = initialized_model
    groups = common.learned_parameters(model)
    before = cloned_state(model)
    arguments = optimizer_arguments(method)
    optimizers = run.make_optimizers(groups, arguments)
    expected_ids = [id(parameter) for values in groups.values() for name, parameter in values]
    actual_ids = [id(parameter) for optimizer in optimizers for group in optimizer.param_groups
                  for parameter in group["params"]]
    assert len(actual_ids) == len(set(actual_ids)) == 241
    assert set(actual_ids) == set(expected_ids)
    assert_state_unchanged(model, before)
    assert isinstance(optimizers[0], SGDG)
    if method == "sgd":
        assert len(optimizers) == 1 and len(optimizers[0].param_groups) == 4
        for group in optimizers[0].param_groups:
            assert group["stiefel"] is (group["name"] == "R")
            assert group["lr"] == group["initial_lr"] == 1e-4
    else:
        assert len(optimizers) == 2 and isinstance(optimizers[1], torch.optim.Adam)
        assert len(optimizers[0].param_groups) == 1
        assert optimizers[0].param_groups[0]["stiefel"]
        assert len(optimizers[1].param_groups) == 224
        named = {name: parameter for values in groups.values() for name, parameter in values}
        for group in optimizers[1].param_groups:
            assert len(group["params"]) == 1
            parameter = named[group["parameter_name"]]
            assert group["params"][0] is parameter
            assert group["lr"] == group["initial_lr"] == 0.001 * float(parameter.detach().mean())
            assert group["eps"] == 1e-12 and group["weight_decay"] == 0
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            initial = group["initial_lr"]
            group["lr"] = initial * run.schedule(9, 100, 10)
            assert group["lr"] == initial
            group["lr"] = initial * run.schedule(50, 100, 10)
            assert group["initial_lr"] == initial
    assert_state_unchanged(model, before)


@pytest.mark.parametrize("method", ["sgd", "adam"])
def test_save_resume_parameters_optimizer_rng_and_next_update(initialized_model, tmp_path, monkeypatch, method):
    model = initialized_model
    common.training_mode(model, "C", 0, 50)
    arguments = optimizer_arguments(method)
    optimizers = run.make_optimizers(common.learned_parameters(model), arguments)
    cuda_state = torch.tensor([4, 2, 7], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_state.clone())

    def update(current, active_optimizers, step):
        for optimizer in active_optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * run.schedule(step, 100, 10)
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, 0.01)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    update(model, optimizers, 0)
    before = cloned_state(model)
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    run.save_resume(model, optimizers, tmp_path, 1, "C")
    assert not (tmp_path / "resume.pt.tmp").exists()
    saved = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=True)
    assert saved["metadata"] == dict(route="C", update_step=1)
    assert len(saved["parameters"]) == 241
    assert len(saved["optimizers"]) == len(optimizers)
    assert saved["python_rng"] == python_rng
    assert torch.equal(saved["torch_rng"], torch_rng)
    assert torch.equal(saved["cuda_rng"], cuda_state)
    assert_state_unchanged(model, before)
    restored = copy.deepcopy(model)
    restored_parameters = dict(restored.named_parameters())
    identities = {name: id(restored_parameters[name]) for name in saved["parameters"]}
    with torch.no_grad():
        for name in saved["parameters"]:
            restored_parameters[name].fill_(0.25)
    common.load_state(restored, tmp_path / "resume.pt")
    restored_optimizers = run.make_optimizers(common.learned_parameters(restored), arguments)
    for optimizer, optimizer_state in zip(restored_optimizers, saved["optimizers"]):
        optimizer.load_state_dict(optimizer_state)
    for name, identity in identities.items():
        assert id(restored_parameters[name]) == identity
    assert_state_unchanged(restored, before)
    random.setstate(python_rng)
    torch.set_rng_state(torch_rng)
    update(model, optimizers, 1)
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["torch_rng"])
    update(restored, restored_optimizers, 1)
    assert_state_unchanged(restored, cloned_state(model))
    for expected_optimizer, actual_optimizer in zip(optimizers, restored_optimizers):
        expected = expected_optimizer.state_dict()
        actual = actual_optimizer.state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        assert actual["state"].keys() == expected["state"].keys()
        for index, entry in expected["state"].items():
            for key, value in entry.items():
                torch.testing.assert_close(actual["state"][index][key], value, rtol=0, atol=0)


def test_save_resume_interruption_preserves_previous_snapshot(initialized_model, tmp_path, monkeypatch):
    optimizers = run.make_optimizers(common.learned_parameters(initialized_model), optimizer_arguments())
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(3, dtype=torch.uint8))
    run.save_resume(initialized_model, optimizers, tmp_path, 1, "C")
    committed = (tmp_path / "resume.pt").read_bytes()

    def fail_save(value, destination):
        Path(destination).write_bytes(b"interrupted-write")
        raise OSError("injected interrupted save")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_save)
        with pytest.raises(OSError, match="injected interrupted save"):
            run.save_resume(initialized_model, optimizers, tmp_path, 2, "C")
    assert (tmp_path / "resume.pt").read_bytes() == committed
    assert common.load_state(initialized_model, tmp_path / "resume.pt")["metadata"]["update_step"] == 1
    run.save_resume(initialized_model, optimizers, tmp_path, 2, "C")
    assert not (tmp_path / "resume.pt.tmp").exists()
    assert common.load_state(initialized_model, tmp_path / "resume.pt")["metadata"]["update_step"] == 2


def test_actual_training_loop_clamps_adam_scale_overshoot(initialized_model, tmp_path, monkeypatch):
    model = initialized_model
    groups = common.learned_parameters(model)
    learned_names = {name for values in groups.values() for name, parameter in values}
    original_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                        if name not in learned_names}
    arguments = optimizer_arguments(relative_scale_lr=2.0)
    arguments.__dict__.update(initial=tmp_path / "unused.pt", resume=None, output=tmp_path,
                              route="C", steps=1, schedule_steps=100, warmup=0, switch_step=50,
                              accumulation=1, scale_only_steps=False, checkpoints=[])
    before_clamp = []
    actual_make = run.make_optimizers

    def capturing_optimizers(current_groups, args):
        optimizers = actual_make(current_groups, args)
        actual_step = optimizers[-1].step

        def capture_step(*args, **kwargs):
            result = actual_step(*args, **kwargs)
            before_clamp.extend(float(parameter.detach().min()) for group in optimizers[-1].param_groups
                                for parameter in group["params"])
            return result

        monkeypatch.setattr(optimizers[-1], "step", capture_step)
        return optimizers

    def synthetic_loss(current, tokens):
        return sum(parameter.sum() for values in common.learned_parameters(current).values()
                   for name, parameter in values if parameter.requires_grad)

    def cpu_module(module, *args, **kwargs):
        assert all(parameter.device.type == "cpu" for parameter in module.parameters())
        return module

    def cpu_tensor(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        return tensor

    monkeypatch.setattr(run, "build_training_model", lambda path: model)
    monkeypatch.setattr(run, "make_optimizers", capturing_optimizers)
    monkeypatch.setattr(run, "token_nll", synthetic_loss)
    monkeypatch.setattr(run, "evaluate_checkpoint", lambda model, args, step, *rest: dict(step=step))
    monkeypatch.setattr(run, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.nn.Module, "cuda", cpu_module)
    monkeypatch.setattr(torch.Tensor, "cuda", cpu_tensor)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(3, dtype=torch.uint8))
    run.train(arguments, torch.ones(1, 8, dtype=torch.long), [], [], [])
    assert before_clamp and min(before_clamp) < 0
    for group in ("SA", "SW", "SP2"):
        assert all(torch.isfinite(parameter).all() and (parameter >= 1e-8).all()
                   for name, parameter in groups[group])
    for name, parameter in model.named_parameters():
        if name in original_weights:
            assert not parameter.requires_grad and torch.equal(parameter, original_weights[name]), name
    assert (tmp_path / "resume.pt").exists()


def test_optimizer_scale_reference_fixes_rates_without_loading_parameters(initialized_model, tmp_path):
    model = initialized_model
    groups = common.learned_parameters(model)
    reference_path = tmp_path / "initial.pt"
    saved = common.save_state(model, reference_path)
    reference_bytes = reference_path.read_bytes()
    original = cloned_state(model)
    parameter_ids = {name: id(parameter) for values in groups.values() for name, parameter in values}
    default_arguments = optimizer_arguments()
    assert not hasattr(default_arguments, "optimizer_scale_reference")
    reference_arguments = optimizer_arguments()
    reference_arguments.optimizer_scale_reference = reference_path

    def scale_rates(optimizers):
        assert len(optimizers) == 2 and isinstance(optimizers[1], torch.optim.Adam)
        scale_groups = optimizers[1].param_groups
        assert len(scale_groups) == 224
        named = {name: parameter for values in groups.values() for name, parameter in values}
        for group in scale_groups:
            assert group["lr"] == group["initial_lr"]
            assert len(group["params"]) == 1
            assert group["params"][0] is named[group["parameter_name"]]
        rates = {group["parameter_name"]: group["initial_lr"] for group in scale_groups}
        assert len(rates) == 224
        return rates

    original_default_rates = scale_rates(run.make_optimizers(groups, default_arguments))
    original_reference_rates = scale_rates(run.make_optimizers(groups, reference_arguments))
    assert original_reference_rates == original_default_rates
    assert_state_unchanged(model, original)
    with torch.no_grad():
        for group_name in ("SA", "SW", "SP2"):
            for index, (name, parameter) in enumerate(groups[group_name]):
                parameter.mul_(2.0 + index / 128)
    changed = cloned_state(model)
    changed_reference_rates = scale_rates(run.make_optimizers(groups, reference_arguments))
    changed_default_rates = scale_rates(run.make_optimizers(groups, default_arguments))
    none_arguments = optimizer_arguments()
    none_arguments.optimizer_scale_reference = None
    changed_none_rates = scale_rates(run.make_optimizers(groups, none_arguments))
    assert changed_reference_rates == original_reference_rates
    assert changed_none_rates == changed_default_rates
    for group_name in ("SA", "SW", "SP2"):
        for name, parameter in groups[group_name]:
            assert changed_reference_rates[name] == 0.001 * float(saved["parameters"][name].mean())
            assert changed_default_rates[name] == 0.001 * float(parameter.detach().mean())
            assert changed_default_rates[name] != changed_reference_rates[name]
    assert parameter_ids == {name: id(parameter) for values in groups.values() for name, parameter in values}
    assert_state_unchanged(model, changed)
    assert reference_path.read_bytes() == reference_bytes


@pytest.mark.parametrize("task", [None, "train", "postprocess", "distill", "sequential", "local-d"])
def test_tmux_launch_explicit_child_environment_quoting_pid_and_records(tmp_path, monkeypatch, capsys, task):
    from experiments.phase3 import launch

    project = tmp_path / "project root 'quoted';$literal"
    source_file = project / "worktrees/SpinQuant-phase3-joint/experiments/phase3/launch.py"
    directory = project / "runs/phase3"
    directory.mkdir(parents=True)
    run_name = "tmux.cpu 'quoted';$literal"
    initial = "checkpoint with 'quotes';$(not-executed) $HOME.pt"
    session = "phase3-" + run_name.replace(".", "_").replace(":", "_")
    output = directory / run_name
    log_path = directory / (run_name + ".log")
    launch_path = directory / (run_name + ".launch.json")
    driver = {"postprocess": "postprocess.py", "distill": "distill.py",
              "sequential": "sequential_postprocess.py", "local-d": "local_d.py"}.get(task, "run.py")
    forwarded = (["--parent", initial, "--reference-state", initial, "--mode", "sp2"]
                 if task == "postprocess" else ["--initial", initial, "--route", "C"])
    if task == "distill":
        forwarded = ["--parent", initial, "--steps", "100"]
    if task == "sequential":
        forwarded = ["--parent", initial, "--mode", "sp2"]
    if task == "local-d":
        forwarded = ["--parent", initial, "--reference-state", initial, "--layer", "1"]
    command = ["/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python", "-u",
               str(source_file.parent / driver), "--output", str(output), *forwarded]
    environment = dict(CUDA_VISIBLE_DEVICES="7", PYTHONDONTWRITEBYTECODE="1",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       HF_HOME=str(project / "cache/huggingface"), GIT_OPTIONAL_LOCKS="0",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    existing_server = dict(CUDA_VISIBLE_DEVICES="3", HF_HOME="stale cache", UNRELATED="keep")
    server_before = existing_server.copy()
    tmux_calls = []
    queries = []

    def fake_query(arguments, **kwargs):
        assert arguments == ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"]
        assert kwargs == dict(text=True)
        queries.append(arguments)
        return "1, GPU-first, 4096, 24576, 80\n7, GPU-selected, 512, 24576, 5\n"

    def fake_run(arguments, **kwargs):
        assert arguments[:-1] == ["tmux", "-L", "rotation-quant-phase3", "new-session", "-d",
                                  "-s", session, "-c", str(project), "-P", "-F", "#{pane_pid}"]
        assert kwargs == dict(check=True, text=True, capture_output=True)
        payload = shlex.split(arguments[-1])
        assert payload[:2] == ["exec", "env"]
        supplied_environment = dict(assignment.split("=", 1) for assignment in payload[2:2 + len(environment)])
        assert supplied_environment == environment
        assert payload[2 + len(environment):-3] == command
        assert payload[-3:] == [">", str(log_path), "2>&1"]
        child_environment = dict(existing_server, **supplied_environment)
        assert child_environment["CUDA_VISIBLE_DEVICES"] == "7"
        assert child_environment["HF_HOME"] == environment["HF_HOME"]
        assert existing_server == server_before
        assert log_path.exists() and log_path.read_text() == ""
        tmux_calls.append(arguments)
        return SimpleNamespace(stdout=" 314159 \n")

    monkeypatch.setattr(launch, "__file__", str(source_file))
    monkeypatch.setattr(launch.shutil, "which", lambda executable: "/mock/tmux" if executable == "tmux" else None)
    monkeypatch.setattr(launch.subprocess, "check_output", fake_query)
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    task_arguments = [] if task is None else ["--task", task]
    monkeypatch.setattr(sys, "argv", [str(source_file), "--gpu", "7", "--name", run_name,
                                      *task_arguments, "--", *forwarded])
    launch.main()
    record = json.loads(launch_path.read_text())
    assert json.loads(capsys.readouterr().out) == record
    assert record["pid"] == 314159 and record["gpu"] == 7
    assert record["backend"] == "tmux"
    assert record["tmux_socket"] == "rotation-quant-phase3" and record["tmux_session"] == session
    assert shlex.split(record["attach_command"]) == ["tmux", "-L", "rotation-quant-phase3", "attach", "-t", session]
    assert record["command"] == command and shlex.split(record["shell_command"]) == command
    assert record["child_environment"] == environment
    assert record["output"] == str(output) and record["log"] == str(log_path)
    assert record["selected_gpu"] == dict(index=7, uuid="GPU-selected", memory_used_mib=512,
                                           memory_total_mib=24576, utilization=5)
    assert len(record["gpu_snapshot"]) == 2
    assert len(tmux_calls) == len(queries) == 1
    saved_record = launch_path.read_bytes()
    with pytest.raises(SystemExit) as failure:
        launch.main()
    assert failure.value.code == 2
    assert len(tmux_calls) == len(queries) == 1
    assert launch_path.read_bytes() == saved_record and log_path.read_text() == ""


@pytest.fixture
def postprocess_toy():
    from utils.quant_utils import ActQuantWrapper, RotationStaticActQuantizer

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    records = {}
    for suffix, input_features, output_features in (("up_proj", 2, 4), ("gate_proj", 2, 4), ("down_proj", 4, 2)):
        linear = torch.nn.Linear(input_features, output_features, bias=False)
        codes = (torch.arange(input_features * output_features).reshape(output_features, input_features) % 9 - 4).to(torch.int8)
        scale = torch.full((output_features, 1), 0.125)
        with torch.no_grad():
            linear.weight.copy_(codes.float() * scale)
        wrapper = ActQuantWrapper(linear)
        if suffix == "down_proj":
            wrapper.quantizer = quantization.SP2Quantizer(2.54)
        else:
            wrapper.quantizer = RotationStaticActQuantizer()
            wrapper.quantizer.load_scale(torch.tensor([0.125 if suffix == "up_proj" else 0.25]))
        setattr(layer.mlp, suffix, wrapper)
        records["model.layers.0.mlp." + suffix] = dict(packed=quantization.pack_int4(codes),
                                                     shape=tuple(codes.shape), scale=scale)
    model.requires_grad_(False).eval()
    return model, records


def assert_records_unchanged(actual, expected):
    assert actual.keys() == expected.keys()
    for name, record in actual.items():
        assert record["shape"] == expected[name]["shape"]
        assert torch.equal(record["packed"], expected[name]["packed"]), name
        assert torch.equal(record["scale"], expected[name]["scale"]), name


def test_postprocess_load_static_cold_exact_coverage_without_calibration(initialized_model, calibration, tmp_path, monkeypatch):
    from experiments.phase3 import postprocess
    from utils.quant_utils import ActQuantizer, RotationStaticActQuantizer

    common.training_mode(initialized_model, "C", 0, 50)
    frozen, records = common.frozen_model(initialized_model)
    path = tmp_path / "parent.pt"
    common.save_frozen(frozen, records, path, dict(test="cold-load"))
    before = cloned_state(frozen)
    expected_probe = common.evaluate(frozen, calibration)
    parent_bytes = path.read_bytes()
    reload_calls = []
    original_reload = postprocess.reload_frozen

    def reject_calibration(*args, **kwargs):
        pytest.fail("Cold frozen load attempted calibration or dynamic quantization")

    def record_reload(model, package):
        reload_calls.append(package)
        return original_reload(model, package)

    monkeypatch.setattr(postprocess, "reload_frozen", record_reload)
    monkeypatch.setattr(RotationStaticActQuantizer, "begin_calibration", reject_calibration)
    monkeypatch.setattr(RotationStaticActQuantizer, "finish_calibration", reject_calibration)
    monkeypatch.setattr(ActQuantizer, "find_params", reject_calibration)
    loaded, loaded_records = postprocess.load_static(path, device="cpu")
    assert reload_calls == [path]
    assert isinstance(loaded, EvaluationModel) and loaded is not frozen
    assert not hasattr(loaded, "R1") and not loaded.config.use_cache and not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert len(loaded_records) == len(common.wrappers(loaded)) == 112
    assert sum(isinstance(wrapper.quantizer, quantization.SP2Quantizer)
               for wrapper in common.wrappers(loaded).values()) == 16
    assert sum(isinstance(wrapper.quantizer, RotationStaticActQuantizer)
               for wrapper in common.wrappers(loaded).values()) == 96
    assert all(wrapper.quantizer.bits == 8 for wrapper in common.wrappers(loaded).values())
    assert_state_unchanged(loaded, before)
    assert_records_unchanged(loaded_records, records)
    assert common.evaluate(loaded, calibration) == expected_probe
    assert_state_unchanged(loaded, before)
    assert path.read_bytes() == parent_bytes


def test_postprocess_capture_quantized_raw_gate_and_aligned_split(postprocess_toy, monkeypatch):
    from experiments.phase3 import postprocess

    model, records = postprocess_toy
    layer = model.model.layers[0].mlp
    before = cloned_state(model)
    positions = torch.arange(32 * 128, dtype=torch.float32)
    features = torch.stack((positions / 256 + 0.13, torch.sin(positions / 15) + 0.17), dim=-1)
    windows = list(features.reshape(32, 1, 128, 2).unbind(0))

    def toy_backbone(current, inputs):
        mlp = current.model.layers[0].mlp
        return mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(inputs)) * mlp.up_proj(inputs))

    monkeypatch.setattr(postprocess, "backbone", toy_backbone)
    samples = postprocess.capture_inputs(model, windows)
    quantized_up = (features / 0.125).round().clamp(-128, 127) * 0.125
    quantized_gate = (features / 0.25).round().clamp(-128, 127) * 0.25
    expected_gate = torch.nn.functional.linear(quantized_gate, layer.gate_proj.weight)
    expected_up = torch.nn.functional.linear(quantized_up, layer.up_proj.weight)
    raw_down = torch.nn.functional.silu(expected_gate) * expected_up
    quantized_down = oracle_sp2(raw_down, layer.down_proj.quantizer.alpha, layer.down_proj.quantizer.levels)
    up_name, gate_name, down_name = ["model.layers.0.mlp." + suffix
                                    for suffix in ("up_proj", "gate_proj", "down_proj")]
    assert torch.equal(samples[up_name]["quantized"], quantized_up[::8])
    assert torch.equal(samples[gate_name]["quantized"], quantized_gate[::8])
    assert torch.equal(samples[gate_name]["output"], expected_gate[::8])
    assert torch.equal(samples[down_name]["raw"], raw_down[::8])
    assert torch.equal(samples[down_name]["quantized"], quantized_down[::8])
    assert not torch.equal(samples[down_name]["raw"], samples[down_name]["quantized"])
    assert all(tensor.shape[0] == 512 for sample in samples.values() for tensor in sample.values())
    assert torch.equal(samples[up_name]["quantized"][383], quantized_up[23 * 128 + 120])
    assert torch.equal(samples[up_name]["quantized"][384], quantized_up[24 * 128])
    split = postprocess.split_mse(torch.arange(512, dtype=torch.float32))
    assert split == dict(fit_mse=float(torch.arange(384, dtype=torch.float32).mean()),
                         heldout_mse=float(torch.arange(384, 512, dtype=torch.float32).mean()))
    for wrapper in common.wrappers(model).values():
        assert not wrapper._forward_pre_hooks and not wrapper._forward_hooks
        assert not wrapper.module._forward_pre_hooks
    assert_state_unchanged(model, before)


def test_postprocess_fixed_grid_round_parent_grid_and_pure_helper(postprocess_toy, tmp_path, monkeypatch):
    from experiments.phase3 import postprocess

    model, records = postprocess_toy
    name = "model.layers.0.mlp.up_proj"
    samples = {}
    reference = {}
    generator = torch.Generator().manual_seed(13)
    for module_name, wrapper in common.wrappers(model).items():
        records[module_name]["packed"] = quantization.pack_int4(torch.zeros_like(wrapper.weight, dtype=torch.int8))
        reference[module_name] = torch.zeros_like(wrapper.weight, dtype=torch.bfloat16)
        samples[module_name] = dict(quantized=torch.randn(16, wrapper.module.in_features, generator=generator))
    target_codes = torch.tensor([[1, -2], [3, 1], [-2, 2], [4, -1]], dtype=torch.int8)
    reference[name] = (target_codes.float() * records[name]["scale"]).to(torch.bfloat16)
    postprocess.apply_records(model, records)
    parent_records = copy.deepcopy(records)
    parent_reference = {key: value.clone() for key, value in reference.items()}
    before = cloned_state(model)
    imports = []
    original_spec = postprocess.importlib.util.spec_from_file_location
    original_path = list(sys.path)

    def capture_import(module_name, path, *args, **kwargs):
        imports.append(Path(path))
        return original_spec(module_name, path, *args, **kwargs)

    monkeypatch.setattr(postprocess.importlib.util, "spec_from_file_location", capture_import)
    monkeypatch.setattr(postprocess, "progress", lambda *args, **kwargs: None)
    candidates = postprocess.rounding_candidates(model, records, reference, samples,
                                                SimpleNamespace(output=tmp_path, top_modules=1))
    assert imports == [common.PROJECT_ROOT / "scripts/phase2/fixed_grid_rounding.py"]
    assert sys.path == original_path
    diagnostics = json.loads((tmp_path / "rounding.json").read_text())[name]
    assert diagnostics["trials"][0]["step"] == 0
    assert diagnostics["trials"][0]["changed_codes"] == 0
    assert diagnostics["selected_step"] > 0
    assert diagnostics["selected_step"] == min(diagnostics["trials"], key=lambda trial: trial["heldout_mse"])["step"]
    assert candidates and any(name in weights for label, weights, alphas in candidates)
    for label, weights, alphas in candidates:
        assert not alphas
        for module_name, record in weights.items():
            codes = quantization.unpack_int4(record["packed"], record["shape"])
            assert ((codes >= -8) & (codes <= 7)).all()
            assert torch.equal(record["scale"], parent_records[module_name]["scale"])
    assert_records_unchanged(records, parent_records)
    assert all(torch.equal(reference[key], value) for key, value in parent_reference.items())
    assert_state_unchanged(model, before)


def test_postprocess_fused_d_identity_selected_columns_and_bf16_grid():
    from experiments.phase3 import postprocess

    up_codes = torch.tensor([[-8, 7], [-3, 2], [1, -2], [4, 6]], dtype=torch.int8)
    down_codes = torch.tensor([[-7, 3, 2, -1], [4, -2, -5, 6]], dtype=torch.int8)
    up = dict(packed=quantization.pack_int4(up_codes), shape=(4, 2), scale=torch.tensor([[0.125], [0.25], [0.0625], [0.5]]))
    down = dict(packed=quantization.pack_int4(down_codes), shape=(2, 4), scale=torch.tensor([[0.25], [0.5]]))
    original = copy.deepcopy(dict(up=up, down=down))
    reference = torch.tensor([[0.0625, 1.0625, -0.25, 1.5], [-0.125, -1.5, 0.375, -2.0]], dtype=torch.bfloat16)
    original_reference = reference.clone()
    identity_up, identity_down = postprocess.fused_d_records(up, down, reference, torch.ones(4))
    assert_records_unchanged(dict(up=identity_up, down=identity_down), original)
    diagonal = torch.tensor([2.0, 1.0, 0.5, 1.0])
    transformed_up, transformed_down = postprocess.fused_d_records(up, down, reference, diagonal)
    expected_codes = down_codes.clone()
    for output_channel in range(2):
        for input_channel in (0, 2):
            value = float(reference[output_channel, input_channel]) * float(diagonal[input_channel])
            integer = round(value / float(down["scale"][output_channel, 0]))
            expected_codes[output_channel, input_channel] = min(7, max(-8, integer))
    assert torch.equal(transformed_up["packed"], up["packed"])
    assert torch.equal(transformed_up["scale"], up["scale"] / diagonal[:, None])
    assert torch.equal(transformed_down["scale"], down["scale"])
    actual_codes = quantization.unpack_int4(transformed_down["packed"], (2, 4))
    assert torch.equal(actual_codes, expected_codes)
    assert torch.equal(actual_codes[:, [1, 3]], down_codes[:, [1, 3]])
    expected_up_weight = (up_codes.float() * (up["scale"] / diagonal[:, None])).to(torch.bfloat16)
    expected_down_weight = (expected_codes.float() * down["scale"]).to(torch.bfloat16)
    assert torch.equal(postprocess.dequant_record(transformed_up, "cpu"), expected_up_weight)
    assert torch.equal(postprocess.dequant_record(transformed_down, "cpu"), expected_down_weight)
    inputs = torch.tensor([[0.5, -0.25]], dtype=torch.bfloat16)
    gate = torch.tensor([[0.75, -0.5, 1.0, 0.25]], dtype=torch.bfloat16)
    expected_pair = torch.nn.functional.linear(gate * torch.nn.functional.linear(inputs, expected_up_weight), expected_down_weight)
    actual_pair = torch.nn.functional.linear(
        gate * torch.nn.functional.linear(inputs, postprocess.dequant_record(transformed_up, "cpu")),
        postprocess.dequant_record(transformed_down, "cpu"))
    assert torch.equal(actual_pair, expected_pair)
    assert_records_unchanged(dict(up=up, down=down), original)
    assert torch.equal(reference, original_reference)
    for invalid in (torch.ones(3), torch.ones(4, 1), torch.tensor([1.0, 0.0, 1.0, 1.0]),
                    torch.tensor([1.0, -1.0, 1.0, 1.0]), torch.tensor([1.0, math.nan, 1.0, 1.0]),
                    torch.tensor([1.0, math.inf, 1.0, 1.0])):
        with pytest.raises(ValueError):
            postprocess.fused_d_records(up, down, reference, invalid)


def test_postprocess_range_candidates_include_parent_without_mutation(postprocess_toy, tmp_path):
    from experiments.phase3 import postprocess

    model, records = postprocess_toy
    before = cloned_state(model)
    name = "model.layers.0.mlp.down_proj"
    parent = float(model.get_submodule(name).quantizer.alpha)
    samples = {name: dict(raw=torch.linspace(-4, 4, 512 * 4).reshape(512, 4))}
    candidates = postprocess.range_candidates(model, records, samples, SimpleNamespace(output=tmp_path))
    result = json.loads((tmp_path / "sp2_ranges.json").read_text())[name]
    assert len(result["trials"]) == 33
    assert result["parent"]["alpha"] == parent and result["parent"]["exponent"] == 0
    assert result["selected"] == min(result["trials"], key=lambda trial: trial["heldout_mse"])
    assert all(trial["alpha"] > 0 and math.isfinite(trial["alpha"]) for trial in result["trials"])
    assert candidates[0][1] == {}
    if candidates[0][2]:
        assert candidates[0][2][name] == result["selected"]["alpha"]
        assert result["selected"]["heldout_mse"] < result["parent"]["heldout_mse"]
    assert_state_unchanged(model, before)


@pytest.mark.parametrize("outcome", ["improves", "parent", "probe-error"])
def test_postprocess_candidate_isolation_and_train_only_selection(postprocess_toy, tmp_path, monkeypatch, outcome):
    from experiments.phase3 import postprocess

    model, records = postprocess_toy
    before = cloned_state(model)
    record_before = copy.deepcopy(records)
    names = list(records)
    down_name = "model.layers.0.mlp.down_proj"
    original_scale = model.get_submodule(down_name).quantizer.scale.clone()
    first = dict(records[names[0]], packed=quantization.pack_int4(torch.zeros(records[names[0]]["shape"], dtype=torch.int8)))
    second = dict(records[names[1]], packed=quantization.pack_int4(torch.ones(records[names[1]]["shape"], dtype=torch.int8)))
    alpha = float(model.get_submodule(down_name).quantizer.alpha) * 2
    calibration_marker, probe_marker, validation_marker = object(), object(), object()
    samples_marker = object()
    events = []
    evaluation_count = []
    args = SimpleNamespace(output=tmp_path / "run", parent=tmp_path / "parent.pt", mode="round",
                           reference_state=tmp_path / "state.pt", top_modules=1)

    def capture(current, calibration):
        assert calibration is calibration_marker
        events.append("capture-train")
        return samples_marker

    def propose(current, parent_records, reference, samples, arguments):
        assert samples is samples_marker
        events.append("generate")
        return [("first", {names[0]: first}, {down_name: alpha}), ("second", {names[1]: second}, {})]

    def probe(current, windows):
        assert windows is probe_marker
        index = len(evaluation_count)
        evaluation_count.append(index)
        events.append("probe-" + str(index))
        if index == 0:
            assert_state_unchanged(current, before)
            return dict(nll=2.0)
        if index == 1:
            assert torch.equal(current.get_submodule(names[0]).weight, postprocess.dequant_record(first, "cpu", torch.float32))
            assert torch.equal(current.get_submodule(down_name).quantizer.scale, original_scale.new_tensor([alpha / 127]))
            if outcome == "probe-error":
                raise RuntimeError("injected probe failure")
            return dict(nll=2.1)
        assert index == 2
        assert torch.equal(current.get_submodule(names[0]).weight, before[names[0] + ".module.weight"])
        assert torch.equal(current.get_submodule(down_name).quantizer.scale, original_scale)
        assert torch.equal(current.get_submodule(names[1]).weight, postprocess.dequant_record(second, "cpu", torch.float32))
        return dict(nll=1.9 if outcome == "improves" else 2.2)

    def validation(current, windows):
        assert windows is validation_marker and events[-1] == "probe-2"
        assert outcome == "improves"
        events.append("validation")
        assert torch.equal(current.get_submodule(names[0]).weight, before[names[0] + ".module.weight"])
        assert torch.equal(current.get_submodule(down_name).quantizer.scale, original_scale)
        return dict(nll=100.0, ppl=1e20)

    model.config = SimpleNamespace(to_dict=lambda: {})
    monkeypatch.setattr(postprocess, "data_windows", lambda output: ([], calibration_marker, probe_marker, validation_marker, {}))
    monkeypatch.setattr(postprocess, "reference_weights", lambda path: {})
    monkeypatch.setattr(postprocess, "load_static", lambda path: (model, records))
    monkeypatch.setattr(postprocess, "capture_inputs", capture)
    monkeypatch.setattr(postprocess, "rounding_candidates", propose)
    monkeypatch.setattr(postprocess, "evaluate", probe)
    monkeypatch.setattr(postprocess, "full_validation", validation)
    monkeypatch.setattr(postprocess, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(postprocess, "source_record", lambda: {})
    monkeypatch.setattr(postprocess.subprocess, "check_output", lambda *args, **kwargs: b"")
    monkeypatch.setattr(postprocess.shutil, "copyfile", lambda *args, **kwargs: None)
    if outcome == "probe-error":
        with pytest.raises(RuntimeError, match="injected probe failure"):
            postprocess.run(args)
        assert_state_unchanged(model, before)
        assert "validation" not in events
        return
    postprocess.run(args)
    result = json.loads((args.output / "result.json").read_text())
    assert events[:3] == ["probe-0", "capture-train", "generate"]
    assert events[3:5] == ["probe-1", "probe-2"]
    assert_records_unchanged(records, record_before)
    if outcome == "improves":
        assert result["selected"] == "second" and result["best_probe"] == 1.9
        assert result["validation"]["nll"] == 100.0 and events[-1] == "validation"
    else:
        assert result["selected"] is None and result["best_probe"] == 2.0
        assert "validation" not in events
        assert_state_unchanged(model, before)


def test_postprocess_pair_range_bf16_oracle_and_no_mutation(postprocess_toy):
    from experiments.phase3 import postprocess

    model, records = postprocess_toy
    up_name, down_name = "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"
    inputs = torch.linspace(-3, 5, 512 * 2).reshape(512, 2).to(torch.bfloat16)
    gate = torch.linspace(-1, 2, 512 * 4).reshape(512, 4).to(torch.bfloat16)
    reference_up = (model.get_submodule(up_name).weight.detach() + 0.03125).to(torch.bfloat16)
    reference_down = (model.get_submodule(down_name).weight.detach() - 0.0625).to(torch.bfloat16)
    target = torch.nn.functional.linear(gate * torch.nn.functional.linear(inputs, reference_up), reference_down)
    levels = model.get_submodule(down_name).quantizer.levels
    parent_alpha = float(model.get_submodule(down_name).quantizer.alpha)
    records_before = copy.deepcopy(records)
    tensors = (inputs, gate, target, levels, reference_down)
    tensors_before = [value.clone() for value in tensors]
    state_before = cloned_state(model)
    transformed = postprocess.fused_d_records(records[up_name], records[down_name], reference_down,
                                             torch.tensor([4.0, 1.0, 0.25, 1.0]))
    for up, down in ((records[up_name], records[down_name]), transformed):
        pair_before = copy.deepcopy(dict(up=up, down=down))
        up_weight = (quantization.unpack_int4(up["packed"], up["shape"]).float() * up["scale"]).to(torch.bfloat16)
        down_weight = (quantization.unpack_int4(down["packed"], down["shape"]).float() * down["scale"]).to(torch.bfloat16)
        values = gate * torch.nn.functional.linear(inputs, up_weight)
        expected = []
        for exponent in range(-8, 5):
            alpha = parent_alpha * 2 ** exponent
            actual = torch.nn.functional.linear(oracle_sp2(values, alpha, levels), down_weight)
            errors = (actual.float() - target.float()).square().mean(dim=1)
            expected.append(dict(alpha=alpha, range_exponent=exponent,
                                 fit_mse=float(errors[:384].mean()), heldout_mse=float(errors[384:].mean())))
        trials = postprocess.pair_range_trials(inputs, gate, up, down, target, parent_alpha, levels)
        assert trials == expected
        assert [trial["range_exponent"] for trial in trials] == list(range(-8, 5))
        assert trials[8]["alpha"] == parent_alpha
        assert any(trial["fit_mse"] != trial["heldout_mse"] for trial in trials)
        assert_records_unchanged(dict(up=up, down=down), pair_before)
    for actual, original in zip(tensors, tensors_before):
        assert torch.equal(actual, original)
    assert_records_unchanged(records, records_before)
    assert_state_unchanged(model, state_before)


@pytest.mark.parametrize("extra_d_gain", [False, True])
def test_postprocess_diagonal_paired_range_control_and_binding(postprocess_toy, tmp_path, monkeypatch, extra_d_gain):
    from experiments.phase3 import postprocess

    model, records = postprocess_toy
    up_name, gate_name, down_name = ("model.layers.0.mlp." + suffix
                                    for suffix in ("up_proj", "gate_proj", "down_proj"))
    with torch.no_grad():
        model.get_submodule(down_name).weight[:, 3].zero_()
    down_codes = quantization.unpack_int4(records[down_name]["packed"], records[down_name]["shape"])
    down_codes[:, 3] = 0
    records[down_name]["packed"] = quantization.pack_int4(down_codes)
    parent_alpha = float(model.get_submodule(down_name).quantizer.alpha)
    samples = {
        up_name: dict(quantized=torch.linspace(-2, 3, 16).reshape(8, 2).to(torch.bfloat16)),
        gate_name: dict(output=torch.linspace(-1, 2, 32).reshape(8, 4).to(torch.bfloat16)),
        down_name: dict(raw=torch.linspace(-0.3, 0.3, 32).reshape(8, 4).to(torch.bfloat16)),
    }
    samples[down_name]["raw"][:, 3] = parent_alpha
    reference = {name: model.get_submodule(name).weight.detach().to(torch.bfloat16)
                 for name in (up_name, down_name)}
    reference[down_name] = reference[down_name] + 0.03125
    state_before = cloned_state(model)
    records_before = copy.deepcopy(records)
    samples_before, reference_before = copy.deepcopy(samples), copy.deepcopy(reference)
    expected_gate = torch.nn.functional.silu(samples[gate_name]["output"])
    expected_target = torch.nn.functional.linear(
        expected_gate * torch.nn.functional.linear(samples[up_name]["quantized"], reference[up_name]),
        reference[down_name])
    pair_calls = []

    def measured_pair(inputs, gate, up, down, target, alpha, levels):
        assert torch.equal(inputs, samples[up_name]["quantized"])
        assert torch.equal(gate, expected_gate) and torch.equal(target, expected_target)
        assert alpha == parent_alpha and levels is model.get_submodule(down_name).quantizer.levels
        diagonal = (records[up_name]["scale"] / up["scale"]).flatten()
        changed = diagonal != 1
        identity = not bool(changed.any())
        factor = 1.0 if identity else float(diagonal[changed][0])
        special = int(changed.sum()) == 1 and factor == 16.0
        pair_calls.append(dict(up=up, down=down, factor=factor, diagonal=diagonal, special=special))
        if identity:
            assert up is records[up_name] and down is records[down_name]
        trials = []
        for exponent in range(-8, 5):
            error = 20.0 + abs(exponent)
            if identity and exponent == 0:
                error = 10.0
            if identity and exponent == -1:
                error = 2.0
            if not identity and exponent == -2:
                error = 3.0
            if special and exponent == -3:
                error = 1.0 if extra_d_gain else 2.0
            trials.append(dict(alpha=alpha * 2 ** exponent, range_exponent=exponent,
                               fit_mse=0.0 if exponent == 4 else 5.0, heldout_mse=error))
        return trials

    def reject_evaluation(*args, **kwargs):
        pytest.fail("Local paired search must not evaluate probe or validation")

    monkeypatch.setattr(postprocess, "pair_range_trials", measured_pair)
    monkeypatch.setattr(postprocess, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(postprocess, "evaluate", reject_evaluation)
    monkeypatch.setattr(postprocess, "full_validation", reject_evaluation)
    proposals = postprocess.diagonal_candidates(model, records, reference, samples, SimpleNamespace(output=tmp_path))
    raw = samples[down_name]["raw"]
    projected = oracle_sp2(raw, parent_alpha, model.get_submodule(down_name).quantizer.levels).float()
    expected_activation = (projected - raw.float()).square().mean(dim=0) * reference[down_name].float().square().sum(dim=0)
    expected_weight = projected.square().mean(dim=0) * (
        model.get_submodule(down_name).weight.float() - reference[down_name].float()).square().sum(dim=0)
    expected_total = expected_activation + expected_weight
    ranking = json.loads((tmp_path / "diagonal_ranking.json").read_text())
    assert len(ranking) == 1 and ranking[0]["name"] == down_name
    assert ranking[0]["activation_contributions"] == expected_activation.tolist()
    assert ranking[0]["weight_contributions"] == expected_weight.tolist()
    assert ranking[0]["channel_contributions"] == expected_total.tolist()
    assert ranking[0]["score"] == float(expected_total.sum())
    assert ranking[0]["channels"] == expected_total.argsort(descending=True).tolist()
    assert expected_activation[3] == 0 and expected_weight[3] > 0
    assert expected_activation.argmax().item() != 3 and ranking[0]["channels"][0] == 3
    assert len(pair_calls) == 22
    assert pair_calls[0]["factor"] == 1.0
    assert [call["factor"] for call in pair_calls[1:]] == [0.25, 0.5, 2.0, 4.0, 16.0, 64.0, 256.0] * 3
    diagnostics = json.loads((tmp_path / "diagonal_candidates.json").read_text())[down_name]
    assert len(diagnostics["trials"]) == 22 * 13
    assert diagnostics["parent"]["alpha"] == parent_alpha
    assert diagnostics["parent"]["range_exponent"] == 0
    assert diagnostics["best_identity"]["alpha"] == parent_alpha / 2
    assert diagnostics["best_identity"]["heldout_mse"] == 2.0
    assert proposals[0] == ("range-only-" + down_name, {}, {down_name: parent_alpha / 2})
    if extra_d_gain:
        assert len(proposals) == 2
        label, weights, alphas = proposals[1]
        assert label == "local-d-" + down_name and set(weights) == {up_name, down_name}
        assert alphas == {down_name: parent_alpha / 8}
        selected_call = next(call for call in pair_calls if call["special"])
        assert weights[up_name] is selected_call["up"] and weights[down_name] is selected_call["down"]
        assert diagnostics["selected"]["factor"] == 16.0 and diagnostics["selected"]["range_exponent"] == -3
        assert diagnostics["selected"]["channels"] == (selected_call["diagonal"] != 1).nonzero().flatten().tolist()
        assert diagnostics["selected"]["heldout_mse"] == 1.0
    else:
        assert len(proposals) == 1
        assert diagnostics["selected"] == diagnostics["best_identity"]
    assert_records_unchanged(records, records_before)
    assert_state_unchanged(model, state_before)
    for name, record in samples.items():
        for key, value in record.items():
            assert torch.equal(value, samples_before[name][key])
    for name, value in reference.items():
        assert torch.equal(value, reference_before[name])


@pytest.fixture
def distill_student(initialized_model, tmp_path):
    from experiments.phase3 import distill, postprocess

    common.training_mode(initialized_model, "C", 0, 50)
    frozen, records = common.frozen_model(initialized_model)
    path = tmp_path / "distill-parent.pt"
    common.save_frozen(frozen, records, path, dict(test="distill-parent"))
    model, records = postprocess.load_static(path, device="cpu")
    parent = copy.deepcopy(model)
    old_weights = [wrapper.module.weight for wrapper in common.wrappers(model).values()]
    distill.prepare_student(model, records)
    return model, parent, records, old_weights


def distill_arguments(parent):
    return SimpleNamespace(parent=parent, weight_lr=0.1, momentum=0.9, relative_scale_lr=0.001,
                           data_start=800, accumulation=8, temperature=1.0, ce_weight=0.1)


def assert_optimizer_states_equal(actual_optimizers, expected_optimizers):
    assert len(actual_optimizers) == len(expected_optimizers)
    for actual_optimizer, expected_optimizer in zip(actual_optimizers, expected_optimizers):
        actual, expected = actual_optimizer.state_dict(), expected_optimizer.state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        assert actual["state"].keys() == expected["state"].keys()
        for index, entry in expected["state"].items():
            assert actual["state"][index].keys() == entry.keys()
            for key, value in entry.items():
                torch.testing.assert_close(actual["state"][index][key], value, rtol=0, atol=0)


def test_distill_parent_grid_aliases_real_checkpoint_backward(distill_student, calibration, monkeypatch):
    from experiments.phase3 import distill

    student, parent, records, old_weights = distill_student
    groups = distill.parameter_groups(student)
    assert {name: len(values) for name, values in groups.items()} == dict(W=112, SA=96, SW=112, SP2=16)
    trainable = {id(parameter) for values in groups.values() for _, parameter in values}
    assert len(trainable) == 336
    assert trainable == {id(parameter) for parameter in student.parameters() if parameter.requires_grad}
    assert not {id(parameter) for parameter in old_weights} & {id(parameter) for parameter in student.parameters()}
    for name, wrapper in common.wrappers(student).items():
        assert isinstance(wrapper.module, distill.TrainableQuantLinear)
        assert wrapper.quantizer.bits == 8
        assert wrapper.weight is wrapper.module.weight and wrapper.bias is None
        assert wrapper.module.weight.dtype == wrapper.module.quantizer.scale.dtype == torch.float32
        assert torch.equal(wrapper.module.quantizer.quantize(wrapper.module.weight).to(torch.bfloat16),
                           parent.get_submodule(name).module.weight)
    assert not hasattr(student, "R1")
    assert student.model.gradient_checkpointing
    assert student.model._gradient_checkpointing_func.keywords["use_reentrant"] is False
    high_precision = {name: parameter.detach().clone() for name, parameter in student.named_parameters()
                      if id(parameter) not in trainable}
    with torch.no_grad():
        for ids in calibration:
            expected = parent.lm_head(common.backbone(parent, ids))
            assert torch.equal(student.lm_head(common.backbone(student, ids)), expected)
    teacher = distill.teacher_model()
    teacher_before = cloned_state(teacher)
    checkpoints = []
    original_checkpoint = distill.checkpoint

    def record_checkpoint(function, *args, **kwargs):
        checkpoints.append(kwargs)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(distill, "checkpoint", record_checkpoint)
    optimizers = distill.make_optimizers(groups, distill_arguments("parent.pt"))
    assert isinstance(optimizers[0], torch.optim.SGD) and isinstance(optimizers[1], torch.optim.Adam)
    assert optimizers[0].defaults["momentum"] == 0.9 and optimizers[0].defaults["foreach"] is False
    assert len(optimizers[1].param_groups) == 224
    assert optimizers[1].defaults["eps"] == 1e-12
    optimized = [parameter for optimizer in optimizers for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimized) == len({id(parameter) for parameter in optimized}) == 336
    assert {id(parameter) for parameter in optimized} == trainable
    before = {id(parameter): parameter.detach().clone() for parameter in optimized}
    student.train()
    loss, ce, divergence = distill.distillation_loss(student, teacher, calibration[0], chunk_size=3)
    loss.backward()
    assert len(checkpoints) == 3 and all(row["use_reentrant"] is False for row in checkpoints)
    assert all(torch.isfinite(value) for value in (loss, ce, divergence))
    for values in groups.values():
        for name, parameter in values:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert any(torch.count_nonzero(parameter.grad) for _, parameter in values)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())
    torch.nn.utils.clip_grad_norm_(optimized, 1.0, error_if_nonfinite=True)
    for optimizer in optimizers:
        optimizer.step()
    with torch.no_grad():
        for name in ("SA", "SW", "SP2"):
            for _, parameter in groups[name]:
                parameter.clamp_(min=1e-8)
    for values in groups.values():
        assert any(not torch.equal(parameter, before[id(parameter)]) for _, parameter in values)
    for name, expected in high_precision.items():
        parameter = student.get_parameter(name)
        assert not parameter.requires_grad and parameter.grad is None
        assert torch.equal(parameter, expected)
    assert_state_unchanged(teacher, teacher_before)


def test_distill_chunk_objective_direction_temperature_and_sums():
    from experiments.phase3 import distill

    student = torch.linspace(-2, 3, 2 * 5 * 7).reshape(2, 5, 7).flip(-1).requires_grad_()
    teacher = torch.cos(torch.arange(student.numel()).reshape_as(student).float()).requires_grad_()
    labels = torch.arange(10).reshape(2, 5) % 7
    ce_weight = 0.1
    for temperature in (1.0, 2.0):
        student.grad = None
        actual = distill.chunk_objective(student, teacher, labels, temperature, ce_weight)
        student_values, teacher_values = student.detach().double(), teacher.detach().double()
        student_logp = student_values / temperature - torch.logsumexp(student_values / temperature, -1, keepdim=True)
        teacher_logp = teacher_values / temperature - torch.logsumexp(teacher_values / temperature, -1, keepdim=True)
        ce_logp = student_values - torch.logsumexp(student_values, -1, keepdim=True)
        ce = -ce_logp.gather(-1, labels.unsqueeze(-1)).sum()
        divergence = (teacher_logp.exp() * (teacher_logp - student_logp)).sum() * temperature ** 2
        expected = (ce_weight * ce + (1 - ce_weight) * divergence, ce, divergence)
        for value, target in zip(actual, expected):
            assert value.dtype == torch.float32
            torch.testing.assert_close(value.double(), target, rtol=2e-6, atol=1e-6)
        reverse = (student_logp.exp() * (student_logp - teacher_logp)).sum() * temperature ** 2
        assert not torch.isclose(divergence, reverse, rtol=1e-3)
        actual[0].backward()
        assert student.grad is not None and torch.count_nonzero(student.grad)
        assert teacher.grad is None


def test_distill_loss_chunk_tail_shift_and_token_normalization(monkeypatch):
    from experiments.phase3 import distill

    student = torch.nn.Module()
    student.embed = torch.nn.Embedding(11, 4).to(torch.bfloat16)
    student.lm_head = torch.nn.Linear(4, 11, bias=False).to(torch.bfloat16).requires_grad_(False)
    teacher = copy.deepcopy(student).requires_grad_(False)
    with torch.no_grad():
        teacher.embed.weight.add_(0.125)
    monkeypatch.setattr(distill, "backbone", lambda model, ids: model.embed(ids))
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 6, 4, 2, 9, 7, 5, 3]])
    with torch.no_grad():
        student_values = student.lm_head(student.embed(ids)[:, :-1]).double()
        teacher_values = teacher.lm_head(teacher.embed(ids)[:, :-1]).double()
        student_logp = student_values - torch.logsumexp(student_values, -1, keepdim=True)
        teacher_logp = teacher_values - torch.logsumexp(teacher_values, -1, keepdim=True)
        ce = -student_logp.gather(-1, ids[:, 1:].unsqueeze(-1)).sum() / 14
        divergence = (teacher_logp.exp() * (teacher_logp - student_logp)).sum() / 14
        expected = (0.1 * ce + 0.9 * divergence, ce, divergence)
    for chunk_size in (1, 3, 128):
        student.zero_grad(set_to_none=True)
        actual = distill.distillation_loss(student, teacher, ids, chunk_size=chunk_size)
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value.double(), target, rtol=3e-6, atol=1e-6)
        actual[0].backward()
        assert student.embed.weight.grad is not None and torch.isfinite(student.embed.weight.grad).all()
        assert student.lm_head.weight.grad is None
        assert all(parameter.grad is None for parameter in teacher.parameters())


def test_distill_export_coldload_exact_forward_codes_and_hp(distill_student, calibration, tmp_path):
    from experiments.phase3 import distill, postprocess

    student, parent, records, _ = distill_student
    original_records = copy.deepcopy(records)
    name = next(iter(records))
    wrapper = student.get_submodule(name)
    original_code = int(quantization.unpack_int4(records[name]["packed"], records[name]["shape"])[0, 0])
    next_code = original_code + 1 if original_code < 7 else original_code - 1
    with torch.no_grad():
        wrapper.module.weight[0, 0] = (next_code + 0.1) * wrapper.module.quantizer.scale[0, 0]
        student.get_submodule("model.layers.0.mlp.down_proj").quantizer.scale.mul_(1.125)
    before = cloned_state(student)
    output = tmp_path / "student.pt"
    changed = distill.export_student(student, records, output, dict(test="distill-export"))
    state = torch.load(output, map_location="cpu", weights_only=True)
    assert len(changed) == len(state["weights"]) == 112
    assert changed[name]["changed_codes"] > 0
    for module_name, current in common.wrappers(student).items():
        expected = (current.module.weight.detach().float() / current.module.quantizer.scale.detach()).round().clamp(-8, 7).to(torch.int8)
        packed = state["weights"][module_name]
        codes = quantization.unpack_int4(packed["packed"], packed["shape"])
        assert torch.equal(codes, expected) and ((codes >= -8) & (codes <= 7)).all()
        parent_codes = quantization.unpack_int4(records[module_name]["packed"], records[module_name]["shape"])
        assert changed[module_name] == dict(changed_codes=int((expected != parent_codes).sum()), total_codes=expected.numel())
    for key, value in state["high_precision"].items():
        torch.testing.assert_close(value, parent.state_dict()[key], rtol=0, atol=0, equal_nan=True, msg=key)
    loaded, loaded_records = postprocess.load_static(output, device="cpu")
    assert len(loaded_records) == 112
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    with torch.no_grad():
        for ids in calibration:
            expected = student.lm_head(common.backbone(student, ids))
            assert torch.equal(loaded.lm_head(common.backbone(loaded, ids)), expected)
    assert_state_unchanged(student, before)
    assert_records_unchanged(records, original_records)


def test_distill_resume_optimizer_rng_next_step_and_atomic_save(distill_student, tmp_path, monkeypatch):
    from experiments.phase3 import distill

    model, _, _, _ = distill_student
    args = distill_arguments(tmp_path / "parent.pt")
    optimizers = distill.make_optimizers(distill.parameter_groups(model), args)
    cuda_state = dict(value=torch.tensor([7, 2, 1], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_state["value"].clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: cuda_state.update(value=value.clone()))

    def update(active_optimizers, step):
        for optimizer in active_optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * run.schedule(step, 100, 10)
                for parameter in group["params"]:
                    parameter.grad = torch.randn_like(parameter) * 0.001 + random.random() * 0.001
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    update(optimizers, 0)
    path = tmp_path / "resume.pt"
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    before = cloned_state(model)
    distill.save_resume(model, optimizers, path, 1, args)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert len(saved["parameters"]) == 336 and len(saved["optimizers"]) == 2
    assert all(value.dtype == torch.float32 for value in saved["parameters"].values())
    assert saved["python_rng"] == python_rng and torch.equal(saved["torch_rng"], torch_rng)
    assert torch.equal(saved["cuda_rng"], cuda_state["value"])
    assert not path.with_suffix(".pt.tmp").exists()
    assert_state_unchanged(model, before)
    restored = copy.deepcopy(model)
    restored_groups = distill.parameter_groups(restored)
    identities = {name: id(parameter) for values in restored_groups.values() for name, parameter in values}
    with torch.no_grad():
        for values in restored_groups.values():
            for _, parameter in values:
                parameter.fill_(0.25)
    restored_optimizers = distill.make_optimizers(restored_groups, args)
    update(optimizers, 1)
    cuda_state["value"].zero_()
    assert distill.load_resume(restored, restored_optimizers, path, args) == 1
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.equal(cuda_state["value"], saved["cuda_rng"])
    assert_state_unchanged(restored, before)
    assert identities == {name: id(parameter) for values in distill.parameter_groups(restored).values()
                          for name, parameter in values}
    update(restored_optimizers, 1)
    assert_state_unchanged(restored, cloned_state(model))
    assert_optimizer_states_equal(restored_optimizers, optimizers)
    committed = path.read_bytes()

    def interrupted_save(state, destination):
        Path(destination).write_bytes(b"injected interruption")
        raise OSError("injected interrupted distill save")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupted_save)
        with pytest.raises(OSError, match="injected interrupted distill save"):
            distill.save_resume(model, optimizers, path, 2, args)
    assert path.read_bytes() == committed
    distill.save_resume(model, optimizers, path, 2, args)
    assert not path.with_suffix(".pt.tmp").exists()
    assert torch.load(path, map_location="cpu", weights_only=False)["metadata"]["step"] == 2


@pytest.mark.parametrize("corruption", ["coverage", "weight_shape", "scale_nonpositive", "optimizer_count"])
def test_distill_resume_rejects_malformed_without_mutation(distill_student, tmp_path, monkeypatch, corruption):
    from experiments.phase3 import distill

    model, _, _, _ = distill_student
    args = distill_arguments(tmp_path / "parent.pt")
    groups = distill.parameter_groups(model)
    optimizers = distill.make_optimizers(groups, args)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(3, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: None)
    path = tmp_path / "resume.pt"
    distill.save_resume(model, optimizers, path, 1, args)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    weight_name = groups["W"][0][0]
    if corruption == "coverage":
        del saved["parameters"][weight_name]
    elif corruption == "weight_shape":
        saved["parameters"][weight_name] = torch.ones(1)
    elif corruption == "scale_nonpositive":
        saved["parameters"][groups["SA"][0][0]].zero_()
    else:
        saved["optimizers"] = saved["optimizers"][:1]
    torch.save(saved, path)
    before = cloned_state(model)
    with pytest.raises(ValueError):
        distill.load_resume(model, optimizers, path, args)
    assert_state_unchanged(model, before)


def test_distill_teacher_real_local_from_pretrained_preserves_weights_and_fp32_rope(tmp_path, monkeypatch):
    from experiments.phase3 import distill

    config = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                        tie_word_embeddings=True, attention_dropout=0.0)
    config.head_dim = 4
    config._attn_implementation = "sdpa"
    source = EvaluationModel(config).eval()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "norm" in name:
                parameter.copy_(torch.linspace(0.75, 1.25, parameter.numel()).reshape_as(parameter))
    rope = {name: value.clone() for name, value in source.named_buffers() if name.endswith("inv_freq")}
    source.to(torch.bfloat16)
    source.save_pretrained(tmp_path / "teacher-source")
    expected = cloned_state(source)
    monkeypatch.setattr(distill, "MODEL_PATH", tmp_path / "teacher-source")
    teacher = distill.teacher_model()
    assert not teacher.training and not teacher.config.use_cache and not hasattr(teacher, "R1")
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert not common.wrappers(teacher)
    assert_state_unchanged(teacher, expected)
    assert teacher.lm_head.weight is not teacher.model.embed_tokens.weight
    assert torch.equal(teacher.lm_head.weight, teacher.model.embed_tokens.weight)
    assert rope
    for name, value in rope.items():
        actual = teacher.get_buffer(name)
        assert actual.dtype == torch.float32 and torch.equal(actual, value)


def test_distill_cold_validation_rng_and_optimizer_identity_cpu_mock(distill_student, calibration, tmp_path, monkeypatch):
    from experiments.phase3 import distill

    student, _, records, _ = distill_student
    teacher = distill.teacher_model()
    args = distill_arguments(tmp_path / "parent.pt")
    args.output, args.validation_steps = tmp_path / "evaluation", [1]
    args.output.mkdir()
    optimizers = distill.make_optimizers(distill.parameter_groups(student), args)
    before = cloned_state(student)
    teacher_before = cloned_state(teacher)
    identities = {name: id(parameter) for name, parameter in student.named_parameters()}
    optimizer_ids = [[id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
                     for optimizer in optimizers]
    events = []
    cuda_rng = dict(value=torch.tensor([4, 7, 2], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *args, **kwargs: cuda_rng["value"].clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, *args, **kwargs: cuda_rng.update(value=value.clone()))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty-cache"))
    monkeypatch.setattr(student, "cpu", lambda: (events.append("student-cpu"), student)[1])
    monkeypatch.setattr(teacher, "cpu", lambda: (events.append("teacher-cpu"), teacher)[1])
    monkeypatch.setattr(student, "cuda", lambda: (events.append("student-cuda"), student)[1])
    monkeypatch.setattr(teacher, "cuda", lambda: (events.append("teacher-cuda"), teacher)[1])
    original_export, original_load = distill.export_student, distill.load_static

    def export(*args, **kwargs):
        events.append("export")
        return original_export(*args, **kwargs)

    def load(path):
        assert events == ["export", "student-cpu", "teacher-cpu", "empty-cache"]
        events.append("cold-load")
        torch.rand(7)
        cuda_rng["value"].add_(1)
        return original_load(path, device="cpu")

    validation_marker = object()

    def validation(frozen, windows):
        assert frozen is not student and windows is validation_marker
        assert events[-1] == "cold-load"
        events.append("validation")
        return common.evaluate(frozen, calibration)

    monkeypatch.setattr(distill, "export_student", export)
    monkeypatch.setattr(distill, "load_static", load)
    monkeypatch.setattr(distill, "full_validation", validation)
    monkeypatch.setattr(distill, "progress", lambda *args, **kwargs: None)
    expected_torch_rng, expected_cuda_rng = torch.get_rng_state().clone(), cuda_rng["value"].clone()
    result = distill.evaluate_checkpoint(student, teacher, records, args, 1, calibration, validation_marker)
    assert result["step"] == 1 and math.isfinite(result["validation_nll"])
    assert events[-3:] == ["empty-cache", "student-cuda", "teacher-cuda"]
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)
    assert torch.equal(cuda_rng["value"], expected_cuda_rng)
    assert identities == {name: id(parameter) for name, parameter in student.named_parameters()}
    assert optimizer_ids == [[id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
                             for optimizer in optimizers]
    assert_state_unchanged(student, before)
    assert_state_unchanged(teacher, teacher_before)


def test_distill_train_builds_teacher_before_restoring_rng(distill_student, tmp_path, monkeypatch):
    from experiments.phase3 import distill

    student, parent, records, _ = distill_student
    args = distill_arguments(tmp_path / "parent.pt")
    args.output, args.resume, args.steps = tmp_path / "resumed", tmp_path / "resume.pt", 1
    args.output.mkdir()
    optimizers = distill.make_optimizers(distill.parameter_groups(student), args)
    cuda_rng = dict(value=torch.tensor([5, 1, 9], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_rng["value"].clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: cuda_rng.update(value=value.clone()))
    distill.save_resume(student, optimizers, args.resume, 1, args)
    expected_torch_rng, expected_python_rng = torch.get_rng_state().clone(), random.getstate()
    expected_cuda_rng = cuda_rng["value"].clone()
    events = []
    teacher = copy.deepcopy(parent)
    monkeypatch.setattr(teacher, "cuda", lambda: (events.append("teacher-cuda"), teacher)[1])

    def build_teacher():
        events.append("teacher-build")
        torch.rand(5)
        random.random()
        cuda_rng["value"].add_(1)
        return teacher

    original_restore = distill.load_resume

    def restore(*args, **kwargs):
        assert events == ["teacher-build", "teacher-cuda"]
        events.append("restore")
        return original_restore(*args, **kwargs)

    monkeypatch.setattr(distill, "data_windows", lambda output: (torch.zeros(1, 2), [], [], [], {}))
    monkeypatch.setattr(distill, "load_static", lambda path: (parent, records))
    monkeypatch.setattr(distill, "teacher_model", build_teacher)
    monkeypatch.setattr(distill, "load_resume", restore)
    monkeypatch.setattr(distill, "progress", lambda *args, **kwargs: None)
    distill.train(args)
    assert events == ["teacher-build", "teacher-cuda", "restore"]
    assert torch.equal(torch.get_rng_state(), expected_torch_rng) and random.getstate() == expected_python_rng
    assert torch.equal(cuda_rng["value"], expected_cuda_rng)


def test_distill_adam_weight_coverage_state_update_and_sgd_default(distill_student, tmp_path):
    from experiments.phase3 import distill

    student, _, _, _ = distill_student
    groups = distill.parameter_groups(student)
    args = distill_arguments(tmp_path / "parent.pt")
    default_optimizers = distill.make_optimizers(groups, args)
    explicit_args = copy.copy(args)
    explicit_args.weight_optimizer = "sgd"
    explicit_optimizers = distill.make_optimizers(distill.parameter_groups(copy.deepcopy(student)), explicit_args)
    assert isinstance(default_optimizers[0], torch.optim.SGD)
    assert default_optimizers[0].defaults["momentum"] == 0.9
    assert default_optimizers[0].defaults["foreach"] is False
    assert_optimizer_states_equal(default_optimizers, explicit_optimizers)
    args.weight_optimizer, args.weight_lr = "adam", 2e-5
    optimizers = distill.make_optimizers(groups, args)
    assert all(isinstance(optimizer, torch.optim.Adam) for optimizer in optimizers)
    weight_optimizer, scale_optimizer = optimizers
    assert weight_optimizer.defaults["eps"] == 1e-8
    assert weight_optimizer.defaults["weight_decay"] == 0 and weight_optimizer.defaults["foreach"] is False
    assert len(weight_optimizer.param_groups) == 1 and len(scale_optimizer.param_groups) == 224
    assert weight_optimizer.param_groups[0]["initial_lr"] == weight_optimizer.param_groups[0]["lr"] == 2e-5
    weight_ids = {id(parameter) for _, parameter in groups["W"]}
    optimized = [parameter for optimizer in optimizers for group in optimizer.param_groups for parameter in group["params"]]
    assert len(weight_ids) == 112
    assert {id(parameter) for parameter in weight_optimizer.param_groups[0]["params"]} == weight_ids
    assert len(optimized) == len({id(parameter) for parameter in optimized}) == 336
    assert {id(parameter) for parameter in optimized} == {id(parameter) for parameter in student.parameters()
                                                        if parameter.requires_grad}
    frozen = {name: parameter.detach().clone() for name, parameter in student.named_parameters()
              if not parameter.requires_grad}
    before = {id(parameter): parameter.detach().clone() for parameter in optimized}
    for index, parameter in enumerate(optimized):
        parameter.grad = torch.full_like(parameter, 0.125 if index % 2 else -0.375)
    for optimizer in optimizers:
        optimizer.step()
    assert len(weight_optimizer.state) == 112 and len(scale_optimizer.state) == 224
    for _, parameter in groups["W"]:
        state = weight_optimizer.state[parameter]
        assert parameter.dtype == state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        assert state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape
        assert int(state["step"]) == 1
        torch.testing.assert_close(state["exp_avg"], parameter.grad * 0.1, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(state["exp_avg_sq"], parameter.grad.square() * 0.001, rtol=1e-6, atol=1e-8)
        expected = before[id(parameter)] - args.weight_lr * parameter.grad / (parameter.grad.abs() + 1e-8)
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-8)
        assert not torch.equal(parameter, before[id(parameter)])
    for name, values in groups.items():
        assert all(not torch.equal(parameter, before[id(parameter)]) for _, parameter in values), name
    for name, value in frozen.items():
        assert torch.equal(student.get_parameter(name), value)


def test_distill_teacher_body_offload_loss_grad_state_equivalence(distill_student, calibration, monkeypatch):
    from experiments.phase3 import distill

    student, parent, _, _ = distill_student
    teacher = EvaluationModel(copy.deepcopy(parent.config)).requires_grad_(False).eval()
    for parameter in teacher.parameters():
        parameter.data = parameter.detach().to(torch.bfloat16)
    teacher.lm_head.weight.data = teacher.model.embed_tokens.weight.detach().clone()
    teacher_before, student_before = cloned_state(teacher), cloned_state(student)
    teacher_buffers = {name: value.clone() for name, value in teacher.named_buffers()}
    assert teacher.model.rotary_emb.inv_freq.dtype == torch.float32
    assert teacher.lm_head.weight.data_ptr() != teacher.model.embed_tokens.weight.data_ptr()
    groups = distill.parameter_groups(student)
    targets = {name: parameter for values in groups.values() for name, parameter in values}
    assert len(targets) == 336
    events = []
    ids = calibration[0]
    original_to, original_cpu = teacher.model.to, teacher.model.cpu

    def body_to(device):
        assert not torch.is_grad_enabled() and device == ids.device
        events.append("body-to")
        return original_to(device)

    def body_cpu():
        assert not torch.is_grad_enabled()
        events.append("body-cpu")
        return original_cpu()

    def reject_head_move(*args, **kwargs):
        pytest.fail("Teacher head must not be offloaded with its body")

    def teacher_forward(module, inputs):
        assert not torch.is_grad_enabled()
        events.append("teacher-forward")

    def student_forward(module, inputs):
        assert torch.is_grad_enabled()
        events.append("student-forward")

    def teacher_head(module, inputs):
        assert not torch.is_grad_enabled() and not inputs[0].requires_grad
        assert inputs[0].device == ids.device

    monkeypatch.setattr(teacher.model, "to", body_to)
    monkeypatch.setattr(teacher.model, "cpu", body_cpu)
    monkeypatch.setattr(teacher.lm_head, "to", reject_head_move)
    monkeypatch.setattr(teacher.lm_head, "cpu", reject_head_move)
    handles = [teacher.model.register_forward_pre_hook(teacher_forward),
               student.model.register_forward_pre_hook(student_forward),
               teacher.lm_head.register_forward_pre_hook(teacher_head)]
    student.train()
    baseline = None
    try:
        for offload in (False, True):
            events.clear()
            student.zero_grad(set_to_none=True)
            losses = distill.distillation_loss(student, teacher, ids, chunk_size=3, offload_teacher_body=offload)
            losses[0].backward()
            assert events == (["body-to", "teacher-forward", "body-cpu", "student-forward"]
                              if offload else ["teacher-forward", "student-forward"])
            for name, parameter in targets.items():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            for values in groups.values():
                assert any(torch.count_nonzero(parameter.grad) for _, parameter in values)
            current = (tuple(value.detach().clone() for value in losses),
                       {name: parameter.grad.clone() for name, parameter in targets.items()})
            if baseline is None:
                baseline = current
            else:
                for value, expected in zip(current[0], baseline[0]):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                for name, gradient in current[1].items():
                    torch.testing.assert_close(gradient, baseline[1][name], rtol=0, atol=0, msg=name)
            assert all(parameter.grad is None for parameter in teacher.parameters())
            assert all(parameter.grad is None for parameter in student.parameters() if not parameter.requires_grad)
            assert_state_unchanged(student, student_before)
            assert_state_unchanged(teacher, teacher_before)
            for name, value in teacher_buffers.items():
                actual = teacher.get_buffer(name)
                assert actual.dtype == value.dtype and torch.equal(actual, value)
    finally:
        for handle in handles:
            handle.remove()


def test_distill_resume_optimizer_method_mismatch_and_legacy_default(distill_student, tmp_path, monkeypatch):
    from experiments.phase3 import distill

    student, _, _, _ = distill_student
    cuda_rng = torch.tensor([3, 8, 2], dtype=torch.uint8)
    restored_cuda = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_rng.clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: restored_cuda.append(value.clone()))
    for saved_method, requested_method in (("adam", "sgd"), ("sgd", "adam"), ("legacy-sgd", "adam")):
        source = copy.deepcopy(student)
        saved_args = distill_arguments(tmp_path / "parent.pt")
        if saved_method != "legacy-sgd":
            saved_args.weight_optimizer = saved_method
        saved_optimizers = distill.make_optimizers(distill.parameter_groups(source), saved_args)
        for optimizer in saved_optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, 0.001)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        path = tmp_path / (saved_method + ".pt")
        distill.save_resume(source, saved_optimizers, path, 1, saved_args)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["metadata"]["weight_optimizer"] == ("sgd" if saved_method == "legacy-sgd" else saved_method)
        if saved_method == "legacy-sgd":
            del saved["metadata"]["weight_optimizer"]
            torch.save(saved, path)
        destination = copy.deepcopy(student)
        with torch.no_grad():
            for values in distill.parameter_groups(destination).values():
                for _, parameter in values:
                    parameter.fill_(0.25)
        requested_args = copy.copy(saved_args)
        requested_args.weight_optimizer = requested_method
        requested_optimizers = distill.make_optimizers(distill.parameter_groups(destination), requested_args)
        optimizers_before = copy.deepcopy(requested_optimizers)
        before = cloned_state(destination)
        python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        restored_cuda.clear()
        with pytest.raises(ValueError, match="changes weight optimizer"):
            distill.load_resume(destination, requested_optimizers, path, requested_args)
        assert_state_unchanged(destination, before)
        assert_optimizer_states_equal(requested_optimizers, optimizers_before)
        assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
        assert restored_cuda == []
        if saved_method in ("adam", "legacy-sgd"):
            matching = distill.make_optimizers(distill.parameter_groups(destination), saved_args)
            assert distill.load_resume(destination, matching, path, saved_args) == 1
            assert_state_unchanged(destination, cloned_state(source))
            assert_optimizer_states_equal(matching, saved_optimizers)
            assert len(restored_cuda) == 1 and torch.equal(restored_cuda[0], cuda_rng)


def test_distill_cell_reference_all_codes_saturation_and_no_mutation():
    from experiments.phase3 import distill

    offsets = torch.tensor([-1000.0, -0.51, -0.5, -0.25, 0.0, 0.25, 0.5, 0.51, 1000.0])
    codes = torch.arange(-8, 8, dtype=torch.int8)[:, None].expand(16, offsets.numel()).clone()
    scale = torch.logspace(-4, 2, 16)[:, None]
    record = dict(packed=quantization.pack_int4(codes), shape=tuple(codes.shape), scale=scale)
    reference = (codes.float() + offsets) * scale
    record_before, reference_before = copy.deepcopy(record), reference.clone()
    initialized = distill.cell_reference_initialization(record, reference)
    expected = torch.empty_like(reference)
    for row in range(codes.shape[0]):
        for column in range(codes.shape[1]):
            code = int(codes[row, column])
            value = float(reference[row, column]) / float(scale[row, 0])
            lower = -math.inf if code == -8 else code - 0.49
            upper = math.inf if code == 7 else code + 0.49
            expected[row, column] = min(upper, max(lower, value)) * float(scale[row, 0])
    assert (scale > 0).all() and torch.isfinite(initialized).all()
    torch.testing.assert_close(initialized, expected, rtol=2e-7, atol=1e-10)
    assert initialized[0, 0] / scale[0, 0] < -100
    assert initialized[-1, -1] / scale[-1, 0] > 100
    quantized_codes = (initialized / scale).round().clamp(-8, 7).to(torch.int8)
    assert torch.equal(quantized_codes, codes)
    linear = distill.TrainableQuantLinear(initialized, scale)
    inputs = torch.eye(codes.shape[1], dtype=torch.bfloat16)
    expected_weight = (codes.float() * scale).to(torch.bfloat16)
    assert torch.equal(linear.quantizer.quantize(linear.weight).to(torch.bfloat16), expected_weight)
    assert torch.equal(linear(inputs), torch.nn.functional.linear(inputs, expected_weight))
    for invalid in (torch.zeros(16, 8), torch.zeros(reference.numel()), torch.zeros(1, reference.numel())):
        with pytest.raises(ValueError, match="reference shape"):
            distill.cell_reference_initialization(record, invalid)
    assert_records_unchanged(dict(weight=record), dict(weight=record_before))
    assert torch.equal(reference, reference_before)


def test_distill_master_reference_tiny_initial_and_cold_export_match_parent(distill_student, calibration, tmp_path, monkeypatch):
    from experiments.phase3 import distill, postprocess
    from utils.quant_utils import ActQuantizer, RotationStaticActQuantizer

    default_student, parent, records, _ = distill_student
    student = copy.deepcopy(parent)
    references = {name: (quantization.unpack_int4(record["packed"], record["shape"]).float() + 0.25) * record["scale"]
                  for name, record in records.items()}
    records_before, references_before = copy.deepcopy(records), copy.deepcopy(references)
    parent_before = cloned_state(parent)
    high_precision = {name: (id(parameter), parameter.detach().clone()) for name, parameter in student.named_parameters()
                      if not any(name.startswith(prefix + ".") for prefix in records)}

    def reject_calibration(*args, **kwargs):
        pytest.fail("Master initialization must not recalibrate")

    monkeypatch.setattr(RotationStaticActQuantizer, "begin_calibration", reject_calibration)
    monkeypatch.setattr(ActQuantizer, "find_params", reject_calibration)
    distill.prepare_student(student, records, master_reference=references)
    groups = distill.parameter_groups(student)
    assert {name: len(values) for name, values in groups.items()} == dict(W=112, SA=96, SW=112, SP2=16)
    trainable = {id(parameter) for values in groups.values() for _, parameter in values}
    assert len(trainable) == 336
    assert trainable == {id(parameter) for parameter in student.parameters() if parameter.requires_grad}
    assert not hasattr(student, "R1")
    for name, wrapper in common.wrappers(student).items():
        assert wrapper.weight is wrapper.module.weight and wrapper.module.weight.dtype == torch.float32
        assert wrapper.quantizer.bits == 8
        default_weight = default_student.get_submodule(name).module.weight
        assert torch.equal(default_weight, parent.get_submodule(name).module.weight.float())
        assert not torch.equal(wrapper.module.weight, default_weight)
        assert torch.equal(wrapper.module.quantizer.scale, records[name]["scale"])
        assert torch.equal(wrapper.quantizer.scale, parent.get_submodule(name).quantizer.scale)
        assert torch.equal(wrapper.module.quantizer.quantize(wrapper.module.weight).to(torch.bfloat16),
                           parent.get_submodule(name).module.weight)
    for name, (identity, value) in high_precision.items():
        parameter = student.get_parameter(name)
        assert id(parameter) == identity and not parameter.requires_grad and torch.equal(parameter, value)
    before_export = cloned_state(student)
    path = tmp_path / "cell-reference.pt"
    changes = distill.export_student(student, records, path, dict(test="master-reference"))
    assert len(changes) == 112 and all(row["changed_codes"] == 0 for row in changes.values())
    loaded, loaded_records = postprocess.load_static(path, device="cpu")
    assert_records_unchanged(loaded_records, records)
    with torch.no_grad():
        for ids in calibration:
            expected = parent.lm_head(common.backbone(parent, ids))
            assert torch.equal(student.lm_head(common.backbone(student, ids)), expected)
            assert torch.equal(loaded.lm_head(common.backbone(loaded, ids)), expected)
    assert_state_unchanged(student, before_export)
    assert_state_unchanged(parent, parent_before)
    assert_records_unchanged(records, records_before)
    for name, value in references.items():
        assert torch.equal(value, references_before[name])


def test_distill_reference_for_direct_d_selected_rows_columns_without_disk_mutation(tmp_path, monkeypatch):
    from experiments.phase3 import distill

    up_name, down_name, gate_name = ("model.layers.1.mlp." + suffix for suffix in ("up_proj", "down_proj", "gate_proj"))
    matrices = {
        up_name: torch.tensor([[1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]], dtype=torch.bfloat16),
        down_name: torch.tensor([[0.25, -0.5, 0.75], [-1.0, 1.5, -2.0]], dtype=torch.bfloat16),
        gate_name: torch.arange(6, dtype=torch.bfloat16).reshape(3, 2),
        "model.layers.0.mlp.down_proj": torch.ones(2, 3, dtype=torch.bfloat16),
    }
    matrices_before = copy.deepcopy(matrices)
    reference_state = tmp_path / "reference-state.pt"
    torch.save(dict(parameters=dict(marker=torch.tensor([42]))), reference_state)
    reference_bytes = reference_state.read_bytes()
    calls = []

    def fresh_reference(path):
        assert path == reference_state
        calls.append(path)
        return {name: value.clone() for name, value in matrices.items()}

    monkeypatch.setattr(distill, "reference_weights", fresh_reference)
    selected = dict(channels=[0, 2], factor=2.0, alpha=4.0, range_exponent=-1, heldout_mse=0.1)
    for case, metadata in (
        ("direct-d", dict(mode="d", candidate="local-d-" + down_name)),
        ("range-only", dict(mode="d", candidate="range-only-" + down_name)),
        ("not-d", dict(mode="round", candidate="local-d-" + down_name)),
    ):
        directory = tmp_path / case
        directory.mkdir()
        parent_path = directory / "static_w4a8.pt"
        torch.save(dict(metadata=metadata), parent_path)
        parent_bytes = parent_path.read_bytes()
        diagnostic_path = directory / "diagonal_candidates.json"
        if case == "direct-d":
            diagnostic_path.write_text(json.dumps({down_name: dict(selected=selected),
                "model.layers.0.mlp.down_proj": dict(selected=dict(channels=[1], factor=64.0))}))
            diagnostic_bytes = diagnostic_path.read_bytes()
        actual, transform = distill.reference_for_parent(reference_state, parent_path)
        expected = copy.deepcopy(matrices)
        if case == "direct-d":
            expected[up_name][[0, 2]] /= 2
            expected[down_name][:, [0, 2]] *= 2
            assert transform == dict(name=down_name, **selected)
            assert diagnostic_path.read_bytes() == diagnostic_bytes
        else:
            assert transform is None and not diagnostic_path.exists()
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            assert torch.equal(value, expected[name]), name
            assert torch.equal(matrices[name], matrices_before[name]), name
        assert parent_path.read_bytes() == parent_bytes
        assert reference_state.read_bytes() == reference_bytes
    assert calls == [reference_state] * 3


@pytest.fixture
def local_d_cpu(sequential_cpu, monkeypatch):
    from experiments.phase3 import local_d

    original_dequant = local_d.dequant_record

    def dequant_cpu(record, device):
        assert device == "cuda"
        return original_dequant(record, "cpu")

    monkeypatch.setattr(local_d, "dequant_record", dequant_cpu)
    return local_d


def test_local_d_capture_real_112_wrappers_full_windows_and_cleanup(distill_student, local_d_cpu, monkeypatch):
    local_d = local_d_cpu
    _, parent, records, _ = distill_student
    assert len(common.wrappers(parent)) == len(records) == 112
    before = cloned_state(parent)
    mlp = parent.model.layers[0].mlp
    windows = [(torch.arange(2048).reshape(1, 2048) + index) % 32 for index in range(32)]
    raw_up, expected_gate = [], []

    def observe_up(module, inputs):
        raw_up.append(inputs[0].detach().clone())

    def observe_gate(module, inputs, output):
        expected_gate.append(output.detach().clone())

    handles = [mlp.up_proj.register_forward_pre_hook(observe_up),
               mlp.gate_proj.module.register_forward_hook(observe_gate)]
    try:
        captured = local_d.capture_pair(parent, 0, windows)
    finally:
        for handle in handles:
            handle.remove()
    assert all(len(values) == 32 for values in captured.values())
    assert len(raw_up) == len(expected_gate) == 32
    expected_down = []
    for index, raw in enumerate(raw_up):
        scale = mlp.up_proj.quantizer.scale
        quantized = ((raw.float() / scale).round().clamp(-128, 127) * scale).to(raw.dtype)
        assert raw.shape == captured["up_input"][index].shape == (1, 2048, 8)
        assert torch.equal(captured["up_input"][index], quantized)
        assert torch.equal(captured["gate_output"][index], expected_gate[index])
        down = torch.nn.functional.silu(expected_gate[index]) * torch.nn.functional.linear(quantized, mlp.up_proj.weight)
        assert down.shape == (1, 2048, 16)
        assert torch.equal(captured["down_input"][index], down)
        expected_down.append(oracle_sp2(down, mlp.down_proj.quantizer.alpha, mlp.down_proj.quantizer.levels))
    assert any(not torch.equal(raw, quantized) for raw, quantized in zip(raw_up, captured["up_input"]))
    assert any(not torch.equal(raw, quantized) for raw, quantized in zip(captured["down_input"], expected_down))
    captured_before = copy.deepcopy(captured)
    fit, heldout = local_d.candidate_inputs(captured, records["model.layers.0.mlp.up_proj"],
        float(mlp.down_proj.quantizer.alpha), mlp.down_proj.quantizer.levels)
    expected = torch.cat(expected_down).reshape(-1, 16)
    assert fit.shape == (49152, 16) and heldout.shape == (16384, 16)
    assert torch.equal(fit, expected[:49152]) and torch.equal(heldout, expected[49152:])
    for name in captured:
        assert all(torch.equal(actual, old) for actual, old in zip(captured[name], captured_before[name]))
    for error in (ValueError, RuntimeError):
        def broken(*args, **kwargs):
            raise error("unrelated local-D capture failure")

        monkeypatch.setattr(local_d, "backbone", broken)
        with pytest.raises(error, match="unrelated local-D capture failure"):
            local_d.capture_pair(parent, 0, windows[:1])
        assert not mlp.up_proj.module._forward_pre_hooks
        assert not mlp.gate_proj.module._forward_hooks and not mlp.down_proj._forward_pre_hooks
    monkeypatch.setattr(local_d, "backbone", lambda *args: None)
    with pytest.raises(RuntimeError, match="Incomplete full-window local-D capture"):
        local_d.capture_pair(parent, 0, windows[:1])
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in parent.modules())
    assert_state_unchanged(parent, before)


def test_local_d_sparse_direction_and_bf16_transformed_grid_oracles():
    from experiments.phase3 import local_d

    reference_up = torch.tensor([[0.17, -0.85], [0.03, 0.47], [0.99, -0.13], [-0.34, 0.27]], dtype=torch.bfloat16)
    reference_down = torch.tensor([[0.81, 0.17, 0.26, -0.031], [-0.67, -0.48, 0.14, 0.011]], dtype=torch.bfloat16)
    parent = {
        "up": dict(shape=(4, 2), scale=torch.tensor([[0.1], [0.125], [0.075], [0.2]]),
                   packed=quantization.pack_int4(torch.zeros(4, 2, dtype=torch.int8))),
        "down": dict(shape=(2, 4), scale=torch.tensor([[0.02], [1.0]]),
                     packed=quantization.pack_int4(torch.zeros(2, 4, dtype=torch.int8))),
    }
    before, up_before, down_before = copy.deepcopy(parent), reference_up.clone(), reference_down.clone()
    columns = reference_down.double().abs().amax(dim=0)
    for selected in (1, 3):
        ratio = torch.ones(4, dtype=torch.float64)
        ratio[selected] = math.exp(4)
        channel, direction = local_d.sparse_direction(columns * ratio, reference_down)
        expected = torch.full((4,), -math.log(4) / 4)
        expected[selected] += math.log(4)
        assert channel == selected
        torch.testing.assert_close(direction, expected, rtol=0, atol=1e-7)
        assert torch.equal(direction[torch.arange(4) != channel], direction[0].expand(3))
    for activation, weights in ((torch.zeros(4), torch.zeros(2, 4)), (torch.ones(4), torch.ones(2, 4))):
        channel, constant = local_d.sparse_direction(activation, weights)
        assert channel == 0 and torch.isfinite(constant).all()
        torch.testing.assert_close(constant, torch.zeros_like(constant), rtol=0, atol=1e-12)
    hidden = torch.tensor([[0.17, -0.84], [-0.32, 1.25]], dtype=torch.float64)
    gate = torch.tensor([[0.3, -0.8, 0.2, 1.1], [-0.1, 0.7, 0.8, -0.5]], dtype=torch.float64)
    original = torch.nn.functional.linear(torch.nn.functional.silu(gate) *
        torch.nn.functional.linear(hidden, reference_up.double()), reference_down.double())
    for strength in (0, 0.25, 0.5, 0.75, 1):
        diagonal = (direction * strength).exp()
        assert torch.isfinite(diagonal).all() and (diagonal > 0).all()
        assert float(diagonal.min()) >= 0.25 and float(diagonal.max()) <= 4
        assert abs(float(diagonal.double().log().mean())) < 1e-7
        if strength == 0:
            assert torch.equal(diagonal, torch.ones_like(diagonal))
        paired = torch.nn.functional.linear(torch.nn.functional.silu(gate) *
            torch.nn.functional.linear(hidden, reference_up.double() / diagonal.double()[:, None]),
            reference_down.double() * diagonal.double()[None, :])
        torch.testing.assert_close(paired, original, rtol=1e-12, atol=1e-12)
        actual_up, actual_down, transformed_down = local_d.transformed_records(
            reference_up, reference_down, parent["up"], parent["down"], diagonal)
        expected_up = (reference_up.float() / diagonal[:, None]).to(torch.bfloat16)
        expected_down = (reference_down.float() * diagonal[None, :]).to(torch.bfloat16)
        expected_up_scale = parent["up"]["scale"] / diagonal[:, None]
        expected_down_scale = torch.maximum(parent["down"]["scale"], expected_down.float().abs().amax(1, keepdim=True) / 7)
        assert torch.equal(transformed_down, expected_down) and transformed_down.dtype == torch.bfloat16
        for record, weight, scale in ((actual_up, expected_up, expected_up_scale), (actual_down, expected_down, expected_down_scale)):
            codes = quantization.unpack_int4(record["packed"], record["shape"])
            assert torch.equal(record["scale"], scale)
            assert torch.equal(codes, (weight.float() / scale).round().clamp(-8, 7).to(torch.int8))
            assert codes.min() >= -8 and codes.max() <= 7
        if strength == 0:
            assert not torch.equal(actual_down["scale"], parent["down"]["scale"])
            assert not torch.equal(actual_down["packed"], parent["down"]["packed"])
        assert actual_down["scale"][1] == parent["down"]["scale"][1]
    for invalid in (torch.ones(3), torch.ones(4, 1), torch.tensor([1., 1., 0., 1.]), torch.tensor([1., 1., float("nan"), 1.])):
        with pytest.raises(ValueError, match="D must"):
            local_d.transformed_records(reference_up, reference_down, parent["up"], parent["down"], invalid)
    assert_records_unchanged(parent, before)
    assert torch.equal(reference_up, up_before) and torch.equal(reference_down, down_before)


def test_local_d_search_each_diagonal_independent_inputs_rtn_and_all_nll(postprocess_toy, local_d_cpu, tmp_path, monkeypatch):
    from experiments.phase3 import sequential_postprocess as sequential

    local_d = local_d_cpu
    model, records = postprocess_toy
    for parameter in model.parameters():
        parameter.data = parameter.detach().to(torch.bfloat16)
    assert model.model.layers[0].mlp.down_proj.quantizer.levels.dtype == torch.float32
    up_name, down_name = "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"
    reference = {up_name: torch.tensor([[0.2, 0.7], [-0.3, 0.4], [0.5, -0.8], [-0.1, 0.2]], dtype=torch.bfloat16),
                 down_name: torch.tensor([[0.4, 0.3, 0.01, 0.2], [-0.6, -0.2, 0.01, -0.3]], dtype=torch.bfloat16)}
    before, records_before, reference_before = cloned_state(model), copy.deepcopy(records), copy.deepcopy(reference)
    windows = [torch.full((1, 2048), index, dtype=torch.long) for index in range(32)]
    positions = torch.arange(2048).float()
    captured = dict(up_input=[], gate_output=[], down_input=[])
    for index in range(32):
        captured["up_input"].append(torch.stack((torch.sin(positions / 91) + index / 31, torch.cos(positions / 37)), -1)[None].bfloat16())
        captured["gate_output"].append((torch.sin(positions[:, None] / 51 + torch.arange(4)) + index / 100)[None].bfloat16())
        raw = torch.ones(1, 2048, 4, dtype=torch.bfloat16)
        raw[:, :, 2] = 8 if index < 24 else 1
        raw[:, :, 0] = 1 if index < 24 else 1000
        captured["down_input"].append(raw)
    captures = copy.deepcopy(captured)
    args = SimpleNamespace(layer=0, output=tmp_path, strengths=[0, 0.5, 1], milestones=[512, 2048, 8192])
    alpha = float(model.get_submodule(down_name).quantizer.alpha)
    calls, scored = [], []
    active = dict(index=0)

    def capture(current, layer, calibration):
        assert current is model and layer == 0 and calibration is windows
        return copy.deepcopy(captured)

    def coordinate(weight, scale, fit, heldout, **kwargs):
        index = len(calls)
        active["index"] = index
        channel, direction = local_d.sparse_direction(torch.tensor([1., 1., 8., 1.]), reference[down_name])
        assert channel == 2
        diagonal = (direction * args.strengths[index]).exp()
        up, down, expected_weight = local_d.transformed_records(reference[up_name], reference[down_name], records[up_name], records[down_name], diagonal)
        assert torch.equal(weight, expected_weight) and torch.equal(scale, down["scale"])
        assert kwargs["milestones"] == (512, 2048, 8192) and kwargs["neighbor_search"] is True
        assert "initial_codes" not in kwargs
        up_weight = (quantization.unpack_int4(up["packed"], up["shape"]).float() * up["scale"]).bfloat16()
        expected = torch.cat([oracle_sp2(torch.nn.functional.silu(gate) * torch.nn.functional.linear(inputs, up_weight),
            alpha, model.get_submodule(down_name).quantizer.levels) for inputs, gate in zip(captured["up_input"], captured["gate_output"])]).reshape(-1, 4)
        assert fit.shape == (49152, 4) and heldout.shape == (16384, 4)
        assert torch.equal(fit, expected[:49152]) and torch.equal(heldout, expected[49152:])
        initial = (weight.float() / scale).round().clamp(-8, 7).to(torch.int8)
        trials = []
        for offset, step in enumerate((0, 512, 2048, 8192)):
            codes = initial.clone()
            codes[0, 0] = -4 + offset
            if step == 0:
                codes = initial.clone()
            trials.append(dict(step=step, codes=codes, fit_mse=4.0-offset, heldout_mse=4.0-offset))
        calls.append(dict(up=copy.deepcopy(up), down=copy.deepcopy(down), trials=copy.deepcopy(trials), fit=fit.clone()))
        return trials

    def nll(current, selection):
        assert current is model and all(torch.equal(actual, expected) for actual, expected in zip(selection, windows[24:]))
        assert sum(window.numel() - 1 for window in selection) == 16376
        index, offset = active["index"], len(scored) % 4
        candidate = calls[index]
        expected_weight = (candidate["trials"][offset]["codes"].float() * candidate["down"]["scale"]).bfloat16()
        assert torch.equal(current.get_submodule(down_name).weight, expected_weight)
        expected_up = (quantization.unpack_int4(candidate["up"]["packed"], candidate["up"]["shape"]).float() * candidate["up"]["scale"]).bfloat16()
        assert torch.equal(current.get_submodule(up_name).weight, expected_up)
        assert float(current.get_submodule(down_name).quantizer.alpha) == alpha
        scored.append((index, candidate["trials"][offset]["step"]))
        return [3.1, 2.8, 3.3, 3.5][offset] - index * 0.1

    monkeypatch.setattr(local_d, "capture_pair", capture)
    monkeypatch.setattr(local_d, "rounding_helper", lambda: coordinate)
    monkeypatch.setattr(local_d, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(sequential, "selection_score", nll)
    assert local_d.score_candidate is sequential.score_candidate
    candidates = local_d.search_pair(model, records, reference, windows, windows[24:], 3.0, args)
    assert scored == [(index, step) for index in range(3) for step in (0, 512, 2048, 8192)]
    assert len(candidates) == 3 and all(candidate["step"] == 512 for candidate in candidates)
    assert not torch.equal(calls[0]["fit"], calls[-1]["fit"])
    for index, candidate in enumerate(candidates):
        assert candidate["report"]["channel"] == 2 and candidate["report"]["sp2_alpha"] == alpha
        assert candidate["train_nll"] == 2.8 - index * 0.1
        assert_records_unchanged(candidate["weights"], {up_name: calls[index]["up"],
            down_name: dict(calls[index]["down"], packed=quantization.pack_int4(calls[index]["trials"][1]["codes"]))})
        saved = torch.load(tmp_path / f"candidate-{index:02d}.pt", weights_only=True)
        assert torch.equal(saved["diagonal"], candidate["diagonal"])
    assert_state_unchanged(model, before)
    assert_records_unchanged(records, records_before)
    assert all(torch.equal(value, reference_before[name]) for name, value in reference.items())
    assert all(torch.equal(value, expected) for name in captured for value, expected in zip(captured[name], captures[name]))


@pytest.mark.parametrize("winner", ["reject", "identity", "diagonal"])
def test_local_d_driver_matched_cold_exports_fixed_boundaries_and_provenance(distill_student, local_d_cpu, calibration, tmp_path, monkeypatch, winner):
    from experiments.phase3 import postprocess

    local_d = local_d_cpu
    _, parent, records, _ = distill_student
    metadata = dict(method="joint R/SW/SA PTQ then sequential W4/SP2", original_backbone_trained=False,
                    source_run="B100-prefix", step=100)
    parent_path = tmp_path / "ptq-parent.pt"
    common.save_frozen(parent, records, parent_path, metadata)
    parent_bytes, before = parent_path.read_bytes(), cloned_state(parent)
    parent_package = torch.load(parent_path, weights_only=True)
    args = SimpleNamespace(parent=parent_path, output=tmp_path / winner, reference_state=tmp_path / "B100-R.pt",
                           layer=1, strengths=[0, 1], milestones=[512, 2048, 8192])
    up_name, down_name = "model.layers.1.mlp.up_proj", "model.layers.1.mlp.down_proj"
    reference = {name: parent.get_submodule(name).weight.detach().clone() * 1.125 for name in (up_name, down_name)}
    reference_before = {name: value.clone() for name, value in reference.items()}
    candidate_scores = {"reject": [3.0, 3.1], "identity": [2.8, 3.1], "diagonal": [2.8, 2.5]}[winner]
    candidates = []
    for index, strength in enumerate(args.strengths):
        direction = torch.zeros(16)
        direction[5] = math.log(4)
        direction -= direction.mean()
        diagonal = (strength * direction).exp()
        up, down, _ = local_d.transformed_records(reference[up_name], reference[down_name], records[up_name], records[down_name], diagonal)
        candidates.append(dict(weights={up_name: up, down_name: down}, diagonal=diagonal, train_nll=candidate_scores[index],
            step=512, report=dict(strength=strength, channel=5, layer=1, selected_step=512, selected_train_nll=candidate_scores[index])))
    train = torch.arange(48 * 2048).reshape(48, 2048) % 32
    train[:, 0] = torch.arange(48) % 32
    train[:, 1] = torch.arange(48) // 32
    indices = list(range(31, -1, -1))
    validation_ids = torch.zeros(1, 252852, dtype=torch.long)
    validation = [validation_ids[:, offset:offset + 2048] for offset in range(0, 252852, 2048)]
    loaded, full_calls, searched, references = [], [], [], []

    def load(path, *arguments, **kwargs):
        assert arguments == () and kwargs == {}
        current, current_records = postprocess.load_static(path, device="cpu")
        loaded.append((path, current))
        return current, current_records

    def reference_weights(path):
        assert path == args.reference_state
        references.append(path)
        return reference

    def initial_score(current, selection):
        assert current is loaded[0][1]
        assert_state_unchanged(current, before)
        assert len(selection) == 8 and sum(window.numel() - 1 for window in selection) == 16376
        assert all(torch.equal(actual, train[index:index + 1]) for actual, index in zip(selection, indices[24:]))
        return 3.0

    def search(current, current_records, current_reference, captured_windows, selection, score, arguments):
        assert len(captured_windows) == 32 and all(window.shape == (1, 2048) for window in captured_windows)
        assert all(torch.equal(actual, train[index:index + 1]) for actual, index in zip(captured_windows, indices))
        assert len(selection) == 8 and all(torch.equal(actual, expected) for actual, expected in zip(selection, captured_windows[24:]))
        assert current_reference is reference and score == 3.0 and arguments is args
        assert_records_unchanged(current_records, records)
        searched.append(current)
        return candidates

    def validation_score(current, windows):
        path, loaded_model = loaded[-1]
        assert current is loaded_model and current is not searched[0] and windows is validation
        assert sum(window.numel() - 1 for window in windows) == 252728
        index = 0 if path.parent.name == "identity_rule" else 1
        candidate = candidates[index]
        package = torch.load(path, weights_only=True)
        expected_records = dict(records, **candidate["weights"])
        assert_records_unchanged(package["weights"], expected_records)
        assert len(package["weights"]) == 112 and len(package["activation"]) == 112
        for name, value in package["activation"].items():
            for field, actual in value.items():
                expected = parent_package["activation"][name][field]
                if isinstance(actual, torch.Tensor):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                else:
                    assert actual == expected
        for name, value in package["high_precision"].items():
            torch.testing.assert_close(value, parent_package["high_precision"][name], rtol=0, atol=0, equal_nan=True)
        expected_model = copy.deepcopy(parent)
        postprocess.apply_records(expected_model, candidate["weights"])
        assert_state_unchanged(current, cloned_state(expected_model))
        with torch.no_grad():
            ids = calibration[0]
            assert torch.equal(current.lm_head(common.backbone(current, ids)), expected_model.lm_head(common.backbone(expected_model, ids)))
        saved_metadata = package["metadata"]
        assert saved_metadata["parent"] == str(parent_path) and saved_metadata["parent_metadata"] == metadata
        assert saved_metadata["reference_state"] == str(args.reference_state) and saved_metadata["mode"] == "local-d"
        assert saved_metadata["diagonal"] == dict(layer=1, channel=5, values=candidate["diagonal"].tolist(), strength=index)
        assert saved_metadata["selection"] == candidate["report"]
        full_calls.append(path.parent.name)
        return dict(nll=4.0 + index, ppl=math.exp(4.0 + index), predicted_tokens=252728, token_count=252852)

    monkeypatch.setattr(local_d, "load_static", load)
    monkeypatch.setattr(local_d, "reference_weights", reference_weights)
    monkeypatch.setattr(local_d, "data_windows", lambda: (train, [], [], validation, dict(calibration_window_indices=indices)))
    monkeypatch.setattr(local_d, "selection_score", initial_score)
    monkeypatch.setattr(local_d, "search_pair", search)
    monkeypatch.setattr(local_d, "full_validation", validation_score)
    monkeypatch.setattr(local_d, "source_record", lambda: dict(test="local-D driver CPU"))
    monkeypatch.setattr(local_d, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(local_d.subprocess, "check_output", lambda *args, **kwargs: b"CPU test source boundary")
    local_d.run(args)
    assert references == [args.reference_state] and len(searched) == 1
    result = json.loads((args.output / "result.json").read_text())
    settings = json.loads((args.output / "settings.json").read_text())
    data = json.loads((args.output / "data.json").read_text())
    assert result["parent_metadata"] == settings["parent_metadata"] == metadata
    assert settings["arguments"]["parent"] == str(parent_path) and settings["arguments"]["reference_state"] == str(args.reference_state)
    assert data["fit_window_indices"] == indices[:24] and data["selection_window_indices"] == indices[24:]
    assert (data["calibration_length"], data["fit_rows"], data["heldout_rows"], data["selection_predicted_tokens"]) == (2048, 49152, 16384, 16376)
    if winner == "reject":
        assert result["validation_status"].startswith("NOT TESTED") and not full_calls
        assert len(loaded) == 1 and not list(args.output.glob("*/static_w4a8.pt"))
    else:
        assert result["validation_status"] == "completed"
        assert full_calls == (["identity_rule"] if winner == "identity" else ["identity_rule", "selected_d"])
        assert len(loaded) == len(full_calls) + 1
        for label in full_calls:
            saved_validation = json.loads((args.output / label / "validation.json").read_text())
            assert saved_validation["parent_metadata"] == metadata and saved_validation["parent"] == str(parent_path)
        if winner == "identity":
            assert result["validation"]["selected_d"] == dict(result["validation"]["identity_rule"], reused_identity=True)
            assert not (args.output / "selected_d").exists()
        else:
            assert result["selected"]["strength"] == 1
            assert result["validation"]["selected_d"]["nll"] > result["validation"]["identity_rule"]["nll"]
    assert parent_path.read_bytes() == parent_bytes
    assert_state_unchanged(parent, before)
    assert_state_unchanged(searched[0], before)
    assert_records_unchanged(records, parent_package["weights"])
    assert all(torch.equal(value, reference_before[name]) for name, value in reference.items())


@pytest.fixture
def sequential_cpu(monkeypatch):
    def tensor_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        return tensor

    def module_cuda(module, *args, **kwargs):
        assert all(parameter.device.type == "cpu" for parameter in module.parameters())
        return module

    monkeypatch.setattr(torch.Tensor, "cuda", tensor_cuda)
    monkeypatch.setattr(torch.nn.Module, "cuda", module_cuda)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_sequential_capture_full_quantized_windows_and_exception_cleanup(postprocess_toy, sequential_cpu, monkeypatch):
    from experiments.phase3 import sequential_postprocess as sequential

    model, _ = postprocess_toy
    name = "model.layers.0.mlp.up_proj"
    wrapper = model.get_submodule(name)
    before = cloned_state(model)
    windows = [torch.arange(index * 2048, (index + 1) * 2048).reshape(1, 2048) for index in range(32)]
    visited = []

    def inputs(ids):
        return torch.stack((torch.sin(ids.float() / 101), torch.cos(ids.float() / 79)), dim=-1)

    def backbone(current, ids):
        visited.append(ids.clone())
        current.get_submodule(name)(inputs(ids))
        pytest.fail("Capture should stop before evaluating the inner Linear")

    monkeypatch.setattr(sequential, "backbone", backbone)
    fit, heldout = sequential.capture_module_inputs(model, name, windows)
    raw = torch.cat([inputs(ids).reshape(-1, 2) for ids in windows])
    expected = (raw / wrapper.quantizer.scale).round().clamp(-128, 127) * wrapper.quantizer.scale
    assert fit.shape == (49152, 2) and heldout.shape == (16384, 2)
    assert torch.equal(fit, expected[:49152]) and torch.equal(heldout, expected[49152:])
    assert not torch.equal(torch.cat((fit, heldout)), raw)
    assert len(visited) == 32 and all(torch.equal(actual, original) for actual, original in zip(visited, windows))
    assert not wrapper.module._forward_pre_hooks
    for error_type in (ValueError, RuntimeError):
        def broken(*args, **kwargs):
            raise error_type("unrelated capture failure")

        monkeypatch.setattr(sequential, "backbone", broken)
        with pytest.raises(error_type, match="unrelated capture failure"):
            sequential.capture_module_inputs(model, name, windows[:1])
        assert not wrapper.module._forward_pre_hooks
    monkeypatch.setattr(sequential, "backbone", lambda *args: None)
    with pytest.raises(RuntimeError, match="Incomplete quantized input capture"):
        sequential.capture_module_inputs(model, name, windows[:1])
    assert not wrapper.module._forward_pre_hooks
    assert_state_unchanged(model, before)


def test_sequential_candidate_weight_alpha_exact_rollback_on_failure(postprocess_toy, monkeypatch):
    from experiments.phase3 import sequential_postprocess as sequential
    from experiments.phase3.postprocess import dequant_record

    model, records = postprocess_toy
    weight_name, alpha_name = "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"
    candidate = dict(records[weight_name], packed=quantization.pack_int4(torch.ones(records[weight_name]["shape"], dtype=torch.int8)))
    alpha = float(model.get_submodule(alpha_name).quantizer.alpha) * 1.125
    before, records_before = cloned_state(model), copy.deepcopy(records)
    windows = object()
    for outcome in ("success", "exception", "nonfinite"):
        def evaluate(current, selected_windows):
            assert selected_windows is windows
            assert torch.equal(current.get_submodule(weight_name).module.weight, dequant_record(candidate, "cpu", torch.float32))
            assert torch.equal(current.get_submodule(alpha_name).quantizer.scale, torch.tensor([alpha / 127]))
            if outcome == "exception":
                raise RuntimeError("injected candidate evaluation failure")
            return dict(nll=math.nan if outcome == "nonfinite" else 2.0)

        monkeypatch.setattr(sequential, "full_validation", evaluate)
        if outcome == "success":
            assert sequential.score_candidate(model, windows, {weight_name: candidate}, {alpha_name: alpha}) == 2.0
        else:
            with pytest.raises(RuntimeError, match="injected candidate|Non-finite train selection"):
                sequential.score_candidate(model, windows, {weight_name: candidate}, {alpha_name: alpha})
        assert_state_unchanged(model, before)
        assert_records_unchanged(records, records_before)


def test_sequential_round_nll_prefilter_parent_reuse_and_accepted_capture(postprocess_toy, sequential_cpu, tmp_path, monkeypatch):
    from experiments.phase3 import sequential_postprocess as sequential
    from experiments.phase3.postprocess import dequant_record

    model, records = postprocess_toy
    first, second = "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"
    selection, calibration = object(), object()
    args = SimpleNamespace(output=tmp_path, milestones=[512, 2048, 8192])
    calls, captured, coordinates = [], [], []
    actual_helper = sequential.rounding_helper()
    assert Path(inspect.unwrap(actual_helper).__globals__["__file__"]).resolve() == common.PROJECT_ROOT / "scripts/phase2/fixed_grid_rounding.py"
    active = dict(name=first)
    original_state, original_records = cloned_state(model), copy.deepcopy(records)

    def capture(current, name, windows):
        assert windows is calibration
        active["name"] = name
        captured.append(name)
        if name == second:
            assert int((current.get_submodule(first).weight / records[first]["scale"]).round()[0, 0]) == -2
        width = records[name]["shape"][1]
        return torch.ones(3, width), torch.ones(1, width)

    def coordinate(*arguments, **kwargs):
        inspect.signature(actual_helper).bind(*arguments, **kwargs)
        assert kwargs["milestones"] == (512, 2048, 8192) and kwargs["neighbor_search"] is True
        parent = quantization.unpack_int4(records[active["name"]]["packed"], records[active["name"]]["shape"])
        assert torch.equal(kwargs["initial_codes"], parent)
        kwargs["progress"](16)
        kwargs["progress"](128)
        trials = []
        for step, offset, error in ((0, 0, 10.0), (512, 1, 1.0), (2048, 2, 4.0), (8192, 3, 12.0)):
            codes = parent.clone()
            codes[0, 0] += offset
            trials.append(dict(step=step, codes=codes, heldout_mse=error, fit_mse=error))
        return trials

    def evaluate(current, windows):
        assert windows is selection
        name = active["name"]
        code = int((current.get_submodule(name).weight / records[name]["scale"]).round()[0, 0])
        calls.append((name, code))
        return dict(nll=({-3: 3.5, -2: 2.5}[code] if name == first else 2.75))

    def progress(output, status, **kwargs):
        if status == "sequential-round-coordinate":
            coordinates.append(kwargs["step"])

    monkeypatch.setattr(sequential, "capture_module_inputs", capture)
    monkeypatch.setattr(sequential, "rounding_helper", lambda: coordinate)
    monkeypatch.setattr(sequential, "full_validation", evaluate)
    monkeypatch.setattr(sequential, "progress", progress)
    selected, score, result = sequential.round_module(model, first, records[first], model.get_submodule(first).weight,
                                                       calibration, selection, 3.0, args)
    assert score == 2.5 and result["selected_step"] == 2048
    assert result["trials"][0]["train_nll"] == 3.0 and result["trials"][-1]["train_nll"] is None
    assert torch.equal(model.get_submodule(first).weight, dequant_record(selected, "cpu", torch.float32))
    rejected, final_score, second_result = sequential.round_module(model, second, records[second], model.get_submodule(second).weight,
                                                                   calibration, selection, score, args)
    assert rejected is None and final_score == score and second_result["selected_step"] == 0
    assert calls == [(first, -3), (first, -2), (second, -3), (second, -2)]
    assert captured == [first, second] and coordinates == [128, 128]
    assert torch.equal(model.get_submodule(second).weight, original_state[second + ".module.weight"])
    for name, wrapper in common.wrappers(model).items():
        assert torch.equal(wrapper.quantizer.scale, original_state[name + ".quantizer.scale"])
    assert_records_unchanged(records, original_records)


def test_sequential_selection_actual_bf16_evaluator_and_16376_targets(sequential_cpu, monkeypatch):
    from experiments.phase3 import sequential_postprocess as sequential
    from utils import eval_utils

    config = LlamaConfig(vocab_size=8, hidden_size=4, intermediate_size=8, num_hidden_layers=1,
                        num_attention_heads=1, num_key_value_heads=1, max_position_embeddings=2048,
                        tie_word_embeddings=False, attention_dropout=0.0)
    config.head_dim, config._attn_implementation = 4, "sdpa"
    model = EvaluationModel(config).requires_grad_(False).eval()
    for parameter in model.parameters():
        parameter.data = parameter.detach().to(torch.bfloat16)
    windows = [(torch.arange(2048).reshape(1, -1) + index) % 8 for index in range(8)]
    expected_losses = []
    with torch.no_grad():
        for ids in windows:
            logits = model.lm_head(common.backbone(model, ids))
            losses = torch.nn.functional.cross_entropy(logits[:, :-1].permute(0, 2, 1), ids[:, 1:], reduction="none")
            assert logits.dtype == losses.dtype == torch.bfloat16
            expected_losses.append(losses.float().mean())
    expected = math.log(torch.exp(torch.stack(expected_losses).mean()).item())
    actual_evaluator = eval_utils.evaluator
    calls = []

    def cpu_evaluator(current, encoding, device, arguments):
        assert device == "cuda" and current.seqlen == 2048
        assert torch.equal(encoding.input_ids, torch.cat(windows, dim=1))
        assert arguments.bsz == 1 and arguments.eval_nsamples is None
        calls.append(encoding.input_ids.numel() // current.seqlen * (current.seqlen - 1))
        return actual_evaluator(current, encoding, "cpu", arguments)

    monkeypatch.setattr(eval_utils, "evaluator", cpu_evaluator)
    assert sequential.full_validation is common.full_validation
    assert sequential.selection_score(model, windows) == expected
    assert calls == [16376]


@pytest.mark.parametrize("mode", ["round", "sp2"])
def test_sequential_driver_prefix_resume_data_cold_validation_and_provenance(distill_student, sequential_cpu, tmp_path, monkeypatch, mode):
    from experiments.phase3 import sequential_postprocess as sequential

    _, parent, records, _ = distill_student
    parent_metadata = dict(method="quantization-aware distillation" if mode == "sp2" else "joint R/SW/SA PTQ",
                           source_run="retained-parent", step=100, original_backbone_trained=mode == "sp2")
    parent_path = tmp_path / "parent.pt"
    common.save_frozen(parent, records, parent_path, parent_metadata)
    parent_bytes, original_records = parent_path.read_bytes(), copy.deepcopy(records)
    targets = ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"]
    args = SimpleNamespace(output=tmp_path / "first", parent=parent_path, reference_state=tmp_path / "reference.pt" if mode == "round" else None,
                           resume_prefix=None, mode=mode, targets=targets, family="down", milestones=[512, 2048], alpha_factors=[1, 1.25, 2])
    train = torch.arange(48 * 2048).reshape(48, 2048) % 32
    train[:, 0] = torch.arange(48) % 32
    train[:, 1] = torch.arange(48) // 32
    indices = list(range(31, -1, -1))
    validation_tokens = torch.zeros(1, 252852, dtype=torch.long)
    validation = [validation_tokens[:, offset:offset + 2048] for offset in range(0, validation_tokens.numel(), 2048)]
    selection_expected = [train[index:index + 1] for index in indices[24:]]
    controls = dict(interrupt=True, reject=False)
    evaluated, attempted, captures, loaded, full_calls = [], [], [], [], []
    references = {name: parent.get_submodule(name).weight.detach().clone() for name in records}
    parent_alphas = {name: float(parent.get_submodule(name).quantizer.alpha) for name in targets}
    originals = {name: quantization.unpack_int4(records[name]["packed"], records[name]["shape"]) for name in targets}
    original_load = sequential.load_static
    original_to = torch.nn.Module.to
    requested_devices, rope_snapshots = [], {}
    assert inspect.signature(original_load).parameters["device"].default == "cuda"
    active = dict(name=None)

    def read_model(model_class, *arguments, **kwargs):
        configuration = kwargs["config"]
        configuration._attn_implementation = kwargs["attn_implementation"]
        model = model_class(configuration)
        for parameter in model.parameters():
            parameter.data = parameter.detach().to(dtype=kwargs["torch_dtype"])
        rope_snapshots[id(model)] = {name: value.clone() for name, value in model.named_buffers()
                                     if name.endswith("inv_freq")}
        assert rope_snapshots[id(model)] and all(value.dtype == torch.float32 for value in rope_snapshots[id(model)].values())
        return model

    def cpu_device_boundary(module, *arguments, **kwargs):
        if arguments and isinstance(arguments[0], (str, torch.device)) and torch.device(arguments[0]).type == "cuda":
            assert arguments == ("cuda",) and kwargs == {}
            requested_devices.append("cuda")
            return original_to(module, "cpu")
        return original_to(module, *arguments, **kwargs)

    def load(path, *arguments, **kwargs):
        assert arguments == () and kwargs == {}
        model, state_records = original_load(path)
        for name, expected in rope_snapshots[id(model)].items():
            actual = model.get_buffer(name)
            assert actual.dtype == torch.float32 and torch.equal(actual, expected)
        loaded.append((path, model))
        return model, state_records

    def code_or_scale(current, name):
        if mode == "round":
            code = int((current.get_submodule(name).weight / records[name]["scale"]).round()[0, 0])
            return code - int(originals[name][0, 0])
        ratio = float(current.get_submodule(name).quantizer.alpha) / parent_alphas[name]
        return min(range(3), key=lambda index: abs(ratio - [1, 1.25, 2][index]))

    def validation_score(current, windows):
        assert current.model.rotary_emb.inv_freq.dtype == torch.float32
        if windows is validation:
            assert loaded[-1][0].name == "static_w4a8.pt" and current is loaded[-1][1]
            assert sum(window.numel() - 1 for window in windows) == 252728
            assert [code_or_scale(current, name) for name in targets] == [1, 1]
            full_calls.append(current)
            return dict(nll=4.0, ppl=math.exp(4), token_count=252852, predicted_tokens=252728)
        assert len(windows) == 8 and all(torch.equal(actual, expected) for actual, expected in zip(windows, selection_expected))
        assert all(window.shape == (1, 2048) for window in windows)
        assert sum(window.numel() - 1 for window in windows) == 16376
        state = tuple(code_or_scale(current, name) for name in targets)
        evaluated.append(state)
        if controls["reject"]:
            return dict(nll=3.0 if state == (0, 0) else 3.5)
        return dict(nll={(0, 0): 3.0, (1, 0): 2.7, (2, 0): 3.5, (1, 1): 2.5, (1, 2): 2.8}[state])

    def before_module(current, name):
        attempted.append(name)
        active["name"] = name
        if name == targets[1] and not controls["reject"]:
            assert code_or_scale(current, targets[0]) == 1
            if controls["interrupt"]:
                controls["interrupt"] = False
                raise RuntimeError("injected interruption after saved prefix")

    def capture(current, name, windows):
        before_module(current, name)
        assert len(windows) == 32 and all(window.shape == (1, 2048) for window in windows)
        assert all(torch.equal(window, train[index:index + 1]) for window, index in zip(windows, indices))
        captures.append(name)
        width = records[name]["shape"][1]
        return torch.zeros(3, width), torch.zeros(1, width)

    def coordinate(reference, scale, fit, heldout, **kwargs):
        name = active["name"]
        assert torch.equal(scale, records[name]["scale"]) and torch.equal(kwargs["initial_codes"], originals[name])
        trials = []
        for step, offset in ((0, 0), (512, 1), (2048, 2)):
            codes = originals[name].clone()
            codes[0, 0] += offset
            assert ((codes >= -8) & (codes <= 7)).all()
            trials.append(dict(step=step, codes=codes, fit_mse=3.0 - offset, heldout_mse=3.0 - offset))
        return trials

    original_range = sequential.range_module

    def range_search(current, name, *args):
        before_module(current, name)
        return original_range(current, name, *args)

    def reference_weights(path):
        assert mode == "round" and path == args.reference_state
        return references

    monkeypatch.setattr(sequential, "data_windows", lambda: (train, [torch.zeros(1, 128)], [], validation,
                                                              dict(calibration_window_indices=indices.copy())))
    monkeypatch.setattr(EvaluationModel, "from_pretrained", classmethod(read_model))
    monkeypatch.setattr(torch.nn.Module, "to", cpu_device_boundary)
    monkeypatch.setattr(sequential, "load_static", load)
    monkeypatch.setattr(sequential, "reference_weights", reference_weights)
    monkeypatch.setattr(sequential, "capture_module_inputs", capture)
    monkeypatch.setattr(sequential, "rounding_helper", lambda: coordinate)
    monkeypatch.setattr(sequential, "range_module", range_search)
    monkeypatch.setattr(sequential, "full_validation", validation_score)
    monkeypatch.setattr(sequential, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(sequential, "source_record", lambda: {})
    monkeypatch.setattr(sequential.shutil, "copyfile", lambda *args, **kwargs: None)
    monkeypatch.setattr(sequential.subprocess, "check_output", lambda *args, **kwargs: b"")
    with pytest.raises(RuntimeError, match="after saved prefix"):
        sequential.run(args)
    prefix_path = args.output / "prefix.pt"
    prefix_bytes = prefix_path.read_bytes()
    prefix = torch.load(prefix_path, map_location="cpu", weights_only=False)
    assert prefix["completed_targets"] == targets[:1] and prefix["targets"] == targets
    assert prefix["initial_train_nll"] == 3.0 and prefix["selected_train_nll"] == 2.7
    assert set(prefix["weights"]) == ({targets[0]} if mode == "round" else set())
    assert len(prefix["activation_scales"]) == 16
    assert "high_precision" not in prefix and "parameters" not in prefix
    assert not prefix_path.with_suffix(".pt.tmp").exists()
    assert evaluated == [(0, 0), (1, 0), (2, 0)] and not full_calls
    assert not (args.output / "static_w4a8.pt").exists()
    resumed_args = copy.copy(args)
    resumed_args.output, resumed_args.resume_prefix = tmp_path / "resumed", prefix_path
    sequential.run(resumed_args)
    assert attempted == [targets[0], targets[1], targets[1]]
    assert evaluated == [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)]
    assert captures == (targets if mode == "round" else [])
    assert len(full_calls) == 1 and len(loaded) == 3
    assert requested_devices == ["cuda"] * 3
    assert loaded[0][0] == loaded[1][0] == parent_path and loaded[2][0] == resumed_args.output / "static_w4a8.pt"
    result = json.loads((resumed_args.output / "result.json").read_text())
    assert result["selected_train_nll"] == 2.5 and result["completed_targets"] == targets
    assert len(result["new_module_results"]) == 1 and result["new_module_results"][0]["module"] == targets[1]
    assert result["validation"]["nll"] == 4.0
    for filename in ("settings.json", "result.json", "validation.json"):
        document = json.loads((resumed_args.output / filename).read_text())
        assert document["parent_metadata"] == parent_metadata
    package = torch.load(resumed_args.output / "static_w4a8.pt", map_location="cpu", weights_only=True)
    assert package["metadata"]["parent_metadata"] == parent_metadata
    assert "no gradient training in this stage" in result["validation"]["method"]
    data = json.loads((resumed_args.output / "data.json").read_text())
    assert data["fit_window_indices"] == indices[:24] and data["selection_window_indices"] == indices[24:]
    assert (data["calibration_length"], data["fit_rows"], data["heldout_rows"], data["selection_predicted_tokens"]) == (2048, 49152, 16384, 16376)
    for name, record in package["weights"].items():
        assert torch.equal(record["scale"], original_records[name]["scale"])
        if mode == "sp2" or name not in targets:
            assert torch.equal(record["packed"], original_records[name]["packed"])
    for name, wrapper in common.wrappers(parent).items():
        if not name.endswith("down_proj"):
            assert torch.equal(package["activation"][name]["scale"], wrapper.quantizer.scale)
    for name, value in package["high_precision"].items():
        torch.testing.assert_close(value, parent.state_dict()[name], rtol=0, atol=0, equal_nan=True)
    assert parent_path.read_bytes() == parent_bytes and prefix_path.read_bytes() == prefix_bytes
    broken = copy.deepcopy(prefix)
    broken["completed_targets"] = targets[1:]
    broken_path = tmp_path / "out-of-order.pt"
    torch.save(broken, broken_path)
    invalid_args = copy.copy(resumed_args)
    invalid_args.resume_prefix = broken_path
    check_model, check_records = copy.deepcopy(parent), copy.deepcopy(records)
    before = cloned_state(check_model)
    with pytest.raises(ValueError, match="completed target prefix"):
        sequential.restore_prefix(check_model, check_records, targets, invalid_args)
    assert_state_unchanged(check_model, before)
    assert_records_unchanged(check_records, records)
    if mode == "sp2":
        controls["reject"] = True
        rejected_args = copy.copy(args)
        rejected_args.output = tmp_path / "rejected"
        sequential.run(rejected_args)
        rejected = json.loads((rejected_args.output / "result.json").read_text())
        assert rejected["selected_train_nll"] == rejected["initial_train_nll"] == 3.0
        assert rejected["validation_status"].startswith("NOT TESTED")
        assert not (rejected_args.output / "static_w4a8.pt").exists() and len(full_calls) == 1
        assert parent_path.read_bytes() == parent_bytes
