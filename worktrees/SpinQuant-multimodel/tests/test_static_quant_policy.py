import ast
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.process_args import parser_gen
from utils.quant_utils import ActQuantWrapper
from utils.static_quant_policy import (
    configure_input_activation_quantizers,
    load_rotation_static_activation_scales,
    validate_static_activation_args,
    validate_static_activation_model_config,
    validate_static_activation_request,
)


TARGET_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class FakeDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.mlp = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, _wrapper())
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp, name, _wrapper())


class FakeModel(torch.nn.Module):
    def __init__(self, num_hidden_layers=2):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            FakeDecoderLayer() for _ in range(num_hidden_layers)
        )
        self.lm_head = _wrapper()
        self.config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            pretraining_tp=1,
        )


def _wrapper():
    return ActQuantWrapper(torch.nn.Linear(4, 4, bias=False))


def _args(**overrides):
    values = {
        "a_quant_mode": "dual_phase_static",
        "a_static_encoding": "symmetric_int8",
        "a_bits": 8,
        "a_groupsize": -1,
        "a_asym": False,
        "a_clip_ratio": 0.9,
        "int8_down_proj": False,
        "act_order": False,
        "k_bits": 16,
        "v_bits": 16,
        "save_qmodel_path": None,
        "export_to_et": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _target_wrappers(model):
    wrappers = []
    for layer in model.model.layers:
        for path in TARGET_PATHS:
            value = layer
            for component in path.split("."):
                value = getattr(value, component)
            wrappers.append(value)
    return wrappers


def test_cli_defaults_preserve_legacy_behavior(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spinquant"])
    args, unknown = parser_gen()

    assert unknown == []
    assert args.a_quant_mode == "legacy_dynamic"
    assert args.a_static_encoding is None


@pytest.mark.parametrize("encoding", ["symmetric_int8", "asymmetric_uint8"])
def test_cli_accepts_static_encodings(monkeypatch, encoding):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spinquant",
            "--a_quant_mode",
            "dual_phase_static",
            "--a_static_encoding",
            encoding,
        ],
    )
    args, unknown = parser_gen()

    assert unknown == []
    assert args.a_quant_mode == "dual_phase_static"
    assert args.a_static_encoding == encoding


def test_cli_rejects_unknown_static_encoding(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["spinquant", "--a_static_encoding", "signed_int8"]
    )
    with pytest.raises(SystemExit):
        parser_gen()


def test_static_encoding_is_conditionally_required():
    with pytest.raises(ValueError, match="explicit a_static_encoding"):
        validate_static_activation_request(
            _args(a_static_encoding=None), FakeModel().config
        )

    validate_static_activation_request(
        _args(a_quant_mode="legacy_dynamic", a_static_encoding=None),
        FakeModel().config,
    )


def test_full_request_validation_composes_args_and_model_config_stages():
    with pytest.raises(ValueError, match="a_bits"):
        validate_static_activation_request(_args(a_bits=4), FakeModel().config)

    config = FakeModel().config
    config.pretraining_tp = 2
    with pytest.raises(ValueError, match="pretraining_tp"):
        validate_static_activation_request(_args(), config)


def test_static_policy_configures_exact_seven_per_layer_and_excludes_lm_head(
    monkeypatch,
):
    model = FakeModel(num_hidden_layers=2)
    static_calls = []
    configure_static = type(model.lm_head.quantizer).configure_static

    def forbidden_legacy_configure(*args, **kwargs):
        raise AssertionError("static policy called legacy configure")

    def recording_static_configure(quantizer, *args, **kwargs):
        static_calls.append((quantizer, args, kwargs))
        return configure_static(quantizer, *args, **kwargs)

    monkeypatch.setattr(
        "utils.quant_utils.ActQuantizer.configure", forbidden_legacy_configure
    )
    monkeypatch.setattr(
        "utils.quant_utils.ActQuantizer.configure_static",
        recording_static_configure,
    )
    names = configure_input_activation_quantizers(_args(), model)

    assert len(names) == 14
    assert set(names) == {
        f"model.layers.{layer_index}.{path}"
        for layer_index in range(2)
        for path in TARGET_PATHS
    }
    assert len(static_calls) == 14
    assert {id(call[0]) for call in static_calls} == {
        id(wrapper.quantizer) for wrapper in _target_wrappers(model)
    }
    assert all(call[1] == () for call in static_calls)
    assert all(
        call[2] == {"encoding": "symmetric_int8", "clip_ratio": 0.9}
        for call in static_calls
    )
    assert all(wrapper.quantizer.static_enabled for wrapper in _target_wrappers(model))
    assert all(
        wrapper.quantizer.static_encoding == "symmetric_int8"
        for wrapper in _target_wrappers(model)
    )
    assert model.lm_head.quantizer.static_enabled is False
    assert model.lm_head.out_quantizer.static_enabled is False
    assert all(
        wrapper.out_quantizer.static_enabled is False
        for wrapper in _target_wrappers(model)
    )


def test_static_policy_rejects_missing_extra_and_duplicate_targets():
    missing = FakeModel(num_hidden_layers=1)
    missing.model.layers[0].self_attn.q_proj = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="missing or unwrapped"):
        configure_input_activation_quantizers(_args(), missing)

    extra = FakeModel(num_hidden_layers=1)
    extra.extra = torch.nn.Module()
    extra.extra.q_proj = _wrapper()
    with pytest.raises(RuntimeError, match="missing, extra, or duplicated"):
        configure_input_activation_quantizers(_args(), extra)

    duplicate = FakeModel(num_hidden_layers=1)
    duplicate.model.layers[0].self_attn.k_proj = (
        duplicate.model.layers[0].self_attn.q_proj
    )
    with pytest.raises(RuntimeError, match="duplicated"):
        configure_input_activation_quantizers(_args(), duplicate)


def test_rotation_static_loader_does_not_require_down_proj_scale_when_a16():
    model = FakeModel(num_hidden_layers=1)
    configure_input_activation_quantizers(
        _args(a_quant_mode="rotation_static"), model
    )
    scales = {
        f"model.layers.0.{path}.quantizer": torch.tensor([0.25])
        for path in TARGET_PATHS
        if path != "mlp.down_proj"
    }

    load_rotation_static_activation_scales(
        model, scales, down_proj_fp16=True
    )

    assert model.model.layers[0].mlp.down_proj.quantizer.bits == 16
    assert not model.model.layers[0].mlp.down_proj.quantizer.has_scale
    assert all(
        wrapper.quantizer.has_scale
        for wrapper in _target_wrappers(model)
        if wrapper is not model.model.layers[0].mlp.down_proj
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"a_quant_mode": "unsupported"}, "a_quant_mode"),
        ({"a_static_encoding": None}, "explicit a_static_encoding"),
        ({"a_bits": 4}, "a_bits"),
        ({"a_groupsize": 32}, "per_tensor"),
        ({"a_clip_ratio": 0.0}, "a_clip_ratio"),
        ({"int8_down_proj": True}, "int8_down_proj"),
        ({"act_order": True}, "act_order"),
        ({"k_bits": 8}, "k_bits"),
        ({"v_bits": 8}, "v_bits"),
        ({"save_qmodel_path": "partial.pt"}, "save_qmodel_path"),
        ({"export_to_et": True}, "export_to_et"),
    ],
)
def test_static_policy_rejects_precision_kv_and_export_escapes(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_static_activation_args(_args(**overrides))


def test_static_policy_rejects_pretraining_tensor_parallelism():
    model = FakeModel()
    model.config.pretraining_tp = 2

    with pytest.raises(ValueError, match="pretraining_tp"):
        validate_static_activation_model_config(_args(), model.config)

    # The model-config stage deliberately does not duplicate args-only rules.
    validate_static_activation_model_config(_args(a_bits=4), FakeModel().config)


def test_legacy_validation_does_not_add_static_gates():
    config = FakeModel().config
    config.pretraining_tp = 4

    args = _args(
        a_quant_mode="legacy_dynamic",
        a_static_encoding=None,
        a_bits=4,
        a_groupsize=32,
        a_clip_ratio=0.0,
        int8_down_proj=True,
        act_order=True,
        k_bits=4,
        v_bits=4,
        save_qmodel_path="legacy.pt",
        export_to_et=True,
    )
    validate_static_activation_args(args)
    validate_static_activation_model_config(args, config)
    validate_static_activation_request(args, config)


def test_legacy_policy_preserves_existing_input_mapping(monkeypatch):
    model = FakeModel(num_hidden_layers=1)
    down_proj_calls = []

    def fake_down_proj_groupsize(received_model, groupsize):
        down_proj_calls.append((received_model, groupsize))
        return 10

    monkeypatch.setattr(
        "utils.static_quant_policy.utils.llama_down_proj_groupsize",
        fake_down_proj_groupsize,
    )
    args = _args(
        a_quant_mode="legacy_dynamic",
        a_static_encoding=None,
        a_bits=6,
        a_groupsize=4,
        a_asym=True,
        a_clip_ratio=0.8,
        int8_down_proj=True,
    )

    names = configure_input_activation_quantizers(args, model)

    assert len(names) == 8
    assert down_proj_calls == [(model, 4)]
    layer = model.model.layers[0]
    assert layer.self_attn.q_proj.quantizer.bits == 6
    assert layer.self_attn.q_proj.quantizer.groupsize == 4
    assert layer.self_attn.q_proj.quantizer.sym is False
    assert layer.self_attn.q_proj.quantizer.clip_ratio == 0.8
    assert layer.self_attn.o_proj.quantizer.groupsize == 2
    assert layer.mlp.down_proj.quantizer.bits == 8
    assert layer.mlp.down_proj.quantizer.groupsize == 10
    assert model.lm_head.quantizer.bits == 16
    assert all(
        not wrapper.quantizer.static_enabled
        for wrapper in _target_wrappers(model) + [model.lm_head]
    )


def test_train_and_eval_entries_use_the_same_public_policy_api():
    repository_root = Path(__file__).resolve().parents[1]
    validation_call = "static_quant_policy.validate_static_activation_request"
    configuration_call = (
        "static_quant_policy.configure_input_activation_quantizers"
    )
    expected_calls = {
        validation_call,
        configuration_call,
    }

    for relative_path, function_name in (
        ("eval_utils/main.py", "ptq_model"),
        ("train_utils/main.py", "prepare_model"),
    ):
        source = (repository_root / relative_path).read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        calls = [
            (ast.unparse(node.func), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        call_names = {name for name, _ in calls}
        assert expected_calls <= call_names
        assert isinstance(function.body[0], ast.Expr)
        assert isinstance(function.body[0].value, ast.Call)
        assert ast.unparse(function.body[0].value.func) == validation_call
        add_actquant_lines = [
            lineno for name, lineno in calls if name == "quant_utils.add_actquant"
        ]
        configuration_lines = [
            lineno for name, lineno in calls if name == configuration_call
        ]
        assert add_actquant_lines
        assert len(configuration_lines) == 1
        assert max(add_actquant_lines) < configuration_lines[0]
        assert ".quantizer.configure(" not in source


@pytest.mark.parametrize(
    ("relative_path", "rank_call", "runtime_calls"),
    [
        (
            "ptq.py",
            "utils.get_local_rank",
            (
                "LlamaForCausalLM.from_pretrained",
                "model.cuda",
                "ptq_model",
                "LlamaTokenizerFast.from_pretrained",
                "data_utils.get_wikitext2",
            ),
        ),
        (
            "optimize_rotation.py",
            "get_local_rank",
            (
                "LlamaForCausalLMQuant.from_pretrained",
                "prepare_model",
                "random_hadamard_matrix",
                "RotateModule",
                "LlamaTokenizerFast.from_pretrained",
                "datasets.load_dataset",
                "CustomJsonDataset",
            ),
        ),
    ],
)
def test_top_level_static_validation_precedes_runtime_side_effects(
    relative_path, rank_call, runtime_calls
):
    repository_root = Path(__file__).resolve().parents[1]
    source = (repository_root / relative_path).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    calls = [
        (ast.unparse(node.func), node.lineno)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]

    def direct_statement_call(statement):
        value = statement.value
        assert isinstance(value, ast.Call)
        return ast.unparse(value.func)

    assert [direct_statement_call(statement) for statement in function.body[:4]] == [
        "process_args_ptq",
        "static_quant_policy.validate_static_activation_args",
        "transformers.AutoConfig.from_pretrained",
        "static_quant_policy.validate_static_activation_model_config",
    ]

    def only_call_line(call_name):
        lines = [lineno for name, lineno in calls if name == call_name]
        assert len(lines) == 1, f"expected one {call_name} call in {relative_path}"
        return lines[0]

    process_args_line = only_call_line("process_args_ptq")
    args_validation_line = only_call_line(
        "static_quant_policy.validate_static_activation_args"
    )
    config_load_line = only_call_line("transformers.AutoConfig.from_pretrained")
    config_validation_line = only_call_line(
        "static_quant_policy.validate_static_activation_model_config"
    )
    distributed_init_line = only_call_line("dist.init_process_group")
    rank_line = only_call_line(rank_call)

    assert (
        process_args_line
        < args_validation_line
        < config_load_line
        < config_validation_line
        < distributed_init_line
        < rank_line
    )

    barrier_lines = [
        lineno
        for name, lineno in calls
        if name in {"torch.distributed.barrier", "dist.barrier"}
    ]
    assert barrier_lines
    assert rank_line < min(barrier_lines)

    for runtime_call in runtime_calls:
        runtime_lines = [
            lineno for name, lineno in calls if name == runtime_call
        ]
        assert runtime_lines, f"missing {runtime_call} in {relative_path}"
        assert all(distributed_init_line < lineno for lineno in runtime_lines)
