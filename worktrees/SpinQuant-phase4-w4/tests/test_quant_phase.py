import ast
import importlib
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig
from transformers.cache_utils import StaticCache

import utils.quant_phase as quant_phase_module
from utils.quant_phase import (
    QuantPhase,
    quant_phase_context,
    require_quant_phase,
    resolve_quant_phase,
)


class FakeCache:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.get_seq_length_calls = 0

    def get_seq_length(self):
        self.get_seq_length_calls += 1
        return self.sequence_length


class FakeStaticCache(FakeCache):
    def get_seq_length(self):
        self.get_seq_length_calls += 1
        raise AssertionError("StaticCache.get_seq_length must not be called")


@pytest.fixture
def fake_cache_types(monkeypatch):
    monkeypatch.setattr(quant_phase_module, "Cache", FakeCache)
    monkeypatch.setattr(quant_phase_module, "StaticCache", FakeStaticCache)


def test_none_and_empty_dynamic_cache_resolve_prefill(fake_cache_types):
    assert resolve_quant_phase(None, torch.tensor([0, 1])) is QuantPhase.PREFILL

    cache = FakeCache(0)
    assert resolve_quant_phase(cache, torch.tensor([0, 1])) is QuantPhase.PREFILL
    assert cache.get_seq_length_calls == 1


def test_nonempty_dynamic_cache_resolves_decode(fake_cache_types):
    cache = FakeCache(4)
    assert resolve_quant_phase(cache, torch.tensor([4])) is QuantPhase.DECODE
    assert cache.get_seq_length_calls == 1


@pytest.mark.parametrize(
    ("cache", "cache_position"),
    [
        (None, torch.tensor([1])),
        (FakeCache(0), torch.tensor([1])),
        (FakeCache(3), torch.tensor([0])),
        (FakeCache(3), torch.tensor([4])),
    ],
)
def test_dynamic_cache_and_position_mismatch_fails_closed(
    fake_cache_types, cache, cache_position
):
    with pytest.raises(ValueError, match="does not match"):
        resolve_quant_phase(cache, cache_position)


@pytest.mark.parametrize(
    "cache_position",
    [
        None,
        [0, 1],
        torch.tensor([]),
        torch.tensor(0),
        torch.tensor([[0, 1]]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1, 0]),
        torch.tensor([0, 2]),
        torch.tensor([2, 1]),
    ],
)
def test_invalid_cache_position_fails_closed(cache_position):
    with pytest.raises((TypeError, ValueError)):
        resolve_quant_phase(None, cache_position)


@pytest.mark.parametrize("sequence_length", [-1, True, 1.5, torch.tensor([1, 2])])
def test_invalid_dynamic_cache_length_fails_closed(
    fake_cache_types, sequence_length
):
    with pytest.raises((TypeError, ValueError)):
        resolve_quant_phase(FakeCache(sequence_length), torch.tensor([0]))


@pytest.mark.parametrize(
    ("cache_position", "expected"),
    [
        (torch.tensor([0, 1, 2]), QuantPhase.PREFILL),
        (torch.tensor([7, 8]), QuantPhase.DECODE),
    ],
)
def test_static_cache_uses_position_without_occupancy_reduction(
    fake_cache_types, cache_position, expected
):
    cache = FakeStaticCache(999)
    assert resolve_quant_phase(cache, cache_position) is expected
    assert cache.get_seq_length_calls == 0


def test_missing_and_nested_phase_contexts():
    with pytest.raises(RuntimeError, match="No quantization phase"):
        require_quant_phase()

    with quant_phase_context(QuantPhase.PREFILL):
        assert require_quant_phase() is QuantPhase.PREFILL
        with quant_phase_context(QuantPhase.DECODE):
            assert require_quant_phase() is QuantPhase.DECODE
        assert require_quant_phase() is QuantPhase.PREFILL

    with pytest.raises(RuntimeError, match="No quantization phase"):
        require_quant_phase()


@pytest.mark.parametrize("phase", [None, "prefill", True])
def test_invalid_phase_context_fails_closed(phase):
    with pytest.raises(TypeError, match="explicit QuantPhase"):
        with quant_phase_context(phase):
            pass


MODEL_PATHS = (
    "eval_utils/modeling_llama.py",
    "train_utils/modeling_llama_quant.py",
)


def _tiny_config():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=8,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    return config


def _find_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"Missing {class_name}.{method_name}")


def _is_named_call(call, name):
    return isinstance(call, ast.Call) and (
        isinstance(call.func, ast.Name) and call.func.id == name
    )


def _is_missing_phase_check(test):
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "quant_phase"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


@pytest.mark.parametrize("relative_path", MODEL_PATHS)
def test_model_copies_mirror_phase_propagation(relative_path):
    source_path = Path(__file__).parents[1] / relative_path
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    decoder_forward = _find_method(tree, "LlamaDecoderLayer", "forward")
    decoder_args = [argument.arg for argument in decoder_forward.args.args]
    assert "quant_phase" in decoder_args
    contexts = [node for node in ast.walk(decoder_forward) if isinstance(node, ast.With)]
    assert any(
        _is_named_call(item.context_expr, "quant_phase_context")
        and isinstance(item.context_expr.args[0], ast.Name)
        and item.context_expr.args[0].id == "quant_phase"
        for node in contexts
        for item in node.items
    )

    model_forward = _find_method(tree, "LlamaModel", "forward")
    model_args = [argument.arg for argument in model_forward.args.args]
    assert "quant_phase" in model_args
    calls = [node for node in ast.walk(model_forward) if isinstance(node, ast.Call)]
    resolver_calls = [
        call for call in calls if _is_named_call(call, "resolve_quant_phase")
    ]
    assert len(resolver_calls) == 1
    assert any(
        _is_missing_phase_check(node.test)
        and resolver_calls[0] in tuple(ast.walk(node))
        for node in ast.walk(model_forward)
        if isinstance(node, ast.If)
    )

    checkpoint_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_gradient_checkpointing_func"
    ]
    assert len(checkpoint_calls) == 1
    assert isinstance(checkpoint_calls[0].args[-1], ast.Name)
    assert checkpoint_calls[0].args[-1].id == "quant_phase"

    ordinary_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "decoder_layer"
    ]
    assert len(ordinary_calls) == 1
    phase_keywords = [
        keyword
        for keyword in ordinary_calls[0].keywords
        if keyword.arg == "quant_phase"
    ]
    assert len(phase_keywords) == 1
    assert isinstance(phase_keywords[0].value, ast.Name)
    assert phase_keywords[0].value.id == "quant_phase"

    causal_forward = _find_method(tree, "LlamaForCausalLM", "forward")
    causal_args = [argument.arg for argument in causal_forward.args.args]
    assert "quant_phase" in causal_args
    causal_calls = [
        node for node in ast.walk(causal_forward) if isinstance(node, ast.Call)
    ]
    model_calls = [
        call
        for call in causal_calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "model"
    ]
    assert len(model_calls) == 1
    assert any(keyword.arg == "quant_phase" for keyword in model_calls[0].keywords)

    prepare = _find_method(tree, "LlamaForCausalLM", "prepare_inputs_for_generation")
    prepare_calls = [node for node in ast.walk(prepare) if isinstance(node, ast.Call)]
    assert any(_is_named_call(call, "resolve_quant_phase") for call in prepare_calls)
    prepare_source = ast.get_source_segment(source, prepare)
    assert '"quant_phase": quant_phase' in prepare_source
    assert "is_torchdynamo_compiling()" in ast.get_source_segment(
        source, model_forward
    )
    assert "if using_static_cache or past_key_values is None" in source


@pytest.mark.parametrize("relative_path", MODEL_PATHS)
def test_generation_host_boundary_attaches_static_cache_phase(
    relative_path, monkeypatch
):
    modeling = importlib.import_module(relative_path.replace("/", ".")[:-3])
    config = _tiny_config()
    model = modeling.LlamaForCausalLM(config).eval()
    cache = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=4,
        device="cpu",
        dtype=torch.float32,
    )

    def forbidden_get_seq_length(*args, **kwargs):
        raise AssertionError("StaticCache.get_seq_length must not be called")

    monkeypatch.setattr(StaticCache, "get_seq_length", forbidden_get_seq_length)
    attention_mask = torch.zeros(1, 1, 1, 4)
    for position, expected in ((0, QuantPhase.PREFILL), (2, QuantPhase.DECODE)):
        cache_position = torch.tensor([position])
        model_inputs = model.prepare_inputs_for_generation(
            torch.tensor([[1]]),
            past_key_values=cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=cache_position.unsqueeze(0),
            use_cache=True,
        )
        assert model_inputs["quant_phase"] is expected


@pytest.mark.parametrize("relative_path", MODEL_PATHS)
def test_real_static_cache_fullgraph_prefill_decode_specializations(
    relative_path, monkeypatch
):
    modeling = importlib.import_module(relative_path.replace("/", ".")[:-3])
    config = _tiny_config()
    model = modeling.LlamaModel(config).eval()
    if relative_path.startswith("train_utils"):
        head_dim = config.hidden_size // config.num_attention_heads
        model.layers[0].self_attn.R2 = torch.nn.Linear(
            head_dim, head_dim, bias=False
        )
    cache = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=4,
        device="cpu",
        dtype=torch.float32,
    )

    def forbidden_get_seq_length(*args, **kwargs):
        raise AssertionError("StaticCache.get_seq_length must not be called")

    monkeypatch.setattr(StaticCache, "get_seq_length", forbidden_get_seq_length)
    prefill_position = torch.tensor([0])
    decode_position = torch.tensor([1])
    assert resolve_quant_phase(cache, prefill_position) is QuantPhase.PREFILL
    assert resolve_quant_phase(cache, decode_position) is QuantPhase.DECODE

    def resolver_in_compiled_region(*args, **kwargs):
        raise AssertionError("resolve_quant_phase entered the compiled region")

    monkeypatch.setattr(modeling, "resolve_quant_phase", resolver_in_compiled_region)

    def specialized_forward(inputs_embeds, cache_position, quant_phase):
        return model(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            use_cache=True,
            return_dict=False,
            cache_position=cache_position,
            quant_phase=quant_phase,
        )[0]

    compiled_forward = torch.compile(
        specialized_forward, backend="eager", fullgraph=True
    )
    prefill_output = compiled_forward(
        torch.randn(1, 1, config.hidden_size),
        prefill_position,
        QuantPhase.PREFILL,
    )
    decode_output = compiled_forward(
        torch.randn(1, 1, config.hidden_size),
        decode_position,
        QuantPhase.DECODE,
    )
    assert prefill_output.shape == decode_output.shape == (1, 1, config.hidden_size)
