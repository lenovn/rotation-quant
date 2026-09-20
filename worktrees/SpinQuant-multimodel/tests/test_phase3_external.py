import copy
import inspect
import json
import math
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, LlamaConfig

from eval_utils.modeling_llama import LlamaForCausalLM
from experiments.phase3 import common, external_eval, launch, postprocess, quantization
from utils.quant_utils import ActQuantizer, RotationStaticActQuantizer, add_actquant


TOKEN_PATH = common.PROJECT_ROOT / "runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/input_tokens.pt"


@pytest.fixture(scope="module")
def saved_c4():
    return torch.load(TOKEN_PATH, map_location="cpu", weights_only=True)


def test_external_fixed_c4_tokens_metadata_and_accounting(saved_c4, monkeypatch):
    before = TOKEN_PATH.stat()
    ids, metadata = external_eval.load_c4_tokens(TOKEN_PATH)
    assert ids.dtype == torch.long and ids.shape == (1, 2097152)
    assert torch.equal(ids, saved_c4["input_ids"]) and metadata == saved_c4["metadata"]
    assert (metadata["windows"], metadata["window_length"], metadata["predicted_tokens"]) == (1024, 2048, 2096128)
    assert len(metadata["source_files"]) == 8
    assert metadata["text_separator"] == "\n\n" and metadata["purpose"] == "external evaluation only; no calibration or selection"
    original_load = torch.load
    replacements = [
        dict(metadata=dict(metadata, split="train")),
        dict(metadata=dict(metadata, tokenizer_path="/not-the-saved-tokenizer")),
        dict(metadata=dict(metadata, add_bos_token=True)),
        dict(input_ids=ids.reshape(1024, 2048)),
        dict(input_ids=ids.to(torch.int32)),
        dict(metadata=dict(metadata, windows=1023)),
        dict(metadata=dict(metadata, predicted_tokens=2097151)),
    ]
    for replacement in replacements:
        state = dict(saved_c4, **replacement)

        def read(path, **kwargs):
            assert path == TOKEN_PATH and kwargs == dict(map_location="cpu", weights_only=True)
            return state

        monkeypatch.setattr(external_eval.torch, "load", read)
        with pytest.raises(ValueError):
            external_eval.load_c4_tokens(TOKEN_PATH)
    monkeypatch.setattr(external_eval.torch, "load", original_load)
    assert TOKEN_PATH.stat().st_mtime_ns == before.st_mtime_ns
    assert TOKEN_PATH.stat().st_size == before.st_size
    assert torch.equal(torch.load(TOKEN_PATH, map_location="cpu", weights_only=True)["input_ids"], ids)


def test_external_actual_acceptance_128_window_chunks_and_weighted_oracle(saved_c4, monkeypatch):
    ids = saved_c4["input_ids"]
    model = SimpleNamespace(seqlen=2048)
    calls = []
    actual_evaluator = external_eval.evaluator
    assert Path(inspect.getsourcefile(inspect.unwrap(actual_evaluator))) == common.SOURCE_ROOT / "utils/eval_utils.py"

    def segment_score(current, encoding, device, arguments):
        assert current is model and device == "cuda" and current.seqlen == 2048
        assert vars(arguments) == dict(eval_nsamples=None, bsz=1, capture_layer_io=False)
        start = sum(value.numel() for value in calls)
        assert torch.equal(encoding.input_ids, ids[:, start:start + encoding.input_ids.numel()])
        calls.append(encoding.input_ids.clone())
        return torch.exp(torch.tensor(1.0 + len(calls) / 10, dtype=torch.float32)).item()

    def reject_data(*args, **kwargs):
        pytest.fail("External evaluation must not use WikiText or prepare tokens")

    monkeypatch.setattr(external_eval, "evaluator", segment_score)
    monkeypatch.setattr(common, "data_windows", reject_data)
    result = external_eval.evaluate_tokens(model, ids, 128)
    assert len(calls) == 8 and all(value.shape == (1, 128 * 2048) for value in calls)
    assert torch.equal(torch.cat(calls, dim=1), ids) and model.seqlen == 2048
    assert result["token_count"] == 2097152 and result["predicted_tokens"] == 2096128
    assert result["unscored_tail_tokens"] == 0
    expected_nll = sum(math.log(torch.exp(torch.tensor(1 + index / 10)).item()) for index in range(1, 9)) / 8
    assert result["nll"] == pytest.approx(expected_nll, rel=0, abs=5e-16)
    assert result["ppl"] == math.exp(result["nll"])
    assert result["ppl"] != sum(segment["ppl"] for segment in result["segments"]) / 8
    calls.clear()
    uneven = external_eval.evaluate_tokens(model, ids[:, :129 * 2048], 128)
    assert [segment["windows"] for segment in uneven["segments"]] == [128, 1]
    expected = (128 * math.log(torch.exp(torch.tensor(1.1)).item()) + math.log(torch.exp(torch.tensor(1.2)).item())) / 129
    assert uneven["nll"] == pytest.approx(expected, rel=0, abs=5e-16)


@pytest.fixture
def external_tiny(tmp_path, monkeypatch):
    config = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=16,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048, tie_word_embeddings=True)
    config.head_dim = 4
    torch.manual_seed(42)
    original = LlamaForCausalLM._from_config(config, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    original.requires_grad_(False)
    model_path = tmp_path / "original-bf16"
    original.save_pretrained(model_path, safe_serialization=False)
    frozen = copy.deepcopy(original)
    frozen.config.tie_word_embeddings = False
    frozen.lm_head.weight = torch.nn.Parameter(frozen.model.embed_tokens.weight.detach().clone(), requires_grad=False)
    add_actquant(frozen)
    records = {}
    with torch.no_grad():
        for index, (name, wrapper) in enumerate(common.wrappers(frozen).items()):
            scale = wrapper.module.weight.float().abs().amax(1, keepdim=True).clamp_min(1e-5) / 7
            codes = quantization.int4_codes(wrapper.module.weight, scale)
            wrapper.module.weight.copy_((codes.float() * scale).to(torch.bfloat16))
            records[name] = dict(packed=quantization.pack_int4(codes), shape=tuple(codes.shape), scale=scale)
            if name.endswith("down_proj"):
                wrapper.quantizer = quantization.SP2Quantizer(0.5 + index / 100)
            else:
                wrapper.quantizer = RotationStaticActQuantizer()
                wrapper.quantizer.load_scale(torch.tensor([0.025 + index / 10000]))
    package = tmp_path / "frozen.pt"
    common.save_frozen(frozen, records, package, dict(method="CPU frozen fixture", no_new_training=True))
    original_config_loader, original_model_loader = AutoConfig.from_pretrained, LlamaForCausalLM.from_pretrained
    device_calls, load_calls = [], []
    original_to = torch.nn.Module.to

    def read_config(path, **kwargs):
        assert path == str(common.MODEL_PATH) and kwargs == dict(local_files_only=True)
        return original_config_loader(model_path, **kwargs)

    def read_model(model_class, path, **kwargs):
        assert path == str(common.MODEL_PATH)
        assert kwargs["torch_dtype"] == torch.bfloat16 and kwargs["local_files_only"] is True
        assert kwargs["attn_implementation"] == "sdpa" and kwargs["config"].tie_word_embeddings is False
        load_calls.append(path)
        return original_model_loader(model_path, **kwargs)

    def cpu_to(module, *arguments, **kwargs):
        if arguments and isinstance(arguments[0], (str, torch.device)) and torch.device(arguments[0]).type == "cuda":
            assert arguments == ("cuda",) and not kwargs
            device_calls.append("to-cuda")
            return original_to(module, "cpu")
        return original_to(module, *arguments, **kwargs)

    def cpu_cuda(module, *args, **kwargs):
        assert not args and not kwargs
        assert all(parameter.device.type == "cpu" for parameter in module.parameters())
        device_calls.append("cuda")
        return module

    def reject(*args, **kwargs):
        pytest.fail("External path attempted training, rotation, norm fusion, calibration, or optimizer creation")

    monkeypatch.setattr(AutoConfig, "from_pretrained", read_config)
    monkeypatch.setattr(LlamaForCausalLM, "from_pretrained", classmethod(read_model))
    monkeypatch.setattr(torch.nn.Module, "to", cpu_to)
    monkeypatch.setattr(torch.nn.Module, "cuda", cpu_cuda)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(common, "data_windows", reject)
    monkeypatch.setattr(common, "initialize_scales", reject)
    monkeypatch.setattr(common, "calibrate_sp2", reject)
    monkeypatch.setattr(common, "build_training_model", reject)
    monkeypatch.setattr(common, "fuse_layer_norms", reject)
    monkeypatch.setattr(RotationStaticActQuantizer, "begin_calibration", reject)
    monkeypatch.setattr(ActQuantizer, "find_params", reject)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", reject)
    return SimpleNamespace(original=original, frozen=frozen, package=package, records=records,
                           device_calls=device_calls, load_calls=load_calls)


def test_external_paired_driver_real_coldload_rope_and_immutable_scales(external_tiny, saved_c4, tmp_path, monkeypatch):
    fixture = external_tiny
    package_bytes = fixture.package.read_bytes()
    original_load = external_eval.load_model
    loaded, calls, reloads = {}, {}, []
    original_reload = postprocess.reload_frozen
    assert external_eval.load_static is postprocess.load_static
    assert inspect.signature(postprocess.load_static).parameters["device"].default == "cuda"

    def reload(model, path):
        assert path == fixture.package
        reloads.append(path)
        return original_reload(model, path)

    def load(mode, package):
        model = original_load(mode, package)
        assert not model.training and not model.config.use_cache and model.seqlen == 2048
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
        assert not hasattr(model, "R1") and not any("Rotation" in type(module).__name__ for module in model.modules()
                                                  if not isinstance(module, RotationStaticActQuantizer))
        rotary = [value for name, value in model.named_buffers() if name.endswith("inv_freq")]
        assert rotary and all(value.dtype == torch.float32 for value in rotary)
        assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
        expected = fixture.original if mode == "bf16" else fixture.frozen
        actual_state, expected_state = model.state_dict(), expected.state_dict()
        assert actual_state.keys() == expected_state.keys()
        for name, value in expected_state.items():
            torch.testing.assert_close(actual_state[name], value, rtol=0, atol=0, equal_nan="quantizer." in name)
        if mode == "bf16":
            assert not common.wrappers(model)
            assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
            assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
        else:
            wrappers = common.wrappers(model)
            assert len(wrappers) == 112
            assert sum(isinstance(wrapper.quantizer, quantization.SP2Quantizer) for wrapper in wrappers.values()) == 16
            assert all(wrapper.quantizer.bits == 8 and wrapper.out_quantizer.bits == 16 for wrapper in wrappers.values())
        with torch.no_grad():
            ids = torch.tensor([[1, 4, 7, 3, 9]])
            assert torch.equal(model.lm_head(common.backbone(model, ids)), expected.lm_head(common.backbone(expected, ids)))
        loaded[mode] = model
        calls[mode] = []
        return model

    def evaluator(model, encoding, device, arguments):
        mode = next(name for name, current in loaded.items() if current is model)
        index = len(calls[mode])
        assert not torch.is_grad_enabled() and device == "cuda"
        assert encoding.input_ids.shape == (1, 128 * 2048)
        assert torch.equal(encoding.input_ids, saved_c4["input_ids"][:, index * 128 * 2048:(index + 1) * 128 * 2048])
        calls[mode].append(encoding.input_ids.clone())
        return torch.exp(torch.tensor(2 + index / 20)).item()

    monkeypatch.setattr(postprocess, "reload_frozen", reload)
    monkeypatch.setattr(external_eval, "load_model", load)
    monkeypatch.setattr(external_eval, "evaluator", evaluator)
    for mode in ("bf16", "quantized"):
        output = tmp_path / mode
        output.mkdir()
        args = SimpleNamespace(mode=mode, package=fixture.package if mode == "quantized" else None,
                               tokens=TOKEN_PATH, output=output, chunk_windows=128)
        external_eval.run(args)
        result = json.loads((output / "result.json").read_text())
        assert result["dataset"] == "allenai/c4" and result["split"] == "validation"
        assert result["predicted_tokens"] == 2096128 and result["activation_scales_unchanged"] is True
        assert result["input_token_path"] == str(TOKEN_PATH) and result["input_metadata"] == saved_c4["metadata"]
        assert not result["training"] and not result["calibration"] and not result["candidate_selection"]
        before = torch.load(output / "quantizers_before.pt", weights_only=True)
        after = torch.load(output / "quantizers_after.pt", weights_only=True)
        assert len(before) == len(after) == (112 if mode == "quantized" else 0)
        for name, record in before.items():
            for key, value in record.items():
                assert torch.equal(value, after[name][key])
    assert len(calls["bf16"]) == len(calls["quantized"]) == 8
    assert all(torch.equal(left, right) for left, right in zip(calls["bf16"], calls["quantized"]))
    assert reloads == [fixture.package] and len(fixture.load_calls) == 2
    assert fixture.device_calls == ["cuda", "to-cuda", "cuda"]
    assert fixture.package.read_bytes() == package_bytes
    changed_output = tmp_path / "detected-scale-mutation"
    changed_output.mkdir()
    args.output = changed_output

    def mutate_scale(model, ids, chunk_windows):
        model.model.layers[0].mlp.down_proj.quantizer.scale.mul_(2)
        return dict(predicted_tokens=2096128, unscored_tail_tokens=0)

    monkeypatch.setattr(external_eval, "evaluate_tokens", mutate_scale)
    with pytest.raises(ValueError, match="Fixed-model state or C4 target accounting changed"):
        external_eval.run(args)
    assert not (changed_output / "result.json").exists()
    assert fixture.package.read_bytes() == package_bytes


def test_external_launcher_routing_environment_and_driver_default(tmp_path, monkeypatch, capsys):
    source_file = tmp_path / "worktrees/SpinQuant-phase3-joint/experiments/phase3/launch.py"
    directory = tmp_path / "runs/phase3"
    directory.mkdir(parents=True)
    run_name = "external 'quoted';$literal"
    package = "/frozen parent 'quoted';$(literal).pt"
    forwarded = ["--mode", "quantized", "--package", package, "--tokens", str(TOKEN_PATH)]
    output = directory / run_name
    expected = ["/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python", "-u",
                str(source_file.parent / "external_eval.py"), "--output", str(output), *forwarded]
    launches = []

    def query(arguments, **kwargs):
        assert arguments[0] == "nvidia-smi" and kwargs == dict(text=True)
        return "1, GPU-mock, 1024, 24576, 0\n"

    def tmux(arguments, **kwargs):
        assert arguments[:4] == ["tmux", "-L", "rotation-quant-phase3", "new-session"]
        assert kwargs == dict(check=True, text=True, capture_output=True)
        payload = shlex.split(arguments[-1])
        assert payload[:2] == ["exec", "env"] and "CUDA_VISIBLE_DEVICES=1" in payload
        assert payload[-len(expected)-3:-3] == expected
        assert payload[-3:] == [">", str(directory / (run_name + ".log")), "2>&1"]
        launches.append(arguments)
        return SimpleNamespace(stdout="12345\n")

    monkeypatch.setattr(launch, "__file__", str(source_file))
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/mock/tmux")
    monkeypatch.setattr(launch.subprocess, "check_output", query)
    monkeypatch.setattr(launch.subprocess, "run", tmux)
    monkeypatch.setattr(sys, "argv", [str(source_file), "--gpu", "1", "--name", run_name, "--task", "external", "--", *forwarded])
    launch.main()
    record = json.loads((directory / (run_name + ".launch.json")).read_text())
    assert record["pid"] == 12345 and record["command"] == expected and len(launches) == 1
    assert json.loads(capsys.readouterr().out) == record
    observed = []
    monkeypatch.setattr(external_eval, "run", lambda args: observed.append(args))
    monkeypatch.setattr(external_eval, "source_record", lambda: dict(test="external CLI boundary"))
    monkeypatch.setattr(external_eval.subprocess, "check_output", lambda *args, **kwargs: "mock diff")
    monkeypatch.setattr(sys, "argv", ["external_eval.py", "--output", str(tmp_path / "driver"), *forwarded])
    external_eval.main()
    assert len(observed) == 1 and observed[0].chunk_windows == 128
    assert observed[0].package == Path(package).resolve() and observed[0].tokens == TOKEN_PATH.resolve()
