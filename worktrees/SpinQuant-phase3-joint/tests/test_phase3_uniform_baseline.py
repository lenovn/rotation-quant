import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from datasets import Dataset
import datasets
import pyarrow as arrow
import pytest
import torch
import torch.nn.functional as functional
from transformers import LlamaTokenizerFast

from experiments.phase3 import common, external_eval, launch, quantization, uniform_baseline
from test_phase3_external import TOKEN_PATH, external_tiny
from train_utils.quant_linear import QuantizeLinear
from utils.quant_utils import RotationStaticActQuantizer


B100_ROOT = common.PROJECT_ROOT / "runs/phase3/route-b-adam-100-20260914a"
CALIBRATION_DATA = B100_ROOT / "data.json"
SP2_CALIBRATION = B100_ROOT / "checkpoint-0100/sp2_calibration.json"


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan="quantizer." in name)


def test_uniform_actual_train_calibration_matches_saved_indices(monkeypatch):
    saved = json.loads(CALIBRATION_DATA.read_text())
    path = common.DATA_PATH / "wikitext-train.arrow"
    before = path.stat()
    reads = []
    original_from_file = Dataset.from_file

    def read_train(filename, *args, **kwargs):
        assert Path(filename).resolve() == path.resolve()
        reads.append(filename)
        return original_from_file(filename, *args, **kwargs)

    def reject(*args, **kwargs):
        pytest.fail("Calibration must not use C4, WikiText validation/test, training or dataset downloads")

    monkeypatch.setattr(Dataset, "from_file", staticmethod(read_train))
    monkeypatch.setattr(datasets, "load_dataset", reject)
    monkeypatch.setattr(common, "data_windows", reject)
    monkeypatch.setattr(external_eval, "load_c4_tokens", reject)
    monkeypatch.setattr(external_eval, "load_wikitext2_test_tokens", reject)
    calibration, metadata = uniform_baseline.load_calibration(CALIBRATION_DATA)
    with arrow.memory_map(str(path), "r") as source:
        rows = arrow.ipc.open_stream(source).read_all().column("text").to_pylist()
    tokenizer = LlamaTokenizerFast.from_pretrained(str(common.MODEL_PATH), local_files_only=True,
        model_max_length=2048, padding_side="right", use_fast=True,
        add_eos_token=False, add_bos_token=False)
    encoded = tokenizer(rows, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([token for row in encoded for token in row], dtype=torch.long)
    assert ids.numel() == saved["train_tokens"] == 2435022
    assert len(reads) == 1 and len(calibration) == 32
    for window, index in zip(calibration, saved["calibration_window_indices"]):
        assert window.dtype == torch.long and tuple(window.shape) == (1, 128)
        assert torch.equal(window, ids[index * 2048:index * 2048 + 128].reshape(1, -1))
    assert metadata["split"] == "train"
    assert metadata["dataset"] == "Salesforce/wikitext" and metadata["subset"] == "wikitext-2-raw-v1"
    assert metadata["calibration_window_indices"] == saved["calibration_window_indices"]
    assert metadata["calibration_length"] == 128 and metadata["calibration_tokens"] == 4096
    assert metadata["add_bos_token"] is False and metadata["add_eos_token"] is False
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    print("UNIFORM_TRAIN_ORACLE " + json.dumps(dict(train_tokens=ids.numel(),
        train_windows=saved["train_windows"], calibration_windows=len(calibration),
        calibration_length=128, calibration_tokens=sum(window.numel() for window in calibration),
        calibration_window_indices=saved["calibration_window_indices"]), sort_keys=True))


@pytest.mark.parametrize("fail_forward", [False, True], ids=["success", "exception"])
def test_uniform_capture_stride_full_max_and_restoration(external_tiny, monkeypatch, fail_forward):
    model = external_tiny.frozen
    wrappers = common.wrappers(model)
    downs = {name: wrapper for name, wrapper in wrappers.items() if name.endswith("down_proj")}
    first = next(iter(downs.values()))
    first.quantizer.bits = 16
    original_bits = {name: wrapper.quantizer.bits for name, wrapper in wrappers.items()}
    original_state = copy.deepcopy(model.state_dict())
    retained = first.register_forward_pre_hook(lambda module, inputs: None)
    original_hooks = {name: dict(wrapper._forward_pre_hooks) for name, wrapper in wrappers.items()}
    full_inputs = {name: [] for name in downs}
    calls = []

    def fake_backbone(current, ids):
        assert current is model and ids.device.type == "cpu" and not torch.is_grad_enabled()
        assert all(wrapper.quantizer.bits == 16 for wrapper in downs.values())
        assert all(wrapper.quantizer.bits == original_bits[name]
                   for name, wrapper in wrappers.items() if name not in downs)
        calls.append(ids)
        for index, (name, wrapper) in enumerate(downs.items()):
            values = (torch.arange(128 * wrapper.module.in_features).reshape(1, 128, -1)
                      .to(torch.bfloat16) / 128 + index + len(calls))
            values[:, 1, 0] = 4096 + index * 32
            full_inputs[name].append(values.detach().clone().reshape(128, -1))
            wrapper(values)
        if fail_forward:
            raise RuntimeError("injected calibration forward failure")

    monkeypatch.setattr(uniform_baseline, "backbone", fake_backbone)
    calibration = [torch.ones((1, 128), dtype=torch.long), torch.zeros((1, 128), dtype=torch.long)]
    try:
        if fail_forward:
            with pytest.raises(RuntimeError, match="injected calibration forward failure"):
                uniform_baseline.capture_down_inputs(model, calibration)
        else:
            samples, maximum = uniform_baseline.capture_down_inputs(model, calibration)
            assert samples.keys() == maximum.keys() == downs.keys()
            for name in downs:
                expected = torch.cat([values[::8] for values in full_inputs[name]])
                assert torch.equal(samples[name], expected) and expected.shape[0] == 32
                assert maximum[name] == max(float(values.abs().max()) for values in full_inputs[name])
                assert maximum[name] > float(expected.abs().max())
        assert {name: wrapper.quantizer.bits for name, wrapper in wrappers.items()} == original_bits
        assert all(dict(wrapper._forward_pre_hooks) == original_hooks[name] for name, wrapper in wrappers.items())
        assert_state_equal(model.state_dict(), original_state)
    finally:
        retained.remove()


def test_uniform_search_real_int8_math_and_fifty_candidates(monkeypatch):
    values = torch.tensor([[-16, -1, 1, 16], [-4, 0.125, -0.5, 3],
        [0.25, -0.125, 2, -6], [5, -3, 0.25, -0.5],
        [0.375, 0.75, -1.5, 0.0625]], dtype=torch.bfloat16)
    weight = torch.tensor([[0.75, -0.25, 0.5, 0.125],
        [-0.25, 0.5, 0.0625, -0.125]], dtype=torch.bfloat16)
    before_values, before_weight = values.clone(), weight.clone()
    bound = 20.0
    actual_forward = RotationStaticActQuantizer.forward
    measured = []

    def record_forward(quantizer, inputs):
        assert quantizer.bits == 8 and quantizer.has_scale and not quantizer.observing
        assert not quantizer.scale_learnable and inputs.dtype == torch.bfloat16
        result = actual_forward(quantizer, inputs)
        scale = quantizer.scale.detach().clone()
        expected = ((inputs.float() / scale).round().clamp(-128, 127) * scale).to(inputs.dtype)
        assert torch.equal(result, expected)
        measured.append(dict(scale=float(scale),
            output_mse=float(functional.linear(expected.float() - inputs.float(), weight.float()).square().mean())))
        return result

    monkeypatch.setattr(RotationStaticActQuantizer, "forward", record_forward)
    record = uniform_baseline.search_int8_range(values, weight, bound)
    candidates = record["candidates"]
    assert len(candidates) == len(measured) == 50
    coarse = torch.linspace(-16.0, 1.0, 33).tolist()
    assert [candidate["log2_ratio"] for candidate in candidates[:33]] == coarse
    best = min(range(33), key=lambda index: measured[index]["output_mse"])
    refined = torch.linspace(coarse[max(0, best - 1)], coarse[min(32, best + 1)], 17).tolist()
    assert [candidate["log2_ratio"] for candidate in candidates[33:]] == refined
    for candidate, measurement in zip(candidates, measured):
        assert candidate["alpha"] == bound * 2 ** candidate["log2_ratio"]
        assert measurement["scale"] == float(torch.tensor(candidate["alpha"] / 127, dtype=torch.float32))
        assert candidate["output_mse"] == measurement["output_mse"]
    assert record["selected"] == min(candidates, key=lambda candidate: candidate["output_mse"])
    assert torch.equal(values, before_values) and torch.equal(weight, before_weight)
    historical = json.loads(SP2_CALIBRATION.read_text())
    assert len(historical) == 16
    for comparison in historical.values():
        assert len(comparison["candidates"]) == 50 and comparison["sampled_rows"] == 512
        assert [candidate["log2_ratio"] for candidate in comparison["candidates"][:33]] == coarse


@pytest.mark.parametrize("fail_forward", [False, True], ids=["success", "exception"])
def test_uniform_capture_v_layout_restores_original_storage(external_tiny, monkeypatch, fail_forward):
    model = external_tiny.frozen
    wrappers = common.wrappers(model)
    values = {name: wrapper for name, wrapper in wrappers.items() if name.endswith("self_attn.v_proj")}
    downs = {name: wrapper for name, wrapper in wrappers.items() if name.endswith("down_proj")}
    original_weights = {name: wrapper.module.weight.data for name, wrapper in wrappers.items()}
    original_parameters = {name: wrapper.module.weight for name, wrapper in wrappers.items()}
    original_state = copy.deepcopy(model.state_dict())
    original_bits = {name: wrapper.quantizer.bits for name, wrapper in wrappers.items()}
    first_down = next(iter(downs.values()))
    retained = first_down.register_forward_pre_hook(lambda module, inputs: None)
    original_hooks = {name: dict(wrapper._forward_pre_hooks) for name, wrapper in wrappers.items()}
    calls = []

    if not fail_forward:
        reference = next(iter(values.values())).module.weight
        linear = QuantizeLinear(reference.shape[1], reference.shape[0], bias=False).to(torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(reference)
            rotated = linear.rotated_weight(torch.eye(reference.shape[1]), torch.eye(reference.shape[0]), False)
            scale = rotated.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 7
            codes = quantization.int4_codes(rotated, scale)
            online = (codes.float() * scale).to(torch.bfloat16).cpu()
            assert rotated.stride() == codes.stride() == online.stride() == (1, reference.shape[0])
            packed = quantization.pack_int4(codes)
            unpacked = quantization.unpack_int4(packed, tuple(codes.shape))
            cold = torch.empty_like(online, memory_format=torch.contiguous_format)
            cold.copy_((unpacked.float() * scale).to(cold))
            assert cold.stride() == (reference.shape[1], 1) and torch.equal(cold, online)
            print("V_LAYOUT_SOURCE_CHAIN " + json.dumps(dict(shape=list(reference.shape),
                online_stride=list(online.stride()), cold_stride=list(cold.stride()), values_equal=True)))

    def fake_backbone(current, ids):
        assert current is model and not torch.is_grad_enabled()
        calls.append(ids)
        assert len(values) == 16 and all(wrapper.quantizer.bits == 16 for wrapper in downs.values())
        for name, wrapper in wrappers.items():
            actual, original = wrapper.module.weight, original_weights[name]
            assert actual is original_parameters[name] and wrapper.weight is actual
            assert actual.dtype == original.dtype and actual.device == original.device
            assert torch.equal(actual, original)
            if name in values:
                assert original.is_contiguous() and not actual.is_contiguous()
                assert actual.stride() == (1, actual.shape[0])
                assert actual.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
            else:
                assert actual.stride() == original.stride()
                assert actual.data_ptr() == original.data_ptr()
            if name not in downs:
                assert wrapper.quantizer.bits == original_bits[name]
        for wrapper in downs.values():
            wrapper(torch.ones((1, 128, wrapper.module.in_features), dtype=torch.bfloat16))
        if fail_forward:
            raise RuntimeError("injected V-layout capture failure")

    monkeypatch.setattr(uniform_baseline, "backbone", fake_backbone)
    try:
        calibration = [torch.ones((1, 128), dtype=torch.long)]
        if fail_forward:
            with pytest.raises(RuntimeError, match="injected V-layout capture failure"):
                uniform_baseline.capture_down_inputs(model, calibration)
        else:
            samples, maximum = uniform_baseline.capture_down_inputs(model, calibration)
            assert samples.keys() == maximum.keys() == downs.keys()
            assert all(sample.shape == (16, 16) for sample in samples.values())
            assert all(bound == 1.0 for bound in maximum.values())
        assert len(calls) == 1
        for name, wrapper in wrappers.items():
            actual, original = wrapper.module.weight, original_weights[name]
            assert actual is original_parameters[name] and wrapper.weight is actual
            assert actual.data_ptr() == original.data_ptr()
            assert actual.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
            assert actual.storage_offset() == original.storage_offset() and actual.stride() == original.stride()
            assert torch.equal(actual, original)
            assert wrapper.quantizer.bits == original_bits[name]
            assert dict(wrapper._forward_pre_hooks) == original_hooks[name]
        assert_state_equal(model.state_dict(), original_state)
    finally:
        retained.remove()


def make_overlay(model, package):
    return dict(parent=str(package.resolve()),
        calibration_metadata=dict(dataset="Salesforce/wikitext", subset="wikitext-2-raw-v1",
            split="train", calibration_length=128, calibration_tokens=4096,
            calibration_window_indices=json.loads(CALIBRATION_DATA.read_text())["calibration_window_indices"]),
        scales={name: torch.tensor([0.0125 + index / 10000], dtype=torch.float32)
                for index, (name, wrapper) in enumerate(common.wrappers(model).items()) if name.endswith("down_proj")})


def test_uniform_overlay_down_only_and_invalid_inputs(external_tiny, tmp_path):
    fixture = external_tiny
    overlay = make_overlay(fixture.frozen, fixture.package)
    path = tmp_path / "overlay.pt"
    package_bytes = fixture.package.read_bytes()
    original_state = copy.deepcopy(fixture.frozen.state_dict())
    wrong_parent = dict(overlay, parent=str(tmp_path / "other-parent.pt"))
    invalid = [wrong_parent,
        dict(overlay, calibration_metadata=dict(overlay["calibration_metadata"], split="validation")),
        dict(overlay, calibration_metadata=dict(overlay["calibration_metadata"], dataset="allenai/c4"))]
    first_name = next(iter(overlay["scales"]))
    for scale in (torch.tensor([0.0]), torch.tensor([-0.1]), torch.tensor([float("nan")]),
                  torch.tensor([float("inf")]), torch.ones(2)):
        invalid.append(dict(overlay, scales=dict(overlay["scales"], **{first_name: scale})))
    invalid.append(dict(overlay, scales={name: scale for name, scale in overlay["scales"].items() if name != first_name}))
    invalid.append(dict(overlay, scales=dict(overlay["scales"], unrelated=torch.ones(1))))
    for state in invalid:
        torch.save(state, path)
        with pytest.raises(ValueError):
            uniform_baseline.apply_down_int8(fixture.frozen, path, fixture.package)
        assert_state_equal(fixture.frozen.state_dict(), original_state)
    torch.save(overlay, path)
    overlay_bytes = path.read_bytes()
    result = uniform_baseline.apply_down_int8(fixture.frozen, path, fixture.package)
    assert result == dict(path=str(path.resolve()), parent=str(fixture.package.resolve()),
        down_format="int8", calibration_metadata=overlay["calibration_metadata"])
    assert path.stat().st_size < 65536
    actual_state = fixture.frozen.state_dict()
    for name, value in original_state.items():
        if ".mlp.down_proj.quantizer." not in name:
            torch.testing.assert_close(actual_state[name], value, rtol=0, atol=0, equal_nan="quantizer." in name)
    for name, wrapper in common.wrappers(fixture.frozen).items():
        assert isinstance(wrapper.quantizer, RotationStaticActQuantizer)
        assert wrapper.quantizer.bits == 8 and wrapper.quantizer.has_scale and not wrapper.quantizer.observing
        if name in overlay["scales"]:
            assert torch.equal(wrapper.quantizer.scale, overlay["scales"][name])
            inputs = torch.linspace(-4, 4, 2 * wrapper.module.in_features).reshape(2, -1).to(torch.bfloat16)
            scale = overlay["scales"][name]
            expected = ((inputs.float() / scale).round().clamp(-128, 127) * scale).to(inputs.dtype)
            assert torch.equal(wrapper(inputs), functional.linear(expected, wrapper.module.weight, wrapper.module.bias))
    assert fixture.package.read_bytes() == package_bytes and path.read_bytes() == overlay_bytes
    with pytest.raises(ValueError, match="Invalid matched INT8 overlay"):
        uniform_baseline.apply_down_int8(fixture.frozen, path, fixture.package)


def test_uniform_calibration_driver_history_matching_and_small_overlay(external_tiny, tmp_path, monkeypatch):
    fixture = external_tiny
    original_state = copy.deepcopy(fixture.frozen.state_dict())
    package_bytes = fixture.package.read_bytes()
    overlay = make_overlay(fixture.frozen, fixture.package)
    values = torch.linspace(-2, 3, 32).reshape(2, 16).to(torch.bfloat16)
    bound = float(values.abs().max())
    samples = {name: values.clone() for name in overlay["scales"]}
    maximum = {name: bound for name in samples}
    historical = {name: dict(sampled_rows=2, full_absmax=bound,
        candidates=[dict(alpha=bound)] * 50, selected=dict(alpha=bound, output_mse=0.1)) for name in samples}
    historical_path = tmp_path / "synthetic-sp2.json"
    historical_path.write_text(json.dumps(historical))
    monkeypatch.setattr(uniform_baseline, "load_calibration", lambda path:
        ([torch.ones((1, 128), dtype=torch.long)] * 32, overlay["calibration_metadata"]))
    monkeypatch.setattr(uniform_baseline, "load_static", lambda path: (fixture.frozen, fixture.records))
    monkeypatch.setattr(uniform_baseline, "capture_down_inputs", lambda model, calibration:
        ({name: value.clone() for name, value in samples.items()}, maximum.copy()))
    output = tmp_path / "calibration"
    output.mkdir()
    args = SimpleNamespace(parent=fixture.package, calibration_data=CALIBRATION_DATA,
        sp2_calibration=historical_path, output=output)
    uniform_baseline.run(args)
    saved = torch.load(output / "down_int8_scales.pt", map_location="cpu", weights_only=True)
    result = json.loads((output / "result.json").read_text())
    search = json.loads((output / "range_search.json").read_text())
    assert saved.keys() == {"parent", "calibration_metadata", "scales"}
    assert saved["parent"] == str(fixture.package) and saved["calibration_metadata"] == overlay["calibration_metadata"]
    assert saved["scales"].keys() == samples.keys() and len(search) == 16
    assert (output / "down_int8_scales.pt").stat().st_size < 65536
    assert result["down_layers"] == 16 and result["candidates_per_layer"] == 50
    assert result["parent_scales_unchanged"] is True and result["historical_capture_matched"] is True
    assert result["c4_used"] is False and result["training"] is False and result["calibration"] is True
    for name, record in search.items():
        assert len(record["candidates"]) == 50 and record["sampled_rows"] == 2
        assert record["historical_sp2_selected"] == historical[name]["selected"]
        assert record["historical_sp2_absmax"] == bound
        assert float(saved["scales"][name]) == record["selected"]["scale"]
    assert_state_equal(fixture.frozen.state_dict(), original_state)
    first_name = next(iter(samples))
    for field, replacement in (("sampled_rows", 3), ("candidates", [dict(alpha=bound)] * 49),
                               ("full_absmax", bound + 0.01)):
        changed = copy.deepcopy(historical)
        changed[first_name][field] = replacement
        historical_path.write_text(json.dumps(changed))
        args.output = tmp_path / ("reject-" + field)
        args.output.mkdir()
        with pytest.raises(ValueError, match="calibration inputs or search budget differ"):
            uniform_baseline.run(args)
        assert not (args.output / "down_int8_scales.pt").exists()
        assert not (args.output / "result.json").exists()
    assert_state_equal(fixture.frozen.state_dict(), original_state)
    assert fixture.package.read_bytes() == package_bytes


def test_uniform_external_c4_overlay_and_default_compatibility(external_tiny, tmp_path, monkeypatch):
    fixture = external_tiny
    overlay_path = tmp_path / "overlay.pt"
    overlay = make_overlay(fixture.frozen, fixture.package)
    torch.save(overlay, overlay_path)
    package_bytes = fixture.package.read_bytes()
    original_load = external_eval.load_model
    calls, loaded = [], []

    def reject(*args, **kwargs):
        pytest.fail("External C4 evaluation must never fit calibration or search scales")

    def load(mode, package):
        model = original_load(mode, package)
        loaded.append(model)
        return model

    def fake_evaluator(model, encoding, device, arguments):
        assert device == "cuda" and not torch.is_grad_enabled()
        assert model.seqlen == 2048 and encoding.input_ids.shape == (1, 128 * 2048)
        calls.append(encoding.input_ids.clone())
        return 2.0

    monkeypatch.setattr(uniform_baseline, "load_calibration", reject)
    monkeypatch.setattr(uniform_baseline, "capture_down_inputs", reject)
    monkeypatch.setattr(uniform_baseline, "search_int8_range", reject)
    monkeypatch.setattr(external_eval, "load_model", load)
    monkeypatch.setattr(external_eval, "evaluator", fake_evaluator)
    original_calls = None
    for label in ("original-sp2", "uniform-int8"):
        calls.clear()
        output = tmp_path / label
        output.mkdir()
        args = SimpleNamespace(mode="quantized", package=fixture.package, tokens=TOKEN_PATH,
            output=output, chunk_windows=128)
        if label == "uniform-int8":
            args.down_int8_scales = overlay_path
        external_eval.run(args)
        result = json.loads((output / "result.json").read_text())
        model_record = json.loads((output / "model.json").read_text())
        assert result["dataset"] == "allenai/c4" and result["split"] == "validation"
        assert result["predicted_tokens"] == 2096128 and result["unscored_tail_tokens"] == 0
        assert result["training"] is False and result["calibration"] is False and result["candidate_selection"] is False
        assert result["activation_scales_unchanged"] is True and len(calls) == 8
        expected_formats = {"int8": 112} if label == "uniform-int8" else {"int8": 96, "sp2": 16}
        assert result["activation_formats"] == model_record["activation_formats"] == expected_formats
        assert result["down_int8_overlay"] == model_record["down_int8_overlay"]
        if label == "uniform-int8":
            assert result["down_int8_overlay"]["path"] == str(overlay_path.resolve())
            assert result["down_int8_overlay"]["calibration_metadata"] == overlay["calibration_metadata"]
            assert all(torch.equal(left, right) for left, right in zip(original_calls, calls))
        else:
            assert result["down_int8_overlay"] is None
            original_calls = list(calls)
        before = torch.load(output / "quantizers_before.pt", map_location="cpu", weights_only=True)
        after = torch.load(output / "quantizers_after.pt", map_location="cpu", weights_only=True)
        assert before.keys() == after.keys() and len(before) == 112
        for name in before:
            assert_state_equal(after[name], before[name])
    state = loaded[-1].state_dict()
    for name, value in fixture.frozen.state_dict().items():
        if ".mlp.down_proj.quantizer." not in name:
            torch.testing.assert_close(state[name], value, rtol=0, atol=0, equal_nan="quantizer." in name)
    monkeypatch.setattr(external_eval, "load_model", lambda mode, package: loaded[-1])
    del args.down_int8_scales
    args.output = tmp_path / "reject-unrequested-uniform"
    args.output.mkdir()
    with pytest.raises(ValueError, match="quantization coverage differs"):
        external_eval.run(args)
    args.mode = "bf16"
    args.down_int8_scales = overlay_path
    args.output = tmp_path / "reject-bf16-overlay"
    args.output.mkdir()
    with pytest.raises(ValueError, match="Only quantized mode"):
        external_eval.run(args)
    assert fixture.package.read_bytes() == package_bytes


def test_uniform_cli_overlay_rejection_and_launcher_mapping(tmp_path, monkeypatch):
    observed = []
    parent = tmp_path / "parent with spaces.pt"
    overlay = tmp_path / "overlay with spaces.pt"
    forwarded = ["--parent", str(parent), "--calibration-data", str(CALIBRATION_DATA),
        "--sp2-calibration", str(SP2_CALIBRATION)]
    monkeypatch.setattr(uniform_baseline, "run", lambda args: observed.append(args))
    monkeypatch.setattr(uniform_baseline, "source_record", lambda: {"test": "uniform CLI"})
    monkeypatch.setattr(uniform_baseline.subprocess, "check_output", lambda *args, **kwargs: "mock diff")
    monkeypatch.setattr(uniform_baseline, "set_seed", lambda seed: None)
    monkeypatch.setattr(sys, "argv", ["uniform_baseline.py", "--output", str(tmp_path / "uniform"), *forwarded])
    uniform_baseline.main()
    assert len(observed) == 1 and observed[0].parent == parent.resolve()
    assert observed[0].calibration_data == CALIBRATION_DATA.resolve()
    assert observed[0].sp2_calibration == SP2_CALIBRATION.resolve()
    monkeypatch.setattr(external_eval, "run", lambda args: observed.append(args))
    monkeypatch.setattr(external_eval, "source_record", lambda: {"test": "overlay CLI"})
    monkeypatch.setattr(external_eval, "set_seed", lambda seed: None)
    external_args = ["--mode", "quantized", "--package", str(parent), "--tokens", str(TOKEN_PATH),
        "--down-int8-scales", str(overlay)]
    monkeypatch.setattr(sys, "argv", ["external_eval.py", "--output", str(tmp_path / "external"), *external_args])
    external_eval.main()
    assert len(observed) == 2 and observed[-1].down_int8_scales == overlay.resolve()
    assert observed[-1].tokens == TOKEN_PATH.resolve() and observed[-1].chunk_windows == 128
    rejected = tmp_path / "bf16-overlay"
    monkeypatch.setattr(sys, "argv", ["external_eval.py", "--output", str(rejected),
        "--mode", "bf16", "--tokens", str(TOKEN_PATH), "--down-int8-scales", str(overlay)])
    with pytest.raises(SystemExit) as error:
        external_eval.main()
    assert error.value.code == 2 and not rejected.exists() and len(observed) == 2
    source_file = tmp_path / "worktrees/SpinQuant-phase3-joint/experiments/phase3/launch.py"
    directory = tmp_path / "runs/phase3"
    directory.mkdir(parents=True)
    launches = []

    def fake_tmux(command, project, environment, log_path, session):
        assert environment["CUDA_VISIBLE_DEVICES"] == "1"
        assert command[0] == "/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python"
        launches.append(command)
        return 12345, "mock-socket"

    monkeypatch.setattr(launch, "__file__", str(source_file))
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/mock/tmux")
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *args, **kwargs: "1, GPU-mock, 0, 24576, 0\n")
    monkeypatch.setattr(launch, "start_tmux", fake_tmux)
    for task, args, entrypoint in (("uniform-calibration", forwarded, "uniform_baseline.py"),
                                  ("external", external_args, "external_eval.py")):
        monkeypatch.setattr(sys, "argv", ["launch.py", "--gpu", "1", "--name", task, "--task", task, "--", *args])
        launch.main()
        assert launches[-1][2] == str(source_file.parent / entrypoint)
        assert launches[-1][5:] == args
    assert len(launches) == 2
