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
from transformers import LlamaTokenizerFast

from experiments.phase3 import common, external_eval
from test_phase3_external import TOKEN_PATH, external_tiny


@pytest.fixture(scope="module")
def wikitext_test_oracle():
    path = common.DATA_PATH / "wikitext-test.arrow"
    before = path.stat()
    with arrow.memory_map(str(path), "r") as source:
        rows = arrow.ipc.open_stream(source).read_all().column("text").to_pylist()
    tokenizer = LlamaTokenizerFast.from_pretrained(str(common.MODEL_PATH), local_files_only=True,
        model_max_length=2048, padding_side="right", use_fast=True,
        add_eos_token=False, add_bos_token=False)
    text = "\n\n".join(rows)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    windows = [ids[:, start:start + 2048] for start in range(0, ids.numel() - 1, 2048)]
    full, tail = divmod(ids.numel(), 2048)
    metadata = dict(dataset="Salesforce/wikitext", subset="wikitext-2-raw-v1", split="test",
        tokenizer_path=str(common.MODEL_PATH), add_bos_token=False, add_eos_token=False,
        dataset_path=str(path), rows=len(rows), text_join="double-newline",
        window_length=2048, token_count=ids.numel(),
        windows=len(windows), full_windows=full, tail_tokens=tail,
        predicted_tokens=sum(window.numel() - 1 for window in windows),
        unscored_tail_tokens=int(tail == 1))
    return SimpleNamespace(path=path, before=before, text=text, ids=ids, metadata=metadata)


def test_wikitext2_test_real_arrow_tokenizer_and_metadata(wikitext_test_oracle, monkeypatch):
    oracle = wikitext_test_oracle
    reads, tokenizer_calls = [], []
    original_from_file = Dataset.from_file
    original_tokenizer = LlamaTokenizerFast.from_pretrained

    def read_test(path, *args, **kwargs):
        assert Path(path).resolve() == oracle.path.resolve()
        reads.append(path)
        return original_from_file(path, *args, **kwargs)

    def load_tokenizer(tokenizer_class, path, **kwargs):
        assert tokenizer_class is LlamaTokenizerFast
        assert Path(path).resolve() == common.MODEL_PATH.resolve()
        assert kwargs == dict(local_files_only=True, model_max_length=2048,
            padding_side="right", use_fast=True, add_eos_token=False, add_bos_token=False)
        tokenizer_calls.append(path)
        return original_tokenizer(path, **kwargs)

    def reject(*args, **kwargs):
        pytest.fail("WikiText-2 test loading must not access train/validation or network datasets")

    monkeypatch.setattr(Dataset, "from_file", staticmethod(read_test))
    monkeypatch.setattr(LlamaTokenizerFast, "from_pretrained", classmethod(load_tokenizer))
    monkeypatch.setattr(datasets, "load_dataset", reject)
    monkeypatch.setattr(common, "data_windows", reject)
    ids, metadata = external_eval.load_wikitext2_test_tokens()
    assert len(reads) == len(tokenizer_calls) == 1
    assert ids.dtype == torch.long and ids.device.type == "cpu"
    assert tuple(ids.shape) == (1, oracle.ids.numel()) and torch.equal(ids, oracle.ids)
    for name, value in oracle.metadata.items():
        assert metadata[name] == value, name
    assert oracle.metadata["full_windows"] > 128 and oracle.metadata["tail_tokens"] >= 2
    after = oracle.path.stat()
    assert (after.st_size, after.st_mtime_ns) == (oracle.before.st_size, oracle.before.st_mtime_ns)
    print("WIKITEXT_TEST_ORACLE " + json.dumps(oracle.metadata, sort_keys=True))


def test_wikitext2_test_real_tail_target_weighted_nll(wikitext_test_oracle, monkeypatch):
    oracle = wikitext_test_oracle
    model = SimpleNamespace(seqlen=2048)
    calls, scores = [], []

    def fake_evaluator(current, encoding, device, arguments):
        assert current is model and device == "cuda"
        assert vars(arguments) == dict(eval_nsamples=None, bsz=1, capture_layer_io=False)
        start = sum(value.numel() for value in calls)
        assert torch.equal(encoding.input_ids, oracle.ids[:, start:start + encoding.input_ids.numel()])
        expected_length = 2048 if start < oracle.metadata["full_windows"] * 2048 else oracle.metadata["tail_tokens"]
        assert current.seqlen == expected_length
        calls.append(encoding.input_ids.clone())
        scores.append(torch.exp(torch.tensor(1 + len(calls) / 4, dtype=torch.float32)).item())
        return scores[-1]

    monkeypatch.setattr(external_eval, "evaluator", fake_evaluator)
    result = external_eval.evaluate_tokens(model, oracle.ids, 128)
    full = oracle.metadata["full_windows"]
    group_windows = [min(128, full - first) for first in range(0, full, 128)]
    lengths = [2048 * count for count in group_windows] + [oracle.metadata["tail_tokens"]]
    targets = [count * 2047 for count in group_windows] + [oracle.metadata["tail_tokens"] - 1]
    assert [value.numel() for value in calls] == lengths
    assert [segment["predicted_tokens"] for segment in result["segments"]] == targets
    assert torch.equal(torch.cat(calls, dim=1), oracle.ids)
    assert result["token_count"] == oracle.ids.numel()
    assert result["predicted_tokens"] == sum(targets) == oracle.metadata["predicted_tokens"]
    assert result["unscored_tail_tokens"] == 0 and model.seqlen == 2048
    expected_nll = math.fsum(math.log(score) * count for score, count in zip(scores, targets)) / sum(targets)
    assert result["nll"] == pytest.approx(expected_nll, rel=0, abs=1e-15)
    assert result["ppl"] == math.exp(result["nll"])
    assert abs(result["nll"] - sum(map(math.log, scores)) / len(scores)) > 1e-3
    assert abs(result["ppl"] - sum(scores) / len(scores)) > 1e-3


def test_wikitext2_test_paired_driver_saved_tokens_and_frozen_scales(
        external_tiny, wikitext_test_oracle, tmp_path, monkeypatch):
    fixture, oracle = external_tiny, wikitext_test_oracle
    package_bytes = fixture.package.read_bytes()
    original_load = external_eval.load_model
    loaded, calls = {}, {}

    def load(mode, package):
        model = original_load(mode, package)
        assert not model.training and not model.config.use_cache and model.seqlen == 2048
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        loaded[mode] = model
        calls[mode] = []
        return model

    def fake_evaluator(model, encoding, device, arguments):
        assert not torch.is_grad_enabled() and device == "cuda"
        mode = next(name for name, current in loaded.items() if current is model)
        start = sum(value.numel() for value in calls[mode])
        assert torch.equal(encoding.input_ids, oracle.ids[:, start:start + encoding.input_ids.numel()])
        calls[mode].append(encoding.input_ids.clone())
        return torch.exp(torch.tensor(1 + len(calls[mode]) / 4, dtype=torch.float32)).item()

    def reject_c4(*args, **kwargs):
        pytest.fail("--wikitext2-test must not enter the C4 token loader")

    monkeypatch.setattr(external_eval, "load_model", load)
    monkeypatch.setattr(external_eval, "load_c4_tokens", reject_c4)
    monkeypatch.setattr(external_eval, "evaluator", fake_evaluator)
    for mode in ("bf16", "quantized"):
        output = tmp_path / mode
        output.mkdir()
        args = SimpleNamespace(mode=mode, package=fixture.package if mode == "quantized" else None,
            tokens=None, wikitext2_test=True, output=output, chunk_windows=128)
        external_eval.run(args)
        data = json.loads((output / "data.json").read_text())
        result = json.loads((output / "result.json").read_text())
        saved = torch.load(output / "input_tokens.pt", map_location="cpu", weights_only=True)
        assert torch.equal(saved["input_ids"], oracle.ids)
        assert saved["metadata"] == data["input_metadata"] == result["input_metadata"]
        for name, value in oracle.metadata.items():
            assert saved["metadata"][name] == value, name
        for name in ("dataset", "subset", "split", "token_count", "predicted_tokens", "unscored_tail_tokens"):
            assert result[name] == oracle.metadata[name], name
        for record in (data, result):
            assert Path(record["input_token_path"]).resolve() == (output / "input_tokens.pt").resolve()
            assert record["reused_tokens"] is False
            assert record["training"] is False and record["calibration"] is False
        assert data["selection"] is False and result["candidate_selection"] is False
        assert result["activation_scales_unchanged"] is True
        assert result["package"] == (str(fixture.package) if mode == "quantized" else None)
        assert result["chunk_windows"] == 128
        before = torch.load(output / "quantizers_before.pt", map_location="cpu", weights_only=True)
        after = torch.load(output / "quantizers_after.pt", map_location="cpu", weights_only=True)
        assert before.keys() == after.keys() and len(before) == (112 if mode == "quantized" else 0)
        for name, record in before.items():
            assert record.keys() == after[name].keys()
            for key, value in record.items():
                assert torch.equal(value, after[name][key])
        expected_model = fixture.original if mode == "bf16" else fixture.frozen
        expected_state, actual_state = expected_model.state_dict(), loaded[mode].state_dict()
        assert expected_state.keys() == actual_state.keys()
        for name, value in expected_state.items():
            torch.testing.assert_close(actual_state[name], value, rtol=0, atol=0, equal_nan="quantizer." in name)
    assert len(calls["bf16"]) == len(calls["quantized"])
    assert all(torch.equal(left, right) for left, right in zip(calls["bf16"], calls["quantized"]))
    assert torch.equal(torch.cat(calls["bf16"], dim=1), oracle.ids)
    assert fixture.package.read_bytes() == package_bytes

    def mutate_scale(model, ids, chunk_windows):
        model.model.layers[0].mlp.down_proj.quantizer.scale.mul_(2)
        return dict(predicted_tokens=oracle.metadata["predicted_tokens"], unscored_tail_tokens=0)

    monkeypatch.setattr(external_eval, "load_model", lambda mode, package: loaded["quantized"])
    monkeypatch.setattr(external_eval, "evaluate_tokens", mutate_scale)
    args.output = tmp_path / "rejected-scale-mutation"
    args.output.mkdir()
    with pytest.raises(ValueError, match="Fixed-model state"):
        external_eval.run(args)
    assert not (args.output / "result.json").exists()
    assert fixture.package.read_bytes() == package_bytes


def test_wikitext2_test_singleton_tail_is_reported_without_scoring(tmp_path, monkeypatch):
    ids = torch.arange(2049, dtype=torch.long).reshape(1, -1)
    model = torch.nn.Module()
    model.register_buffer("inv_freq", torch.tensor([1.0], dtype=torch.float32))
    model.seqlen = 2048
    calls = []

    def fake_evaluator(current, encoding, device, arguments):
        assert current is model and current.seqlen == 2048 and device == "cuda"
        assert torch.equal(encoding.input_ids, ids[:, :2048])
        calls.append(encoding.input_ids)
        return 2.0

    monkeypatch.setattr(Dataset, "from_file", staticmethod(lambda path: {"text": ["synthetic test row"]}))
    monkeypatch.setattr(LlamaTokenizerFast, "from_pretrained", lambda *args, **kwargs:
        lambda *args, **kwargs: SimpleNamespace(input_ids=ids))
    monkeypatch.setattr(external_eval, "load_model", lambda mode, package: model)
    monkeypatch.setattr(external_eval, "evaluator", fake_evaluator)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    external_eval.run(SimpleNamespace(mode="bf16", package=None, tokens=None,
        output=tmp_path, chunk_windows=128))
    result = json.loads((tmp_path / "result.json").read_text())
    assert len(calls) == 1 and result["token_count"] == 2049
    assert result["predicted_tokens"] == 2047 and result["unscored_tail_tokens"] == 1
    assert result["input_metadata"]["tail_tokens"] == 1
    assert result["input_metadata"]["windows"] == result["input_metadata"]["full_windows"] == 1
    assert result["nll"] == math.log(2.0) and result["ppl"] == 2.0
    assert result["split"] == "test" and model.seqlen == 2048


def test_wikitext2_test_cli_required_exclusive_and_c4_compatible(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(external_eval, "run", lambda args: observed.append(args))
    monkeypatch.setattr(external_eval, "source_record", lambda: dict(test="WikiText-2 test CLI boundary"))
    monkeypatch.setattr(external_eval.subprocess, "check_output", lambda *args, **kwargs: "mock diff")
    monkeypatch.setattr(external_eval, "set_seed", lambda seed: None)
    package = tmp_path / "already-selected.pt"
    for name, selection in (("wikitext", ["--wikitext2-test"]), ("c4", ["--tokens", str(TOKEN_PATH)])):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["external_eval.py", "--output", str(output),
            "--mode", "quantized", "--package", str(package), *selection])
        external_eval.main()
        args = observed[-1]
        assert args.output == output and args.package == package.resolve()
        assert args.mode == "quantized" and args.chunk_windows == 128
        assert args.tokens == (None if name == "wikitext" else TOKEN_PATH.resolve())
        assert args.wikitext2_test is (name == "wikitext")
    assert len(observed) == 2
    for name, selection in (("neither", []), ("both", ["--tokens", str(TOKEN_PATH), "--wikitext2-test"])):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["external_eval.py", "--output", str(output),
            "--mode", "bf16", *selection])
        with pytest.raises(SystemExit) as error:
            external_eval.main()
        assert error.value.code == 2
        assert not output.exists() and len(observed) == 2
