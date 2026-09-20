"""Independent checks for the B512 budget and checkpoint export boundary."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.phase3 import budget, common, run
from test_phase3_joint import (assert_state_unchanged, calibration, cloned_state,
                               initialized_model, tiny_loader)


def test_reference_windows_preserve_observed_order_and_exclude_probe(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text('\n'.join(json.dumps(dict(window_indices=indices))
                              for indices in ([7, 3], [7, 12])))
    assert budget.reference_window_indices(tmp_path, 20, [18, 19]) == [7, 3, 12]
    with pytest.raises(ValueError, match="overlap"):
        budget.reference_window_indices(tmp_path, 20, [12])
    with pytest.raises(ValueError, match="outside"):
        budget.reference_window_indices(tmp_path, 12, [18, 19])


def test_actual_training_loop_reuses_fixed_pool_and_records_original_indices(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    parameter = model.weight
    groups = dict(R=[("weight", parameter)], SA=[], SW=[], SP2=[])
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    optimizer.param_groups[0].update(initial_lr=0.01, name="R")
    original_indices = [7, 3, 12]
    windows = torch.tensor(original_indices).reshape(-1, 1)
    observed, rows, checkpoints = [], [], []
    args = SimpleNamespace(initial=Path("initial.pt"), resume=None, scale_only_steps=False,
                           route="B", switch_step=50, output=tmp_path, steps=512,
                           schedule_steps=512, warmup=10, accumulation=8,
                           checkpoints=[100, 256, 512], gradient_window_indices=original_indices,
                           training_reference=None)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(run, "build_training_model", lambda initial: model)
    monkeypatch.setattr(run, "learned_parameters", lambda current: groups)
    monkeypatch.setattr(run, "make_optimizers", lambda current, arguments: [optimizer])
    monkeypatch.setattr(run, "training_mode", lambda *args: None)
    monkeypatch.setattr(run, "gradient_record", lambda *args: {})
    monkeypatch.setattr(run, "save_resume", lambda *args: None)
    monkeypatch.setattr(run, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "append_json", lambda path, row: rows.append(row))

    def loss(current, ids):
        observed.append(int(ids.item()))
        return current.weight.square().sum()

    def evaluate(current, arguments, step, *data):
        checkpoints.append(step)
        return dict(step=step, training_probe=1.0)

    monkeypatch.setattr(run, "token_nll", loss)
    monkeypatch.setattr(run, "evaluate_checkpoint", evaluate)
    run.train(args, windows, [], [], [])
    assert observed == (original_indices * 1366)[:4096]
    assert [index for row in rows for index in row["window_indices"]] == observed
    assert checkpoints == [0, 100, 256, 512]
    assert rows[-1]["cumulative_train_tokens"] == 8388608
    assert rows[99]["learning_rates"]["R"] == 0.01 * run.schedule(99, 512, 10)
    assert rows[99]["learning_rates"]["R"] != 0.01 * run.schedule(99, 100, 10)


def test_comparison_keeps_old_terminal_and_new_intermediate_separate(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "settings.json").write_text(json.dumps(dict(arguments=dict(steps=100, schedule_steps=100))))
    old_results = [dict(step=100, training_probe=2.9, validation_nll=2.84, validation_ppl=17.1)]
    (old / "results.json").write_text(json.dumps(old_results))
    checkpoints = [dict(step=step, training_probe=2.8) for step in (100, 256, 512)]
    budget.write_comparison(new, old, checkpoints, 512, 8)
    rows = json.loads((new / "comparison.json").read_text())["rows"]
    assert [row["position"] for row in rows] == ["old_schedule_final", "new_schedule_intermediate",
                                               "new_schedule_intermediate", "new_schedule_final"]
    assert [row["schedule_steps"] for row in rows] == [100, 512, 512, 512]
    assert rows[0]["validation_ppl"] == 17.1 and "validation_ppl" not in rows[1]
    assert rows[-1]["cumulative_train_tokens"] == 8388608
    assert "未评测" in (new / "COMPARISON.md").read_text()


def test_checkpoint_export_preserves_learned_training_object(initialized_model, calibration, tmp_path, monkeypatch):
    model = initialized_model
    common.training_mode(model, "B", 100, 50)
    model.train()
    with torch.no_grad():
        for values in common.learned_parameters(model).values():
            for name, parameter in values:
                if name.endswith(".scale"):
                    parameter.mul_(0.81)
    before = cloned_state(model)
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    requirements = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    args = SimpleNamespace(output=tmp_path, route="B", steps=512, switch_step=50, accumulation=8,
                           initial=Path("initial.pt"), validation_steps=[100])
    evaluated = []
    monkeypatch.setattr(run, "progress", lambda *args, **kwargs: None)

    def validation(frozen, windows):
        assert frozen is not model and windows is calibration
        assert all(wrapper.quantizer.bits == 8 for wrapper in common.wrappers(frozen).values())
        evaluated.append(frozen)
        return common.evaluate(frozen, windows)

    monkeypatch.setattr(run, "full_validation", validation)
    result = run.evaluate_checkpoint(model, args, 100, calibration, calibration, calibration)
    assert len(evaluated) == 1 and result["step"] == 100
    assert model.training
    assert_state_unchanged(model, before)
    assert identities == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert requirements == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert all(wrapper.quantizer.bits == (16 if name.endswith("down_proj") else 8)
               for name, wrapper in common.wrappers(model).items())
    package = torch.load(tmp_path / "checkpoint-0100/static_w4a8.pt", weights_only=True)
    for name, wrapper in common.wrappers(model).items():
        assert torch.equal(package["weights"][name]["scale"], wrapper.module.quantizer.scale)
        if not name.endswith("down_proj"):
            assert torch.equal(package["activation"][name]["scale"], wrapper.quantizer.scale)
    records = json.loads((tmp_path / "checkpoint-0100/sp2_calibration.json").read_text())
    assert len(records) == 16 and all(len(record["candidates"]) == 50 for record in records.values())
