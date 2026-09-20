"""Independent Phase5 Uniform protocol checks; synthetic CPU tensors only."""
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.phase3 import common, distill, sequential_postprocess as seq, uniform_baseline as uniform
from experiments.phase3 import architecture
from experiments.phase3.quantization import SP2Quantizer, pack_int4
from utils.quant_utils import ActQuantWrapper, RotationStaticActQuantizer


class Tiny(nn.Module):
    def __init__(self, down_format):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1, model_type='synthetic', to_dict=lambda: {'num_hidden_layers': 1})
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.mlp = nn.Module()
        self.high = nn.Parameter(torch.tensor([3.0]))
        for family, names in [(layer.self_attn, ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                              (layer.mlp, ['gate_proj', 'up_proj', 'down_proj'])]:
            for name in names:
                wrapper = ActQuantWrapper(nn.Linear(4, 4, bias=False))
                wrapper.quantizer = (SP2Quantizer(2.54) if name == 'down_proj' and down_format == 'sp2'
                                     else RotationStaticActQuantizer())
                if isinstance(wrapper.quantizer, RotationStaticActQuantizer):
                    wrapper.quantizer.load_scale(torch.tensor([0.02]))
                family.add_module(name, wrapper)
        self.checkpointing = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing = True

    def cuda(self):
        return self


def records_for(model):
    records = {}
    for index, (name, wrapper) in enumerate(common.wrappers(model).items()):
        codes = ((torch.arange(16).reshape(4, 4) + index) % 16 - 8).to(torch.int8)
        scale = torch.full((4, 1), 0.03 + index * 0.01)
        wrapper.module.weight.data.copy_(codes.float() * scale)
        records[name] = {'packed': pack_int4(codes), 'shape': tuple(codes.shape), 'scale': scale}
    return records


@pytest.mark.parametrize('fmt', ['sp2', 'int8'])
def test_cold_package_keeps_actual_format_and_forward(tmp_path, fmt):
    source = Tiny(fmt)
    records = records_for(source)
    path = tmp_path / 'package.pt'
    common.save_frozen(source, records, path, {'parent': 'synthetic'})
    saved = torch.load(path, weights_only=True)
    name = 'model.layers.0.mlp.down_proj'
    assert saved['activation'][name]['format'] == fmt
    target = Tiny('int8' if fmt == 'sp2' else 'sp2')
    common.reload_frozen(target, path)
    values = torch.tensor([[-4.0, -0.17, 0.021, 1.23], [0.001, 0.09, 2.7, -1.8]])
    for name, wrapper in common.wrappers(source).items():
        other = target.get_submodule(name)
        assert type(wrapper.quantizer) is type(other.quantizer)
        torch.testing.assert_close(wrapper(values), other(values), rtol=0, atol=0)
    torch.testing.assert_close(target.high, source.high, rtol=0, atol=0)
    assert common.format_description(target) == ('7 W4 / 6 static INT8 / 1 static SP2; BF16 KV16 prefill'
        if fmt == 'sp2' else '7 W4 / 7 static INT8; BF16 KV16 prefill')


@pytest.mark.parametrize('fmt', ['sp2', 'int8'])
def test_qat_uses_actual_down_quantizer_and_learns_scale(fmt):
    model = Tiny(fmt)
    records = records_for(model)
    name = 'model.layers.0.mlp.down_proj'
    before_type = type(model.get_submodule(name).quantizer)
    distill.prepare_student(model, records)
    wrapper = model.get_submodule(name)
    assert type(wrapper.quantizer) is before_type
    assert model.checkpointing and not model.high.requires_grad
    values = torch.tensor([[-4.3, -0.173, 0.027, 1.234], [0.017, 0.094, 2.77, -1.81]])
    loss = wrapper(values).square().sum()
    loss.backward()
    assert wrapper.quantizer.scale.requires_grad
    assert wrapper.quantizer.scale.grad is not None
    assert torch.isfinite(wrapper.quantizer.scale.grad).all()
    assert wrapper.quantizer.scale.grad.abs().sum() > 0
    assert wrapper.module.weight.grad is not None
    groups = distill.parameter_groups(model)
    assert any(parameter is wrapper.quantizer.scale for _, parameter in groups['SP2'])
    # Historical optimizer-group name does not choose the forward quantizer.
    optimizers = distill.make_optimizers(groups, SimpleNamespace(weight_lr=1e-5, weight_optimizer='adam', relative_scale_lr=.001))
    before = wrapper.quantizer.scale.detach().clone()
    for optimizer in optimizers:
        optimizer.step()
    assert not torch.equal(before, wrapper.quantizer.scale)


@pytest.mark.parametrize('fmt', ['sp2', 'int8'])
def test_neighbor_capture_observes_actual_quantized_input(monkeypatch, fmt):
    model = Tiny(fmt)
    name = 'model.layers.0.mlp.down_proj'
    wrapper = model.get_submodule(name)
    windows = [torch.tensor([[i, i+1]]) for i in range(4)]
    raw = []
    def forward(current, ids):
        values = torch.tensor([[[-4.1, .013, .191, 1.337], [.29, -.09, 2.57, -1.27]]]) + ids[0, 0] / 100
        raw.append(values)
        return wrapper(values)
    monkeypatch.setattr(seq, 'backbone', forward)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    fit, heldout = seq.capture_module_inputs(model, name, windows)
    expected = [wrapper.quantizer(values).reshape(-1, 4) for values in raw]
    assert torch.equal(fit, torch.cat(expected[:3]))
    assert torch.equal(heldout, expected[3])
    assert not torch.equal(fit, torch.cat(raw[:3]).reshape(-1, 4))
    assert not wrapper.module._forward_pre_hooks


@pytest.mark.parametrize('fmt', ['sp2', 'int8'])
def test_range_search_scores_actual_operator_and_restores_parent(monkeypatch, tmp_path, fmt):
    model = Tiny(fmt)
    name = 'model.layers.0.mlp.down_proj'
    wrapper = model.get_submodule(name)
    original_type = type(wrapper.quantizer)
    values = torch.tensor([[-3.17, .17, .027, 1.234]])
    scales, outputs = [], []
    def score(current, windows):
        assert type(wrapper.quantizer) is original_type
        scales.append(float(wrapper.quantizer.scale))
        outputs.append(wrapper.quantizer(values).clone())
        return 20 + len(scales)  # reject every contraction, preserve parent
    monkeypatch.setattr(seq, 'selection_score', score)
    monkeypatch.setattr(seq, 'progress', lambda *a, **k: None)
    original = wrapper.quantizer.scale.clone()
    args = SimpleNamespace(output=tmp_path, alpha_factors=[1, .875, .75, .5, .25, .125])
    selected, result = seq.range_module(model, name, [torch.tensor([[1, 2]])], 10., args)
    assert selected == 10 and result['selected_factor'] == 1
    assert torch.equal(wrapper.quantizer.scale, original)
    assert len(scales) == 5
    for factor, scale, output in zip(args.alpha_factors[1:], scales, outputs):
        assert scale == pytest.approx(float(original) * factor)
        oracle = original_type(2.54 * factor) if fmt == 'sp2' else RotationStaticActQuantizer()
        if fmt == 'int8': oracle.load_scale(torch.tensor([float(original) * factor]))
        torch.testing.assert_close(output, oracle(values), rtol=1e-6, atol=1e-7)


def test_uniform_calibration_only_reads_train_and_declared_indices(monkeypatch, tmp_path):
    from datasets import Dataset
    data_path = tmp_path / 'data'
    model_path = tmp_path / 'model'
    monkeypatch.setattr(uniform, 'DATA_PATH', data_path)
    monkeypatch.setattr(uniform, 'MODEL_PATH', model_path)
    reads = []
    def read(filename):
        reads.append(Path(filename).name)
        assert Path(filename) == data_path / 'wikitext-train.arrow'
        return {'text': ['train row one', 'train row two']}
    class Tokenizer:
        def __call__(self, rows, **kwargs):
            assert rows == ['train row one', 'train row two']
            assert kwargs == {'add_special_tokens': False}
            return {'input_ids': [list(range(2048)), list(range(2048, 4096))]}
    monkeypatch.setattr(Dataset, 'from_file', staticmethod(read))
    monkeypatch.setattr(architecture, 'tokenizer', lambda path: Tokenizer())
    metadata = dict(model_path=str(model_path), dataset_cache=str(data_path), train_tokens=4096,
                    train_windows=2, calibration_length=128, calibration_window_indices=[1, 0], seed=42,
                    train_protocol='per row no special tokens, concatenate')
    path = tmp_path / 'data.json'
    path.write_text(json.dumps(metadata))
    windows, actual = uniform.load_calibration(path)
    assert reads == ['wikitext-train.arrow'] and actual['split'] == 'train'
    assert windows[0].tolist() == [list(range(2048, 2176))]
    assert windows[1].tolist() == [list(range(128))]
    assert actual['calibration_tokens'] == 256


def test_fixed_qat400_global_token_budget_and_no_target_stop(monkeypatch, tmp_path):
    model = Tiny('int8')
    records = records_for(model)
    train = torch.arange(64).view(-1, 1).expand(-1, 2048)
    sampled, checkpoints, logs = [], [], []
    monkeypatch.setattr(distill, 'data_windows', lambda output: (train, [], ['probe'], ['validation'], {}))
    monkeypatch.setattr(distill, 'load_static', lambda parent: (model, records))
    monkeypatch.setattr(distill, 'teacher_model', lambda: Tiny('int8'))
    monkeypatch.setattr(distill, 'place_student', lambda *a: {'cpu_test': True})
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(distill, 'evaluate', lambda *a: {})
    def loss(student, teacher, ids, *a, **kwargs):
        sampled.append(int(ids[0, 0]))
        value = sum(w.module.weight.square().mean() + w.quantizer.scale.square().mean()
                    for w in common.wrappers(student).values())
        return value, value.detach(), value.detach()
    monkeypatch.setattr(distill, 'distillation_loss', loss)
    monkeypatch.setattr(distill, 'progress', lambda *a, **k: None)
    monkeypatch.setattr(distill, 'save_resume', lambda *a: None)
    monkeypatch.setattr(distill, 'append_json', lambda path, row: logs.append(row))
    def checkpoint(*args):
        checkpoints.append(args[4]); return {'validation_ppl': 0.0}
    monkeypatch.setattr(distill, 'evaluate_checkpoint', checkpoint)
    args = SimpleNamespace(output=tmp_path, parent=tmp_path/'parent.pt', reference_state=None, resume=None,
        steps=400, schedule_steps=400, accumulation=8, warmup=10, weight_lr=1e-5, weight_optimizer='adam',
        relative_scale_lr=.001, data_start=800, temperature=1., ce_weight=.1, checkpoints=[100, 200, 400],
        resume_every=25, target_ppl=None)
    distill.train(args)
    result = json.loads((tmp_path/'result.json').read_text())
    assert result['completed_steps'] == 400 and not result['target_reached']
    assert len(sampled) == 3200 and sampled == [(800+i) % 64 for i in range(3200)]
    assert checkpoints == [100, 200, 400]
    assert logs[-1]['cumulative_train_tokens'] == 6553600
    assert all(len(row['microbatches']) == 8 for row in logs)
    tree = ast.parse(inspect.getsource(distill.main))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and node.args
             and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--target-ppl']
    assert len(calls) == 1
    assert next(keyword.value.value for keyword in calls[0].keywords if keyword.arg == 'default') is None


def test_uniform_export_package_preserves_all_weights_and_actual_int8(monkeypatch, tmp_path):
    model = Tiny('sp2')
    records = records_for(model)
    parent = tmp_path / 'parent.pt'
    common.save_frozen(model, records, parent, {})
    name = 'model.layers.0.mlp.down_proj'
    values = torch.tensor([[-1.2, .031, .271, .9], [.123, -.197, .561, -2.7]])
    metadata = {'dataset': 'Salesforce/wikitext', 'split': 'train'}
    expected_path = tmp_path / 'sp2.json'
    expected_path.write_text(json.dumps({name: {'sampled_rows': 2, 'full_absmax': float(values.abs().max()),
                                              'candidates': [{}] * 50, 'selected': {'alpha': 2.54}}}))
    monkeypatch.setattr(uniform, 'load_calibration', lambda path: (['train-only'], metadata))
    monkeypatch.setattr(uniform, 'load_static', lambda path: (model, records))
    monkeypatch.setattr(uniform, 'capture_down_inputs', lambda *a: ({name: values}, {name: float(values.abs().max())}))
    monkeypatch.setattr(uniform, 'progress', lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    output = tmp_path / 'export'
    output.mkdir()
    uniform.run(SimpleNamespace(output=output, parent=parent, calibration_data=tmp_path/'data.json',
                                sp2_calibration=expected_path, export_package=True))
    package = output / 'static_w4a8.pt'
    cold = Tiny('sp2')
    loaded = common.reload_frozen(cold, package)
    assert all(record['format'] == 'int8' for record in loaded['activation'].values())
    old = torch.load(parent, weights_only=True)
    for key, record in loaded['weights'].items():
        for field in ['packed', 'scale']:
            assert torch.equal(record[field], old['weights'][key][field])
    for key in old['activation']:
        if key != name:
            assert torch.equal(old['activation'][key]['scale'], loaded['activation'][key]['scale'])
    overlay = torch.load(output/'down_int8_scales.pt', weights_only=True)
    assert torch.equal(cold.get_submodule(name).quantizer.scale, overlay['scales'][name])
    assert json.loads((output/'result.json').read_text())['candidates_per_layer'] == 50


@pytest.mark.parametrize('fmt', ['sp2', 'int8'])
def test_external_records_actual_package_format_without_overlay(monkeypatch, tmp_path, fmt):
    from experiments.phase3 import external_eval
    model = Tiny(fmt)
    records_for(model)
    model.register_buffer('inv_freq', torch.ones(2, dtype=torch.float32))
    ids = torch.tensor([[1, 2, 3]])
    metadata = dict(dataset='allenai/c4', subset='en', split='validation', windows=1, predicted_tokens=2)
    monkeypatch.setattr(external_eval, 'load_c4_tokens', lambda path: (ids, metadata))
    monkeypatch.setattr(external_eval, 'load_model', lambda *a: model)
    monkeypatch.setattr(external_eval, 'evaluate_tokens', lambda *a: dict(predicted_tokens=2, unscored_tail_tokens=0, ppl=2., nll=.693))
    monkeypatch.setattr(external_eval, 'progress', lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    external_eval.run(SimpleNamespace(output=tmp_path, tokens=tmp_path/'input.pt', mode='quantized',
                                     package=tmp_path/'package.pt', chunk_windows=128, down_int8_scales=None))
    result = json.loads((tmp_path/'result.json').read_text())
    expected = {'int8': 6, 'sp2': 1} if fmt == 'sp2' else {'int8': 7}
    assert result['activation_formats'] == expected
    assert result['down_int8_overlay'] is None and result['activation_scales_unchanged']
    assert not result['training'] and not result['calibration'] and not result['candidate_selection']
