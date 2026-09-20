"""Independent tiny actual-QAT two-device forward/backward equivalence."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
sys.path.insert(0, str(ROOT / 'worktrees/SpinQuant-multimodel'))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from experiments.phase3 import common, distill
from experiments.phase3.placement import place_student

torch.set_num_threads(2)
torch.manual_seed(123)
with tempfile.TemporaryDirectory(prefix='phase5-placement-') as directory:
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, attention_dropout=0.)
    Qwen3ForCausalLM(config).bfloat16().save_pretrained(directory)
    common.MODEL_PATH = Path(directory)
    ids = torch.tensor([[1,4,8,3,7,2,9,5]])
    training = common.build_training_model()
    common.initialize_scales(training, [ids])
    frozen, records = common.frozen_model(training)
    for name, wrapper in common.wrappers(frozen).items():
        if name.endswith('down_proj'):
            wrapper.quantizer.bits = 8
    distill.prepare_student(frozen, records)
    results = []
    for devices in (1, 2):
        model = copy.deepcopy(frozen)
        placement = place_student(model, devices)
        model.train()
        value = common.token_nll(model, ids.cuda())
        value.backward()
        gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        results.append((float(value), gradients))
        print('devices', devices, 'loss', float(value), 'gradients', len(gradients), flush=True)
        del model, value
        torch.cuda.empty_cache()
    torch.testing.assert_close(torch.tensor(results[0][0]), torch.tensor(results[1][0]), rtol=0, atol=0)
    assert results[0][1].keys() == results[1][1].keys()
    maximum = 0.
    for name in results[0][1]:
        a, b = results[0][1][name], results[1][1][name]
        maximum = max(maximum, float((a-b).abs().max()))
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
    report = dict(status='PASS', loss=results[0][0], learned_gradient_tensors=len(results[0][1]),
        maximum_gradient_abs_difference=maximum, placement=placement,
        scope='tiny Qwen GQA W4/static INT8/SP2 QAT with nonreentrant checkpoint recomputation')
    (ROOT / 'runs/phase5/verifier/placement_gpu.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
