import argparse
from collections import Counter
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
SOURCE = ROOT / 'worktrees/SpinQuant-phase3-joint'
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

import torch
from datasets import Dataset
from transformers import AutoConfig, LlamaConfig, LlamaTokenizerFast, set_seed
from eval_utils.modeling_llama import LlamaForCausalLM
from experiments.phase3.common import MODEL_PATH, DATA_PATH, reload_frozen, wrappers
from experiments.phase3.quantization import SP2Quantizer, unpack_int4
from utils.eval_utils import evaluator
from utils.quant_utils import ActQuantWrapper, RotationStaticActQuantizer, add_actquant


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def snapshot_quantizers(model):
    return {name: {key: value.detach().cpu().clone()
                  for key, value in wrapper.quantizer.state_dict().items()}
            for name, wrapper in wrappers(model).items()}


def exact_saved_tensor(actual, saved, allow_nan=False):
    try:
        torch.testing.assert_close(actual, saved, rtol=0, atol=0, equal_nan=allow_nan)
        return True
    except AssertionError:
        return False


@torch.no_grad()
def load_candidate(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    config = LlamaConfig.from_dict(saved['config'])
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.bfloat16,
                                         attn_implementation='sdpa')
    add_actquant(model)
    for name, wrapper in wrappers(model).items():
        wrapper.quantizer = (SP2Quantizer(saved['activation'][name]['alpha'])
                             if name.endswith('down_proj') else RotationStaticActQuantizer())
    del saved
    state = reload_frozen(model, path)
    weight_matches = {}
    for name, record in state['weights'].items():
        reconstructed = (unpack_int4(record['packed'], record['shape']).float()
                         * record['scale']).to(torch.bfloat16)
        weight_matches[name] = torch.equal(model.get_submodule(name).module.weight, reconstructed)
    high_precision_matches = {name: exact_saved_tensor(model.state_dict()[name], tensor,
                                                      allow_nan='quantizer.' in name)
                              for name, tensor in state['high_precision'].items()}
    boundaries = {name: dict(input_bits=wrapper.quantizer.bits,
                            format=state['activation'][name]['format'],
                            output_bits=wrapper.out_quantizer.bits,
                            online_full_had=wrapper.online_full_had,
                            online_partial_had=wrapper.online_partial_had,
                            scale_shape=list(wrapper.quantizer.scale.shape),
                            observing=getattr(wrapper.quantizer, 'observing', False))
                  for name, wrapper in wrappers(model).items()}
    assert len(weight_matches) == 112 and all(weight_matches.values())
    assert all(high_precision_matches.values())
    assert Counter(row['format'] for row in boundaries.values()) == {'int8': 96, 'sp2': 16}
    assert all(row['input_bits'] == 8 and row['output_bits'] == 16
               and not row['online_full_had'] and not row['online_partial_had']
               and not row['observing'] for row in boundaries.values())
    evidence = dict(checkpoint=str(path), metadata=state['metadata'], cold_load=True,
                    pretrained_weights_loaded=False, trained_or_recalibrated=False,
                    packed_weight_equality=weight_matches,
                    high_precision_equality=high_precision_matches, boundaries=boundaries,
                    rotation='R1/R2 already fused in saved weights; no live R; R3/R4 off')
    return model, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['bf16', 'c-adam10'])
    options = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    runtime = dict(mode=options.mode, pid=os.getpid(), command=sys.argv,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), started_at=time.time(),
                   model_path=str(MODEL_PATH), model_revision=(MODEL_PATH / '.mv').read_text())
    write_json(OUTPUT / (options.mode + '.runtime.json'), runtime)
    print('AUDITOR_START ' + json.dumps(runtime), flush=True)
    if options.mode == 'bf16':
        config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        original_tie = config.tie_word_embeddings
        config.tie_word_embeddings = False
        model = LlamaForCausalLM.from_pretrained(str(MODEL_PATH), config=config,
                    torch_dtype=torch.bfloat16, local_files_only=True)
        if original_tie:
            model.lm_head.weight.data = model.model.embed_tokens.weight.detach().clone()
        assert not any(isinstance(module, ActQuantWrapper) for module in model.modules())
        load_evidence = dict(original_bf16=True, rotation=False, norm_fusion=False,
                             original_tie_word_embeddings=original_tie,
                             lm_head_matches_embedding=torch.equal(model.lm_head.weight,
                                                                    model.model.embed_tokens.weight),
                             quantized_linear_count=0)
    else:
        path = ROOT / 'runs/phase3/route-c-adam-100-20260914a/checkpoint-0010/static_w4a8.pt'
        model, load_evidence = load_candidate(path)
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    model.seqlen = 2048
    load_evidence.update(attention_class=type(model.model.layers[0].self_attn).__name__,
                         parameter_dtypes=dict(Counter(str(parameter.dtype) for parameter in model.parameters())),
                         rotary_buffer_dtypes={name: str(value.dtype) for name, value in model.named_buffers()
                                               if name.endswith('inv_freq')},
                         kv_bits=16, use_cache=False, seed=42)
    write_json(OUTPUT / (options.mode + '.load.json'), load_evidence)
    tokenizer = LlamaTokenizerFast.from_pretrained(str(MODEL_PATH), local_files_only=True,
        model_max_length=2048, padding_side='right', use_fast=True,
        add_eos_token=False, add_bos_token=False)
    dataset = Dataset.from_file(str(DATA_PATH / 'wikitext-validation.arrow'))
    encoding = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt')
    token_path = OUTPUT / 'validation_input_tokens.pt'
    if token_path.exists():
        assert torch.equal(encoding.input_ids, torch.load(token_path, weights_only=True))
    else:
        torch.save(encoding.input_ids, token_path)
    assert encoding.input_ids.numel() == 252852
    acceptance_path = ROOT / 'scripts/phase2/validation_acceptance.py'
    specification = importlib.util.spec_from_file_location('auditor_acceptance', acceptance_path)
    acceptance = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(acceptance)
    from types import SimpleNamespace
    arguments = SimpleNamespace(eval_nsamples=None, bsz=1, capture_layer_io=False)
    modules = {name: module.__file__ for name, module in tuple(sys.modules.items())
               if getattr(module, '__file__', None)
               and (str(SOURCE) in module.__file__ or name == 'auditor_acceptance')}
    evaluator_source = inspect.getsourcefile(inspect.unwrap(evaluator))
    assert evaluator_source == str(SOURCE / 'utils/eval_utils.py')
    assert not any('repos/SpinQuant/' in str(getattr(module, '__file__', ''))
                   for module in tuple(sys.modules.values()))
    write_json(OUTPUT / (options.mode + '.modules.json'), modules)
    before = snapshot_quantizers(model)
    model.cuda()
    torch.cuda.reset_peak_memory_stats()
    evaluation_started = time.monotonic()
    result = acceptance.evaluate_full_validation(model, encoding, 'cuda', arguments, evaluator)
    after = snapshot_quantizers(model)
    unchanged = all(torch.equal(value, after[name][key])
                    for name, record in before.items() for key, value in record.items())
    assert unchanged and result['predicted_tokens'] == 252728
    result.update(runtime, evaluation_seconds=time.monotonic() - evaluation_started,
                  total_seconds=time.monotonic() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                  acceptance_source=str(acceptance_path), evaluator_source=evaluator_source,
                  precision='BF16 logits CE reduction none; losses to FP32; float32 segment PPL then log; target-weighted NLL',
                  activation_scales_unchanged=unchanged if before else None,
                  input_tokens=str(token_path), status='COMPLETED')
    write_json(OUTPUT / (options.mode + '.result.json'), result)
    print('AUDITOR_RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
