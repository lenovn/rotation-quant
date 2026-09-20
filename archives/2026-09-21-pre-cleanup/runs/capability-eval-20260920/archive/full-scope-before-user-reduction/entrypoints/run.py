"""Frozen FIRON packages through the official lm-eval task and scoring pipeline."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/capability-eval-20260920'
SOURCE = ROOT / 'worktrees/SpinQuant-multimodel'
sys.path[:0] = [str(RUN / 'deps'), str(SOURCE)]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['qwen', 'llama'], required=True)
    p.add_argument('--method', choices=['bf16', 'firon', 'parent', 'w4a16'], required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--batch-size', type=int, default=4, help='Scoring batch size; generation uses fixed per-family/task batches')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    task_lock = (args.output / '.running.lock').open('a')
    try:
        fcntl.flock(task_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Task already running in another worker; preserve its output', flush=True)
        return
    if (args.output / 'results.json').exists():
        raise FileExistsError('Completed result already exists')
    generation_path = args.output / ('generation_batches-%03d.jsonl' % (len(list(args.output.glob('generation_batches-*.jsonl'))) + 1))
    registry = json.loads((RUN / 'models.json').read_text())
    selected = registry[args.model][args.method]
    os.environ['PHASE5_MODEL_PATH'] = selected['model_path']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if args.task == 'mmlu':
        # All 57 official configurations were downloaded by prepare_mmlu.py.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    import torch
    from transformers import set_seed
    from experiments.phase3.architecture import load_model, tokenizer
    from experiments.phase3.postprocess import load_static
    from experiments.phase3.common import wrappers
    from experiments.phase3.external_eval import quantizer_snapshot
    from lm_eval.models.huggingface import HFLM
    import lm_eval

    class FrozenHFLM(HFLM):
        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return self.tokenizer.apply_chat_template(chat_history, tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt, enable_thinking=False)

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            output = super()._model_generate(context, max_length, stop, **generation_kwargs)
            # Preserve completed generation batches even if a downstream scorer fails.
            with generation_path.open('a') as stream:
                for prompt, row in zip(context.detach().cpu().tolist(), output.detach().cpu().tolist()):
                    continuation = row[len(prompt):]
                    stream.write(json.dumps(dict(prompt_token_ids=prompt,
                        generated_token_ids=continuation, max_new_tokens=max_length-len(prompt),
                        stop=stop, eos_token_ids=self.model.generation_config.eos_token_id, raw_text=self.tokenizer.decode(continuation, skip_special_tokens=False)),
                        ensure_ascii=False) + '\n')
            return output

        def _model_call(self, inps, attn_mask=None, labels=None):
            with torch.no_grad():
                return self.model(input_ids=inps, attention_mask=attn_mask,
                                  use_cache=False).logits

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    started = time.time()
    generation = args.task in ('gsm8k_cot', 'ifeval')
    effective_batch_size = (64 if args.model == 'qwen' and args.task == 'ifeval' else 16) if generation else args.batch_size
    settings = dict(arguments=vars(args), effective_batch_size=effective_batch_size, selected=selected, seed=42,
        harness_version='0.4.8', generation_chat_template=generation,
        enable_thinking=False, fewshot_as_multiturn=False,
        num_fewshot=5 if args.task == 'mmlu' else 8 if args.task == 'gsm8k_cot' else 0,
        max_length=8192, generation_max_new_tokens=256 if args.task == 'gsm8k_cot' else 1280 if generation else None,
        scoring_use_cache=False, generation_use_cache=True, kv_bits=16,
        calibration=False, training=False, cuda_allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), source=str(SOURCE), generation_log=str(generation_path) if generation else None)
    save(args.output / 'settings.json', settings)
    try:
        if args.method == 'bf16':
            model = load_model(selected['model_path']).cuda()
        elif args.method == 'w4a16':
            from w4a16 import load_w4a16
            model, w4_record = load_w4a16(selected['model_path'], selected['package'])
            save(args.output / 'w4a16_loading.json', w4_record)
        else:
            model, records = load_static(Path(selected['package']))
            del records
        model.requires_grad_(False).eval()
        quantizers = wrappers(model)
        before = quantizer_snapshot(model)
        counts = {name: Counter() for name in quantizers}
        handles = []
        for name, wrapper in quantizers.items():
            def record(module, inputs, name=name):
                counts[name]['decode' if inputs[0].shape[-2] == 1 else 'prefill'] += 1
                if module.bits != 8 or getattr(module, 'observing', False):
                    raise RuntimeError('Frozen activation quantization disabled: ' + name)
            handles.append(wrapper.quantizer.register_forward_pre_hook(record))
        tok = tokenizer(selected['model_path'])
        if model.config.model_type == 'llama':
            # Legacy LlamaTokenizerFast adds an out-of-vocabulary <unk>; never use it as padding.
            tok.pad_token_id = tok.eos_token_id
        save(args.output / 'tokenizer_runtime.json', dict(pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, unk_token_id=tok.unk_token_id, embedding_rows=model.get_input_embeddings().weight.shape[0]))
        lm = FrozenHFLM(pretrained=model, tokenizer=tok, backend='causal',
                        batch_size=effective_batch_size, max_length=8192, add_bos_token=False)
        results = lm_eval.simple_evaluate(model=lm, tasks=[args.task],
            num_fewshot=settings['num_fewshot'], batch_size=effective_batch_size,
            apply_chat_template=generation, fewshot_as_multiturn=False,
            log_samples=True, bootstrap_iters=0,
            random_seed=42, numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42)
        after = quantizer_snapshot(model)
        unchanged = before.keys() == after.keys() and all(
            before[n].keys() == after[n].keys() and all(torch.equal(v, after[n][k]) for k,v in before[n].items())
            for n in before)
        if not unchanged or any(not c['prefill'] or (generation and not c['decode']) for c in counts.values()):
            raise RuntimeError('Quantizer state/coverage verification failed')
        save(args.output / 'quantization.json', dict(scales_unchanged=unchanged,
            quantizer_calls=counts, backbone_linears=len(quantizers),
            weight_source=selected.get('package'), elapsed_seconds=time.time()-started))
        samples = results.pop('samples', {})
        for task, rows in samples.items():
            with (args.output / ('samples_' + task + '.jsonl')).open('w') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        save(args.output / 'results.json', results)
    except BaseException as e:
        save(args.output / 'failure.json', dict(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
        raise

if __name__ == '__main__':
    main()
