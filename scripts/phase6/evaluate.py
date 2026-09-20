"""Fixed Phase6 package/BF16 evaluation, reusing the existing task protocols."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / 'runs/capability-eval-20260920'
sys.path.insert(0, str(LEGACY / 'deps'))
import joint as j
import torch
from experiments.phase3.external_eval import load_wikitext2_test_tokens, evaluate_tokens, quantizer_snapshot

TASKS = ['boolq', 'piqa', 'social_iqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']


def c4_tokens():
    name = j.c.MODEL_PATH.name
    old = {'llama-3.2-1b-instruct': ROOT/'runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/input_tokens.pt',
           'qwen3-1.7b': ROOT/'runs/phase5/qwen3-1p7b/c4-data/input_tokens.pt'}
    path = old.get(name, ROOT/'runs/phase6/data'/name/'c4.pt')
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        return state['input_ids'], state['metadata'], path
    docs = ROOT/'runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/documents.jsonl'
    texts = [json.loads(line)['text'] for line in docs.open()]
    tokenizer = j.c.model_tokenizer(j.c.MODEL_PATH)
    ids = tokenizer('\n\n'.join(texts), add_special_tokens=False, return_tensors='pt').input_ids[:, :2097152]
    full, tail = divmod(ids.numel(), 2048)
    metadata = dict(dataset='allenai/c4', subset='en', split='validation', document_source=str(docs),
                    tokenizer_path=str(j.c.MODEL_PATH), add_bos_token=False, add_eos_token=False,
                    window_length=2048, token_count=ids.numel(), predicted_tokens=full*2047+max(tail-1, 0),
                    full_windows=full, tail_tokens=tail, unscored_tail_tokens=int(tail==1))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(input_ids=ids, metadata=metadata), path)
    return ids, metadata, path


def reuse_bf16(task):
    family = {'llama-3.2-1b-instruct': 'llama', 'qwen3-1.7b': 'qwen'}.get(j.c.MODEL_PATH.name)
    if not family:
        return None
    if task in TASKS:
        path = LEGACY/family/'bf16'/task/'results.json'
        return path if path.exists() else None
    rows = json.loads((LEGACY/'ppl_reuse.json').read_text())
    for row in rows:
        if row['model_key'] == family and row['method'] == 'bf16' and row['dataset'] == ('allenai/c4' if task == 'c4' else 'Salesforce/wikitext'):
            path = Path(row['evidence_path'])
            if path.exists():
                return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--package', type=Path)
    p.add_argument('--tasks', nargs='+', default=['test', 'c4', *TASKS], choices=['test', 'c4', *TASKS])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('PHASE6_MEMORY_FRACTION', '.85')))
    from transformers import set_seed
    set_seed(42)
    model = None
    for task in a.tasks:
        out = a.output/task
        out.mkdir(exist_ok=True)
        if (out/'results.json').exists() or (out/'reuse.json').exists():
            continue
        reused = reuse_bf16(task) if a.package is None else None
        if reused:
            j.c.write_json(out/'reuse.json', dict(source=str(reused), model_path=str(j.c.MODEL_PATH), protocol='existing matched BF16'))
            continue
        started = time.monotonic()
        j.c.write_json(out/'settings.json', dict(model_path=str(j.c.MODEL_PATH), package=str(a.package) if a.package else None,
                       task=task, seed=42, training=False, calibration=False, selection=False,
                       num_fewshot=0, chat_template=False, use_cache=False, batch_size=4,
                       pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        try:
            if model is None:
                if a.package:
                    model, state = j.load_package(a.package)
                    del state
                else:
                    model = j.c.load_model(j.c.MODEL_PATH).cuda().requires_grad_(False).eval()
                model.config.use_cache = False
                model.seqlen = 2048
            model.cuda()
            before = quantizer_snapshot(model)
            for wrapper in j.c.wrappers(model).values():
                wrapper.quantizer.calls = 0
            if task in ('test', 'c4'):
                if task == 'test':
                    ids, metadata = load_wikitext2_test_tokens()
                    token_path = None
                else:
                    ids, metadata, token_path = c4_tokens()
                with torch.no_grad():
                    result = evaluate_tokens(model, ids, 128)
                if result['predicted_tokens'] != metadata['predicted_tokens']:
                    raise ValueError('Target count differs')
                result.update(input_metadata=metadata, input_token_path=str(token_path) if token_path else None)
            else:
                import lm_eval
                from lm_eval.models.huggingface import HFLM
                class FrozenHFLM(HFLM):
                    def _model_call(self, inps, attn_mask=None, labels=None):
                        with torch.no_grad():
                            return self.model(input_ids=inps, attention_mask=attn_mask, use_cache=False).logits
                tokenizer = j.c.model_tokenizer(j.c.MODEL_PATH)
                tokenizer.pad_token_id = tokenizer.eos_token_id
                lm = FrozenHFLM(pretrained=model, tokenizer=tokenizer, backend='causal', batch_size=4, max_length=8192, add_bos_token=False)
                result = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=0, batch_size=4,
                         apply_chat_template=False, fewshot_as_multiturn=False, log_samples=True, bootstrap_iters=0,
                         random_seed=42, numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42)
                for name, rows in result.pop('samples', {}).items():
                    with (out/('samples_'+name+'.jsonl')).open('w') as stream:
                        for row in rows:
                            stream.write(json.dumps(row, default=str, ensure_ascii=False)+'\n')
                del lm
            after = quantizer_snapshot(model)
            unchanged = before.keys() == after.keys() and all(torch.equal(v, after[n][k]) for n,row in before.items() for k,v in row.items())
            calls = {n:w.quantizer.calls for n,w in j.c.wrappers(model).items()}
            if not unchanged or any(v == 0 for v in calls.values()):
                raise RuntimeError('Quantization state or forward coverage changed')
            j.c.write_json(out/'quantization.json', dict(scales_unchanged=unchanged, calls=calls, elapsed_seconds=time.monotonic()-started))
            j.c.write_json(out/'results.json', json.loads(json.dumps(result, default=str)))
            print(task, 'completed', time.monotonic()-started, flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as error:
            j.c.write_json(out/'failure.json', dict(type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
