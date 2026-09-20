# Independent verifier report, 2026-09-20

Status: PASS for the frozen-package scoring/generation adaptation reviewed in `scripts/capability_eval/run.py`; downstream dataset execution and numerical result validity remain subject to actual task completion.

The verifier did not edit application code, train models, select checkpoints, or execute benchmark tasks. Only the independent diagnostic script and this report were added here.

## GPU evidence

`check_models.py qwen` and `check_models.py llama` ran separately using `runs/phase5/env/bin/python`, CUDA_VISIBLE_DEVICES=6, with bytecode writes disabled. The sandbox could not initialize CUDA; the authorized elevated execution succeeded. Each final package ran one short scoring forward with `use_cache=False`, followed by exactly two generated tokens with an explicit `DynamicCache`, `use_cache=True`, and greedy generation. No calibration occurred.

| Model | Frozen backbone activation quantizers | Finite score logits | Score / prefill / decode coverage | All quantizer state tensors unchanged |
|---|---:|---|---|---|
| Qwen3-1.7B | 196 | PASS | Every quantizer called in all three stages | PASS |
| Llama-3.2-1B-Instruct | 112 | PASS | Every quantizer called in all three stages | PASS |

Raw reports: `qwen.json`, `llama.json`. Each report records the exact package path, GPU, output token IDs and per-module call counts. The tests are runtime compatibility/quantization checks and not ability estimates.

## Code review

- `models.json` selects the seed-42 final Qwen package and accepted final Llama package, with separate parent package paths. It passes `PHASE5_MODEL_PATH` before importing the loader, avoiding the module-level model-path binding trap.
- HFLM receives the already loaded model object. The inspected harness constructor assigns that object directly rather than reloading a high-precision model. No adapter code calibrates or modifies frozen scales.
- `FrozenHFLM._model_call` uses the supplied model with cache disabled for scoring. Generation inherits harness 0.4.8 `_model_generate`, which explicitly passes `use_cache=True`. The Llama class declares `_supports_cache_class=True`. The direct GPU test validates explicit DynamicCache operation.
- Frozen W4 package records are dequantized onto the fixed grid during loading; INT8/SP2 activation quantizers are called on scoring, prefill and decode. This is fake-quant BF16 execution with KV16, not a native integer kernel measurement.
- Both methods within each model use the same architecture loader, tokenizer, task configuration, generation limits and prompt formatting. Generation templates explicitly set Qwen `enable_thinking=False`.
- Official installed GSM8K CoT YAML provides its eight fixed demonstrations and greedy generation. Harness default generation length is 256 tokens, matching the recorded settings. Official IFEval YAML uses 1280 tokens, prompt-level strict accuracy among its outputs, and greedy generation. The script uses 5-shot MMLU and zero-shot for the eight requested multiple-choice tasks.
- Full-run quantizer snapshots and call counts are saved before results are accepted. Call labels use sequence length (1 = decode), suitable for these benchmark prompts but not a universal proof of cache state. The independent check used a real DynamicCache.

## Scope limits

At the time of this check, importing HFLM still failed because the in-progress dependency installation lacked `lxml` (through sacrebleu). The coordinator was notified. Code-review PASS does not mean that the dependency environment or a benchmark task has completed successfully.

The check does not verify benchmark data availability, result aggregation, PPL protocol reuse, package provenance beyond the explicit selected paths, long generation quality, or generation truncation frequency. Those require the coordinator's run artifacts. The final paper should state generation caps and any incomplete tasks. Current compatibility warnings about future GenerationMixin inheritance, tokenizer class naming and sampling-only settings are nonfatal in Transformers 4.51.3 and do not require source changes for this execution.
