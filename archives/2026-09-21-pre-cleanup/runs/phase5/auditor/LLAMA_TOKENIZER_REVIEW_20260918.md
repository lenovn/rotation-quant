# Llama tokenizer compatibility review

2026-09-18. **PASS for the architecture-specific tokenizer fix and declared production-environment consistency.** Qwen tests and full data reconstruction were not rerun.

The main executor's first real Llama `data_windows` check correctly failed its inherited token-count guard: generic `AutoTokenizer` produced 252853 validation tokens instead of 252852. The fix makes `architecture.tokenizer` inspect `AutoConfig.model_type`: Llama uses the historical `LlamaTokenizerFast`, Qwen retains `AutoTokenizer`. Both keep local-only lookup and `add_bos_token=False`/`add_eos_token=False`; no dataset or optimization changes were introduced.

Independent auditor check in Phase5 isolated Transformers4.51.3: a short real text with a paragraph break tokenized identically under the repaired entry and an explicitly instantiated historical Llama tokenizer. Default tokenization equals explicit `add_special_tokens=False`, and no BOS/EOS appears. Actual class is `LlamaTokenizerFast`. Evidence: `auditor/llama_tokenizer_branch_20260918.json`. This was deliberately a small regression, not another full dataset tokenization.

Main executor full-data evidence was read from `verifier/llama_tokenizer_regression.json`: it executed `data_windows()[4] == json.load(open(historical_path))` against `runs/phase3/route-b-adam-100-20260914a/data.json` and recorded PASS. The historical record includes 2435022 train tokens, 1188 full train windows, 1998 dropped train tokens, all 32 calibration indices, 128-token calibration length, probe indices, 252852 validation inputs, 252728 targets and seed42. Its full-metadata equality is primary executor evidence, not an independent full-data rerun.

Environment choice was explicitly clarified and verified against actual launch JSONs:

- Llama full-data check and formal Llama runs use the preserved old `/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python` (Transformers4.44.2), keeping the historical Llama implementation environment.
- `llama32-1b/uniform-initial-s42.launch.json` and `initial-sp2-qat400-s42.launch.json` both name that exact Python and the Llama model path.
- Qwen runs use `runs/phase5/env/bin/python`, independently observed as Transformers4.51.3. The old environment is not upgraded.

Different model-family compatibility environments are explicit; the metric/data rules are shared. The isolated-env short-text result is supplementary and is not mislabeled as the environment of the main executor's full-data regression. Formal launch configuration agrees with that full-data regression, so no redundant full-tokenization rerun was requested.
