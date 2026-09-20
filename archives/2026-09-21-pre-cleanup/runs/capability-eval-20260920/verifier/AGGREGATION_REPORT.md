# Independent aggregation and protocol audit

Status: PASS for aggregation-code/protocol/PPL-reuse checks, 2026-09-20. Benchmark tasks are still running; this is not final-result acceptance.

Reviewed `scripts/capability_eval/summarize.py`, `PROTOCOL.md`, `ppl_reuse.json`, task YAML files in the installed lm-evaluation-harness 0.4.8 tree, and all twelve PPL source result/settings pairs.

## Metric logic

- BoolQ, SIQA, WinoGrande use official `acc`; PIQA, HellaSwag, ARC-Easy, ARC-Challenge, OpenBookQA use the available official `acc_norm`. The mixed selection is explicit in the protocol and consistent across methods. These selected scores are multiplied by 100 once.
- The eight-task average is an unweighted arithmetic mean, computed only when all eight values exist. Missing results are not converted into zero or partial-task averages.
- Official MMLU uses dev `first_n` examples and test evaluation; the adapter sets five shots. The official top-level group has `weight_by_size: True`. The summarizer reads that group accuracy directly rather than averaging 57 subjects equally.
- GSM8K uses `exact_match,strict-match`, not flexible extraction. IFEval uses prompt-level strict accuracy. Both keys match their task definitions.
- Accuracy deltas are score minus the same-model BF16 score, in percentage points. PPL deltas are absolute PPL differences. None propagate to absent values.
- The initially missing PPL entries in `incomplete.json` were reported and the coordinator added explicit PPL missing reasons. This fix was inspected. An absent Qwen W4A16 package is represented rather than silently dropping that row.

## PPL reuse

All twelve records passed exact comparisons of PPL, NLL, dataset/subset/split, token count, predicted targets, token-artifact path, evaluation package in original settings, and selected package in the current model registry. Independently recomputing target-weighted NLL from the saved segments matched each saved NLL to 1e-12. See `ppl_audit.json` for per-record checks and source paths.

WikiText metadata confirms full test rows, no added BOS/EOS, double-newline joining, disjoint 2048-token windows and included partial tails. C4 metadata identifies the fixed validation subset; each model has 2097152 input tokens and 2096128 predicted targets. The protocol accurately avoids describing it as full C4. Existing validation or 8-window PPL is not used in the twelve-record reuse table.

## Generation logging follow-up

The new `_model_generate` override calls the unchanged parent method, then saves output IDs/text. It does not change generation arguments. Raw batch outputs contain padding; future truncation accounting must identify EOS/terminators and configured stop strings before deciding whether a row exhausted its budget. Raw token-array length alone is not a valid per-example truncation test. Append-only logs can include duplicate attempts after retries; use the successful attempt's records when auditing completed tasks.

Final checks still needed after tasks finish: full requested sample counts, matched actual prompts across methods for all tasks, MMLU subgroup/sample totals, non-thinking generation prompts and terminators, truncation frequency, final CSV cells and all documented missing reasons.
