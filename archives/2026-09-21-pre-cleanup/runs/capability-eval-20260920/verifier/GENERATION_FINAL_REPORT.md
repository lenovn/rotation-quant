# Independent generation result audit

Final status: PASS for all 14 completed generation evaluations. Reproducible checks: `audit_generation.py`; structured results: `generation_audit.json`.

Audited all seven GSM8K runs (1319 examples and both official filters per run), plus all seven IFEval runs (541 examples each). The final Qwen parent IFEval run passes every check and reproduces 37.33826247689464 percent prompt strict accuracy. All three Qwen IFEval methods use batch64 as prescribed. Total generation coverage is 13020 model-example outputs, with GSM outputs scored under both filters.

For completed runs, sample IDs, actual prompt strings, generation arguments and targets match the same-model BF16 reference exactly. GSM has eight demonstrations and the target last; all Qwen prompts end with the empty think block required by non-thinking formatting. Configured shots, greedy settings and generation batch sizes match the protocol. Successful logs are selected exclusively by current settings, contain exactly the expected number of records, and their prompt-token multiset after removal of leading padding equals retokenized raw-sample prompts. This excludes mixing the superseded Qwen BF16 partial IFEval attempt into its completed result.

Both GSM exact-match percentages were recomputed from their separate per-sample filter records. IFEval's four metrics were recomputed, flattening instruction-level correctness lists before averaging. Every metric and BF16-relative difference matches `generation_all_metrics.csv`; numeric primary columns match strict GSM and prompt-level strict IFEval. Paper-table GSM cells uniformly show strict / flexible from the same generation, with no per-method best-filter selection.

All completed FIRON/parent runs report unchanged quantizer states and full prefill/decode coverage (196 Qwen or 112 Llama quantizers). Selected generation logs retain EOS/stop metadata, and budget exhaustion is computed only when neither EOS nor stop occurs before the budget ends.

## Interpretation boundary

Qwen BF16 GSM strict/flexible is 4.700530705079606 / 61.56178923426838 percent; FIRON is 11.296436694465505 / 42.2289613343442 percent. The higher FIRON strict score does not show better mathematical reasoning: the same outputs show lower flexible-extraction correctness, while compliance with the fixed strict answer phrase differs. Both official filters must stay visible, and neither is changed after observing the results. This is the existing declared single-user-chat 8-shot protocol, not a fresh prompt optimization.

No available-model generation evaluation remains pending. Final table/delta/missing-cell audit is PASS (`final_table_audit.json`): every available-model metric is filled, all BF16-relative deltas match numeric main-table values, and the only 13 missing cells are Qwen W4A16's two PPL and eleven capability tasks. The nine saved capability-evaluation Python entrypoints were byte-for-byte compared with the live scripts and all match. This completes generation acceptance alongside the separate zero-shot, MMLU and PPL audit reports; no benchmark was rerun for the final audits.
