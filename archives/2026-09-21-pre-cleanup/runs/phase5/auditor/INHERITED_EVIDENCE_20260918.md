# Phase5 inherited evidence audit

Date: 2026-09-18. Independent evidence auditor; read-only inspection of old artifacts, no GPU, no rerun, no modification to Phase2/3/4. New outputs are confined to `runs/phase5/auditor/`.

## Result

**PASS for inherited metric/document consistency.** `inherited_results.csv` contains 13 historical results at original JSON precision, with original evidence/package paths and `inherited_no_rerun` status. `inherited_checks.json` records the performed checks. This is not a new model-quality measurement or an independent GPU reproduction.

All 13 results reproduce their saved aggregate NLL and PPL exactly from saved segments using the inherited target-weighted protocol. Targets sum correctly; each segment NLL equals log(segment PPL). The seven external result files have completed progress, no failure JSON, and record no training/calibration/candidate selection and unchanged activation scales. These flags were read; large tensor snapshots were not reloaded in this audit.

Both WikiText test files have identical input metadata and scoring windows. All five C4 files have identical input token path, metadata, metric protocol, targets and windows. Machine summaries agree with original result JSONs, and saved C4 comparison deltas agree with subtraction of raw result values. The test and C4 final SP2-QAT400 records reference the **same exact package path**.

| Llama state | WikiText-2 test PPL | Fixed C4 PPL |
| --- | ---: | ---: |
| Original BF16 | 13.162650325300651 | 21.843397624525274 |
| B100 matched down INT8 format control | not required/measured | 1330.1567504736056 |
| B100 initial SP2 | not required/measured | 29.18423776720624 |
| Exact SP2-PTQ parent | missing | 27.43961641336637 |
| SP2-QAT400 | 14.154416405154963 | 26.98478050458106 |

Exact raw evidence is linked in every CSV row. Summary inputs checked: `runs/phase3/WIKITEXT2_TEST_20260917.json` and `runs/phase3/C4_ATTRIBUTION_20260918.json`. New C4 attribution supersedes the older `C4_RESULTS_20260915.md` statement that the exact PTQ parent had not yet been evaluated.

## Fixed C4 documents and Qwen entry

- Original documents: `/home/dongpeiyan/projects/rotation-quant/runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/documents.jsonl`.
- Metadata and original Llama token tensor are in the same directory.
- All 4480 document records contain row index, URL and original text. Their row indices exactly match the first 4480 entries of `random.Random(42).shuffle(list(range(364608)))`, verified without loading C4 or downloading anything.
- Original metadata: eight English validation shards; join with two newlines; no BOS/EOS; Llama joined stream 2150947 tokens; retain 2097152-token prefix, discarding 53795 suffix tokens; 1024 independent 2048-token windows and 2096128 targets.
- The raw-document preparation source is `scripts/phase2/prepare_c4_evaluation.py`. It uses `LlamaTokenizerFast` and selects documents until its token budget is reached. **Do not run its document-selection loop for Qwen:** preserve these 4480 saved documents and their order, then tokenize with the actual Qwen tokenizer.
- Reuse the existing prefix cap, record Qwen full concatenated count and retained/discarded counts, and do not append documents if Qwen produces fewer tokens. Preserve a scoreable final short window. Different Qwen counts are acceptable; never feed Llama token IDs into Qwen.
- `experiments/phase3/external_eval.py::load_c4_tokens` in the inherited implementation currently requires complete windows and exact `windows * 2048` input length. Supporting a Qwen tail therefore needs a narrow metadata/accounting adaptation; the underlying `validation_acceptance.evaluate_full_validation` already supports short tails.

WikiText test is the cached `Salesforce/wikitext`, `wikitext-2-raw-v1`, `b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow`: 4358 raw rows joined in order by two newlines, no special tokens. Llama has 289076 inputs, 141 full windows plus 308-token tail, 288934 targets. Training remains per-row no-special-token tokenization followed by ID concatenation; do not silently change it to joined text.

## Core matrix and representative ablation

| Llama seed42 item | Inherited state | Required remaining work |
| --- | --- | --- |
| BF16 | validation/test/C4 complete | reuse |
| SP2-PTQ exact parent | validation/C4 complete | test if in final PTQ table |
| SP2-QAT400 | validation/test/C4 complete | reuse |
| Uniform-PTQ matched full postprocessing | not found | implement and measure |
| Uniform-QAT400 matched full training | not found | implement and measure |
| Initial SP2 directly followed by matched QAT400 | not found | train this one branch, pair with existing SP2-QAT400 |
| Complete method and key control seeds43/44 | not found in inspected settings | execute after seed42 matrix |

Search scope was all `settings.json` beneath current `runs/phase3`, `runs/phase4`, and `runs/phase5`, plus original metric paths. The two distinct 400-step historical distillation configurations both use the fully refined parent, with Adam master LR 1e-5 or 2e-5. Neither is the required direct-initial-SP2 ablation. Runs using a D-modified parent, 200-step schedule, 800-step continuation, missing reference initialization, or different LR cannot stand in for it. Archived/deleted trees were not exhaustively searched; no matching artifact was found in active runs.

The B100 INT8 overlay used only training calibration, 50 scale candidates/layer and the same B100 package as initial SP2. It is a useful format comparison, **not Uniform-PTQ or Uniform-QAT400**. Its scale overlay remains at `runs/phase3/b100-uniform-calibration-20260918c/down_int8_scales.pt`.

The fully matched representative ablation should use the existing `route-b-adam-100-20260914a/checkpoint-0100/static_w4a8.pt` as initial-SP2 parent and the same directory's `state.pt` as floating reference. Match existing final run settings: clean start, Adam master LR 1e-5, relative scale LR .001, 400 schedule/updates, warmup10, accumulation8, temperature1, CE weight .1, data_start800, same cell-reference initialization and WikiText-only data. No new Joint100 or postprocessing is needed for that branch.

## Interpretation boundaries

These are BF16 fake-quant prefill, KV16, `use_cache=False` results. No new decode, native NPU, full-C4, mixed-data-QAT or downstream accuracy evidence was produced. Eight C4 chunks are scoring segments, not eight training seeds. Existing Llama final test/C4 and exact-parent C4 should not be rerun because of this handoff.
