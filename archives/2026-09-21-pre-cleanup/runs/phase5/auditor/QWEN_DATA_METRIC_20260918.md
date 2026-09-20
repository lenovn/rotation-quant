# Qwen data and metric audit

Independent evidence auditor, 2026-09-18. **PASS for data reconstruction and seven new CPU protocol tests.** No GPU use and no full PPL reevaluation; this is not an acceptance result for the running BF16 or trained model.

## Actual C4 artifact

Read-only inspected `runs/phase5/qwen3-1p7b/c4-data/metadata.json` and `input_tokens.pt`; their metadata are exactly equal. Independently read the original Phase2 `documents.jsonl`, joined all original texts with two newlines, and used local `AutoTokenizer` with no special tokens to reconstruct the complete Qwen stream.

- Documents: **4480**, unchanged order; saved row indices match both the original documents and the first 4480 indices from `Random(42)` over all 364608 source rows.
- Text length: **10098392 characters**.
- Tokenizer: `/home/dongpeiyan/projects/rotation-quant/cache/models/qwen3-1.7b`, actual class `Qwen2TokenizerFast` (the official tokenizer class used for this Qwen3 checkpoint).
- Local checkpoint revision record: `Qwen/Qwen3-1.7B`, `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, read from `phase5_revision.json`; no network identity audit was repeated.
- Full reconstructed stream: **2191766 tokens**. Historical prefix cap: **2097152 tokens**. Discarded suffix: **94614 tokens**.
- Saved `[1, 2097152]` INT64 tensor matches the independently reconstructed prefix **exactly, every token**.
- **1024** independent 2048-token windows; **2096128** predicted targets; tail0 and unscored_tail0. No new document sampling, Llama token IDs, BOS/EOS, chat template or thinking prompt.

The same raw documents and prefix rule are preserved across models. The retained character span can differ because the tokenizer changes; the equality of target counts here follows from both streams exceeding the historical cap, not from reusing token IDs.

Reproduction script: `runs/phase5/auditor/verify_qwen_data.py`. Machine evidence: `qwen_c4_reconstruction_20260918.json`; tokenizer informational output is retained in the adjacent `.stderr`. This CPU reconstruction does not call a model or compute PPL.

## New split and numerical path

Added `worktrees/SpinQuant-multimodel/tests/test_phase5_eval_protocol.py`, without modifying application code. Command from that worktree:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/dongpeiyan/projects/rotation-quant/runs/phase5/env/bin/python -m pytest -q -p no:cacheprovider tests/test_phase5_eval_protocol.py
```

**7 passed in 10.42s**, first run, log `runs/phase5/auditor/qwen_eval_protocol_20260918.log`. No old suite or repeat run. `git diff --check` passed.

Checked:

1. `load_wikitext2_test_tokens()` still defaults to official test. Explicit validation reads only `wikitext-validation.arrow`; train is rejected. Both splits preserve source-row order, double-newline join and `add_special_tokens=False`.
2. External driver without the new flag uses test; `wikitext2_validation=True` selects validation and persists that split in saved input metadata.
3. Short tail>=2 is scored, tail1 is explicitly unscored, and no tail produces no extra window. Targets omit each independent window's first token.
4. A tiny official Transformers `Qwen3ForCausalLM` runs on **CPU BF16** using its native decoder and head. `architecture.qwen_evaluator` exactly matches an independent oracle: BF16 logits cross-entropy with reduction none, per-token losses cast to FP32, FP32 window/segment means and exponentiation. Segment PPL is then converted to Python log and weighted by target count in the inherited acceptance function.
5. Two full synthetic windows plus a short tail reproduce the expected weighted NLL/PPL exactly; a one-token tail is omitted and counted. Original `model.seqlen` is restored after evaluation. Qwen evaluator dispatch selects `architecture.qwen_evaluator`.

The tests do not replace real Qwen-1.7B forward equivalence, GPU backend verification, model-quality scores or final cold-package acceptance. Full BF16 test/C4 and quantized results remain the responsibility of their actual runs and subsequent result audit.
