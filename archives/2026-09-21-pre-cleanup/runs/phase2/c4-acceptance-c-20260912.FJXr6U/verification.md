# Independent verification: PASS

Run: `c4-acceptance-c-20260912.FJXr6U` (2026-09-12). Verifier: `verify_c4`, separate from the implementation role.

## Code and CPU checks

- Command: `PYTHONDONTWRITEBYTECODE=1 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -q -p no:cacheprovider scripts/phase2/test_validation_acceptance.py scripts/phase2/test_c4_acceptance.py`.
- Result: **15 passed in 4.84s** (6 existing acceptance tests and 9 independent C4 tests).
- Tests cover original behavior, chunk limits and exact input order, unequal chunk/tail weighting, no cross-window target, restoration on exceptions, fixed-token input without dataset loading/calibration, and joined-text token preparation. Actual 1024-window accounting is tested on CPU.
- `bash -n scripts/phase2/44_run_c4_acceptance_c_local.sh`: PASS.
- Runtime source copies for the launcher, acceptance entrypoint, token preparation, and codebooks match the checked files byte for byte.

## Actual data verification

- Source: eight official `allenai/c4` English validation gzip files, 45,576 records each, 364,608 records in total. Only the validation split was downloaded/prepared for this run.
- Independently scanned all eight cached gzip files. Every saved selected document's text and URL matches its source row. Prepared-document counts by shard 0–7: 563, 574, 535, 525, 573, 580, 582, 548 (total 4,480).
- Independently reproduced the seed-42 full-document permutation: saved row order matches exactly and contains no repeated source row.
- Re-tokenized the 4,480 saved documents joined by two newlines with the same local tokenizer, no BOS/EOS, and no truncation during tokenization: 2,150,947 tokens. The first 2,097,152 exactly equal the saved `[1, 2097152]` int64 tensor. File metadata and embedded metadata match.
- 53,795 suffix tokens were omitted to obtain exactly 1,024 disjoint 2,048-token windows. The prepared-document count does not mean all 4,480 documents were fully scored.

## Completed model results

Both result files and the summary were independently read and recomputed. Each has the same saved token path and metadata, eight 128-window chunks, no partial tail, and 2,096,128 predicted tokens (1,024 × 2,047). NLL is weighted by predicted tokens and PPL is its exponential; chunk PPLs are not averaged.

| Configuration | PPL | Mean NLL |
| --- | ---: | ---: |
| Original BF16, unrotated | 21.843397624525 | 3.083898707666 |
| Fixed C RTN W4 / non-down INT8 / down-SP2 | 29.539535457545 | 3.385729551101 |

- PPL increase: 7.696137833019; relative increase: 35.233245145%.
- Mean NLL increase: 0.301830843435 nat per predicted token.
- Original BF16 uses the unrotated preparation branch, with zero quantized linear inputs; no quantized checkpoint or rotation was loaded in that branch.
- Candidate loads the existing C `rtn/w4_rtn_model.pt`, matched C `rotation/R.bin` and `rotation/quant_scales.pt`, and the existing SP2 scales from `down-codebooks-c-20260909.6YRIty/results/down_scales.json`.
- Runtime checks passed for 112 backbone W4 linear weights matching the existing checkpoint, 96 fixed non-down INT8 inputs, and 16 fixed down-SP2 inputs. The post-evaluation buffer comparison confirms all activation quantizer buffers, including SP2 alpha/levels, stayed unchanged.
- Both modes use BF16 execution, KV16, `use_cache=False`, and explicit PREFILL. The loaded-checkpoint branch bypasses RTN/GPTQ calibration; saved-data loading rejects `eval_mode=False`. No training, scale fitting, or format selection occurs in this evaluation.

## Scope and limits

PASS establishes consistent evaluation of this fixed pair on the specified C4 validation subset. It is not a test of INT8/PoT alternatives, calibration sufficiency, decode/KV quantization, native SP2/W4 kernels, or phone-backend latency. Independence is with respect to this project's local quantization development; base-model pretraining overlap was not audited. The underlying evaluator retains its float32 PPL aggregation and BF16 computation. Its legacy `WikiText2` log label is static text; the saved-input branch and recorded data demonstrate that this run evaluated C4.
