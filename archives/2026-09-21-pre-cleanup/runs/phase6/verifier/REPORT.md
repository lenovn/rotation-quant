# Phase 6 independent verification

Date: 2026-09-21. Status: **PASS for new CPU-tested path and protocol review**. No GPU experiment was run by verifier. Application source was not modified.

Reviewed: `scripts/phase6/down_precision.py`, `runs/phase6/PLAN.md`, imported Phase3 wrapper, static loader, common.full_validation and existing evaluator.

Command: `PYTHONDONTWRITEBYTECODE=1 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider -q /home/dongpeiyan/projects/rotation-quant/runs/phase6/verifier/test_down_precision.py`

Result: **7 passed in 8.55s**.

Verified:

- INT8 and INT16 round on the FP32 signed integer grid; endpoint clipping and tie-to-even examples pass; BF16 input returns BF16, scale remains FP32.
- Real `ActQuantWrapper` bypasses old SP2 with bits=16 and calls the registered Linear input hook. INT16 is actually executed, not interpreted as floating bypass. Two synthetic layers with distinct alpha values verify proper layer association.
- Float arm returns input unchanged through an identity Linear. Original weights and preexisting buffers stay exactly equal (including existing NaNs), and non-down bitwidth stays 8.
- Quantizer is a registered child of Linear; moving one layer to the meta device moves its scale and diagnostic counters without moving the other layer. Actual evaluator uses layer.to(device)/cpu without dtype casting, so this registration follows its movement.
- All 16 full_absmax values match independent historical capture metadata exactly; range is 2.5625 to 816.0. Reuse is train-only parallel down-bypass calibration, not new cascaded integer calibration.
- New validation tokenizer output equals the old tokenizer invocation token-for-token. 252852 input tokens, 252728 targets, 124 windows including 948-token tail.
- Full validation delegates to unchanged existing evaluator/acceptance code, preserving BF16 logits cross entropy and token-weighted segmented NLL.

The first test invocation had three verifier-test failures because torch.equal considers NaN unequal to itself in unused legacy output-quantizer buffers. The test comparison was corrected to exact `assert_close(..., equal_nan=True)`; no application change was needed.

Result reuse audit: no matched full-validation results were found for absmax INT8/INT16/down-float on old B100. Existing uniform INT8 calibration searches 50 output-MSE candidates and is therefore a different range rule. Phase5 rounded/range-adjusted/QAT packages change weights or ranges; C4 is a different dataset. Existing SP2 validation may only be a separately labeled contextual row. CPU PASS does not certify future GPU results or native integer/NPU execution.

## Independent completed-result audit

Status: **PASS**. Audited run `runs/phase6/b100-down-precision-20260921` after progress became completed. No failure.json exists. No GPU rerun performed.

| Arm | NLL | PPL |
|---|---:|---:|
| down BF16 bypass | 2.7809410428134664 | 16.134196776282934 |
| down INT8 | 7.3318433173780155 | 1528.19612779757 |
| down INT16 | 2.7897502791337607 | 16.276954599024368 |

All three results share the exact parent package, per-layer alpha, calibration source, seed, dataset/split, target count, and evaluation precision. Archived entrypoint plus common/postprocess/quantization/quant_utils/eval_utils are byte-identical to the source files reviewed. Each segment's reported NLL equals log(segment PPL); target-weighted aggregation and exp(aggregate NLL) exactly reproduce each result.

Both integer arms have exactly 16 layer diagnostics, 124 calls per layer, and 252852*8192 elements per layer. Saved scale tensors exactly equal FP32(alpha/127) or FP32(alpha/32767); diagnostic scale and signed integer limits agree. Every result reports unchanged original and added static scales. Float has no additional integer quantizer or saved down scale.

INT16 minus float NLL is +0.008797640112124583 for the 123 full windows and +0.011892346023756328 for the tail; weighted total +0.00880923632029429. PPL difference is +0.14275782274143367, or +0.8848151830606499 percent. Thus both segments support the same direction; this is not an aggregation sign artifact.

Interpretation supported: with this B100 checkpoint and reused 4096-token train absmax calibration, INT16 avoids the severe INT8 degradation while retaining a small measurable loss against BF16 bypass. The evidence does not establish lossless INT16, absence of benefit from learning down-SA, or native NPU performance.
