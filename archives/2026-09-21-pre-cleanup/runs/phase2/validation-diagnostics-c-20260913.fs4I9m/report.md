# C RTN full-validation activation diagnostics

Run completed on 2026-09-13; all three GPU evaluations exited successfully.
Launcher: `scripts/phase2/45_run_validation_diagnostics_c_local.sh`.
Environment: `rotation-quant-p0`, physical GPU 3; cached dataset in offline mode.

| Configuration | PPL | NLL (nat/token) | PPL above historical BF16 |
| --- | ---: | ---: | ---: |
| Original BF16, historical matched reference (not rerun) | 13.634657725643203 | 2.612614913352714 | 0 |
| C RTN W4, all activation inputs A16 | 16.231066669209774 | 2.7869271014377417 | 2.596408943566571 |
| C RTN W4, 96 non-down fixed INT8, 16 down A16 | 16.292812357698388 | 2.790724050911014 | 2.6581546320551848 |
| C RTN W4, 96 non-down fixed INT8, 16 down fixed SP2 | 17.64239997588965 | 2.8703050943792365 | 4.007742250246446 |

All new results use the same existing `learned-sw-c-20260909.ByFYAM/C/rtn/w4_rtn_model.pt`, matching `rotation/R.bin` and `rotation/quant_scales.pt`. SP2 uses the existing `down-codebooks-c-20260909.6YRIty/results/down_scales.json`; it was not recalibrated. No weights were requantized. The historical BF16 reference was read from `../validation-acceptance-c-20260909.9WVDue/w16a16/result.json`. New SP2 PPL and NLL exactly reproduce that run's W4A8 result.

Protocol: complete WikiText-2 raw validation, 252852 input tokens, 252728 scored predictions, 123 disjoint 2048-token windows and one 948-token tail. Batch 1, seed 42, BF16 fake quantization, KV16, `use_cache=False`, prefill teacher-forced next-token scoring. Segment NLL is log of the existing evaluator's float32 PPL, aggregated by prediction-token count, matching the previous acceptance method.

Conditional changes in this order:

- Historical BF16 to current W4/all-A16: +0.17431218808502758 NLL.
- Adding non-down INT8: +0.0037969494732723597 NLL, +0.06174568848861384 PPL.
- Adding down SP2: +0.07958104346822248 NLL, +1.3495876181912614 PPL.

These are ordered conditional effects, not independent additive causal contributions. At fixed C weights, restoring down to A16 still leaves PPL 16.29281, above the BF16+1 target of 14.63466. Reaching that target therefore requires improving the effective W4 model as well as down activation quantization. This does not establish a lower bound for retrained R/D/SW candidates or imply A16 is globally optimal after retraining.

Runtime checks passed for all 112 per-output-channel W4 checkpoint weights and unchanged activation buffers. Result JSONs record the actual input bit widths: 112 A16; 96 A8 plus 16 A16; and 112 A8 respectively. Output quantizers remain A16. Embeddings, lm_head, norms and KV retain their existing high precision. These measurements do not validate native integer kernels or device latency.

Independent verifier `/root/verify_diagnostics`: PASS. Reviewed the load path, actual artifacts and historical comparison; six CPU aggregation tests passed, the launcher's shell syntax passed, and a real ActQuantWrapper/RotationStaticActQuantizer CPU check confirmed A16 bypass does not invoke quantization and exactly matches the original Linear output. The checkpoint contains both lm_head weight keys and strict full-state loading replaces initial model values.
