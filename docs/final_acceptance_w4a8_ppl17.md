# Final W4A8 Validation Acceptance Result

## Executive result

The final fixed-configuration acceptance run achieved a WikiText-2 validation
PPL of **17.6423999759** for the C RTN W4A8 candidate. This is the result to
report for the final acceptance stage because it was measured together with a
matched BF16 baseline under the same full-validation protocol.

| Configuration | Validation PPL | Validation NLL (nat/token) |
| --- | ---: | ---: |
| Original BF16 W16A16 | 13.6346577256 | 2.6126149134 |
| C RTN W4A8 candidate | **17.6423999759** | 2.8703050944 |

Relative to the matched BF16 baseline, the candidate adds **4.0077422502 PPL**
and **0.2576901810 nat/token**, corresponding to a **29.393787% relative PPL
increase**. This is an end-to-end language-model PPL change; it is not a
29.39-percentage-point task-accuracy decrease.

## Candidate configuration

- Model: local Llama-3.2-1B-Instruct.
- Weights: 112 backbone Linear layers using symmetric W4 per output channel
  (`groupsize=-1`, RTN checkpoint).
- Activations: 96 non-`down_proj` inputs using fixed per-tensor INT8 scales;
  16 `down_proj` inputs using fixed SP2-A8 per-tensor scales.
- Rotation: R1/R2.
- KV precision: KV16.
- Embeddings, `lm_head`, and normalization layers remain high precision.

The validation path loaded the fixed candidate configuration without
re-quantizing weights or recalibrating activation scales. The quantizer buffers
were unchanged before and after evaluation.

## Evaluation protocol

- Dataset: the complete `wikitext-2-raw-v1` validation split.
- 252,852 input tokens and 252,728 scored next-token predictions.
- 123 disjoint 2,048-token windows plus one 948-token tail window.
- Teacher-forced next-token evaluation, batch size 1, BF16, `use_cache=False`,
  PREFILL, and KV16.
- Segment NLLs were combined using prediction-token weighting before computing
  the final PPL.

The acceptance run was recorded on 2026-09-09 under
`validation-acceptance-c-20260909.9WVDue`. The launcher was
`scripts/phase2/43_run_validation_acceptance_c_local.sh`.

## Interpretation and limits

This result validates the fixed quantization configuration under a matched
BF16-fake-quant evaluation. It does not by itself establish native integer
kernels, decode behavior, quantized KV-cache behavior, phone-NPU execution, or
latency. It also does not provide a causal W4-versus-A8 error decomposition.

The SP2 configuration had been selected using earlier short-sample diagnostics;
therefore, this full validation run is a fixed-configuration acceptance
measurement, not an untouched independent generalization test.

The model weights, rotations, scales, caches, and run directories are
intentionally not included in this repository submission. This document keeps
the result and its measurement contract available for review without uploading
large binary artifacts.
