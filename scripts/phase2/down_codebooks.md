# Down-input A8 codebook experiment

## Sources and adaptations

- PoT: [APoT author's code](https://github.com/yhhhli/APoT_Quantization/blob/master/ImageNet/models/quant_layer.py), `build_power_value(B=7, additive=False)` and the signed projection in `apot_quantization`. We retain its normalized value set and absolute-distance nearest-neighbour semantics. A sorted midpoint search replaces the large all-pairs distance matrix; equal-distance ties choose the smaller magnitude deterministically. We use FP32 for construction/projection and cast to BF16 for the existing GEMM path. The original nonnegative activation clamp is not used for signed down inputs. No APoT additive codebook is substituted for SP2.
- SP2: [Mix and Match, arXiv:2012.04240v2](https://arxiv.org/pdf/2012.04240), Eq. (8). A8 has one sign bit and two exponent fields of 4 and 3 bits: `q1={0,2^-1,...,2^-15}`, `q2={0,2^-1,...,2^-7}`. Enumerate all `q1+q2` and deduplicate. The 128 magnitude encodings give 94 unique nonnegative values, hence **187 signed values**, not 255 unique values. The 0.75-to-1 gap is retained. This is a paper-formula reproduction; no verified official SP2 implementation was found.
- INT8: the project's `utils/quant_utils.py::STEQuantize` numerical convention: integer codes `[-128,127]`, FP32 step `alpha/127`, ties-to-even. INT8 has 256 values versus PoT's 255 and SP2's 187; all use an 8-bit encoding budget. INT8's negative endpoint is `-128*alpha/127`.
- [DeepShift](https://github.com/mostafaelhoushi/DeepShift/blob/master/pytorch/deepshift/utils.py) was inspected but its logarithmic rounding is not used: geometric boundaries differ from numerical nearest-neighbour projection.

## Fixed input model

Use `runs/phase2/learned-sw-c-20260909.ByFYAM/C/rtn/w4_rtn_model.pt` with C's `rotation/R.bin` and `rotation/quant_scales.pt`. User switched to C if it performs better: same test 8x2048 reference PPL is B-GPTQ 15.1518078, C-GPTQ 15.3142958, B-RTN 15.0265388, C-RTN **14.6552505**. C-RTN is selected, not C-GPTQ.

Load the existing checkpoint once via the existing PTQ entrypoint. No weight quantization or rotation optimization runs. Existing load validation stays intact. Install down quantizers only after checkpoint loading. Verify all 96 actual non-down scales equal C's exported scales and are unchanged after evaluation. R1/R2 only; no R3/R4; KV16; prefill only.

## Calibration and evaluation

1. Reloaded down-A16 reference uses the original test evaluator, tokenizer configuration, batch 1 and test 8x2048.
2. Calibration uses 32 disjoint train 2048-token windows shuffled with seed 42. Capture 64 uniformly sampled complete token rows per window (same rows for all formats/layers); full-window maxima set each layer's search range. Each format has 33 log-spaced alpha candidates from `max/2^16` to `2*max`, then 17 candidates between the neighbouring coarse points of its optimum. All use the same 50 evaluations and minimize `mean(((Q_bf16(X)-X) @ W4.T)^2)` in FP32. Search-edge optima are recorded. Sampling is a first-round computational compromise and can miss rare influential tokens.
3. Local damage uses the first eight disjoint **validation** windows, all token rows. All down inputs come from the same down-A16 reference trajectory. Hooks measure hypothetical quantization without returning modified tensors, so candidates do not contaminate later layers' inputs. Report input MSE, linear-output MSE/NMSE, clipping/zeroing fractions, maximum output error and per-window output MSE. These are FP32 linear responses to BF16-quantized input differences, not independently rounded BF16 output-subtraction errors.
4. Separately evaluate all 16 down inputs as INT8, PoT, then SP2, using their independently calibrated per-layer alphas. The original evaluator measures propagated end-to-end test PPL/NLL. Test loss is never used for scale selection. No mixed-per-layer search is included in this round.

Run `bash scripts/phase2/42_run_down_codebooks_c_local.sh`. Output goes to a fresh `runs/phase2/down-codebooks-c-20260909.XXXXXX/` directory. JSON contains settings, codebooks, all scale-search candidates, local damage and PPL. Saved down scales are separate from source SA. These are fake-quant numerical measurements, not phone-backend latency or packed-format deployment results.
