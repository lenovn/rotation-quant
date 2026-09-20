# Qwen Uniform-QAT400 seed42 audit

2026-09-18. **Training, validation and same-package test/C4 acceptance PASS. Qwen seed42's complete SP2/Uniform × PTQ/QAT core matrix is accepted; three-seed statistics remain incomplete.** Read-only CPU checks of newly completed Uniform QAT evidence, with no GPU, forward, calibration, retokenization or repeated architecture/PTQ suite. Training evidence: `QWEN_UNIFORM_QAT400_20260918.json`; external evidence and full-precision core matrix: `QWEN_UNIFORM_QAT400_EXTERNAL_20260918.json`.

Run: `runs/phase5/qwen3-1p7b/uniform-qat400-s42`. Final package: `checkpoint-0400/static_w4a8.pt`. The exact parent is the complete previously audited `uniform-down-readapt-s42/static_w4a8.pt`; the floating reference is the common `joint100-s42/checkpoint-0100/state.pt`. This is full Uniform QAT following all matched PTQ stages, not the B100-only INT8 format control.

## Fixed budget and matched SP2 configuration

Training has exactly steps1–400,8 microbatches each,2,048 inputs/2,047 targets per microbatch: **6,553,600 input-token visits /6,550,400 prediction-target visits**. All3,200 window indices exactly match the SP2 run and `(800 + (step - 1) * 8 + microbatch) % 1221`. Data metadata and seed42 are identical; the loop uses only WikiText train windows, with the final8 corpus windows excluded. No resume, target-PPL early stop or missing updates occurred. Full validation is at400 only;100/200 checkpoints use train probes.

Every argument matches completed Qwen SP2-QAT400 except output and exact parent package. This includes400 schedule steps, warmup10, Adam weight LR1e-5, relative scale LR0.001, temperature1, CE weight0.1, accumulation8, data start800, teacher offload and two-device layer placement. Actual per-update W learning rates match exactly. The28 decoder layers are divided14/14 between the two local devices; global batch remains8. Absolute scale learning rates can differ because their initial scale values/formats differ, under the same relative-rate rule.

Saved distillation, native architecture, quantization and placement source files are byte-identical to the completed SP2 run. Thus the same previously audited native unrotated frozen BF16 Qwen teacher, original tied teacher head, no norm fusion, no-cache objective and cell-reference initialization apply. The initialization record points to the common Joint100 floating reference, the exact Uniform parent and no diagonal transform. Teacher/source evidence limits remain those stated in `QWEN_SP2_QAT400_20260918.md`; this audit did not recreate or compare a separately saved teacher checkpoint.

## Actual INT8 training and frozen high-precision state

Every update reports finite objective/CE/KL/gradient norm and gradient coverage W196/SA168/SW196/down-scale28. The saved learned state contains588 parameters,196 weight Adam states and392 scale Adam states, all at optimizer step400. No R, embedding, LM head or norm parameter is optimized. The legacy group label `SP2` contains28 **INT8** down scales here: the actual type-dependent prepare path enables INT8 scale learning.

The final package has **196 W4 matrices and196 static INT8 activation quantizers**. All196 packed-weight matrices,196 weight scales and196 activation scales differ from the exact PTQ parent. Model config and all **154 high-precision tensors, including56 Q/K norm tensors**, are bitwise unchanged from that parent. These checks distinguish actual full Uniform QAT from a late format overlay or a misleading optimizer-group name.

Result/progress report completed400, target not reached and no failure file. Final validation has262,208 targets, **PPL15.715716054138214 /NLL2.7546612342218633**. Independent segment-weighted recomputation matches exactly, and the official seed42 `Uniform-QAT400` summary row matches the exact final package and values.

## Completed Uniform external acceptance

Both `uniform-qat400-test-s42` and `uniform-qat400-c4-s42` report completed with no failure file. Their result/model records identify the exact step400 package used by validation, with196 W4/196 static INT8, no down overlay, KV16, `use_cache=false` and FP32 RoPE. All external calibration/training/candidate-selection flags are false.

All196 saved quantizer states match before/after each evaluation; test/C4 initial states also match one another, and all actual scales match the package. Every INT8 quantizer has loaded static scale, observer disabled and zero calibration samples. This uses direct tensor equality, not just `activation_scales_unchanged=true`.

Test IDs and complete metadata exactly equal corresponding Qwen BF16, both exact PTQ packages and SP2-QAT400: **299,078 inputs /298,931 targets**,146 full2048-token windows plus70-token tail. C4 uses the same existing Qwen token artifact: **2,097,152 inputs /2,096,128 targets**,1,024 full windows without tail. All five methods have matching segment structure/target counts. Every new Uniform segment NLL equals log(PPL), and target-weighted NLL/exp(NLL) recompute exactly. Summary's three Uniform-QAT400 rows match official model, seed42, INT8, WikiText-only, exact parent/final package, counts and raw metric precision.

Uniform-QAT400 final test is **PPL15.055184987698569 /NLL2.7117224493174277**; C4 is **PPL27.10478088408928 /NLL3.2997101287131727**. The validation artifact limitation remains explicit: QAT did not save separate validation IDs or before/after quantizer snapshots, so its package source, counts and segment structure are checked without claiming an additional validation tensor-ID comparison. No independent full-test rerun was added for Uniform.

## Completed matched Qwen seed42 core matrix

Each PTQ entry is the exact three-stage postprocessed package; each QAT entry is its own step400 descendant. B100 initial-format controls are excluded. Both QAT runs have **400 optimizer updates ×global batch8 ×2048 input tokens**, i.e.6,553,600 input visits and6,550,400 target visits, with identical train-window sequence and W LR schedule. Their teacher, objective, reference, optimizer rule, validation schedule and two-device placement are matched; the exact parent/activation format and resulting initial state differ. The inherited convergence snapshots in coordinate postprocessing are documented in the PTQ audit; no additional search budget was granted to either method.

| Method | WikiText validation PPL | WikiText test PPL | Fixed C4 PPL |
|---|---:|---:|---:|
| Original BF16 | 17.7149122731184 | 16.715764347250076 | 23.136144236064272 |
| Full SP2-PTQ | 14.935424095235582 | 14.302163393485294 | 24.903332773203157 |
| Full Uniform-PTQ | 17.491760837320335 | 16.78537625006989 | 29.322131492633147 |
| Full SP2-QAT400 | 14.813941218960903 | 14.151216160997155 | 24.24703443394153 |
| Full Uniform-QAT400 | 15.715716054138214 | 15.055184987698569 | 27.10478088408928 |

| Split | SP2−Uniform ΔNLL, PTQ | SP2−Uniform ΔNLL, QAT400 | SP2 QAT−PTQ ΔNLL | Uniform QAT−PTQ ΔNLL |
|---|---:|---:|---:|---:|
| WikiText validation | -0.15799411343775738 | -0.059092522577013806 | -0.008167135685384963 | -0.10706872654612853 |
| WikiText test | -0.16009723395145103 | -0.06192188114782882 | -0.010610243911363781 | -0.108785596714986 |
| Fixed C4 | -0.16334093111462034 | -0.11141581019173596 | -0.02670732230598638 | -0.07863244322887075 |

For this matched seed42 matrix, SP2 has lower NLL than Uniform in both PTQ and QAT400 on every split. QAT improves both formats from their exact PTQ parents; Uniform's larger recovery narrows the format gap but does not reverse it. Both final QAT methods remain above original BF16 on C4, even though their WikiText metrics are below BF16. This establishes the specified seed42 comparison, not statistical reliability across seeds, universal dataset improvement or superiority to an independently reproduced external method.

Prior accepted evidence is reused from `QWEN_PTQ_ACCEPTANCE_20260918.json`, `QWEN_SP2_QAT400_EXTERNAL_20260918.json` and the original BF16 records; their postprocessing/architecture checks were not repeated. The one independent complete SP2 seed42 test already passed as recorded by the verifier. Seeds43/44 remain required for the planned three-seed summary; this completed seed42 result requires no repetition or split-specific package switch.
