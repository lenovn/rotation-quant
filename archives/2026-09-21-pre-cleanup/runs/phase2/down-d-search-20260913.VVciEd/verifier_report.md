# Independent static SP2 alpha-expansion verification

Verdict: PASS. Accept VVciEd as an improved matched-validation strict W4A8 configuration over weight parent k0nDFs. This is not an untouched test or NPU deployment certification.

Independent source review: sp2_alpha_loop.py, precision_loop.py dispatch/configure/measure, static SP2 codebook and real SpinQuant precision wrapper. No verifier GPU work or implementation edits. CPU tests ran with CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1.

## CPU and artifact evidence

- Full run_alpha CPU simulation inspected every one of 64 evaluator calls: previous accepted alpha prefix retained, only current alpha replaced, future original alphas retained, parent down weights and checkpoint non-down weights unchanged. Factor-1 score reuses preceding accepted score correctly; selected alpha explicitly restored after each layer. Original alpha map untouched. Final alpha save/reload and single final measurement path passed.
- Independently audited all 16 actual layer files and all 80 trial records. Each factor list equals [1,1.125,1.25,1.5,2], trial alpha equals original alpha times factor, factor-1 score equals prior selected score, selected factor minimizes actual eight-window train NLL, and final saved alphas equal selections.
- Expanded layers are 0:2, 2:1.5, 3:1.25, 13:1.25; all other 12 retain original alpha. Final selected train NLL is 2.7730014588220326.
- Weight parent remains k0nDFs with all 112 SW, its 16 down integer-code records and all 96 non-down SA frozen. Earlier same-session CPU artifact check established each parent down SW exactly equals original learned C SW. Only SP2 alpha changes; no teacher, gradients, MSE fitting, code changes or high-precision exceptions.
- Independent segment-weighted NLL and exp(NLL) recomputation passed; 252852 total tokens and 252728 predictions with final 948-token segment retained. Both restored precision lists empty; runtime frozen_checks_pass=true.
- budget.json records one full validation, 242.33267829200486 seconds, 4.072265625 GiB peak reserved. GPU timing is runner evidence, not a verifier rerun.

| Configuration | PPL | NLL |
| --- | ---: | ---: |
| k0nDFs frozen weight parent | 17.134821523344698 | 2.8411127393601405 |
| VVciEd expanded static SP2 | 16.701812082653035 | 2.815517221479444 |

PPL improvement is 0.4330094406916629; NLL improvement is 0.0255955178806966. Parent delta fields independently checked. Legacy delta fields still reference original C.

The down_scales.json is an activation overlay requiring parent integer weights and original checkpoint/non-down SA. It is not a standalone model export. Expanded alpha never shrinks thresholds. For fixed input this cannot increase out-of-range saturation, but changed upstream activations can alter downstream clipping counts; no claim is made that whole-model observed saturation necessarily decreases.

## Bounded boundary-followup readiness

PASS independent CPU mock of updated sp2_alpha_boundary branch. It resolves separate alpha and weight parents, loads alpha parent's scales over supplied original scales, searches only previous factor-2 layers (actual VVciEd: layer 0), and keeps other 15 parent alphas fixed. Four candidates [2,4,8,16] times current alpha plus reused identity, best restoration, output reload and one final measurement were exercised. This readiness check does not certify forthcoming boundary GPU results. The branch is reviewed for the requested one-step continuation from VVciEd, not a general recursive search procedure.
