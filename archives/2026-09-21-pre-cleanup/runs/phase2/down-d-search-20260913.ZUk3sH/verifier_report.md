# Independent residual-diagnostics verification

Verdict: PASS for the four completed diagnostic measurements and their implementation/state protocol. None of these high-precision conditions is promoted as a strict W4A8 candidate.

Parent: runs/phase2/down-d-search-20260913.k0nDFs, rounded_downs PPL 17.134821523344698 and NLL 2.8411127393601405. The reused full_a8 result exactly matches every original parent field. Source: scripts/phase2/precision_loop.py residual_diagnostics branch, configure and measure; actual SpinQuant ActQuantWrapper bits<16 branch. Verifier performed CPU/read-only inspection only, plus this report write; no GPU rerun or implementation edits.

## Shared state and execution

- Parent's 16 down packed-weight records load once. Independent CPU comparison confirmed every stored down SW equals the original learned C SW exactly. Non-down weights come from the same original C checkpoint.
- Four cases restore respectively 112 A16 inputs, 16 down A16 inputs, 16 down W16 weights, and 96 non-down W16 weights. Every other input/weight remains on the shared parent background. Original 96 non-down SA and 16 SP2 alpha retained where active, with no calibration or selection.
- down_w16 correctly removes restored modules from replacements before configure, so parent W4 replacements cannot shadow restored FP weights. Each case resets all 112 wrappers, preventing precision or weight-state carryover.
- Earlier same-session CPU tests independently exercised all four restoration combinations and the subsequent strict reset using actual RotationStaticActQuantizer; all passed. Actual wrapper source confirms A16 bypass via bits<16.
- measure verifies exact expected weights, requested precision, and unchanged quantizer buffers after evaluation. All four runtime frozen_checks_pass flags are true. This is reviewed runner evidence, not an independent GPU rerun.
- progress.json records precision_loop_completed, four full validations, 127.16195278195664 seconds and 4.189453125 GiB peak reserved. The parent result was reused, not rerun.

## Independent output checks

For all five result records, independently recomputed token-weighted segment NLL and exp(NLL). All agree within 1e-12 NLL / 1e-10 PPL. Each has 252852 total tokens, 252728 predicted tokens, 123 full 2048-token windows plus a 948-token final segment, and no unscored tail. All four individual result.json files exactly equal their aggregate entries. Restoration lists exactly match loop_settings.json. delta_parent fields correctly reference the accepted parent; legacy delta fields continue to reference original C.

| Condition | PPL | NLL | NLL change vs parent |
| --- | ---: | ---: | ---: |
| Parent strict W4A8 | 17.134821523344698 | 2.8411127393601405 | 0 |
| All A16 | 15.863821573150512 | 2.764041143874271 | -0.07707159548586962 |
| Down A16 | 15.922558633306572 | 2.7677368856749784 | -0.07337585368516208 |
| Down W16 | 15.77839572928717 | 2.758641645438812 | -0.08247109392132845 |
| Non-down W16 | 16.548714018571108 | 2.8063083959986814 | -0.0348043433614591 |

## Interpretation boundary

Both down activation quantization and down weight quantization retain meaningful conditional recovery potential on this parent. Non-down W16 also improves NLL, so residual weight effects are not exclusive to down projections. All-A16 versus down-A16 differs by only about 0.00369574 NLL on these measured backgrounds, suggesting down inputs dominate the measured activation recovery; this is a conditional comparison, not a proof of an independently attributable percentage.

Recovery values cannot be added: weight/activation errors interact, and restoring one family changes subsequent activations. These are matched prefill use_cache=False, KV16 validation diagnostics. They are not all-W4A8 deployment results, NPU measurements, or an untouched external test. The accepted strict parent remains unchanged.
