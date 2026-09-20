# Independent verifier report

Verdict: PASS implementation/protocol and saved-artifact checks; NO PROMOTION of sparse D.

Scope: independently reviewed scripts/phase2/sparse_d_sp2.py, precision_loop.py configure/measure/residual_diagnostics, down_d_search.py helpers, fixed_grid_rounding.py and the actual SpinQuant input-wrapper precision branch. CPU only; no GPU execution, installation or implementation edits by verifier. Live nested SpinQuant HEAD: 24918316ed594848d4de797c356b120f2a4ee0f3.

## Protocol

- Five strengths 0/.25/.5/.75/1, selected channel 1976 from first 24 full train windows; disjoint final 8 train windows used for local heldout MSE then end-to-end NLL selection. This heldout data participates in selection and is not an untouched generalization estimate.
- Paired up/D and down*D begin from unquantized fixed-R weights for every candidate. Gate/other weights and original non-down SA/SP2 alpha retained. GM(D)=1 and [.25,4] checked. Up SW transports C/D; down SW is max(original C SW,current BF16 transformed row absmax/7).
- D=I applies the same SW and rounding rule; it is intentionally distinct from the frozen parent. Runtime records show 1258 down SW rows expand even at D=I.
- Actual quantized gate/up and SP2 down-input values feed rounding. Milestones 0/512/2048/8192 are local-MSE selected; eight-window end-to-end train NLL chooses strength .25.
- configure resets every candidate from the fixed checkpoint plus explicit replacements, preventing prior-candidate state leakage. Parent records copied before replacing only layer-1 up/down.
- Two full validations completed; budget.json records 216.15464245300973 seconds, 5.728515625 GiB torch peak reserved, 6330 MiB process peak. These are runner-recorded GPU metrics, not verifier reruns.

## Independent CPU verification

Executed existing Python environment with PYTHONDONTWRITEBYTECODE=1 and CUDA_VISIBLE_DEVICES='':

- FP64 paired-fusion functional equivalence on small random SwiGLU tensors.
- All 16 signed INT4 values pack/unpack exactly.
- Four residual restoration configurations retain parent down weights except where W16 restoration is requested; subsequent strict reconfiguration clears A16 state.
- Actual RotationStaticActQuantizer exercised for non-down state; SpinQuant wrapper source confirms bits<16 controls down A16 bypass.
- Coordinate rounding fit MSE monotonically decreases on a small independent random case.
- Reloaded both saved packed_model.pt artifacts on CPU. Exactly 17 replacement records, 15 untouched parent down records bit-identical, all nibble roundtrips exact; D bounds/GM1, identity D, transported up SW and non-shrunk down SW verified.
- Independently recomputed token-weighted NLL and exp(NLL) from both segment records; 252728 predicted tokens from 252852 total tokens, final 948-token segment retained.

## Results and decision

| Condition | PPL | NLL | Delta PPL vs parent |
| --- | ---: | ---: | ---: |
| Frozen parent k0nDFs | 17.134821523344698 | 2.8411127393601405 | 0 |
| Same-rule D=I | 17.17204209888376 | 2.8432826019598076 | +0.037220575539063105 |
| Selected D=.25 | 17.155105697678067 | 2.8422958376250413 | +0.020284174333369265 |

D=.25 improves on D=I by about 0.0169364 PPL but does not improve on the accepted parent. Retain the parent. Existing result delta_ppl/delta_nll fields refer to original C PPL 17.64239997588965; they must not be interpreted as deltas versus parent.

Residual-diagnostics branch readiness: PASS source and CPU state checks. This does not certify its forthcoming GPU measurement results. Restoration modules are correctly excluded from parent replacements. No activation calibration or gradient training occurs. High-precision cases remain diagnostics, not strict W4A8 candidates.

Limits: no independent full-model GPU rerun. Frozen full-model weight/quantizer checks and FP64 actual-model equivalence are supported by reviewed runner assertions and completed logs. Packed artifacts are overlays requiring their declared base checkpoint and original activation-scale/SP2-alpha sources; they are not standalone NPU exports. Validation has been used repeatedly in the broader experiment loop, so this is matched validation evidence, not an untouched test estimate.
