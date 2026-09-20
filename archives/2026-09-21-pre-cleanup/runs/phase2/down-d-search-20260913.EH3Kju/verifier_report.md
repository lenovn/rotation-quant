# Independent frozen C4 verification

Verdict: PASS external same-token measurement protocol and arithmetic. Frozen Ds4vSf improves over historical C+SP2 on this fixed C4 sample, while remaining worse than BF16.

Parent is Ds4vSf, selected before this C4 run. Reviewed frozen_c4_loop.py and actual settings: loads saved parent integer records and alpha, verifies original learned SW, keeps96 non-down SA, and performs no C4 calibration or candidate selection. All input quantizers remain A8, weights and quantizer buffers are asserted unchanged after evaluation. No verifier GPU rerun; these full-model preservation checks are reviewed runtime evidence.

Independent CPU readiness loaded actual saved int64 tokens [1,2097152], compared metadata with historical BF16/C results, and exercised evaluator chunk routing on those real tokens. All8 chunks of128 disjoint2048-token windows matched exact slices and historical segment order.2096128 predicted tokens, length restoration and weighted arithmetic passed.

Final independent artifact checks confirmed exact historical input_token_path, metadata and segment starts/lengths/windows/prediction counts. Recomputed weighted NLL, exp(NLL), and original-C deltas agree. All1024 windows evaluated; no unscored tail. Runtime frozen_checks_pass=true and external_evaluations=1. budget.full_validation_evaluations=0 refers to the WikiText counter and does not mean no C4 evaluation occurred.

| Same-token condition | PPL | NLL |
| --- | ---: | ---: |
| Historical BF16 | 21.843397624525274 | Historical reference |
| Historical C+SP2 | 29.539535457544623 | 3.385729551100929 |
| Frozen Ds4vSf | 27.062581939332095 | 3.2981520335394494 |

Delta versus C: PPL -2.476953518212529 (-8.385214864913848 percent); NLL -0.08757751756147947. PPL remains5.219184314806821 above BF16. Historical controls were reused from the exact saved-token protocol, not rerun.

Runner budget:209.54867263202323 seconds,9.48046875 GiB peak reserved,9144 MiB process peak. This is fixed-sample C4 validation with KV16 and use_cache=False prefill; it is not native NPU/decode performance or an exhaustive distribution guarantee. Candidate was not changed using these C4 results during this measurement.

## Non-down sensitivity branch readiness

Independent CPU PASS on a full112-wrapper,48-case simulation using actual configure and RotationStaticActQuantizer. Each case restores only its requested W16 group while all activation quantizers remain A8; other weights reset to parent. A preexisting non-down packed replacement was included to verify restored keys cannot shadow FP restoration. No case carryover observed. Top-two selection exercised two3-matrix qkv groups, confirming maximum6 matrices; final parent configuration restored. No GPU diagnostic results are certified by this readiness check. W16 conditions are localization diagnostics only, not deployable candidates, and no C4 scores enter their selection.
