# Independent sequential neighbor-family verification

Verdict: PASS. Ds4vSf improves matched strict W4A8 validation over parent9Zycd9 and can be retained as the new measured candidate.

Source and CPU readiness were independently established with a complete16-layer simulation: parent initialization,33 NLL calls checking accepted prefix/current trial/future-parent weight states, fixed alphas, exact parent initial codes, best restoration after deliberately inferior last candidate, propagation, save/reload and final configuration all passed. No verifier GPU execution.

## Actual artifact audit

- Initial actual eight-window train NLL=2.7387435590788605, exactly equal parent9Zycd9 final selected train score.
- Independently checked every one of16 layer files: trial0 identity score equals preceding accepted score; selected step minimizes evaluated true train NLL; zero-change identity preserved. Final train NLL=2.730179279517435.
- Accepted layers/steps: 0/8192,1/2048,3/8192,4/512,9/512,10/8192,11/2048,12/512,14/2048. Layers2,5,6,7,8,13,15 retain identity.
- Loaded all16 packed records on CPU. Names, shapes and every SW exactly equal parent; nibble roundtrip passes; actual code differences match each selected trial's changed_codes. Identity-selected layers have byte-identical packed codes.
- Saved SP2 alpha JSON exactly equals parent. Non-down weights and96 SA remain frozen through reviewed configure/runtime checks.
- Independently recomputed full-validation token-weighted NLL and exp(NLL);252728 predictions,252852 total tokens, retained final948-token segment. Both restored precision lists empty and runtime frozen_checks_pass=true.

| Configuration | PPL | NLL |
| --- | ---: | ---: |
| Parent9Zycd9 | 16.161902488363097 | 2.7826567744038146 |
| Sequential neighbor family | 16.122226577621436 | 2.7801988527027004 |

Delta PPL=-0.03967591074166066; delta NLL=-0.002457921701114251. Runner records one full validation,961.9992316610296 seconds,5.91796875 GiB peak reserved and6522 MiB process peak. These timings and full-model runtime assertions are reviewed runner evidence, not verifier GPU reruns.

Interpretation: small aggregate validation improvement; accepted train-prefix changes are conditional and cannot be added as independent contributions. All-W4A8/KV16 prefill validation does not certify NPU execution or untouched-test generalization. Saved weights/alphas require their declared base checkpoint and non-down scales.
