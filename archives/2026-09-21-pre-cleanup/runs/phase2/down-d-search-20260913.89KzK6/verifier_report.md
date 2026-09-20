# Independent boundary-alpha verification

Verdict: PASS. Accept 89KzK6 as the better measured strict W4A8 configuration over VVciEd. This is matched validation evidence, not external test or NPU certification.

Independent checks: reviewed actual parent/weight-parent loading and alpha isolation; earlier complete CPU branch simulation confirmed four candidate calls, unchanged other 15 alphas, selected restoration and saved reload. Current CPU/read-only artifact review confirmed only model.layers.0.mlp.down_proj changes from VVciEd, with factors [1,2,4,8,16] and identity score exactly 2.7730014588220326. The chosen factor 8 sets alpha=25.76784531118807; factor 16 alpha=51.53569062237614 worsens train NLL from 2.765303596393192 to 2.7739136898008456. All other 15 alpha values exactly match VVciEd. Saved selection_summary matches selected trial.

All 112 weight scales and the k0nDFs integer weights remain frozen, along with 96 non-down activation scales. No gradients, threshold shrinking or precision exceptions. Fixed inputs have non-decreasing thresholds, but no claim is made about downstream observed clipping counts when inputs change.

Independent segment-weighted NLL and exponentiation reproduce PPL, with 252728 predictions and retained tail. Both precision-restoration lists empty and runtime frozen_checks_pass=true. GPU runtime assertions are reviewed evidence, not an independent GPU rerun.

| Configuration | PPL | NLL |
| --- | ---: | ---: |
| VVciEd parent | 16.701812082653035 | 2.815517221479444 |
| Boundary expansion | 16.508818358818367 | 2.803894684130134 |

Runner budget records one full validation, 78.39811332599493 seconds and 4.287109375 GiB peak reserved. Artifacts are alpha overlays requiring declared k0nDFs weights and original non-down scales.

## Neighbor-code branch readiness

PASS source review and independent CPU mathematics. Reviewed neighbor_rounding_loop.py and optional initial_codes/neighbor_search changes to fixed_grid_rounding.py. Default floor/ceil path remains unchanged. Neighbor candidates are code +/-1 clipped to [-8,7], use actual BF16 dequantization deltas, exact quadratic per-coordinate cost 2*d*g+d^2*Hjj and rank-one gradient updates. No autograd optimization.

Independent test across 12 random BF16 problems compared the first selected coordinate step against direct exhaustive per-row +/-1/0 objective evaluation for every candidate column. Objectives agree within 1e-5; all subsequent measured fit MSE values non-increasing within 1e-6, initial codes unchanged, codes remain in INT4 range.

Source review confirms alpha parent loads 89KzK6 while weight parent resolves k0nDFs; actual parent SP2 quantizer outputs are captured on 24 fit/8 heldout windows. Every NLL candidate reconfigures the full model with fixed parent alphas/SA and explicit weight replacements. Only layer-1 down codes may change. Identity is included, heldout MSE merely prefilters, actual eight-window train NLL selects. Saved code reload checks all scales and unaffected codes before final configure/measure. Readiness does not certify forthcoming neighbor GPU results.
