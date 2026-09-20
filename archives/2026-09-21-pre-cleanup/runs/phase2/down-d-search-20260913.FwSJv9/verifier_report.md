# Independent matched floor/ceil control verification

Verdict: PASS. Matched comparison supports an additional neighbor-domain benefit for the measured protocol.

Independently loaded control packed weights: all 16 record names match k0nDFs, all SW unchanged, other 15 down packed records identical; saved alpha JSON exactly matches 89KzK6. Selected step512 is the minimum actual train NLL among eligible trials. Runtime precision lists and preservation checks retain strict W4A8 semantics.

Compared actual loop_settings to neighbor run 9Zycd9: same parent, weight parent, target, calibration-window indices, 24/8 full-window row counts, milestones, SW and activation settings. Source dispatch uses the same implementation, with neighbor_search=False versus True. Initial weights/input background and NLL-selection procedure therefore match.

Independent weighted segment NLL and exp(NLL) reproduce the full-validation result over 252728 predictions, including the final tail segment.

| Condition | PPL | NLL |
| --- | ---: | ---: |
| Parent89KzK6 | 16.508818358818367 | 2.803894684130134 |
| Floor/ceil continuation512 | 16.491503247836636 | 2.802845293595272 |
| Neighbor8192 | 16.161902488363097 | 2.7826567744038146 |

Neighbor-minus-control PPL is -0.3296007594735393. Additional floor/ceil optimization on updated inputs recovers a small amount relative to parent, while the neighbor search yields a further improvement under the matched search/selection protocol. This is not a universal guarantee or an additive independent error decomposition. Parent-to-neighbor gain must not all be attributed to the expanded code domain without this control.

## Sequential neighbor-down family readiness

PASS source review of rounding_family.py neighbor_downs_nll branch and full16-layer CPU simulation. No verifier GPU work.

Simulation exercised parent weight/alpha initialization, 33 NLL calls (initial plus two candidates per layer), and checked every call's accepted prefix/current candidate/future-parent states and fixed alpha. Each layer starts solver from exact parent codes. A deliberately inferior last candidate forces best restoration before propagation; all16 restores passed. Reused identity scores match prior accepted score, saved reload matches final weights, final configure/measure state correct. Existing original-mode paths remain present.

Actual Ds4vSf measurements are not covered by this readiness result; final artifact/result verification remains separate. These results are matched prefill validation with KV16, not NPU performance or untouched external-test certification.
