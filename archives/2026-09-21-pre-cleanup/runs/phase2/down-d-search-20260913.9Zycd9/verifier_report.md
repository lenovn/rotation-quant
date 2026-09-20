# Independent neighbor-code verification

Verdict: PASS implementation/artifact/measurement checks; retain as improved measured strict W4A8 configuration. Attribution specifically to relaxed neighbor-code domain remains pending matched floor/ceil continuation control.

Reviewed neighbor_rounding_loop.py and fixed_grid_rounding.py. Prior independent CPU check used 12 random BF16 problems: exact first-step objective agrees with direct exhaustive row +/-1/0 evaluation for every column, subsequent fit MSE non-increasing, initial codes not mutated, all codes within [-8,7]. Quadratic coordinate cost and rank-one gradient update are valid; no autograd training. Existing default floor/ceil branch remains available.

Parent alpha configuration is 89KzK6; integer weight parent is k0nDFs. Actual SP2 quantizer outputs under the new alpha background are captured from 24 fit and 8 heldout full train windows. Candidate 0 uses exact parent codes and score. MSE only prefilters; actual eight-window next-token train NLL chooses among evaluated milestones. Every candidate resets full model state via configure.

Actual train NLL: identity 2.765303596393192; step512 2.7388301210590185; step2048 2.7401242659847846; selected step8192 2.7387435590788605. Selection is consistent with minimum actual train NLL, not minimum MSE alone.

Independent CPU artifact reload confirms exactly 16 down records; all SW exactly equal weight parent, other 15 packed records byte-identical, all INT4 records roundtrip. Target layer-1 down has 2898198 changed codes, final range [-8,7], maximum net code movement 13 after successive neighboring steps. Saved alpha JSON exactly equals 89KzK6. Thus no scale change or alpha change explains this incremental run.

Independent token-weighted NLL and exp(NLL) calculation passes, with 252728 predictions, 252852 total tokens and retained final 948-token segment. Both restoration lists empty; runtime frozen_checks_pass=true.

| Configuration | PPL | NLL |
| --- | ---: | ---: |
| 89KzK6 parent | 16.508818358818367 | 2.803894684130134 |
| Neighbor selected8192 | 16.161902488363097 | 2.7826567744038146 |

PPL delta is -0.3469158704552697; NLL delta -0.021237909726319337. progress.json records completion, one full validation, 128.90671524504432 seconds, 4.869140625 GiB peak reserved. No verifier GPU rerun; runtime preservation assertions are reviewed runner evidence.

Matched control readiness: rounding_continuation dispatch uses the same run_neighbor function, same parent loading, inputs, milestones, identity and NLL selection; neighbor_search=False selects the preserved floor/ceil code branch. No other experimental-state difference identified in source. The control is needed because recalculating the objective on expanded-alpha inputs and spending additional optimization steps can also yield improvement. Do not attribute the full parent-to-neighbor change to the unrestricted code domain before that control finishes.

Limits: artifacts require the declared base model and non-down SA; this is prefill/KV16 matched validation rather than NPU or untouched external-test evidence.
