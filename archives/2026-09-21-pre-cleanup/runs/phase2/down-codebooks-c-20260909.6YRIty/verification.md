# Independent verification

Final status: **PASS** for implementation and this numerical experiment.

Application code and this verifier were handled by separate roles. The verifier
only added `scripts/phase2/test_down_codebooks.py`; no application code was edited.

CPU command (2026-09-09):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider scripts/phase2/test_down_codebooks.py -q
```

Result: **7 passed in 4.57s**.

The tests independently construct exact rational PoT/SP2 sets from paper formulas.
PoT A8 has 255 signed values; SP2 A8 with sign/field allocation 1/4/3 has 94
nonnegative unique values and 187 signed values. Duplicate SP2 sums are not counted
as new numerical levels. SP2 retains the 0.75-to-1 gap. Random exhaustive
nearest-neighbor comparisons, all adjacent midpoint ties, minimum PoT level,
clipping, signs, BF16 output casts, noncontiguous tensors, and the existing signed
INT8 range pass. This tests the adapted numerical projection rather than claiming
hardware acceleration or a complete reproduction of the paper's training method.

A two-layer synthetic model verifies that calibration collection and held-out
damage hooks preserve actual reference outputs; local output MSE matches direct
matrix calculations at both layers. Calibration runs exactly 50 candidates per
format and returns the candidate minimizing the stated BF16-input/FP32-linear
output-error objective.

Source review confirms the launcher loads C's RTN W4 checkpoint, C's R, and C's
non-down SA. Actual post-load non-down scales are compared with the export. Scale
selection uses train, local damage uses validation, and PPL uses test. All down
quantizers are installed together for each candidate PPL; original quantizers are
restored afterward and non-down scales are rechecked. No blocking defect was
identified before the real run. These source checks alone do not establish that
the real model experiment completed successfully.

## Real experiment verification

The completed `results/summary.json`, per-format PPL JSON, complete experiment
log, 16 calibration JSON files, saved scale JSON, and held-out local-damage JSON
were independently inspected. Direct assertions verify:

- Source paths consistently name C RTN W4, C rotation, and C activation scales.
- All 96 non-down scales remained unchanged; all 16 down layers were evaluated.
- Every layer has 50 candidate evaluations per format, selected alpha matches
  `down_scales.json`, and selected loss is the minimum recorded candidate loss.
- Each layer/format measured 8 validation windows, 134217728 input elements
  (8*2048*8192), and 33554432 output elements (8*2048*2048).
- PPL JSON matches summary and explicit per-format log entries.
- Reloaded down-A16 reference PPL is exactly 14.655250549316406, matching C RTN's
  existing reported reference.

| All-down format | PPL | Aggregate held-out linear output MSE |
| --- | ---: | ---: |
| A16 reference | 14.6552505493 | — |
| INT8 | 42.3567810059 | 0.0016165880865 |
| PoT | 16.8535957336 | 0.0011816991500 |
| SP2 | 15.7965917587 | 0.0005817837704 |

SP2 minimizes held-out local output MSE in 14 of 16 layers; PoT wins layer 0
and INT8 wins layer 6. PoT's selected alpha in layers 13 and 15 is the search
upper endpoint, 2*full_absmax: its result is limited to the chosen search range.
No claim of globally optimal scales, backend latency, or generalization beyond
the evaluated windows is made. The severe INT8 PPL loss is a measured outcome,
not a validation failure; the local-error and PPL checks use different input
trajectories by design.
