# Independent validation acceptance verification

Final verification status: **PASS** for implementation, model configuration,
and numerical accounting. This does not assert a user-defined accuracy threshold
has been met; no threshold was specified.

Application and verifier roles were separate. The verifier added only
`scripts/phase2/test_validation_acceptance.py` and this report.

## CPU checks

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider scripts/phase2/test_validation_acceptance.py -q
```

Result: **6 passed in 2.58s**. Independent stub-evaluator tests verify complete
windows plus partial tail, token-weighted NLL, complete windows only, tail only,
single-token unscorable tail, insufficient input rejection, exception-safe
sequence-length restoration, and unchanged input args. Launcher syntax passes
`bash -n`.

Independent CPU inspection of C's source RTN checkpoint checks all **112 Linear
weights / 973078528 elements**: metadata is symmetric 4-bit, per output channel,
groupsize -1, with scale shape `[out,1]`. Recovered integer codes are all in
`[-8,7]`; multiplying them by saved channel scales and casting to stored BF16
reproduces every saved weight exactly via `torch.equal`.

Source review confirms W16A16 uses original BF16 weights with no rotation and
bypasses the PTQ preparation function. W4A8 loads C RTN/R/non-down SA and the
already-selected SP2 scales; no calibration or parameter selection runs on
validation. Runtime checks compare loaded weights with the checkpoint, validate
96 non-down static per-tensor A8 quantizers, install 16 static SP2 quantizers,
and compare all quantizer buffers before and after evaluation.

## Completed run

Both mode result JSON files equal the corresponding combined summary entries
and explicit `VALIDATION_ACCEPTANCE` JSON lines in their logs. Both consume
**252852 validation tokens**, partitioned into 123 full 2048-token windows and
one 948-token tail. There are **252728 predicted tokens**: 251781 from full
windows plus 947 from the tail. Window-initial tokens have no preceding context
and are not scored. The independent recomputation of segment-weighted NLL and
its exponential matches saved totals exactly.

| Configuration | Validation PPL | Validation NLL |
| --- | ---: | ---: |
| Original W16A16 (BF16) | 13.6346577256 | 2.6126149134 |
| C RTN W4, non-down INT8, down SP2-A8 | 17.6423999759 | 2.8703050944 |

Overall degradation is **+4.0077422502 PPL**, **+29.393787% relative PPL**, and
**+0.2576901810 NLL per predicted token**. These differences were independently
recomputed from the two results. Runtime reports 112 quantized Linear weights,
96 non-down INT8 inputs, 16 down SP2 inputs, weights matching checkpoint, and
unchanged activation quantizer buffers.

The measurement is teacher-forced prefill on WikiText-2 validation using
disjoint context windows. The baseline and candidate share identical token and
window accounting. NLL is reconstructed from the original evaluator's FP32 PPL
for each segment, so it includes that exponent/log rounding. This is total model
degradation, with no W/A attribution, and is not a phone-backend latency or
task-accuracy measurement.
