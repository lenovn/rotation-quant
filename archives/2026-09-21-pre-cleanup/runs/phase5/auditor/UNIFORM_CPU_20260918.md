# Phase5 Uniform implementation and protocol audit

Independent auditor, 2026-09-18. Application source: `worktrees/SpinQuant-multimodel`. Added only `tests/test_phase5_uniform.py`; no application edits, no GPU, no old suite, no modification to old data/model artifacts.

**Limited CPU PASS: all 13 distinct narrow cases passed.** This is not a real Qwen/GPU/multiple-device training result or final package acceptance.

## Commands and outcome

Environment: `runs/phase5/env/bin/python`, inherited PyTorch plus isolated Phase5 dependencies. CWD: `worktrees/SpinQuant-multimodel`.

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/dongpeiyan/projects/rotation-quant/runs/phase5/env/bin/python -m pytest -q -p no:cacheprovider tests/test_phase5_uniform.py
```

First run: **9 passed, 1 failed in 10.27s**. Log: `runs/phase5/auditor/uniform_cpu_20260918.log`. During test development the main executor added `place_student`, which the CPU training-loop harness had not mocked. Failure was the expected device-visibility check on a CPU-only harness, not a failed numerical application test. Added only a CPU placement stub to this harness; no application change for this failure.

Then ran the one failed test plus three new export/external-format cases:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/dongpeiyan/projects/rotation-quant/runs/phase5/env/bin/python -m pytest -q -p no:cacheprovider tests/test_phase5_uniform.py -k 'fixed_qat400 or uniform_export_package or external_records'
```

Result: **4 passed, 9 deselected in 10.25s**. Log: `runs/phase5/auditor/uniform_cpu_budget_export_20260918.log`. Previously passing cases were not rerun. `git diff --check` was clean at audit time.

## What was actually exercised

1. SP2 and INT8 packages independently saved and cold-loaded into a model initialized with the opposite down format. Actual quantizer classes, wrapper forward outputs, all weight records and high-precision state match; format descriptions reflect actual counts.
2. `distill.prepare_student` retains the actual down operator. Both SP2 and static INT8 produce finite nonzero scale gradients and scale updates; master weights get gradients, high-precision parameters stay frozen. Historical optimizer group `SP2` contains all down scales, including INT8, but does not select the operator.
3. Neighbor-code input capture attaches to the inner Linear and receives **quantized** inputs produced by the installed SP2/INT8 quantizer. Tested 3/1 synthetic fit/heldout split and hook removal.
4. Range selection runs actual SP2 or static INT8 operators at factors 1/.875/.75/.5/.25/.125, compares to independent quantizer instances, and restores the parent when every candidate is worse. No conversion from INT8 to SP2 occurs.
5. Uniform calibration data loading reads only `wikitext-train.arrow`, follows recorded window indices/order and 128-token prefixes, and invokes no-special-token row tokenization. Synthetic loader rejects every other file. No actual C4/test files are involved.
6. `uniform_baseline.run(export_package=True)` executes the real 50-candidate INT8 range search and exports a full cold-loadable all-INT8 package. W4 packed codes/scales and non-down SA are unchanged; down scale equals the saved train-only overlay. Model capture/GPU accounting are stubbed for this narrow export test.
7. External evaluation execution for SP2 and full-INT8 cold-package identities (without overlay) records the correct actual activation format counts and unchanged scales; training/calibration/selection remain false. Model compute is stubbed; this checks coverage and export metadata, not PPL.
8. The real `distill.train` orchestration ran **400 optimizer updates × 8 microbatches × 2048 synthetic token positions = 6553600 token visits**. Synthetic scalar objective and CPU placement replace expensive model compute. All 3200 indices follow `(800+i) % train_pool`; checkpoint callbacks are exactly 100/200/400. Even callbacks reporting PPL zero do not stop training when `target_ppl=None`. CLI default is verified as `None`.

Joint100 was read-only inspected, not executed in this audit: `run.train` indexes `(step * accumulation + microstep) % train_windows`, divides loss by accumulation and records cumulative `(step+1)*accumulation*2048`; fixed 100×8 implies 1638400 visits. Real launcher arguments and GPU logs still need to confirm actual production budgets. No claim of numerical Joint100 training validation is made here.

## Findings and resolutions

- **F-U01, fixed:** newly added `isinstance(..., SP2Quantizer)` in `distill.prepare_student` initially lacked the import, which would raise `NameError` before training. Reported immediately; main executor added import. Both actual-format learning tests pass after the fix.
- **F-U02, fixed:** inherited hardcoded `112/96/16 SP2` result labels would misdescribe Qwen or Uniform packages. Main executor added actual-format descriptions in common/distill/sequential/postprocess. Cold-package description cases pass.
- **F-U03, current source corrected; runtime delegated:** an earlier live read showed the legacy final `model.cuda()` after `model.cpu()`, which would lose multiple-device placement. On follow-up, current `distill.py` (mtime 2026-09-18 01:55:03 +0800) ends checkpoint evaluation with `place_student(model, getattr(args, "layer_devices", 1))`; the main executor had concurrently corrected it. This audit confirms the current source restoration call, while actual device transfer/optimizer resume is for the architecture verifier. Do not extend this CPU Uniform PASS to multiple-device behavior.
- **Parameter note:** `sequential_postprocess --alpha-factors` retains historical expansion defaults. Phase5 launchers must explicitly pass the fixed contraction set for both SP2 and INT8, as PLAN declares. Tests used that declared contraction set; main executor confirmed all formal commands will pass it. Main executor also confirmed only step400 receives full validation in the production run, with 100/200 kept as recovery states.

The final experiment matrix remains incomplete until real matched Uniform postprocessing/QAT400 and external test/C4 evaluations finish. The existing B100 INT8 C4 result still does not fill either final Uniform row.
