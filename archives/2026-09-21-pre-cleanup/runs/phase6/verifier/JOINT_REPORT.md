# Phase6 joint training independent verification

CPU tests: **6 PASS in 6.79s** on 2026-09-21. Command:

`PYTHONDONTWRITEBYTECODE=1 runs/phase5/env/bin/python -m pytest -p no:cacheprovider -q runs/phase6/verifier/test_joint.py`

Application source reviewed: `scripts/phase6/joint.py` and its imported Phase3 initialization, freezing, quantizer, evaluator, and scheduler functions. No application code modified by verifier; no GPU started by verifier.

Coverage:

- Explicit INT16 down wrapper prehook is executed once, including after reinstallation; existing wrapper does not quantize twice.
- Signed INT8/INT16 forward matches FP32 grid/clipping formula, and backward matches analytic LSQ scale and input gradients. BF16 input still yields FP32 scale gradient.
- Freezing down-SA preserves scale, removes scale gradients, and retains input gradients.
- Synthetic frozen package export/cold-load preserves outputs, bitwidths and nonlearnable scales. Missing activation coverage or learned parameter coverage rejects load.
- Calibration windows are 32 deterministic full 2048-token windows drawn exclusively from the first 800 training windows. Training steps index eight consecutive windows modulo 800; scheduler uses fixed horizon 512 with 10 warmup updates.
- Static source review confirms pretrained weights frozen by build_training_model, reproducible seed42 initialization there, joint scales separated from R, and fixed-down disables only DOWN_SA among otherwise trainable joint groups.

Blocking implementation issue found and reported to coordinator: `checkpoint_eval` evaluates full validation before obtaining its step0 reference. Existing evaluator leaves embedding and decoder layers on CPU; `backbone(frozen, probe.cuda())` then has mixed devices. Requires restoring model to CUDA or changing operation order before initial cold-reload comparison. CPU tests pass but application milestone remains blocked until this is fixed and its real startup path succeeds.

CPU tests do not establish real-model cold-reload equality or CUDA memory budget. In particular historical v_proj storage layout can affect exact BF16 equality and needs the coordinator's actual cold-load check.

## Revision review

- Device bug fixed with `frozen.cuda()` before reference probe. Added a regression test that simulates evaluator CPU offload and rejects probe execution before restoration. Updated narrow suite: **7 passed in 7.52s**.
- R-only export now calls `freeze(..., current_minmax=True)` after step0, forwarding to the established weight minmax packing; activation recalibration follows.
- `evaluate.py` uses the same legacy zero-shot choices: batch4, max_length8192, no BOS/chat template, seed42, use_cache=False. Independently resolved all 10 BF16 evidence paths for each of Llama and Qwen; all exist, eight capability task configs are zero-shot. Reused PPL evidence is WikiText test and C4 validation, with target counts Llama 288934/2096128 and Qwen 298931/2096128.
- `launch.py` initially allowed nonpositive process memory fractions at 88–90% existing GPU use. Reported and now fixed by rejecting fraction<=0.
- Resume is now one atomically replaced file containing parameters, optimizers, RNG, arm and step. This removes the two-file cross-step inconsistency.

Remaining resume defects reported to coordinator: direct parameter copying bypasses prior coverage/shape/finite/positive validation; and a crash after saving step64/512 resume but during checkpoint evaluation leads resume to skip that pending evaluation, potentially reporting completed without a final result. These are recovery-path issues, not evidence of incorrect uninterrupted training updates. No GPU work performed by verifier during this revision review.
