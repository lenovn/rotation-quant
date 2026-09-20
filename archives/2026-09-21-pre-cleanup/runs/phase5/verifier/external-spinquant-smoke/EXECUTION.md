# Local SpinQuant external candidate smoke

Source: `/home/dongpeiyan/projects/rotation-quant/repos/SpinQuant`, HEAD `24918316ed594848d4de797c356b120f2a4ee0f3` plus preserved dirty source (`source.diff`, `result.json:git_status`). Python: old `rotation-quant-p0`, torch 2.4.1+cu121 / Transformers 4.44.2. GPU2 only.

Completed one 2048-token WikiText train W16/dynamic asymmetric A8/KV16 R-only update, 17 trainable R tensors, SGDG LR1.5, grad clipping1, no formal100-step schedule. All17 gradients finite and nonzero; all17 parameters actually changed. Saved `R.one_update.bin` and `input_windows.pt`.

Completed phase: GPTQ W4/group32 with native weight clipping, one native seed42 WikiText train calibration window. All16 layers /112 backbone matrices completed through the existing `ptq_model`/`gptq_fwrd`. 112 dynamic asymmetric inputA8 wrappers enabled before GPTQ,16 online down Hadamard, o_proj group64. No low-bit K/V or post-RoPE wrapper. Embedding/head remainBF16, explicitly untied. Local A8-before-GPTQ differs from official current ordering.

Runs:
- `phase5-external-spinquant-smoke`, pane2245736: argument unpack error before model/GPU work, preserved `attempt1-argument-unpack.*`.
- `phase5-external-spinquant-smoke2`, pane2249229: one R update completed, then harness missed original ptq.py caller model.cuda(); RoPE device mismatch before GPTQ solving. Preserved `attempt2-missing-cuda.*`, original `run.log`.
- `phase5-external-spinquant-smoke-ptq`, pane2258824 / actual Python PID2258826: `EXTERNAL_SMOKE_RESUME_PTQ=1`, exact saved R/windows reused, missing original caller CUDA placement restored; only GPTQ/forward remain, `ptq-resume.log`. No R update repeated.

Only independent harness files changed. No application code/environment/history modifications. Uses core callable functions, not full CLI Trainer/DDP orchestration; dataset loader supplied existing Arrow train cache, but native CustomJsonDataset/get_wikitext2 sampling implementations retained. This is not official full reproduction, not formal external PPL, not Qwen dynamic/GPTQ support, not100-step R/128-window GPTQ budget, not equal400-step recovery. Final status is in `result.json`.

Actual PTQ-resume command:

```bash
env CUDA_VISIBLE_DEVICES=2 EXTERNAL_SMOKE_RESUME_PTQ=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python /home/dongpeiyan/projects/rotation-quant/runs/phase5/verifier/external-spinquant-smoke/smoke.py > /home/dongpeiyan/projects/rotation-quant/runs/phase5/verifier/external-spinquant-smoke/ptq-resume.log 2>&1
```

## Final independent conclusion: PASS

The unique candidate may accurately be registered as **locally adapted callable SpinQuant learned-rotation + GPTQ W4/group32, dynamic asymmetric A8, KV16 runnable on the existing Llama-3.2-1B-Instruct**. This closes bounded runnability, not formal external quality/replication.

Measured:
- One R-only training update: 2048 input tokens /2047targets, train NLL2.905704975128174,17 finite nonzero gradients and17 actual R changes. No second rotation update was run.
- All112 backbone matrices through existing GPTQ: W4 symmetric/group32, clipping enabled, act_order=false, percdamp=0.01, one2048-token calibration window. All quantized weights finite.
- Final full2048-token smoke forward: all112 dynamicA8 backbone wrappers were called exactly once,16 down online Hadamard; finite logits and train-window NLL2.9911930561065674. This number is only a finite-forward observation on train data, not validation/test/C4 PPL or a method-quality comparison.
- PTQ-resume completed in281.1803867816925seconds, peak allocated9.15443468093872GiB for that process. This excludes the earlier process's R-update peak/time. GPU2 released on normal completion.

Covered core loading/untying, R-only checkpoint backward/SGDG update, save/use learned R, paired down Hadamard, actual local PTQ orchestration and all-layer GPTQ Hessian/solving, dynamicA8/grouped o_proj and no-cache finite forward. Did not cover100R updates,128GPTQ calibration windows, global batch8 full training, Trainer/DDP CLI orchestration, full PPL, static package coldload, Qwen dynamic/GPTQ, ExecuTorch/NPU/decode, or equivalence to the official unmodified algorithm.

Official/local differences remain: local inputA8 configured beforeGPTQ; generic PTQ keeps embedding/headBF16 rather than ExecuTorch8bit packaging; model uses this project's existing modified checkout. Smoke intentionally reduces update/calibration budgets and does not use the400-stepQAT mainline. No second external candidate introduced.

Artifacts retained: `result.json`, `settings.json`, `ptq_settings.json`, `input_windows.pt`, `R.one_update.bin`, `source.diff`, original and resumed logs, and both harness-setup failure attempts. No application source/environment/history changes.

Completion audit: live tracked Git diff and raw status match the recorded start; `completion_audit.json`. Both windows originate from the existing `wikitext-train.arrow` through native CustomJsonDataset/get_wikitext2 train selection. No validation/test/C4 inputs used. Process2258826 exited; GPU2 returned to20MiB/0%util.
