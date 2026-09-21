# Phase6 migration snapshot (2026-09-21)

Training is stopped for server maintenance. Do not start jobs on the old server.

Migration is staged: phase6-code-results.tar.gz contains effective source (including uncommitted changes), small existing result files and documentation ONLY. Checkpoints, pretrained models, dataset caches and large dependencies are pending separate transfer. No experiments are started by unpacking.

First-stage extraction: `tar -xzf phase6-code-results.tar.gz -C /path/to/rotation-quant`. This first package alone is NOT sufficient to resume training.

Reassemble: `cat phase6-data.tar.part-* | tar -xf - -C /path/to/rotation-quant`

For initial migration, use the SAME absolute project path `/home/dongpeiyan/projects/rotation-quant` on the destination. Existing JSON/PT metadata contains absolute paths. A different root requires relocation of both JSON and torch metadata, not merely changing launcher arguments.

The old Python environment uses Python 3.9.23 with system packages from a conda environment. It is NOT a portable virtualenv. Recreate Python 3.9, torch 2.4.1+cu121, transformers 4.51.3, then use requirements-observed.txt as the dependency inventory. Build repos/fast-hadamard-transform on the destination for its CUDA/PyTorch environment. Create the runtime at runs/phase5/env. Verify CPU imports before launching.

Resume with scripts/phase6/launch.py --resume, keeping cosine horizon 512. Read resume.pt for actual saved step. Pilot additionally uses resume_parameters.pt (its step must match). Prior last verified saved steps: pilot481, Qwen1.7 247, Llama3B101, fixed-down286, R-only290. A short attempted restart was cancelled for maintenance; verify latest saved steps on destination.

Before reusing a --name/--steps combination, archive its existing *-resume-T.launch.json and matching .log outside the launch glob; the launcher deliberately refuses duplicate markers. Do not delete previous results/checkpoints. GPU assignments must be reviewed for the new server before launching. Qwen0.6B already finished512: do not rerun it.

Current targets: pilot512 (GPU0), Qwen1.7 256 (GPU1), Llama3B128 (GPU5), fixed-down320 (GPU2), R-only320 (GPU6). Final unified T remains undecided until pilot512 validation. After explicit resumes are live, run scripts/phase6/schedule.py in tmux to select T and schedule remaining training/evaluation. Do not start the scheduler alone expecting it to recover interrupted training automatically.

Precision: projection W4 per-output-channel; non-down input A8 static per-tensor; down input signed INT16 static per-tensor. Embedding/head/norm and KV remain high precision. BF16 fakequant evaluation, not native integer NPU execution. Original pretrained initialization; no B100 continuation. All eight capability tasks, WT2 test and fixed C4 validation subset remain required. Existing validation PPL must not be compared directly with BF16 test PPL.

This directory contains the first-stage code/results archive. Training checkpoints and large dependencies are not included yet.
