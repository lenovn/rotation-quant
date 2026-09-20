# D experiment loop

Authorization: same original thread, no distillation or gradient training; frozen original weights, C R and 96 non-down SA. Single GPU, cumulative 1800 seconds, process GPU memory <=12 GiB, at most nine complete validation evaluations.

## Round 1 hypothesis and controls

Hypothesis: nonuniform positive diagonal balancing can improve full-range down INT8 without losing more W4 precision than it recovers. Five strengths per layer: 0, .25, .5, .75, 1; geometric mean D=1, D in [.25,4]. No alpha/clipping search. Three models: identity+C learned SW; identity+all 112 recomputed SW; selected D+same recomputed SW. Fixed train 32x2048, seed42 and original settings window indices; full range over 65536 rows/layer, MLP MSE on fixed 2048 rows. Final three diagnostics share each model's frozen packed INT4 and SA.

Command: `PRIOR_GPU_SECONDS=34.35333547502523 bash scripts/phase2/46_run_down_d_search_local.sh`.
GPU selection: launcher queries nvidia-smi immediately before starting and selects one card with used/total <35%. Exact snapshot in gpu_before.csv. Python PID and active stage are in progress.json; command.txt and experiment.log contain execution evidence.

Independent startup review: /root/verify_d_loop PASS. Six CPU tests, real tiny BF16 Llama integration/candidate isolation, and all 96 SA quantizer equivalence tests passed. No GPU result is implied by startup PASS.

## Initialization failure and correction

Prior attempt ../down-d-search-20260913.apFXU0 failed before calibration because model-level RoPE cache was on CPU and token positions on CUDA. Its 34.353335475 seconds are charged to this loop. Corrected by moving model.model.rotary_emb to CUDA, matching existing ptq behavior. Independent correction review PASS. No candidates or evaluation results were produced by the failed attempt.

## Status

Round 1 running. Results and next hypothesis will be appended only after measurement. The subsequent bounded experiment must be justified by the three diagnostic NLLs; nine full-validation runs and cumulative resource limits remain unchanged.

## Round 1 measured result and analysis

identity_learned: all_a16 PPL=16.2310666692, NLL=2.7869271014, down_a16 PPL=16.2928123577, NLL=2.7907240509, full_a8 PPL=2083.9034719386, NLL=7.6419980832

identity_recomputed: all_a16 PPL=16.6027304955, NLL=2.8095671695, down_a16 PPL=16.6714053567, NLL=2.8136949978, full_a8 PPL=1746.3508050985, NLL=7.4652836355

search_recomputed: all_a16 PPL=17.7117411449, NLL=2.8742277612, down_a16 PPL=17.7980072029, NLL=2.8790864961, full_a8 PPL=161.3686601098, NLL=5.0836915617

D versus matched I lowers full-A8 NLL by 2.3815920738, while all-A16 NLL increases by 0.0646605917. D recovers substantial activation accuracy but leaves a large down-A8 loss. Independent verifier /root/verify_d_loop: code and all nine result protocols PASS. Cumulative GPU budget used 370.891119334 seconds, including failed startup.

## Round 2 hypothesis and finite configuration

Layerwise MLP MSE may choose strengths differently from end-to-end train NLL. Measure the same 32 train windows for frozen I/recomputed and selected-D/recomputed controls; construct only two new candidates, fixed per-layer strength t=.25 and t=1, each with its own actual-upstream direction and full-range SA. All use the same 112 recomputed SW rule, frozen 96 non-down SA, and independent original effective weights. Record all-A16/down-A16/full-A8 train NLL and newly zeroed down input fractions. No additional validation calls and no claim of validated candidate improvement. Budget includes model initialization and prior attempts.

Round 2 launched after independent startup PASS. Command: `PRIOR_GPU_SECONDS=370.89111933403183 FOLLOWUP_PARENT=/home/dongpeiyan/projects/rotation-quant/runs/phase2/down-d-search-20260913.bpL0NE bash scripts/phase2/46_run_down_d_search_local.sh`. Run directory ../down-d-search-20260913.mHkMqo, GPU1, Python PID3428030, launch snapshot 2420/24564 MiB. Initial sandbox nvidia-smi could not access driver; read-only query and launch outside sandbox succeeded under authorized automatic review. No candidate ran in the rejected-by-device attempt. Exact commands, progress, settings and logs in the follow-up directory.

## Round 2 measured result and decision

identity_recomputed: full-A8 train PPL=1745.6102294921875, NLL=7.464859475219855

search_recomputed: full-A8 train PPL=168.66998291015625, NLL=5.127944041948736

fixed_t025: full-A8 train PPL=1388.3023681640625, NLL=7.235836961840687

fixed_t100: full-A8 train PPL=182.7207794189453, NLL=5.20795919210514

Both additional fixed-strength candidates fail to beat selected D on train. Do not promote either. Measured newly-zeroed input fraction in layer1 remains about 99.9999% with selected D. Future sparse extreme-channel direction is a hypothesis only, not executed. Full-validation count is already 9; no more validation runs are authorized within this cap. GPU cumulative 558.0677173670265 seconds, remaining 1241.9322826329735 seconds. Followup report: ../down-d-search-20260913.mHkMqo/report.md.
