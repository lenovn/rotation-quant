# Initial-SP2-QAT400 representative ablation: training audit

2026-09-18. **PASS for completed training evidence and the intended matched-method comparison; final same-package test/C4 evidence audit also PASS.** No GPU use, model forward, recalibration or experiment rerun. Independent CPU inspection used JSON/JSONL, saved source, and read-only memory-mapped checkpoint metadata/Adam counters. The comparison has opposite directions on WikiText and C4, as recorded below.

Evidence roots:

- First 25 updates: `runs/phase5/llama32-1b/initial-sp2-qat400-s42`.
- Resumed 375 updates: `runs/phase5/llama32-1b/initial-sp2-qat400-s42-r25`.
- Historical full-postprocess comparison: `runs/phase3/distill-b100-refined-ref-adam1e5-400-20260914a`.
- Reproducible audit: `auditor/check_initial_sp2_training.py`; machine output: `auditor/initial_sp2_training_checks_20260918.json`.

## Resume continuity and effective budget

The two training logs contain exactly steps **1–25** and **26–400**, without duplicate or missing committed updates. Both saved `resume.pt` files contain all 336 learned parameter tensors and both Adam states; every weight/scale optimizer counter is respectively **25** and **400**. The resumed settings point to the first run's exact resume path. Saved sources for distillation, quantization, common data/model operations, postprocessing, architecture and scheduling are byte-identical between these two launches. Resume code loads learned parameters, both optimizer states and Python/CPU/CUDA RNG states before entering the loop at the saved absolute step.

All 400 records have eight microbatches, each with 2,047 prediction targets. Their window IDs exactly match the historical run and `(800 + (step - 1) * 8 + microbatch) % 1180`. The tokenizer produces 1,188 complete train windows; `data_windows` withholds the final eight, including probe windows 1180–1183. All 400 logged weight learning rates exactly equal the historical schedule. All updates report gradients for 112 W, 96 SA, 112 SW and 16 SP2 tensors.

Thus the **effective committed budget is 400 × 8 × 2048 = 6,553,600 input-token visits**, or **6,550,400 predicted-target visits**. This is not the total physical computation across the migration: the old traceback is inside the next `distillation_loss` call at `teacher.model.cpu()`, so some uncommitted next-step work was discarded before restoring update 25. Its amount is not logged. `migration.json` records the deliberate SIGINT because of other-process GPU6 memory fluctuations and migration to GPU0. The old `KeyboardInterrupt` is consistent with that migration, not a failed final method result.

Both the resumed `result.json` and `progress.json` report completed step 400. `target_ppl` is null and `target_reached` false; there was no PPL-triggered early stop. Historical logs also contain all 400 steps, despite their enabled target threshold.

## What is matched and what differs

The method-level intervention is the QAT parent package:

- Ablation: `runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/static_w4a8.pt`, the initial SP2 package before down-neighbor/range/readaptation postprocessing.
- Full method: `runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`, the accepted full postprocessed SP2 package.

All three `data.json` files and parameter-coverage JSONs are exactly equal. Settings match reference state, 400 steps and schedule length, accumulation 8, warmup 10, Adam weight learning rate 1e-5, relative scale rate 0.001, temperature 1, CE weight 0.1, data start 800, teacher-body offload, resume interval 25 and checkpoints 100/200/400. The floating reference is the same B100 `checkpoint-0100/state.pt`. Both initialization records declare no diagonal transform and projection of that reference into the **respective parent's INT4 cells**, retaining each parent's initial quantized forward. Source ASTs for the exact projection rule, reference transform, trainable weight operator, parameter grouping, objective and optimizer construction match historical source. Quantizer source is unchanged.

The teacher remains the original unrotated BF16 Llama, no norm fusion, frozen, with the same local model/revision and custom evaluation model class. The architecture dispatch selects the same Llama class and BF16/SDPA configuration; the checkpoint has `tie_word_embeddings=true`, so cloning embedding into the head reproduces the historical explicit clone. The extra Qwen-compatible head placement is a device operation on this already separate Llama head. Formal Llama launches continue to use the old environment (Transformers 4.44.2, PyTorch 2.4.1+cu121). FP32 master weights, static W4/INT8/SP2 scale learning and the same objective/gradient clipping remain in use.

It would be inaccurate to call every execution setting identical. Operational differences are migration/GPU assignment and Phase5 architecture plumbing; full validation runs only at step 400 here, versus 100/200/400 historically; the historical target PPL threshold is disabled here. Relative scale learning rates follow the same rule but can differ numerically because the parent SP2 scales differ, which is part of the parent intervention.

**Observed evidence:** same committed step count, window sequence, weight LR schedule, objective/optimizer implementation, parameter coverage and saved optimizer counters. **Source-supported inference:** the changed validation schedule should not change effective optimization: checkpoint evaluation is outside optimizer updates, uses no-grad, evaluates a separately exported frozen model, forks RNG around cold model construction, restores training mode, and neither path stopped before 400. Resume restores all optimized state and RNG, and Llama attention dropout is zero. This audit does not prove bitwise equality against a hypothetical uninterrupted ablation run; none was repeated. The parent is the sole intended algorithmic treatment, while these explicitly recorded operational differences remain.

## Completed validation

The final package has **112 W4 weights, 96 INT8 and 16 SP2 activation quantizers**. Step-400 validation has 252,852 input tokens and 252,728 targets, including the 948-token tail window. Independent target-weighted recomputation from segment PPL gives **NLL 2.6916587969227876; PPL 14.756133058231322**, exactly matching the result/checkpoint JSON. Historical full-postprocess QAT400 validation is 14.581654675328117; this validation comparison alone does not close the external test/C4 ablation acceptance.

Audit implementation note: its first attempted window-order check incorrectly used all 1,188 windows; live source inspection confirmed the inherited `windows[:-8]` training split. Correcting the auditor to 1,180 produced PASS. No experiment or application change was needed.

## Grouped seed42 comparison (external acceptance complete)

Values below are read directly from each raw JSON, retaining Python/JSON full precision. ΔNLL is **full postprocessing QAT400 minus initial-SP2 direct QAT400**; negative favors the full method.

| Split | Initial-SP2 direct QAT400 PPL | Full postprocessing QAT400 PPL | ΔNLL (full − direct) |
|---|---:|---:|---:|
| WikiText-2 validation | 14.756133058231322 | 14.581654675328117 | -0.011894587390206102 |
| WikiText-2 test | 14.320253706327318 | 14.154416405154963 | -0.011648189467362347 |
| Fixed C4 subset | 26.730584258553062 | 26.98478050458106 | 0.009464634363233415 |

Raw validation paths are the respective training roots' `checkpoint-0400/validation.json`. Direct test: `runs/phase5/llama32-1b/initial-sp2-qat400-test-s42/result.json`; full-method test: `runs/phase3/wiki2-test-best-fixed-20260917a/result.json`. Historical C4: `runs/phase3/c4-best-fixed-20260915a/result.json`; direct C4: `runs/phase5/llama32-1b/initial-sp2-qat400-c4-s42/result.json`.

The full-postprocessing method improves WikiText validation and test relative to direct QAT, while **direct QAT performs better on the fixed C4 subset**. This seed42 ablation does not support an improvement across all three evaluations. The predefined mainline and its package remain fixed; these external metrics are not authorization to select different packages by split, recalibrate, or tune.

## Final external evidence audit

`auditor/initial_sp2_external_checks_20260918.json` records the completed read-only checks. Both external runs report completed and have no failure file. Validation, test and C4 all reference the exact resumed `checkpoint-0400/static_w4a8.pt`, containing 112 W4 matrices and 96 INT8/16 SP2 activation records. Both `model.json` files agree on that package, KV16, `use_cache=false`, FP32 RoPE, and no INT8 overlay; external result flags record no calibration, training or candidate selection.

All saved activation-quantizer tensors are exactly equal before/after each external evaluation, and test/C4 initial quantizer states are equal to one another. All 96 INT8 scales exactly match the package and remain non-observing with zero calibration samples. All 16 SP2 scales exactly match the package's alpha-to-scale cold-load conversion; SP2 level tensors are unchanged. This checks the saved quantizer state directly, rather than relying only on `activation_scales_unchanged=true`.

Test input IDs and complete tokenizer/data metadata exactly equal the historical BF16 and full-method QAT test inputs: 289,076 input tokens, 288,934 targets, including the 308-token tail. C4 uses the same existing input artifact and exact IDs/metadata as historical BF16 and full-method QAT: 2,097,152 input tokens, 2,096,128 targets, 1,024 full windows. All segment starts, window lengths/counts and target counts match the corresponding historical full-method run. Every segment NLL equals `log(segment PPL)`; independent target-weighted recomputation exactly matches final NLL/PPL on all three splits. The three `Initial-SP2-QAT400` summary rows match their raw metrics, counts, official model identity, seed42 and common package exactly.

Direct-QAT test NLL is **2.6616748782978554**; C4 NLL is **3.285808387695454**. The historical-comparison directions above therefore follow the same token/metric protocol, without package switching or test-set recalibration.
