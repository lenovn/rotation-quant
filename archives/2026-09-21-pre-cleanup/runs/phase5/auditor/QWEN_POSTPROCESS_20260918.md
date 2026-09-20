# Qwen seed42 postprocessing and exact PTQ package audit

2026-09-18. **PASS for all six completed stages, both exact final PTQ package identities, and completed fixed-package test/C4 acceptance for seed42.** Read-only CPU comparison of saved packages, data/settings, candidate records, source and metrics; no GPU, forward, calibration, tokenization or modification of in-flight runs. Detailed postprocessing evidence: `QWEN_POSTPROCESS_20260918.json`; final external evidence: `QWEN_PTQ_ACCEPTANCE_20260918.json`; bounded postprocessing checker: `check_qwen_postprocess.py`. Follow-ups inspected only the newly completed readaptation and external artifacts; earlier passed stages were not rerun or re-audited.

## Shared train-only selection and coverage

All four stages use official `Qwen/Qwen3-1.7B`, seed42 and the same32 saved Joint100 calibration indices. Here calibration windows are expanded to full2048-token windows: first24 for fitting (49,152 rows), last8 for candidate selection (16,384 rows /16,376 predicted targets). These are WikiText **train** windows; validation is separately evaluated after stage completion. Each stage covers exactly `model.layers.0.mlp.down_proj` through `model.layers.27.mlp.down_proj` in order, without omissions or duplicates.

Round stages use the exact Joint100 floating reference `joint100-s42/checkpoint-0100/state.pt`. Range settings retain that path for provenance, but live source loads the reference only for `mode=round`; range candidates operate on the existing package. All settings/result/package parent paths match the intended chain:

- SP2: `joint100-s42/checkpoint-0100/static_w4a8.pt` → `sp2-down-round-s42/static_w4a8.pt` → `sp2-range-s42/static_w4a8.pt`.
- INT8: `uniform-initial-s42/static_w4a8.pt` → `uniform-down-round-s42/static_w4a8.pt` → `uniform-range-s42/static_w4a8.pt`.

Every module includes the unchanged current parent candidate: round step0 or range factor1. The selected NLL is the minimum eligible train-window NLL, and each selected prefix becomes the next module's parent. The recorded previous/selected scores form an exact continuous chain through28 modules. Round candidates with held-out reconstruction MSE no better than the parent are skipped before train NLL scoring; the held-out MSE and NLL selection windows both come from the train split. The capture hook is on the underlying linear input after its activation quantizer, preserving each branch's actual SP2/INT8 input behavior.

## Candidate budgets and the inherited convergence case

SP2 rounding has28×`[0,512,2048,8192]` = **112 recorded candidates**. Uniform rounding has27 such sets plus layer2 `[0,14]` = **110 recorded candidates**. Both retain the same declared coordinate limit8192 and checkpoints512/2048/8192.

Uniform layer2 selected step14 is the inherited coordinate-search convergence branch: when the best possible column gain is `>= -1e-15`, the algorithm snapshots the current state and exits before applying that numbered update. Therefore step14 records the state after up to13 actual coordinate updates, not a separately configured14-step budget. The saved helper is byte-identical to the historical Phase3 helper. This is source-supported provenance; the exact terminating gain is not logged and was not recomputed.

Both range stages have28×6 = **168 candidates**, with exactly `[1, 0.875, 0.75, 0.5, 0.25, 0.125]`. All candidate factors, parent reuse, selected minima and exported scale conversion were checked. There is no expansion candidate in the executed range stages.

## Artifact changes and preserved state

All packages retain196 W4 records. SP2 packages retain168 INT8 +28 SP2 activations; Uniform packages retain196 INT8. Configs and **all154 high-precision state tensors**, including **56 Q/K norm tensors**, remain bitwise equal to the respective immediate parent. This comparison handles identical inactive NaN placeholder buffers by their bytes.

Across every stage all W4 scales and non-down weight codes stay unchanged; all168 non-down activation scales stay unchanged. Round stages change only accepted down W4 packed codes and preserve every activation scale:13 SP2 down layers and7 Uniform down layers changed, exactly corresponding to nonzero selected steps. Range stages preserve every packed W4 code and change only selected down input scales:8 SP2 layers and11 Uniform layers contracted. INT8 scales equal the chosen alpha/127 FP32 value; SP2 exported alpha equals the corresponding chosen scale×127 with its actual serialization rounding. Unselected factor1 preserves the parent range.

## Intermediate validation and summary correspondence

All four runs are completed with no failure file. Validation uses262,337 input tokens /262,208 targets. Independent segment-log(PPL), target-weighted NLL and final exp(NLL) recomputation match raw JSON exactly; the four summary rows have the correct official model, seed, format, split, parent/package paths and metrics.

| Completed intermediate stage | Selection train NLL before → after | Validation PPL | Validation NLL | Summary stage |
|---|---|---:|---:|---|
| sp2-down-round-s42 | 2.8335653562108885 → 2.8035783728731345 | 15.930924848715014 | 2.768262179280969 | SP2-down-round |
| sp2-range-s42 | 2.8035783728731345 → 2.766265582072881 | 15.384419587171699 | 2.7333552821716736 | SP2-range |
| uniform-down-round-s42 | 3.0333426037460383 → 3.0204473088695813 | 19.521916498772708 | 2.9715377574550645 | Uniform-down-round |
| uniform-range-s42 | 3.0204473088695813 → 2.9303037690963833 | 17.887933411651215 | 2.884126374450734 | Uniform-range |

These four validation measurements describe intermediate packages. Their original audit did not close either final PTQ control, any QAT400 control, external test/C4 acceptance or multi-seed statistics. Readaptation was left untouched while in flight; its later completed-package audit follows below.

## Completed final readaptation follow-up

The two final runs now report completed, without failure files. Their settings, result metadata and package parent metadata link `sp2-down-readapt-s42` to the exact `sp2-range-s42/static_w4a8.pt` and `uniform-down-readapt-s42` to the exact `uniform-range-s42/static_w4a8.pt`. Both preserve seed42, the same32 complete train windows split24/8, all28 ordered down targets, the unchanged Joint100 floating reference and round milestones512/2048/8192. Every module retains step0, selects the minimum eligible train NLL, and continues the selected prefix; the recorded initial NLL exactly matches the preceding range stage's final selection NLL.

SP2 readaptation has112 trial records (28×4), accepting changed down codes in7 layers. Uniform readaptation has110 records, accepting changes in12 layers; layer2 has `[0,193]` with selected step193 and all other layers have `[0,512,2048,8192]`. The identical inherited helper's no-improving-coordinate branch snapshots before applying that numbered update, so193 represents the state after up to192 actual coordinate updates. The exact terminating gain was not saved. This does not change the configured8192-coordinate limit or introduce another search budget.

For each completed package, every W4 scale, activation scale, non-down packed code, model config and all154 high-precision tensors are unchanged from its range parent; all56 Q/K norm tensors are bitwise unchanged. Only the accepted down packed codes differ, and their layer identities agree with nonzero selected steps. Actual final formats remain196 W4 /168 INT8 +28 SP2 for the SP2 path and196 W4 /196 INT8 for Uniform. No calibration or gradient-training stage was added in this final readaptation.

| Exact PTQ package run | Selection train NLL before → after | Validation PPL | Validation NLL | Summary stage |
|---|---|---:|---:|---|
| sp2-down-readapt-s42 | 2.766265582072881 → 2.7368355058754466 | 14.935424095235582 | 2.7037358473302344 | SP2-PTQ |
| uniform-down-readapt-s42 | 2.9303037690963833 → 2.907194164907816 | 17.491760837320335 | 2.861729960767992 | Uniform-PTQ |

Each validation uses262,337 input tokens /262,208 targets. Segment-weighted recomputation is exact, and both formal PTQ summary rows match the raw values, official model identity, seed, format and exact parent/evaluation package. Follow-up machine evidence is `qwen_readapt_checks_20260918.json`, also appended to the main JSON without repeating earlier checks.

The exact accepted PTQ artifacts for subsequent fixed-package validation/test/C4 comparison are:

- SP2: `runs/phase5/qwen3-1p7b/sp2-down-readapt-s42/static_w4a8.pt`.
- Uniform: `runs/phase5/qwen3-1p7b/uniform-down-readapt-s42/static_w4a8.pt`.

The readaptation follow-up closed only the postprocessing/package-identity audit. External acceptance was still pending at that time; its later completed-artifact audit follows below. No QAT or multi-seed completion is claimed.

## Final fixed-package test/C4 acceptance

**PASS for both exact PTQ packages.** Four newly completed external runs were audited: `sp2-ptq-test-s42`, `sp2-ptq-c4-s42`, `uniform-ptq-test-s42`, and `uniform-ptq-c4-s42`. All report completed, with no failure files. Each method's validation and external results identify its own unchanged final `*-down-readapt-s42/static_w4a8.pt` listed above. `model.json` confirms196 W4 matrices, actual168 INT8 +28 SP2 or196 INT8 activations, KV16, `use_cache=false`, FP32 RoPE and no down INT8 overlay.

For each external run, all196 saved quantizer states are exactly equal before/after evaluation, and the initial test/C4 states for the same method are equal. All actual scales match their package, including SP2 alpha-to-scale storage conversion. INT8 observers remain disabled with loaded scales and zero observed samples. External calibration, training and candidate-selection flags are false. The audit did not inspect or operate on either in-flight QAT400 run.

Saved test input IDs and complete metadata exactly equal Qwen BF16: **299,078 inputs,298,931 targets**,146 full2048-token windows plus a70-token tail. Both C4 runs use the exact same existing Qwen C4 cache as BF16: **2,097,152 inputs,2,096,128 targets**,1,024 full windows and no tail. The tokenizer is Qwen's own; Llama IDs are not reused. Source-text coverage is the previously audited Qwen prefix:4,325 complete source documents plus40,167/67,339 characters of document4,326 (`row_index=230792`); see `C4_TEXT_COVERAGE_20260918.md`. No new document or token sampling was performed.

All three splits' segment starts, window counts/lengths and target counts match corresponding Qwen BF16 results. Every segment's log(PPL), target-weighted final NLL and exp(NLL) recompute exactly. All **six** `SP2-PTQ`/`Uniform-PTQ` summary rows match raw metrics, counts, official model, seed42, actual format and exact parent/final package paths.

Validation evidence has a narrower storage boundary than external evidence: the postprocessing stages did not save separate validation input IDs or quantizer before/after snapshots. Their same-package cold-load source, saved tokenizer/dataset metadata,262,337 input/262,208 target counts and segment structure were checked against BF16; this audit does not claim a new validation-ID tensor comparison or retokenization.

| Split | SP2-PTQ PPL | Uniform-PTQ PPL | ΔNLL (SP2 − Uniform) |
|---|---:|---:|---:|
| WikiText validation | 14.935424095235582 | 17.491760837320335 | -0.15799411343775738 |
| WikiText test | 14.302163393485294 | 16.78537625006989 | -0.16009723395145103 |
| Fixed C4 subset | 24.903332773203157 | 29.322131492633147 | -0.16334093111462034 |

Raw NLLs (validation/test/C4) are **2.7037358473302344 /2.6604108120809626 /3.215001640827423** for SP2 and **2.861729960767992 /2.8205080460324137 /3.3783425719420435** for Uniform. The final SP2 PTQ package is better on these three matched seed42 metrics. This closes exact PTQ external evidence only; QAT400 and later seeds remain separate unfinished results. No split-specific package switching, external-data recalibration or tuning is inferred or authorized by these measurements.
