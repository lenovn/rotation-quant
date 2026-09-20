# Summary renderer review

2026-09-18. **PASS for current stage mapping and displayed values; final three-seed statistics are not complete.** Read-only source/output/CSV inspection, no application changes, GPU, PPL rerun or test suite.

Reviewed `runs/phase5/summarize_results.py` and its current `CORE_RESULTS.md` against **22** summary rows. All **15** displayed completed PPL cells match the uniquely linked evidence row at six-decimal display precision; raw precision remains in CSV/JSON.

- Exact case-insensitive stage matching includes only BF16, full SP2/Uniform PTQ, full SP2/Uniform QAT400 and Llama's initial-SP2-QAT400 ablation. B100 format controls and intermediate rounding/range stages are excluded.
- Qwen never receives the representative initial-SP2 ablation row. BF16 is not multiplied across seeds. Duplicate matches become “多条记录，待核对” rather than selecting the first result.
- Same-seed ΔNLL is SP2 minus Uniform, and is shown numerically only when both rows exist and targets match. Current three Llama seed42 rows lack Uniform QAT and correctly show no numerical Δ.
- All six model/dataset three-seed rows currently show **0/3 complete pairs, not aggregated**. The code requires all seed42/43/44 pairs and equal targets before computing mean and `statistics.stdev`, which is the sample standard deviation (ddof=1). It does not report best-seed statistics or fill missing seeds from PTQ/B100.
- Test/C4 package equality is displayed from explicit package paths; completed current pairs agree. The renderer clearly delegates actual data/architecture identity to raw runs and auditors, so its “same” field is not an independent numerical reproduction.

No incorrect current stage, model assignment or metric value was found.

**Reporting item closed after narrow follow-up:** the new “相对本模型原始 BF16 的指标” section contains all 15 currently completed core records. Independently recomputed every row from CSV: PPL (six decimals), NLL and own-model/same-dataset BF16-relative ΔNLL (nine decimals), PPL ratio (six decimals), and exact targets all match. Each row references the correct same-model/split BF16; BF16 rows give ΔNLL0 and ratio1. This closes the previously reported handoff §6.3 display-field omission. Other sections were not re-reviewed. Three-seed statistics remain incomplete until real paired data exist; the new BF16-relative section does not change that status.
