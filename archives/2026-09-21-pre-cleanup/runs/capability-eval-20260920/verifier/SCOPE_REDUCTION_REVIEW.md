# Independent review of the user-requested scope reduction

Status: PASS. This report supersedes older verifier completion/missing-scope statements for the current deliverable. Only CPU file comparisons and source inspection were performed; no GPU or evaluation rerun was launched.

The current main table has seven model/method rows and exactly four metrics: WikiText-2, C4-subset, Zero-shot Avg. (8), and MMLU. All retained cells, including their unrounded numerical strings, exactly match the archived full-scope numeric table after renaming C4 to C4-subset. `main_results.csv` and `main_results_numeric.csv` agree. The Qwen W4A16 row and both generation columns are absent.

All seven retained eight-task appendix rows and main-metric delta rows match the archived values exactly. The 63 current task-detail records match their archived records in every field. Thus the current scope is seven rows, 14 PPL results and 63 likelihood-scoring tasks. `incomplete.json` is empty; the excluded Qwen W4A16 model is no longer presented as pending.

The scheduler's default list is exactly BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA and MMLU. Its `--tasks` choices use the same nine-task list, preventing generation scheduling through this current entrypoint. Historical generation support in the single-task code remains available only as preserved implementation history; it is not scheduled or referenced in current tables.

The paper wrapper references only the current main table, eight-task score appendix, eight-task delta appendix, and four-metric delta table. It references no generation table. All saved capability-evaluation Python scripts that have source snapshots match their live counterparts byte-for-byte.

Archive comparison source: `archive/full-scope-before-user-reduction/`. Structured checks: `scope_reduction_audit.json`. Previously completed raw-task audits remain evidence for the retained results; the prior generation audit and its old missing-Qwen-W4A16 statement describe the former scope only.
