# Independent final MMLU audit

Status: PASS for all seven completed model/method evaluations. Evidence: `audit_mmlu.py` and `mmlu_audit.json`.

All 399 subject-result files (57 subjects x seven methods) were read in full. Every run has exactly 14042 test samples, with unique within-subject doc IDs and no dropped examples. Within a model family, actual scoring arguments, doc IDs and targets match BF16 for every sample. Every prompt has six `Answer:` markers (five demonstrations plus the target), and task configuration/n-shot metadata specify five shots.

Each subject accuracy was independently reconstructed from per-example correctness. Aggregating all 14042 examples, rather than equally averaging subjects, exactly reproduces the official group result and matches main-table MMLU percentages to 1e-10:

| Model | BF16 | FIRON | Parent | Historical W4A16 |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 60.14812704742914 | 54.401082466885065 | 45.76983335707164 | unavailable |
| Llama-3.2-1B-Instruct | 45.54194559179604 | 37.85785500640934 | 33.734510753453925 | 37.316621563879785 |

All FIRON/parent runs report unchanged quantizer states and complete prefill coverage of 196 Qwen or 112 Llama quantizers. No GPU execution or evaluation rerun was performed.

## Generation-summary code review

`generation_summary.py` counts budget exhaustion only if neither EOS tokens nor configured stop strings appear. This avoids classifying padded early-stopped sequences as truncation for the current EOS-padding and stopping configuration. It is an explicit **budget exhausted without EOS/stop** statistic, not a reconstruction of each example's unpadded token length. It restricts inputs to a completed result and the generation-log path in that result directory's current settings, thereby avoiding stale failed-attempt files when attempt paths differ.

The added `generation_all_metrics.csv` extraction retains all numeric non-stderr task metrics, multiplies by 100 once, and computes matching BF16 percentage-point differences correctly. This preserves GSM strict/flexible and IFEval's four metrics without changing the main metric selection. Generation result completeness and prompt matching remain pending separate final audit.
