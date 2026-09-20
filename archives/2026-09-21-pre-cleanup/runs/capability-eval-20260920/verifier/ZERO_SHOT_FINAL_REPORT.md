# Independent eight-task result audit

Status: PASS for all 56 completed zero-shot evaluations (seven available model/method rows x eight tasks), 2026-09-20. No GPU execution or benchmark rerun was performed.

Reproducible audit: `audit_zero.py`; complete per-result checks: `zero_shot_audit.json`.

For every task and available method, the audit loaded all raw sample rows, confirmed the official expected count and unique doc IDs, recomputed the selected accuracy from per-sample scores, and checked zero shots. Within each model family, `doc_id`, actual scoring arguments, and target were compared directly against BF16 for every example and all matched. This covers 156933 scored sample rows across the 56 result sets.

All FIRON and parent tasks report unchanged quantizer states and prefill execution of all 196 Qwen or 112 Llama activation quantizers. BF16 and historical W4A16 correctly have no A8 wrappers. Each appendix score and BF16-relative percentage-point delta matched the corresponding raw result to 1e-10. All seven independently recomputed eight-task means matched the main CSV to 1e-10:

| Model | Method | Eight-task mean (%) |
|---|---|---:|
| Qwen3-1.7B | BF16 | 58.221483767942416 |
| Qwen3-1.7B | FIRON | 56.36920144163138 |
| Qwen3-1.7B | Parent | 54.16533562024014 |
| Llama-3.2-1B-Instruct | BF16 | 55.27503085976077 |
| Llama-3.2-1B-Instruct | FIRON | 52.19608151254703 |
| Llama-3.2-1B-Instruct | Parent | 50.990144858630124 |
| Llama-3.2-1B-Instruct | Historical W4A16 | 51.28851581339208 |

Qwen W4A16 is explicitly unavailable and is not included as an evaluated row. This PASS covers zero-shot tasks only; MMLU and generation tasks remain outside this report.
