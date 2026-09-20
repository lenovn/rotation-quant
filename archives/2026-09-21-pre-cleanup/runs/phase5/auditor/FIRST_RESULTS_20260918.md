# First Phase5 result audit

2026-09-18. **PASS: four completed results and the current 17-row summary agree with their raw evidence.** Independent read-only JSON/token/scale-tensor audit; no GPU and no PPL rerun.

| Run | PPL | NLL | Targets | Tail inputs |
| --- | ---: | ---: | ---: | ---: |
| Qwen BF16 WikiText test | 16.715764347250076 | 2.816352247049716 | 298931 | 70 |
| Qwen BF16 WikiText validation | 17.7149122731184 | 2.8744067861808946 | 262208 | 193 |
| Qwen BF16 fixed C4 | 23.136144236064272 | 3.1413960802266425 | 2096128 | 0 |
| Llama exact SP2-PTQ WikiText test | 15.560178512473657 | 2.7447149912084505 | 288934 | 308 |

For all four: `progress.json` completed, no `failure.json`; segment targets and weighted NLL/exp reproduce the reported values exactly. Saved input-token metadata equals data/result metadata. Targets follow independent 2048 windows and include scoreable tails. Tokenizer paths match actual launcher model identities. Quantizer before/after saved tensor dictionaries are exactly unchanged (empty for the three BF16 runs; all 112 wrappers for Llama PTQ).

`model.json` records KV16, use_cache false, no training/calibration, FP32 RoPE buffers. Qwen BF16 has no activation quantizers and backbone weight bits16. Llama exact PTQ records W4, 96 INT8 plus 16 SP2, no overlay. These are full-sequence/PREFILL quality scores, not decode/native-kernel/NPU measurements.

The Llama evaluated package is exactly:

```text
runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt
```

Both settings and results reference it. New PTQ test tokens are **elementwise identical** to both historical `wiki2-test-bf16-fixed-20260917a/input_tokens.pt` and `wiki2-test-best-fixed-20260917a/input_tokens.pt`; metadata, segment boundaries/target counts and metric descriptions also match. The previous missing Llama SP2-PTQ test cell is now filled, without rerunning its existing C4 or either historical test result.

All 17 current `summary.csv` rows match raw evidence PPL/NLL/targets/input count/tail count; dataset/split/package fields match wherever present. Historical 13 rows remain marked inherited and these four marked new Phase5 measurements. Audit script: `check_first_results.py`; machine evidence: `first_results_checks_20260918.json` in this auditor directory.

## QAT migration state, not a completed result

Also read the Llama representative-ablation migration evidence. Old `initial-sp2-qat400-s42` has exactly logged steps1–25 and `KeyboardInterrupt`, consistent with the main executor's declared resource-driven SIGINT migration after saving step25. New `initial-sp2-qat400-s42-r25` points to that `resume.pt`, has PID1426248/GPU0 and its first logged optimizer update is step26. Parent package, floating reference, total400 schedule/updates, accumulation8, data_start800 and disabled target-PPL stopping match.

This is an interrupted-and-resumed run, not evidence of algorithm failure or a new independent seed. Final completion must still establish a continuous 1–400 update sequence (25 retained +375 resumed) and unchanged sampling/schedule/optimizer continuation; this report only confirms the observed 25→26 boundary. No completed QAT result is added to the table above.
