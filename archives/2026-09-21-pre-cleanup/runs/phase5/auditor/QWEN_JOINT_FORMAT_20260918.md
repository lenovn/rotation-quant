# Qwen seed42 Joint100 and matched initial down-format audit

2026-09-18. **PASS: actual Joint100 training, matched initial INT8 materialization, and both frozen initial-format C4 runs.** These are **initial Joint100 format controls**, not full postprocessed SP2/Uniform PTQ or Uniform-QAT400 results. Read-only CPU evidence analysis; no GPU, PPL rerun, retokenization, application edit or in-flight artifact modification. Detailed checks: `QWEN_JOINT_FORMAT_20260918.json`; training/package/calibration helper: `check_qwen_joint_format.py`.

## Actual Joint100 budget and seed

`qwen3-1p7b/joint100-s42/training.jsonl` contains exactly updates **1–100**. Each update has eight 2048-token microbatches and 16,376 prediction targets; window indices are exactly 0–799 in order. The effective global budget is **1,638,400 input-token visits / 1,637,600 prediction-target visits**. No updates are missing or duplicated; progress is completed with no failure file.

The actual saved settings use route B, seed42, accumulation8, 100-step schedule, warmup10, rotation LR1.5, Adam scale optimization with relative rate0.001, and the exact `init-s42/initial.pt`. Both initialization and Joint data records agree: 2,517,232 training tokens, 1,229 complete windows, 240-token dropped tail, 32 calibration windows ×128, and probe indices1221–1224. Independently applying `torch.randperm(1221, generator=seed42)[:32]` reproduces every saved calibration index. No tokenization was repeated for this audit.

The initial artifact records seed42 and unoptimized randomized-Hadamard initialization. Saved source calls `set_seed(SEED)` before model construction and creates one R1 plus28 R2 matrices. All **29 R tensors** differ between the initial and actual step100 state. Every update records29 R,168 non-down SA and196 SW gradient tensors and nonzero changed elements in those groups. Route B performs W4-aware R/SA/SW optimization with down activation quantization bypassed; its28 SP2 scales have no training gradients/updates. The final down-SP2 ranges are calibrated after freezing step100, not inferred from train-time SP2 learning.

Step100 validation was independently recomputed from its segment metrics: **PPL16.340940643824144 / NLL2.793673654716446**, 262,208 targets. The corresponding official `Qwen/Qwen3-1.7B`, seed42, `B100-initial-SP2` summary row matches the raw JSON.

## Matched calibration and frozen package identity

SP2 parent: `qwen3-1p7b/joint100-s42/checkpoint-0100/static_w4a8.pt`. INT8 materialization: `qwen3-1p7b/uniform-initial-s42/static_w4a8.pt`, whose metadata/settings identify that exact parent. The complete INT8 file exists and is not an external overlay applied during evaluation.

Direct artifact comparison established:

- All **196 W4 records** have bitwise-identical packed codes, scales (SW), shapes and other record fields.
- All **154 high-precision state tensors** are bitwise identical, including embedding/head, layer norms, RoPE state and all **56 Q/K norm tensors**. Model config is identical.
- All **168 non-down INT8 activation scales** are identical.
- The SP2 package has168 INT8 +28 SP2 activations; the matched package has196 INT8. Only the28 down activation format/ranges and descriptive package metadata differ.

Frozen packages contain fused rotations, not separate R tensors. Equality of their rotated W, embedding/head and all high-precision tensors establishes the same effective rotated basis; there is no separately optimized INT8 R. Sixteen inactive head-quantizer placeholder buffers contain NaN on both sides. Comparing raw tensor bytes confirmed they are unchanged; ordinary `torch.equal` would falsely mark matching NaNs unequal. This is an audit-comparison detail, not a model change or nonfinite learned weight.

INT8 uses the same recorded train-only32×128 calibration indices as final SP2. Both capture every eighth activation row, giving **512 sampled rows per layer**, and have **28 layers ×50 candidates**. Each layer's full absolute maximum and sampled-row count match exactly between saved SP2 records and the INT8 capture metadata. Source uses the same down-input hook/bypass convention and restores the original frozen-model column-major v_proj layout for matched capture; temporary layout and quantizer settings are restored afterward. All saved196 parent quantizer states before/after INT8 calibration are bitwise identical.

Both searches use33 coarse log2 ratios from−16 to1, then17 refinement points between the adjacent coarse ratios around that format's own coarse minimum. Every saved candidate-grid position and selected minimum was independently checked; exported SP2 alpha and INT8 scales agree with their selected candidates and FP32 storage conversion. Thus the search **budget, coarse bounds and refinement rule** are matched; the format-dependent winning ranges and refined grids need not be identical. Full layer ranges and selected alphas remain in the JSON.

Capture evidence is bounded: arrays were not saved, so this audit does not claim a fresh elementwise equality check of captured activations. It establishes common source algorithm/layout, identical selected windows, identical W/SW/non-down state, and exact per-layer capture bounds/counts. No capture or calibration forward was repeated.

## Completed frozen C4 format control

Both external runs completed with no failure file and identify their respective exact initial packages, with no down overlay, no calibration, no training and no candidate selection. `model.json` confirms196 W4 matrices, the actual activation format counts above, KV16, `use_cache=false` and FP32 RoPE.

For each run, all196 saved activation-quantizer states are exactly equal before/after evaluation; actual scales match the package. All INT8 observers remain disabled with zero observed calibration samples. Non-down states are also exactly equal across SP2/INT8 runs. Both use the exact existing Qwen C4 input path and IDs/complete metadata also used by Qwen BF16: **2,097,152 input tokens /2,096,128 targets**,1,024 complete2048-token windows, no tail. Segment boundaries/counts match BF16; every segment NLL/log(PPL) and final target-weighted recomputation agrees exactly. Both summary rows match their raw metrics, packages, seed and official model identity.

| Initial down format | Actual activations | C4 PPL | C4 NLL | Summary stage |
|---|---|---:|---:|---|
| SP2 | 168 INT8 +28 SP2 | 27.18323240302331 | 3.3026003274437667 | B100-initial-SP2 |
| INT8 | 196 INT8 | 33.33932032786121 | 3.5067374910279026 | B100-initial-INT8 |

**ΔNLL (initial SP2 − initial INT8) = −0.2041371635841358** on this fixed Qwen C4 input. This identifies the matched initial down-format effect only. In-flight down-neighbor/range/readaptation/QAT stages are outside this completed initial-package audit, and these C4 metrics do not authorize package switching or tuning on external data.

## External-candidate registry clarification

The separate `verifier/external-spinquant-smoke/EXECUTION.md` now reports a bounded PASS for the existing Llama local callable SpinQuant R-update +GPTQ path. This supersedes the earlier “runtime not validated” registry state only for that tested local core path. It does not establish full CLI orchestration, formal external PPL, official-method equivalence or Qwen GPTQ support, and it is unrelated to the internal Uniform format comparison audited here.
