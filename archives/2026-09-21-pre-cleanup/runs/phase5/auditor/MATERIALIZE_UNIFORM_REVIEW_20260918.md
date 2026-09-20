# Uniform materialization entry review

2026-09-18, independent auditor. **PASS for this thin entry's source review and historical overlay identity.** No application edits, no new test suite, no model forward, calibration, PPL or GPU execution by the auditor.

Reviewed `experiments/phase3/materialize_uniform.py` and the `launch.py` task mapping in `worktrees/SpinQuant-multimodel`.

- `materialize-uniform` maps to the new script; launcher passes a fresh Phase5 output directory and retains its existing used-name refusal, tmux command/identity logging and GPU4 exclusion. Llama invocation must specify `--model llama32-1b` because the launcher default is Qwen.
- The script calls only the already exercised `load_static(parent)`, `apply_down_int8(model, scales, parent)` and `save_frozen(model, records, new_package)` path. It does not call a forward, tokenizer, dataset loader, range search, calibration or optimizer. `load_static` loads onto its default CUDA device but performs no model forward.
- `apply_down_int8` reads the overlay with CPU `torch.load(..., weights_only=True)`, checks resolved parent-path equality, requires WikiText train calibration metadata, complete down-layer coverage, existing SP2 inputs and finite positive single-element INT8 scales. Replacement is in-memory only.
- `load_static` reads the old parent; overlay source is only read. Existing W4 records are passed unchanged to `save_frozen`. New output creation uses `exist_ok=False`; writes are confined to that new directory (source snapshot, settings, model package, result/progress or failure).
- Settings/result declare calibration false and training false; result identifies reused calibration and the overlay source. Package metadata correctly describes an initial matched INT8 parent with neither discrete postprocessing nor QAT yet.

Independently read the tiny historical overlay file (no parent model load):

```text
runs/phase3/b100-uniform-calibration-20260918c/down_int8_scales.pt
```

It contains **16** down scales and exact parent identity:

```text
runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/static_w4a8.pt
```

Its metadata identifies Llama tokenizer, WikiText train, seed42, the original 32 calibration indices, 128 tokens/window and 4096 calibration tokens. This source review does not claim the new materialized package has already been generated or evaluated. Actual creation/cold-package identity should be checked from that run's output; no recalibration is needed for this reuse.
