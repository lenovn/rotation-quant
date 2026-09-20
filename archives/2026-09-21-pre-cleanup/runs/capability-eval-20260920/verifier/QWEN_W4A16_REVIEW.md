# Qwen final-weight A16 control: source review

Initial conclusion: unwrapping backbone `ActQuantWrapper.module` after `load_static` is appropriate for the current Qwen final-package loading path. New loader and runtime/result acceptance remain pending; no GPU execution was performed in this review.

## Implemented loader acceptance

PASS for the new loader and entrypoint code. `w4a16.py` checks the online/output-quantization flags and ordinary Linear type before unwrapping, preserves each same module, and verifies no backbone activation wrappers remain. `run.py` records execution hooks for all saved W4 module names. `ppl_w4a16.py --model qwen` binds the matching model path before imports and uses the Qwen BF16 C4 token artifact. The separate MMLU scheduler can skip PPL to avoid duplicate parallel PPL runs.

Independent CPU-only loading in `check_qwen_w4a16_cpu.py` confirms all 196 backbone weights equal unpacked saved INT4 codes times their saved scales after BF16 casting. All 154 saved high-precision tensors match, including embedding, head, norms and inactive wrapper state; identical-position NaN placeholders in inactive lm_head quantizer buffers are treated as equal. There are exactly 196 expected backbone module names and zero remaining backbone activation wrappers. Raw evidence: `qwen_w4a16_cpu.json`. No GPU smoke test, calibration, forward evaluation or package rewrite was performed by the verifier. Formal result audit remains pending.

Evidence from the live multimodel source:

- `architecture.load_model(..., training=False)` loads the official Qwen3 architecture. Training-only rotation hooks are not installed.
- `postprocess.load_static` calls that loader, adds activation wrappers, and reloads the saved package. It does not activate online full/partial Hadamard operations.
- `common.reload_frozen` restores all saved high-precision tensors and copies unpacked INT4 codes times saved scales into each backbone linear weight. Qwen Q/K normalization and other architecture components remain outside the removed wrappers.
- `ActQuantWrapper.forward` can contain online Hadamard transforms and output quantization in addition to input quantization. Therefore unwrapping is justified only after confirming both online flags are False and output quantizer bits are 16. The current loader's defaults satisfy these conditions. The wrapped module should be an ordinary `torch.nn.Linear`, rather than a training rotation-aware linear.

Recommended implementation: snapshot `list(wrappers(model).items())`, check those conditions, replace each named wrapper with the same `.module` object, and preserve the full surrounding loaded model. This removes only A8/SP2 input fake quantization for the selected frozen final weights, without recalibration, weight search, recomputed rotations or new checkpoint training. The original BF16 dtype remains the A16 execution type. `lm_head` may retain its already inactive 16-bit wrapper; if it too is unwrapped, use the same conditions.

The package and model-path binding must match Qwen before importing the phase3 modules. The control label should explicitly say final FIRON weights / A16, distinguishing it from Llama's historical Phase2 GPTQ control.
