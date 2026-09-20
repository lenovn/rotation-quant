# Invalid auditor measurement: cold construction precision

This directory preserves an actually completed, but protocol-mismatched, auditor evaluation. It is not an algorithm improvement or a reproduced formal result.

- Observed C-Adam10 PPL: 22.066430771720142; NLL: 3.0940574841143964.
- Driver incorrectly constructed an FP32 evaluation model and then called `.to(torch.bfloat16)`. This also rounded nonpersistent RoPE `inv_freq` buffers, which are absent from `state_dict` and the static artifact.
- The primary source uses `from_pretrained(torch_dtype=torch.bfloat16)`, which preserves explicitly constructed FP32 RoPE frequencies. A CPU-only check shows 31/32 frequencies change under the incorrect whole-model cast; native BF16 construction retains exact FP32 frequencies.
- Only the auditor driver is corrected to `_from_config(torch_dtype=torch.bfloat16, attn_implementation='sdpa')`, followed by loading all weights/scales/high-precision tensors from the same saved `static_w4a8.pt`.
- No training, recalibration, artifact change, algorithm implementation, or candidate expansion. Original BF16 is not repeated. One replacement full evaluation of the same C10 is needed because this measurement is invalid for the requested matched protocol.
- See `../rope_cpu_diagnosis.json`. This is an auditor-driver FAIL, not evidence that the main run failed.
