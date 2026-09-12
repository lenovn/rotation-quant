# ADR 0028: Algorithm/deployment boundary for host-resolved phase

- Status: Accepted (amended 2026-08-11)
- Date: 2026-08-11
- Supersedes: ADR 0027

Dual-phase remains cache-derived, and phase resolution is an eager host-boundary responsibility in the current PyTorch algorithm reference. Transformers 4.44.2 `StaticCache.get_seq_length()` performs occupancy reduction, so the host validates cache/position evidence, produces an explicit `QuantPhase`, and propagates it through the model, checkpoint path, and decoder layers. Missing or inconsistent evidence fails closed.

For Hugging Face generation, each model copy's `prepare_inputs_for_generation()` is the boundary that calls `resolve_quant_phase(...)` and adds the result to `model_inputs`. `LlamaForCausalLM.forward`, `LlamaModel.forward`, checkpoint calls, and decoder layers consume the resolved phase; StaticCache calls without it fail closed. DynamicCache direct eager calls may resolve at the same eager boundary.

The algorithm stage modifies SpinQuant, performs training/calibration, and ultimately exports a static W4A8KV8 model package containing W4 weights, frozen dual-phase A8/K8/V8 qparams, and a manifest. mllm converter/IR/runtime integration, numerical alignment, QNN AOT compilation, and mobile NPU validation are a later deployment stage. The model package is therefore not itself evidence of mllm or NPU readiness.

`torch.compile(..., fullgraph=True)` remains useful as a diagnostic for Python frontend compatibility, but it is not an M002 acceptance gate and cannot substitute for later mllm/QNN evidence. In particular, a TorchDynamo failure caused only by a Python context-manager implementation does not justify backend-specific algorithm changes. M002 acceptance instead requires independent verification of cache-derived phase resolution, explicit propagation, checkpoint/context correctness, and fail-closed behavior.
