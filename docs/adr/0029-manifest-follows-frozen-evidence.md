# ADR 0029: Manifest follows frozen quantization evidence

- Status: Accepted
- Date: 2026-08-11

M004 introduces a shared dual-phase static A8 policy and CLI, but deliberately does not write `quant_manifest.json`. A valid manifest must be derived from the model's actually frozen A8/K8/V8 qparams, calibration records, exact coverage, and checkpoint hash; copying requested CLI values into JSON would create an apparently strict artifact without evidence that those settings were realized.

Therefore M004 is named the static A8 preparation profile and must fail closed when asked to use the existing dynamic K/V or incomplete save/export paths. The manifest schema/profile meaning is frozen now, while its builder and file output wait until neutral static K/V, calibration, coverage, and package serialization are implemented. M004 output is not a static W4A8KV8 model package and carries no mllm/QNN or phone NPU claim.
