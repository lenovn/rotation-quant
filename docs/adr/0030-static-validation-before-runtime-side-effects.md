# ADR 0030: Static validation precedes runtime side effects

- Status: Accepted
- Date: 2026-08-11

M004's first implementation validated static requests at the start of `ptq_model()` and `prepare_model()`, but the real `ptq.py` and `optimize_rotation.py` entry points had already initialized NCCL and loaded model weights; evaluation had also moved the model to GPU. Passing policy unit tests therefore did not prove operational fail-closed behavior on the shared server.

Static validation is split into args-only and model-config stages. Each top-level entry must parse and validate args, load only lightweight `AutoConfig`, validate model constraints, and only then permit distributed initialization, model-weight or data loading, or GPU work. Downstream full-request validation remains as defense in depth. This expands M004 from five to seven files and intentionally accepts a small host-side config read before runtime initialization to prevent invalid static jobs from consuming shared compute resources.
