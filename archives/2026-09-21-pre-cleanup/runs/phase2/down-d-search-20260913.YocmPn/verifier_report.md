# Independent same-input MLP-target rejection audit

Verdict: PASS correct rejection/identity handling; NO algorithmic improvement selected.

Earlier independent CPU/source readiness checks verified32-window MLP prehook/quantizer-output alignment,24/8 target shapes, native BF16 SwiGLU reference equivalence, and12 random arbitrary-target neighbor-coordinate problems against direct exhaustive objectives. Local objective is FP32 matrix output fitted to BF16 MLP reference; actual next-token NLL remains the selection criterion.

Actual train NLL:

| Step | Train NLL |
| --- | ---: |
| Parent identity0 | 2.730179279517435 |
| 512 | 2.7404165028563283 |
| 2048 | 2.7434329914582594 |
| 8192 | 2.7607426245894042 |

All three candidates reduce local heldout MSE but worsen true train NLL. selected_step=0 is correct. This is a concrete case where local reconstruction improvement does not justify end-to-end promotion.

Independent CPU reload confirms every one of16 saved packed code records, shapes and SW exactly matches Ds4vSf; alpha JSON exactly matches parent. Result explicitly marks reused_frozen_parent=true; PPL16.122226577621436 and NLL2.7801988527027004 exactly reuse parent rather than reporting a new measurement. budget.json confirms zero full validations,83.93237683200277 seconds and4.869140625 GiB peak reserved. No verifier GPU work.

Parent remains accepted. Same-parent weight-local continuation control i5BuzT is separate and not yet included in this report. No conclusion that every MLP reconstruction method fails is supported; this bounded candidate set failed to improve its actual NLL selection objective.
