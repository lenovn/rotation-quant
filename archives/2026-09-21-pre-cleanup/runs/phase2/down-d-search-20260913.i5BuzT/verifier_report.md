# Independent weight-local continuation rejection audit

Verdict: PASS correct identity rejection; parent Ds4vSf retained.

Independently compared settings with MLP-target YocmPn: same parent, weight parent, target layer1, steps, calibration indices,24/8 fit/heldout row counts, SW and activation state. Reference target is the intended experimental difference.

Actual next-token train NLL: identity2.730179279517435; step5122.736020618595766; step20482.7340562535228257; step81922.7337067540435944. Every nonidentity candidate is worse despite lower local MSE. selected_step=0 correct.

CPU reload verifies all16 packed records, shapes and SW exactly match parent; alpha JSON exactly matches parent. reused_frozen_parent=true, PPL16.122226577621436/NLL2.7801988527027004 match parent. budget.full_validation_evaluations=0 confirms no repeated full validation. No verifier GPU work.

Both matched objectives reject this additional candidate search on Ds4vSf. This supports retaining the parent at this stage, without claiming a global optimization limit or universal failure of either reconstruction objective. Frozen external C4 evaluation is separate and not included here.
