# Same-input MLP reference rounding

Full validation PPL=17.40639276660993, NLL=2.856837539125371, delta_NLL=-0.013467555253865449. Better than original17.64239998 but not better than weight-only down1 rounding17.39509828; retain prior best. Reference includes BF16 gate/up/down at same actual normalized MLP input and same R. Candidate local objective is FP32 matmul with BF16 quantized weights/input; final evaluation uses actual BF16 forward. Original SW/SA/SP2 alpha fixed, no gradient or teacher KL.

Next hypothesis: 2048 calibration rows may permit spurious input correlations and row-heldout from same windows may be too weak. Keep same32train windows but fit full24windows (49152rows), select on remaining8fullwindows (16384rows). Compare original and512/2048/8192 coordinate steps. Fixed-SW weight-only objective retained to match current best. Run ../down-d-search-20260913.sWJwrk via PRECISION_LOOP=rounding_coverage bash scripts/phase2/46_run_down_d_search_local.sh.
