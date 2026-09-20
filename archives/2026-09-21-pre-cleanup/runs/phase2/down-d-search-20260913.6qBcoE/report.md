# Sequential down-family rounding

Full validation PPL=17.546664783380706, NLL=2.8648638910188193; delta_NLL vs original C+SP2=-0.005441203360417202. All16 layer local heldout objectives selected512; original C SW/SA/SP2 alpha frozen. Sequential upstream selected weights are propagated to next layer, independent candidates include original codes. Independent verifier PASS.

Result is worse than single down1 rounding17.3950982801. Do not promote family over current best. Local weight reconstruction improvements do not necessarily accumulate in end-to-end NLL.

Next experiment: same down1 scope and candidate budget as GXOlZS, but use same-input fixed-R unquantized BF16 MLP reference, including gate/up and activation error in reference mismatch. Candidate local matmul remains FP32 with BF16 dequantized weights/inputs, so this is a proxy; final validation uses actual BF16 GEMMs. No teacher model/KL, no gradients, no changed scales/clipping. Run ../down-d-search-20260913.v2WHxW; command PRECISION_LOOP=rounding_mlp bash scripts/phase2/46_run_down_d_search_local.sh.
