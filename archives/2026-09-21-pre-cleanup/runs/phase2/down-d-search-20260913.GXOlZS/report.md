# Down1 fixed-SW rounding result

Full validation PPL 17.395098280145866, NLL 2.8561884584957267; delta versus original C+SP2 -0.2473016957437828 PPL, -0.014116635883509865 NLL. Strict original W4A8 quantization scope retained, no extra high-precision module, no new scales/clipping/online transforms.

Selected 512 coordinate updates by disjoint-row train heldout MSE. Changed482403 weight codes in down1, original SW exactly retained. Selection sees actual frozen SP2 inputs; reference is original weight acting on these same inputs. This is weight-local error, not a total MLP loss. Original weights remain frozen. Pack/reload and full-model frozen checks passed. Independent verifier PASS.

Next hypothesis: aggregate down-family rounding may retain benefit. Run ../down-d-search-20260913.6qBcoE via PRECISION_LOOP=rounding_downs bash scripts/phase2/46_run_down_d_search_local.sh; propagate selected preceding W4A8 layers, retain original112SW and96SA/16SP2 alpha, choose each layer among original and32/128/512 update candidates.
