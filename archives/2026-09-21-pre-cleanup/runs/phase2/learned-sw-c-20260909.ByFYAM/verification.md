# Experiment C verification

2026-09-09. Implementation and CPU verification completed before GPU launch.
Independent verifier: /root/verify_c. Startup verdict: PASS.

- 7 CPU tests passed: tests/test_learned_sw_training.py and tests/test_w4_aware_training.py.
- Independent BF16 check: same forward as B, identity weight STE, per-output-row FP32 scale gradient including saturation.
- 16-layer fixture: 17 R, 96 SA, 112 SW trainable tensors; down input A16, down weight W4.
- SW parameters registered before optimizer and DDP creation. No per-step or final min-max overwrite.
- Launcher common arguments match B: 100 steps, seed/data_seed 42, R LR 1.5, SA LR 1.0, cosine, warmup 10, global batch 8.
- C SW LR: 0.01, separate SGD group; row LSQ normalization 1/sqrt(input_channels*7), positive projection at 1e-8.
- C starts from common R0/SA0, not B final checkpoint; SW0 loaded from B/rotation/initial_quant_scales.pt.
- bash -n and actual argparse RTN override checks passed.

Live initialization verification after launch:
- C initial_quant_scales.pt: all 96 SA tensors exactly equal B initial SA, all FP32.
- All 112 SW tensors exactly equal B initial SW, all FP32.
- First logged loss: B=3.4426, C=3.4426 (rounded training log values).

GPU training and final artifact verification are pending. Startup PASS does not establish completed training, improved PPL, or NPU deployment.
