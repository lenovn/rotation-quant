# Layer 1 BF16 mixed-precision validation

User authorized one additional complete validation on original C + SP2, restoring weights and activations in model.layers.1 (zero-based second block). This is not a D search despite the reused launcher directory prefix. No train windows are read, no calibration or requantization occurs. mixed_settings.json is the operative configuration; generic settings.json contains unused D-search defaults from shared setup.

Frozen original C learned-SW checkpoint and original SP2 alpha are retained elsewhere. Donor is original BF16 model after the same LayerNorm fusion and C R1/R2, before W4. Exactly seven backbone projections restored and activation inputs bypassed; 105 other projections remain C W4, 90 non-down INT8 scales and 15 SP2 alphas remain fixed. High precision boundaries including both lm_head aliases are checked against C. Independent CPU startup verification PASS after correcting weight alias exclusion in boundary checks.

Command is recorded in command.txt. GPU1 started at 2412/24564 MiB. Python PID3668406. Prior loop GPU seconds558.067717367. Result and budget are recorded separately on completion. Baseline 17.64239997588965 is the existing matched validation result, not a new baseline rerun.
