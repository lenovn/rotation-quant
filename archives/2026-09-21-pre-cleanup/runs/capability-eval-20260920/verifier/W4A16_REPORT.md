# Independent W4A16 verification

Status: PASS, 2026-09-20. Application files reviewed: `scripts/capability_eval/w4a16.py` and the W4A16 branch of `run.py`. No application source was modified.

An independent GPU7 run (`check_w4a16.py`, raw evidence `w4a16.json`) loaded the existing Phase2-C GPTQ package. All 147 model state tensors exactly equalled the saved checkpoint after wrapper-name mapping, including embedding, lm_head, and norms. The 112 saved W4 quantizer names exactly matched seven projections in each of 16 layers. No activation wrapper remained. One short scoring forward produced finite logits; a two-token greedy generation using DynamicCache succeeded. GPU7 was released; no unrelated process was stopped.

The saved launcher `runs/phase2/learned-sw-c-20260909.ByFYAM/launcher.sh` explicitly uses `--rotation_components r1_r2`. Thus there is no omitted online R3/R4 transform. The checkpoint was saved after norm fusion and weight rotation; exact loading of its saved embedding/head/norm/linear tensors is appropriate and avoids recomputing rotations.

The historical GPTQ solution observed static A8 inputs with down_proj A16. This control evaluates those saved W4 weights with all-A16 inputs, matching the historical `C/eval-w4a16` use, and is a distinct Phase2 recipe. It must not be described as changing only the activation precision of the final FIRON weights. Historical `C/eval-w4a16/result.json` gives PPL 15.238767623901367 on 8x2048; this is not full WikiText-2 test. The adjacent `C/gptq/result.json` gives 15.314295768737793 for A8/down-A16 and must not supply the W4A16 table cell.

## Initial formal-result audit

All six completed BoolQ result sets (two models x BF16/FIRON/parent) were read. Each contains all 3270 validation examples, zero shots, and the official accuracy metric. For each model, every row's `doc_id`, actual scoring `arguments`, and `target` matches across the three methods exactly. No protocol mismatch was found. This confirms only the completed BoolQ artifacts; later tasks and generation prompts require their own completed artifacts before claims about their execution are made.
