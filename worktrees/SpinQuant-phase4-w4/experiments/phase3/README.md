# Phase 3 joint optimization

The source root is this worktree, not `repos/SpinQuant`. Artifacts go to the
outer project's `runs/phase3`. The main thread owns experiment selection;
`launch.py` only starts one specified experiment and records its actual GPU,
command, PID and log. The latest user authorization permits independent
experiments on GPUs 1/3/5/6/7 according to measured headroom, without the old
single-card, 12 GiB, 35 percent or time caps. Other users' processes are never
terminated. Environment variables affect the child only; no installation is
needed.

## Matched comparison

- A: 50 updates of W16 R+non-down SA, then 50 W4 R+SA+SW. At the switch,
  initialize SW from the current rotated FP weights, not a stale starting grid.
- B: R+non-down SA+SW under W4 from the unoptimized starting rotation.
- C: B plus 16 learned down-SP2 range parameters from the same start.
- Original model weights remain frozen. R1 and per-layer R2 are FP32 and use
  the existing SGDG orthogonal update. R3/R4 stay disabled. Every route uses
  microbatch 1, accumulation 8, 2048 tokens, fixed row-tokenized train order.
- Shared R uses the existing randomized Hadamard initializer with seed 42;
  no learned Phase2 R/SA/SW is inherited. Non-down SA is full-range FP-forward
  train calibration; SW is current-R per-output maximum/7. Initial down-SP2
  uses train-only output MSE, 33 coarse plus 17 refined range candidates.
- A/B receive the same final train-only down-SP2 calibration procedure. C
  keeps its learned range; neither learned non-down SA nor SW is overwritten.

## Forward, gradient and metrics

SP2 preserves the existing sign-plus-magnitude codebook and midpoint ties
toward smaller magnitude. Its parameter is `scale=alpha/127`. Forward is
exact nearest SP2 projection in FP32 followed by the GEMM input dtype.
Backward uses an in-range identity input STE, and a range LSQ surrogate:
`127*(projected_normalized - normalized*in_range)/sqrt(N*127)` for the scale.
W4 uses the existing identity input STE and per-row learned-SW gradient;
non-down A8 uses the existing LSQ gradient. All scale parameters stay FP32.

Training and fixed train probes use FP32-logit next-token CE, computed in
128-token chunks with checkpointed head loss. Full validation deliberately
uses the existing acceptance/evaluator precision: BF16-logit per-token CE,
loss to FP32, then log of evaluator float32 PPL and predicted-token weighting.
Validation is the same double-newline joined WikiText2 validation, 252852
tokens, 123 full windows plus a 948-token tail, 252728 scored targets.
The two CE numerics are labelled, not silently compared as the same metric.

The original all-SGD smoke showed tiny SP2 updates. A paired short test uses
existing Adam for scales with each parameter's LR set to 0.001 times its
initial mean scale, epsilon 1e-12, while retaining SGDG for R. This is not a
change to the quantizer forward or gradient. Formal selection between these
optimizer settings must use the actual probe/update evidence, not this note.
For a pure initialization comparison, `--optimizer-scale-reference` points
to the common unoptimized checkpoint so the absolute per-parameter LRs also
use the same initial scales when starting from improved scales. The initial
`init-c-adam-joint100-20260914a` execution computed the reference mean on CPU,
unlike the original C's GPU mean: 54 logged LRs differ in final bits, by at
most 1.57e-7 relative. That run is not a bitwise-LR-matched control; its original
source snapshot and measurements are preserved. Subsequent executions move
the reference to the parameter device before the mean reduction. The equal-total-budget
control resumes the scale-only ten-update state without `--scale-only-steps`:
R becomes trainable, updates 10..99 consume windows 80..799, and the existing
optimizer/RNG/schedule continue. It has ten scale-only plus ninety joint
updates versus the original hundred joint updates, on the same total data.

The LR schedule is shared 100-update cosine with ten-update warmup; first
update uses 1/10 base LR, unlike a Trainer schedule starting at zero. Checkpoint
10 versus 100 measures continuing one schedule, not independently scheduling
a separate ten-update run. `training.jsonl` records each microbatch loss,
target count, LR, finite gradients and actual changes. Fixed probes and
checkpoints are at 0/10/25/50/100; full validation defaults to 10/100.

## Artifacts and recovery

New `launch.py` jobs run inside tmux on the dedicated `rotation-quant-phase3`
socket. Their `.launch.json` records the session, actual pane/Python PID,
explicit child environment, log and attach command. For example:
`tmux -L rotation-quant-phase3 attach -t phase3-RUN_NAME`.
Ctrl-B then D detaches without stopping training. Earlier detached process
jobs remain running unchanged; they are not restarted to migrate launchers.

`initial.pt` / `state.pt` contain the 17 R, 112 SW, 96 SA and 16 SP2 scale
parameters. Per-update `resume.pt` atomically saves parameters, optimizer and
RNG together; the loader also accepts the completed initial smokes' older
`resume_state.pt` plus `optimizer.pt` pair.
`--resume` takes a prior run directory and writes to a new output directory.
The original model is loaded only from the local cache. The accepted source
`utils.eval_utils` is always imported from this worktree; the small outer
acceptance helper is read-only reuse, not old SpinQuant imports.

Each evaluated checkpoint exports `static_w4a8.pt`: 112 packed signed INT4
weight records with frozen SW, all input formats/scales, high-precision
boundaries, and model config. R is fused into weights/embedding/head offline.
Its probe is compared before/after actual pack reload. Original BF16 hidden
unrotation versus fused head differs in BF16 rounding order; do not promise
bitwise training/frozen logits or mistake a small rounding discrepancy for a
quantization gain. Independent numerical verification is required before
closing implementation, and full experiment claims require completed results.

## Train-only discrete postprocessing

`launch.py --task postprocess` runs `postprocess.py` under the same tmux
launcher. It loads an existing frozen package and records its parent and the
matching rotation/scale state used to recover current-R FP reference weights.
It does not change original source weights or reuse a Phase2 learned rotation.
Only a candidate improving the four fixed train-probe windows receives a new
full validation; the main executor compares that result to the parent before
accepting it. No validation tokens enter calibration or candidate fitting.

- `--mode round`: rank all 112 modules by normalized local W4 output error,
  then try existing fixed-grid +/-1 coordinate updates on the top 16. Inputs
  are the actual frozen activation quantizer outputs. SW stays fixed; heldout
  local error selects among 0/8/32/128 updates. Joint top-four and selected-top-
  sixteen packages are compared on the same train probe.
- `--mode sp2`: all 16 down ranges try 33 log-spaced multipliers from 1/16 to
  16, explicitly including the parent. Each layer's heldout local output MSE
  proposes its range; the combined package still needs a train-probe gain.
- `--mode d`: rank every intermediate channel in all 16 layers by SP2 error
  energy times FP down-column energy, plus quantized-input energy times W4
  down-column error energy. Both terms are recorded separately; their sum is
  a ranking heuristic ignoring cross terms, not causal error attribution.
  This also considers outlier channels whose tiny FP down weights round away.
  For the top four layers, try their top
  1/4/16 channels with D factors 1/4, 1/2, 2, 4, 16, 64 or 256. Identity retains the parent
  codes and SW exactly. Other candidates fold D inverse into up SW and D into
  selected down-column codes using unchanged down SW. All remaining codes,
  weights and non-down SA stay fixed. Both identity and nonidentity D search
  the same thirteen SP2 range multipliers, powers of two from 1/256 to 16. A nonidentity
  proposal must beat the best identity-range control in local heldout MSE;
  range-only and joint-D proposals then receive separate fixed train probes.
  This avoids crediting a range-only improvement to D or retaining an oversized
  SP2 range after compressing outlier channels. The local reference uses FP up/down
  with the parent's quantized up input and gate output; it is not end-to-end
  teacher loss. Each layer proposal is evaluated independently on the probe.

Local fit/heldout are the first 24/last 8 of the existing 32 train-calibration
windows, sampled every eighth row (384/128 rows). This is a holdout from the
local fit, not a claim that these rows were unused by initial calibration.
D introduces no online operation or high-precision backbone exception. These
are fake-quant GPU measurements, not native INT4/SP2 device-kernel validation.

## Authorized quantization-aware distillation fallback

`launch.py --task distill` starts `distill.py` only after the main executor
finds that the measured PTQ routes/postprocessing remain above BF16+1.
This is explicitly **not frozen-original-weight PTQ**. It trains 112 FP32
master backbone weights with SGD momentum, while every forward still uses
W4 on the saved per-output grid. The 96 SA, 112 SW and 16 SP2 tensors use the
existing learned-scale surrogates and Adam. Original fused embedding/head/norm
boundaries remain frozen; no online R, adapter, extra inference operation or
high-precision backbone exception is introduced.

The teacher is the original local BF16 model, unrotated and without norm
fusion, constructed with FP32 RoPE buffers. It is frozen and only sees train
windows. The default objective is 0.9 KL(teacher || student) plus 0.1 labeled
next-token CE, temperature one, FP32 logits; chunked 128-token head losses and
non-reentrant block checkpointing limit memory. The default first stage is
100 updates, microbatch1/accum8, SGD LR0.1/momentum0.9, warmup10/cosine100,
scale Adam LR0.001 times each parent scale's mean. These are initial settings,
not a claim of optimal hyperparameters. The train index starts at window800,
wraps modulo the same 1180 training windows and still excludes the eight
reserved train windows. All actual indices, objectives and learning rates
are recorded.

Resume records atomically save FP32 masters, scale parameters, optimizer and
RNG state, generally every25 updates plus requested checkpoints. They are much
larger than rotation-only state. Validation offloads training/teacher models,
exports packed W4 and frozen activation parameters, then **cold loads that
package** for the original complete validation protocol. Only the frozen
export is an inference artifact. Checkpoints default to10/25/50/100, full
validation to25/100; the process may stop early when a completed full result
meets the stated target, without closing the main goal or skipping independent
final audit and other required comparisons.

After the SGD0.1 setting deteriorated and SGD0.001 remained stable but did not
improve its checkpoint25 validation, a separate `--weight-optimizer adam`
configuration is available. Adam uses FP32 master/moment state, epsilon1e-8,
no weight decay and no foreach temporary tensor lists. `--offload-teacher-body`
moves the frozen teacher body onto the active device for its no-grad forward,
then returns that body to CPU before the student forward/backward; the teacher
head and hidden target remain on GPU. This changes placement, not the KL
definition or teacher weights. It avoids keeping the complete teacher resident
alongside the additional Adam moment state. Such runs specify their own LR
and schedule explicitly, rather than claiming a pure optimizer-only comparison.

`--reference-state` provides a separate master-weight initialization study.
It reconstructs current-R FP weights from that saved Phase3 state and, for a
direct local-D parent, applies the D recorded beside the package. The FP32
masters are projected into the parent's INT4 cells with a 0.49-code interior
margin (saturated endpoint cells remain unbounded outward). Thus **the initial
quantized model is unchanged**, while the optimizer starts with FP residuals
instead of everyone at dequantized grid centers. The parent SW/SA/SP2 and HP
boundaries are unchanged; this does not add calibration or training updates.
The source state and any diagonal transformation are recorded separately.

## Sequential full-window postprocessing

`launch.py --task sequential` runs `sequential_postprocess.py`. This supplements,
not retroactively replaces, the smaller first postprocessing search. It uses
the same Phase3 train indices but all2048 tokens:24 fit windows,8 local-heldout
windows and16376 predicted train targets for actual NLL selection. Selection
uses the original BF16-logit evaluator numeric, not the four-window FP32 probe.

`--mode round` defaults to all16 down modules in model order and fixed-SW
INT4 neighbor checkpoints0/512/2048/8192. It reuses the existing pure Phase2
coordinate helper without modifying Phase2. Every layer captures actual inputs
under the already accepted prefix; local heldout MSE only prefilters candidates,
and actual eight-window train NLL decides which code update to retain.
The explicit `--reference-state` must describe the parent's current-R original
FP weights; the rounding mode is not an adapter for distillation-master states.
`--family non-down` or explicit `--targets` supports subsequent bounded passes.

`--mode sp2` searches each down range by actual train NLL, retaining accepted
prefixes rather than combining independent local-MSE winners. Default alpha
multipliers are1/1.125/1.25/1.5/2/4/8/16 relative to each parent layer's range.
It does not require an FP weight reference and does not change weight codes,
SW or the96 non-down activation scales. It may refine a frozen QAT package,
but that parent retains its QAT provenance and is not reclassified as original
frozen-weight PTQ merely because this postprocessing step uses no gradients.

Each completed module saves its trials and a compact atomic `prefix.pt` of
changed packed weights/down scales, scores and completed targets. A new run
can use `--resume-prefix` with the same parent/reference/mode/target order,
without repeating completed targets. Only a train-NLL-improving final prefix
is exported and cold-loaded for one full validation. Failed or rejected
prefixes are not passed off as newly measured complete results. New source
paths require independent focused verification; this README is not a PASS.

## Matched local D after sequential PTQ

`launch.py --task local-d` runs `local_d.py` on an explicit `--layer` of the
new PTQ prefix, using its original current-R `--reference-state`. This path
expects a PTQ parent without a previously fused D, not a QAT master reference.
The layer is an explicit experiment choice; within it, the channel is ranked
by actual pre-SP2 RMS divided by the FP down-weight column absmax on24 full
train windows. A bounded, geometrically centered sparse log-D is searched at
strengths0/.25/.5/.75/1, including the exact identity transformation.

For every D, up/down FP weights are independently transformed and cast to
BF16. Up SW is transported as parent SW/D; down SW is the maximum of parent
SW and transformed BF16 row absmax/7. Each candidate starts from its own RTN
codes and runs0/512/2048/8192 neighbor rounding using its actual quantized
inputs. All checkpoints compete on actual eight-window train NLL, not just
local MSE. Parent SP2 alphas,96 non-down SA, gate and unrelated weights remain
fixed. The I candidate uses the same SW/rounding rule and is **not assumed to
equal the untouched parent**.

Only an improvement over the untouched parent's train NLL triggers export.
The matched I rule and selected nonidentity candidate then receive separate
cold-package complete validation; when I wins it is evaluated once and the
selection result explicitly reuses that measurement. Candidate records,
diagonal vectors, FP-reference/parent provenance and all trial scores are
retained. No online diagonal operator, new activation format or gradient
training is introduced. Source verification and measured PPL remain separate.

## Fixed external C4 evaluation

`launch.py --task external` runs `external_eval.py --mode bf16` or
`--mode quantized --package PATH` with the same required `--tokens PATH`.
It reuses the existing C4-English validation token artifact and the existing
acceptance/evaluator with `--chunk-windows 128`. C4 has no official test split.
The project protocol is 1024 independent 2048-token windows, 2097152 input
tokens and 2096128 scored targets, not the full C4 validation corpus.

The BF16 branch loads the original unrotated model. The quantized branch
cold-loads the selected Phase3 package without training, recalibration,
reference-weight reconstruction or format selection. It preserves FP32 RoPE
buffers and checks frozen activation states before/after evaluation. Both
branches use BF16-logit CE and target-weighted log segment PPL, exactly as
the existing fixed-token C4 protocol. No training `data_windows` are loaded.
Results include source snapshots, token provenance, actual format coverage,
segment accounting, state checks and timing. Reused input data and historical
C4 measurements are explicitly distinguished from new GPU measurements.
