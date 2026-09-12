# Distribution Atlas record list

## Phase 1 decisions

- Profile the BF16 model after materializing the selected `R.bin` with the same
  evaluation rotation path used by the formal experiment.
- Enable R1/R2 only; keep R3/R4 disabled.
- Cover all 112 Transformer Linear weights.  Activations are stored as 64
  unique tensors because Q/K/V share one input and gate/up share one input.
- Store 4096-bin histograms and render them as 256-bin Linear Count and Log
  Count panels.  Each source keeps its complete observed x-axis range.
- Do not collect per-window maxima or force shared axes.

## Phase 1 limitation carried into Phase 2

- `weight_records.json` contains one flattened histogram per Linear.  It does
  not retain the output-channel identity of individual values, so channel
  distributions cannot be reconstructed from that file.

## Phase 2 weight-channel scope

- Reload the same BF16 model and materialize the same R1/R2 rotation.  No
  WikiText-2 data or model forward is needed for weight-only analysis.
- Treat `weight[channel_index, :]` as one channel because this is the row that
  shares one W4 per-channel scale.
- For every channel from all 112 Linear weights, store `P90(abs(weight))`,
  `P99(abs(weight))`, `max_abs`, `T = max_abs / P99`, and
  `B = P99 / P90`, together with the basic scalar statistics.
- First inspect the continuous T and B score distributions for the full model,
  each projection family, and each individual Linear.  Do not choose an
  anomaly threshold or emit normal/abnormal labels in this pass.
- Preserve each channel's paired `(T, B)` values so that isolated spikes,
  broad tails, and their combination can be distinguished with a joint plot.
- The pooled full-model view mixes 2048-element channels with 8192-element
  `down_proj` channels and is descriptive only.  Per-family and per-Linear
  views retain the shape context needed for interpretation.
- Select channels for detailed 4096-bin weight histograms only after reviewing
  the score distributions.
- Activation-channel localization, including `down_proj` input channels, is a
  separate later slice.

## Repository state

- Adding files to Git, committing, testing, and running the CUDA collector are
  outside this implementation step and require separate authorization.
