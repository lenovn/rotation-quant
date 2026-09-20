# FIRON capability evaluation

Run from `/home/dongpeiyan/projects/rotation-quant`. The existing experiment is
`runs/capability-eval-20260920`; its `PROTOCOL.md`, `models.json`, and per-task
`settings.json` specify the models and scoring protocol. This entrypoint loads
existing packages without calibration or training.

The interpreter is `runs/phase5/env/bin/python`. Additional harness dependencies
are isolated under the result directory's `deps`, which `run.py` adds to its
Python import path. `environment.txt` records the combined environment. Cached
datasets are under `cache/huggingface`; IFEval's NLTK resources are under the
result directory's `nltk_data`. Do not install dependencies into the base
environment to reuse this run.

## Resume or run a subset

The scheduler supplies dataset/NLTK environment variables and skips completed
results. Choose a GPU according to current resource availability; `0` below is
only an example. It does not reserve a GPU or stop other processes.

```bash
runs/phase5/env/bin/python scripts/capability_eval/schedule.py \
  --jobs qwen:firon:0 --tasks gsm8k_cot ifeval
```

Omit `--tasks` for all eleven capability tasks. Supported models are `qwen` and
`llama`; methods are `bf16`, `firon`, `parent`, plus the existing `llama:w4a16`.
There is no registered Qwen W4A16 package. Generation batches are fixed in
`run.py`: GSM8K 16, Llama IFEval 16, Qwen IFEval 64. The `--batch-size` argument
to `run.py` controls scoring batches only. Each task has its own process and
records its command, PID, GPU, selected package, and effective settings.

For a separate single-task output without modifying completed results:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME="$PWD/cache/huggingface" \
NLTK_DATA="$PWD/runs/capability-eval-20260920/nltk_data" \
runs/phase5/env/bin/python scripts/capability_eval/run.py \
  --model qwen --method firon --task ifeval \
  --output runs/capability-eval-separate/qwen/firon/ifeval
```

The registry and dependency locations remain those of the existing experiment.
Do not use the separate-output example to replace the paper results selectively.

## Regenerate tables without GPU inference

```bash
runs/phase5/env/bin/python scripts/capability_eval/summarize.py
runs/phase5/env/bin/python scripts/capability_eval/generation_summary.py \
  > runs/capability-eval-20260920/generation_lengths.json
runs/phase5/env/bin/python scripts/capability_eval/report.py
```

`main_results.csv` and `main_results.tex` show GSM8K strict / flexible extraction
uniformly. `main_results_numeric.csv` retains the declared strict primary metric.
All generation metrics and BF16 differences are in `generation_all_metrics.csv`.
`paper_tables.tex` provides captions and includes the main and appendix tables;
include it from the result directory with LaTeX `booktabs` and `graphicx` loaded.

Failed attempts are retained. A successful task's `settings.json` identifies the
raw generation file to use; do not concatenate old attempts. `quantization.json`
records actual prefill/decode coverage and unchanged quantizer states.
