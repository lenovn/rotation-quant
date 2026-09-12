# rotation-quant reproduction workspace

This repository contains the source code, experiment launchers, tests, and
documentation used for the rotation-quant experiments. Trained model weights,
rotation files, quantization scales, downloaded models, caches, checkpoints,
and run outputs are intentionally excluded from GitHub.

## Repository layout

- `repos/SpinQuant/`: the main SpinQuant source tree, including the current
  R/SA and W4-aware training changes.
- `repos/fast-hadamard-transform/`: the CUDA extension used by SpinQuant.
- `scripts/phase0/`: earlier baseline and dynamic-KV experiments.
- `scripts/phase2/`: the current W16A8 -> W4/GPTQ and W4-aware experiment
  launchers.
- `docs/`: experiment decisions and the final validation report.

The nested source trees are included as ordinary source files in this
repository. Their local `.git` metadata is not included.

## Environment setup

The commands below assume a Linux machine with a CUDA-enabled PyTorch install.
Use the Python and CUDA versions supported by the target machine, then install
the project dependencies:

```bash
git clone https://github.com/lenovn/rotation-quant.git
cd rotation-quant
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r repos/SpinQuant/requirement.txt
python -m pip install -v -e repos/fast-hadamard-transform
export PYTHONPATH="$PWD/repos/SpinQuant:${PYTHONPATH:-}"
```

Place a local Llama-3.2-1B-Instruct model at
`cache/models/llama-3.2-1b-instruct`, or point the launchers at another
location:

```bash
export MODEL_PATH=/path/to/llama-3.2-1b-instruct
export PYTHON_BIN=python
export TORCHRUN_BIN=torchrun
```

The WikiText-2 dataset is loaded through the Hugging Face datasets interface
when an experiment is run. Dataset/model downloads and generated outputs stay
under ignored local directories.

## Current Phase 2 reproduction order

The mainline uses one `RUN_NAME` for all three stages. Adjust GPU variables for
the target machine; the defaults are two GPUs for training and GPU 0 for
single-GPU evaluation.

```bash
export GPU_IDS=0,1
export NPROC_PER_NODE=2
export GPU_ID=0
export RUN_NAME=w16a8-joint-r-sa-r12-s42

bash scripts/phase2/30_optimize_joint_r_sa_w16a8_local.sh
bash scripts/phase2/32_eval_w16a8_downa16_local.sh
bash scripts/phase2/31_gptq_w4a8_static_local.sh
```

The scripts write rotations, scales, checkpoints, GPTQ models, logs, and PPL
results under `runs/phase2/`. Those files are required as local intermediate
artifacts for later stages but are not part of this repository submission.

The W4-aware comparison and final C/SP2 validation launchers are also in
`scripts/phase2/`. They require the outputs of preceding stages; set their
`INITIAL_DIR`, `B_DIR`, `C_DIR`, and `DOWN_SCALES_PATH` variables when using
different run names or locally regenerated artifacts.

## Reproducibility boundary

The final acceptance report in `docs/final_acceptance_w4a8_ppl17.md` records the
matched validation protocol and its PPL result. The reported measurement is a
BF16 fake-quant prefill evaluation; it is not a native phone-NPU, decode, KV
cache, or latency result. Re-running the code on another machine requires the
model, dataset access, CUDA/PyTorch environment, and locally generated
intermediate artifacts.
