#!/usr/bin/env bash

# Shared paths for launchers. Override the *_BIN and MODEL_PATH variables
# after activating the environment when running on another machine.
COMMON_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${COMMON_DIR}/../.." && pwd)
SPINQUANT_ROOT=${SPINQUANT_ROOT:-${PROJECT_ROOT}/repos/SpinQuant}
MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/cache/models/llama-3.2-1b-instruct}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
HF_HOME=${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}

export PROJECT_ROOT SPINQUANT_ROOT MODEL_PATH PYTHON_BIN TORCHRUN_BIN HF_HOME
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Output directories are intentionally ignored by Git but must exist before
# mktemp-based launchers create a run directory.
mkdir -p "${PROJECT_ROOT}/runs/phase0" "${PROJECT_ROOT}/runs/phase2"
