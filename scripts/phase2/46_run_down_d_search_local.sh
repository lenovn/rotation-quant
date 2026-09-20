#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_BIN=/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME="${PROJECT_ROOT}/cache/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${PROJECT_ROOT}/repos/SpinQuant"
PRIOR_GPU_SECONDS=${PRIOR_GPU_SECONDS:-0}
GPU=$(/usr/bin/nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F, '$2/$3 < 0.35 {if (!found || $2/$3 < best) {gpu=$1;best=$2/$3;found=1}} END {if(found) print gpu; else exit 1}')
export CUDA_VISIBLE_DEVICES="$GPU"
RUN_DIR=$(mktemp -d "${PROJECT_ROOT}/runs/phase2/down-d-search-20260913.XXXXXX")
printf 'GPU=%s RUN_DIR=%s LAUNCHER_PID=%s\n' "$GPU" "$RUN_DIR" "$$"
/usr/bin/nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv > "${RUN_DIR}/gpu_before.csv"
EXTRA_ARGS=()
if [[ -n ${FOLLOWUP_PARENT:-} ]]; then EXTRA_ARGS+=(--followup-parent "$FOLLOWUP_PARENT"); fi
if [[ -n ${MIXED_LAYER:-} ]]; then EXTRA_ARGS+=(--mixed-layer "$MIXED_LAYER"); fi
if [[ -n ${PRECISION_LOOP:-} ]]; then EXTRA_ARGS+=(--precision-loop "$PRECISION_LOOP"); fi
if [[ -n ${LOOP_PARENT:-} ]]; then EXTRA_ARGS+=(--loop-parent "$LOOP_PARENT"); fi
printf '%q ' "${ENV_BIN}/python" -u "${PROJECT_ROOT}/scripts/phase2/down_d_search.py" --output "$RUN_DIR" --prior-gpu-seconds "$PRIOR_GPU_SECONDS" "${EXTRA_ARGS[@]}" > "${RUN_DIR}/command.txt"
"${ENV_BIN}/python" -u "${PROJECT_ROOT}/scripts/phase2/down_d_search.py" --output "$RUN_DIR" --prior-gpu-seconds "$PRIOR_GPU_SECONDS" "${EXTRA_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/experiment.log"
