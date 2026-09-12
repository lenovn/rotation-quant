#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/paths.sh"

GPU_ID=${GPU_ID:-0}
RUN_NAME=${RUN_NAME:-w16a8-joint-r-sa-r12-s42}
GPTQ_NSAMPLES=${GPTQ_NSAMPLES:-128}
EVAL_NSAMPLES=${EVAL_NSAMPLES:-8}
RUN_DIR=${PROJECT_ROOT}/runs/phase2/${RUN_NAME}
ROTATION_PATH=${ROTATION_PATH:-${RUN_DIR}/rotation/R.bin}
SCALE_PATH=${SCALE_PATH:-${RUN_DIR}/rotation/quant_scales.pt}
GPTQ_DIR=${RUN_DIR}/gptq

mkdir -p "${GPTQ_DIR}"

export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=${PROJECT_ROOT}/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU_ID}

"${TORCHRUN_BIN}" \
  --nnodes=1 \
  --nproc_per_node=1 \
  "${SPINQUANT_ROOT}/ptq.py" \
  --input_model "${MODEL_PATH}" \
  --optimized_rotation_path "${ROTATION_PATH}" \
  --static_scale_path "${SCALE_PATH}" \
  --output_dir "${GPTQ_DIR}" \
  --save_qmodel_path "${GPTQ_DIR}/w4_gptq_model.pt" \
  --model_max_length 2048 \
  --per_device_eval_batch_size 1 \
  --bf16 True \
  --rotate \
  --rotation_components r1_r2 \
  --a_quant_mode rotation_static \
  --a_bits 8 \
  --a_clip_ratio 1.0 \
  --static_down_proj_fp16 \
  --w_bits 4 \
  --w_groupsize -1 \
  --nsamples "${GPTQ_NSAMPLES}" \
  --k_bits 16 \
  --v_bits 16 \
  --eval_nsamples "${EVAL_NSAMPLES}" \
  2>&1 | tee "${GPTQ_DIR}/wikitext2_${EVAL_NSAMPLES}x2048.log"
