#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/paths.sh"

GPU_IDS=${GPU_IDS:-0,2}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MAX_STEPS=${MAX_STEPS:-100}
CALIBRATION_NSAMPLES=${CALIBRATION_NSAMPLES:-32}
CALIBRATION_SEQLEN=${CALIBRATION_SEQLEN:-128}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
SA_LEARNING_RATE=${SA_LEARNING_RATE:-1.0}
RUN_NAME=${RUN_NAME:-w16a8-joint-r-sa-r12-s42}
RUN_DIR=${PROJECT_ROOT}/runs/phase2/${RUN_NAME}

mkdir -p "${RUN_DIR}/rotation" "${RUN_DIR}/trainer" "${RUN_DIR}/logs"

export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=${PROJECT_ROOT}/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

"${TORCHRUN_BIN}" \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${SPINQUANT_ROOT}/optimize_rotation.py" \
  --input_model "${MODEL_PATH}" \
  --output_rotation_path "${RUN_DIR}/rotation" \
  --output_dir "${RUN_DIR}/trainer" \
  --logging_dir "${RUN_DIR}/logs" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --fp16 False \
  --bf16 True \
  --log_on_each_node False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --learning_rate 1.5 \
  --activation_scale_learning_rate "${SA_LEARNING_RATE}" \
  --weight_decay 0.0 \
  --lr_scheduler_type cosine \
  --gradient_checkpointing True \
  --save_strategy no \
  --max_steps "${MAX_STEPS}" \
  --seed 42 \
  --rotation_components r1_r2 \
  --a_quant_mode rotation_static \
  --a_bits 8 \
  --a_clip_ratio 1.0 \
  --learn_activation_scales \
  --w_bits 16 \
  --w_groupsize -1 \
  --k_bits 16 \
  --v_bits 16 \
  --calibration_nsamples "${CALIBRATION_NSAMPLES}" \
  --calibration_seqlen "${CALIBRATION_SEQLEN}" \
  --calibration_seed 42 \
  2>&1 | tee "${RUN_DIR}/logs/rotation.log"
