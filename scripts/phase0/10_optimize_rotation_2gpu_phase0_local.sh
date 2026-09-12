#!/usr/bin/env bash
# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/paths.sh"
GPU_IDS=${GPU_IDS:-0,1}
RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}/runs/phase0/p0-llama32-1b-spinquant-2gpu-ebs8-w16a8kv16-s42-r01}
mkdir -p "${RUN_DIR}/rotation" "${RUN_DIR}/trainer" "${RUN_DIR}/logs"
"${TORCHRUN_BIN}" --nnodes=1 --nproc_per_node=2 "${SPINQUANT_ROOT}/optimize_rotation.py" \
--input_model "${MODEL_PATH}" \
--output_rotation_path "${RUN_DIR}/rotation" \
--output_dir "${RUN_DIR}/trainer" \
--logging_dir "${RUN_DIR}/logs" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--seed 42 \
--w_bits 16 \
--a_bits 8 \
--w_clip \
--a_asym \
--w_groupsize 32 \
  2>&1 | tee "${RUN_DIR}/logs/rotation.log"
