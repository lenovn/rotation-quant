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
GPU_ID=${GPU_ID:-0}
RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}/runs/phase0/p0-llama32-1b-spinquant-2gpu-ebs8-w16a8kv16-s42-r01}
mkdir -p "${RUN_DIR}/logs"
CUDA_VISIBLE_DEVICES=${GPU_ID} "${TORCHRUN_BIN}" --nnodes=1 --nproc_per_node=1 "${SPINQUANT_ROOT}/ptq.py" \
--input_model "${MODEL_PATH}" \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits 16 \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--seed 42 \
  2>&1 | tee "${RUN_DIR}/logs/eval_bf16_w16a16kv16.log"
