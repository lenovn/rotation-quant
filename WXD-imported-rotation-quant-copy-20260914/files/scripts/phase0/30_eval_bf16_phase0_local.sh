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

export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/dongpeiyan/projects/rotation-quant/cache/huggingface
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=5 torchrun --nnodes=1 --nproc_per_node=1 /home/dongpeiyan/projects/rotation-quant/repos/SpinQuant/ptq.py \
--input_model "/home/dongpeiyan/projects/rotation-quant/cache/models/llama-3.2-1b-instruct" \
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
2>&1 | tee "/home/dongpeiyan/projects/rotation-quant/runs/phase0/p0-llama32-1b-spinquant-2gpu-ebs8-w16a8kv16-s42-r01/logs/eval_bf16_w16a16kv16.log"
