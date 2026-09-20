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
CUDA_VISIBLE_DEVICES=3,4 torchrun --nnodes=1 --nproc_per_node=2 /home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase0-dynamic-kv8-8f47aa3f/optimize_rotation.py \
--input_model "/home/dongpeiyan/projects/rotation-quant/cache/models/llama-3.2-1b-instruct" \
--output_rotation_path "/home/dongpeiyan/projects/rotation-quant/runs/phase0/p0-llama32-1b-spinquant-2gpu-ebs8-w16a8kv8-dynamic-s42-r01/rotation" \
--output_dir "/home/dongpeiyan/projects/rotation-quant/runs/phase0/p0-llama32-1b-spinquant-2gpu-ebs8-w16a8kv8-dynamic-s42-r01/trainer" \
--logging_dir "/home/dongpeiyan/projects/rotation-quant/runs/phase0/p0-llama32-1b-spinquant-2gpu-ebs8-w16a8kv8-dynamic-s42-r01/logs" \
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
--k_bits 8 \
--v_bits 8 \
--k_asym \
--v_asym \
--k_groupsize 64 \
--v_groupsize 64 \
--w_groupsize 32 \
2>&1 | tee "/home/dongpeiyan/projects/rotation-quant/runs/phase0/p0-llama32-1b-spinquant-2gpu-ebs8-w16a8kv8-dynamic-s42-r01/logs/rotation.log"
