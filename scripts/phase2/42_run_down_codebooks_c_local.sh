#!/usr/bin/env bash
# Fixed C RTN W4 and C non-down SA; only down activation grids change.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/paths.sh"
C_DIR=${C_DIR:-${PROJECT_ROOT}/runs/phase2/learned-sw-c-20260909.ByFYAM/C}
RUN_DIR=$(mktemp -d "${PROJECT_ROOT}/runs/phase2/down-codebooks-c-20260909.XXXXXX")
echo "Experiment directory: ${RUN_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export OMP_NUM_THREADS=4
"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 \
  "${PROJECT_ROOT}/scripts/phase2/down_codebook_experiment.py" \
  --down-output "${RUN_DIR}/results" --down-weight-method rtn \
  --down-calibration-windows 32 --down-local-windows 8 --down-rows-per-window 64 \
  --down-coarse-steps 33 --down-refine-steps 17 \
  --input_model "${PROJECT_ROOT}/cache/models/llama-3.2-1b-instruct" \
  --load_qmodel_path "${C_DIR}/rtn/w4_rtn_model.pt" \
  --optimized_rotation_path "${C_DIR}/rotation/R.bin" \
  --static_scale_path "${C_DIR}/rotation/quant_scales.pt" \
  --model_max_length 2048 --per_device_eval_batch_size 1 --fp16 False --bf16 True \
  --rotate --rotation_components r1_r2 --a_quant_mode rotation_static \
  --a_bits 8 --a_clip_ratio 1.0 --static_down_proj_fp16 \
  --w_bits 4 --w_groupsize -1 --no-w_asym --no-w_clip --no-w_rtn \
  --no-act_order --percdamp 0.01 --nsamples 128 --seed 42 \
  --k_bits 16 --v_bits 16 --eval_nsamples 8 --report_to none \
  --output_dir "${RUN_DIR}/ptq" 2>&1 | tee "${RUN_DIR}/experiment.log"
