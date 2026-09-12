#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/paths.sh"
A_DIR=${A_DIR:-${PROJECT_ROOT}/runs/phase2/w4aware-ab-20260909.6jhGG9/A}
RUN_DIR=$(mktemp -d "${PROJECT_ROOT}/runs/phase2/a-rtn-supplement-20260909.XXXXXX")
echo "Run directory: ${RUN_DIR}"
cp "${BASH_SOURCE[0]}" "${RUN_DIR}/launcher.sh"
cp "${PROJECT_ROOT}/scripts/phase2/a_rtn_eval.py" "${RUN_DIR}/a_rtn_eval.py"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
common=(--input_model "${PROJECT_ROOT}/cache/models/llama-3.2-1b-instruct"
 --model_max_length 2048 --per_device_eval_batch_size 1 --fp16 False --bf16 True
 --rotate --rotation_components r1_r2 --a_quant_mode rotation_static
 --a_bits 8 --a_clip_ratio 1.0 --static_down_proj_fp16
 --w_bits 4 --w_groupsize -1 --no-w_asym --no-w_clip --w_rtn
 --no-act_order --percdamp 0.01 --nsamples 128 --seed 42
 --k_bits 16 --v_bits 16 --eval_nsamples 8 --report_to none
 --optimized_rotation_path "${A_DIR}/rotation/R.bin"
 --static_scale_path "${A_DIR}/rotation/quant_scales.pt")
for mode in a8 a16; do
 eval_dir=${RUN_DIR}/${mode}
 mkdir "${eval_dir}"
 entry=("${PROJECT_ROOT}/scripts/phase2/a_rtn_eval.py" --a-rtn-scales-output "${RUN_DIR}/quant_scales.pt")
 artifact=(--save_qmodel_path "${RUN_DIR}/w4_rtn_model.pt")
 if [[ ${mode} == a16 ]]; then
  entry=("${PROJECT_ROOT}/scripts/phase2/w4aware_ab_ptq.py")
  artifact=(--load_qmodel_path "${RUN_DIR}/w4_rtn_model.pt" --static_scale_path "${RUN_DIR}/quant_scales.pt")
 fi
 "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 \
  "${entry[@]}" --ab-eval-mode "${mode}" --ab-result-path "${eval_dir}/result.json" \
  --ab-log-path "${eval_dir}/wikitext2_8x2048.log" "${common[@]}" "${artifact[@]}" \
  --output_dir "${eval_dir}" > "${eval_dir}/wikitext2_8x2048.log" 2>&1
done
