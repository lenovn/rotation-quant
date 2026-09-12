#!/usr/bin/env bash
# Requires --w4_aware_training and learnable-SA initialization support.
# A train -> A GPTQ/A8 -> A16 reload -> B train -> B GPTQ/A8 -> A16 reload.
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd)/paths.sh"
INITIAL_DIR=${INITIAL_DIR:-${PROJECT_ROOT}/runs/phase2/w16a8-joint-r-sa-r12-s42/rotation}
GPU_IDS=${GPU_IDS:-0,1}
GPU_ID=${GPU_ID:-0}
RUN_PREFIX=${RUN_PREFIX:-w4aware-ab-20260909}
case "${RUN_PREFIX}" in
  ''|*[!a-zA-Z0-9._-]*) echo 'RUN_PREFIX must be a plain directory name' >&2; exit 2 ;;
esac

# A new parent guarantees no checkpoint/log from an earlier run is overwritten.
RUN_DIR=$(mktemp -d "${PROJECT_ROOT}/runs/phase2/${RUN_PREFIX}.XXXXXX")
echo "Experiment directory: ${RUN_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

train_common=(
  --input_model "${MODEL_PATH}"
  --optimized_rotation_path "${INITIAL_DIR}/R.bin"
  --static_scale_path "${INITIAL_DIR}/quant_scales.pt"
  --model_max_length 2048 --fp16 False --bf16 True
  --log_on_each_node False --per_device_train_batch_size 1
  --gradient_accumulation_steps 4 --logging_steps 1
  --learning_rate 1.5 --activation_scale_learning_rate 1.0
  --weight_decay 0.0 --lr_scheduler_type cosine --warmup_steps 10
  --gradient_checkpointing True --save_strategy no --max_steps 100
  --seed 42 --data_seed 42 --report_to none
  --rotation_components r1_r2 --a_quant_mode rotation_static
  --a_bits 8 --a_clip_ratio 1.0 --learn_activation_scales
  --w_groupsize -1 --no-w_asym --no-w_clip
  --k_bits 16 --v_bits 16
  --calibration_nsamples 32 --calibration_seqlen 128 --calibration_seed 42
)
ptq_common=(
  --input_model "${MODEL_PATH}" --model_max_length 2048
  --per_device_eval_batch_size 1 --fp16 False --bf16 True
  --rotate --rotation_components r1_r2 --a_quant_mode rotation_static
  --a_bits 8 --a_clip_ratio 1.0 --static_down_proj_fp16
  --w_bits 4 --w_groupsize -1 --no-w_asym --no-w_clip --no-w_rtn
  --no-act_order --percdamp 0.01 --nsamples 128 --seed 42
  --k_bits 16 --v_bits 16 --eval_nsamples 8 --report_to none
)

for arm in A B; do
  arm_dir=${RUN_DIR}/${arm}
  mkdir "${arm_dir}"
  mkdir "${arm_dir}/rotation" "${arm_dir}/trainer" "${arm_dir}/logs" \
    "${arm_dir}/gptq" "${arm_dir}/eval-w4a16"
  train_precision=(--w_bits 16)
  if [[ ${arm} == B ]]; then
    train_precision=(--w_bits 4 --w4_aware_training)
  fi
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=2 \
    "${SPINQUANT_ROOT}/optimize_rotation.py" \
    "${train_common[@]}" "${train_precision[@]}" \
    --output_rotation_path "${arm_dir}/rotation" \
    --output_dir "${arm_dir}/trainer" --logging_dir "${arm_dir}/logs" \
    2>&1 | tee "${arm_dir}/logs/rotation.log"

  for mode in a8 a16; do
    eval_dir=${arm_dir}/gptq
    artifact_args=(--save_qmodel_path "${arm_dir}/gptq/w4_gptq_model.pt")
    if [[ ${mode} == a16 ]]; then
      eval_dir=${arm_dir}/eval-w4a16
      artifact_args=(--load_qmodel_path "${arm_dir}/gptq/w4_gptq_model.pt")
    fi
    CUDA_VISIBLE_DEVICES=${GPU_ID} "${TORCHRUN_BIN}" \
      --standalone --nnodes=1 --nproc_per_node=1 \
      "${PROJECT_ROOT}/scripts/phase2/w4aware_ab_ptq.py" \
      --ab-eval-mode "${mode}" --ab-result-path "${eval_dir}/result.json" \
      --ab-log-path "${eval_dir}/wikitext2_8x2048.log" \
      "${ptq_common[@]}" "${artifact_args[@]}" \
      --optimized_rotation_path "${arm_dir}/rotation/R.bin" \
      --static_scale_path "${arm_dir}/rotation/quant_scales.pt" \
      --output_dir "${eval_dir}" \
      2>&1 | tee "${eval_dir}/wikitext2_8x2048.log"
  done
done

"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for arm in ("A", "B"):
    for directory in ("gptq", "eval-w4a16"):
        result_path = root / arm / directory / "result.json"
        with result_path.open() as source:
            result = json.load(source)
        rows.append({"arm": arm, "training_w_bits": 16 if arm == "A" else 4,
                     "training_log": str(root / arm / "logs/rotation.log"),
                     "result_path": str(result_path), **result})
with (root / "summary.json").open("x") as output:
    json.dump(rows, output, indent=2, allow_nan=False)
    output.write("\n")
print(f"Summary: {root / 'summary.json'}")
PY
