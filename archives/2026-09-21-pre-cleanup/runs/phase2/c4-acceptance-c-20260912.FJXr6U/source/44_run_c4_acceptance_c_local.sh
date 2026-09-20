#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/paths.sh"
C_DIR=${C_DIR:-${PROJECT_ROOT}/runs/phase2/learned-sw-c-20260909.ByFYAM/C}
DOWN_SCALES_PATH=${DOWN_SCALES_PATH:-${PROJECT_ROOT}/runs/phase2/down-codebooks-c-20260909.6YRIty/results/down_scales.json}
RUN_DIR=$(mktemp -d "${PROJECT_ROOT}/runs/phase2/c4-acceptance-c-20260912.XXXXXX")
echo "C4 acceptance directory: ${RUN_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export OMP_NUM_THREADS=4

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/phase2/prepare_c4_evaluation.py" \
  --output-dir "${RUN_DIR}/data" --model-path "${MODEL_PATH}" \
  --cache-dir "${HF_HOME}/datasets" --windows 1024 --seqlen 2048 --seed 42 \
  2>&1 | tee "${RUN_DIR}/prepare.log"

for mode in w16a16 w4a8; do
  extra=(--w_bits 16 --a_bits 16)
  if [[ $mode == w4a8 ]]; then
    extra=(--w_bits 4 --a_bits 8 --w_groupsize -1 --no-w_asym --no-w_clip --no-w_rtn
      --load_qmodel_path "${C_DIR}/rtn/w4_rtn_model.pt"
      --optimized_rotation_path "${C_DIR}/rotation/R.bin"
      --static_scale_path "${C_DIR}/rotation/quant_scales.pt"
      --rotate --rotation_components r1_r2 --a_quant_mode rotation_static
      --a_clip_ratio 1.0 --static_down_proj_fp16
      --accept-down-scales "${DOWN_SCALES_PATH}")
  fi
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 \
    "${PROJECT_ROOT}/scripts/phase2/validation_acceptance.py" \
    --accept-mode "$mode" --accept-output "${RUN_DIR}/${mode}" \
    --accept-token-file "${RUN_DIR}/data/input_tokens.pt" --accept-chunk-windows 128 \
    --input_model "${MODEL_PATH}" --model_max_length 2048 \
    --per_device_eval_batch_size 1 --fp16 False --bf16 True \
    --k_bits 16 --v_bits 16 --seed 42 --report_to none \
    --output_dir "${RUN_DIR}/ptq-${mode}" "${extra[@]}" \
    2>&1 | tee "${RUN_DIR}/${mode}.log"
done

"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
b, q = [json.loads((root / mode / 'result.json').read_text()) for mode in ('w16a16', 'w4a8')]
for key in ('token_count', 'predicted_tokens', 'dataset', 'subset', 'split',
            'window_length', 'input_token_path', 'input_metadata'):
    assert b[key] == q[key], key
assert [(s['start_token'], s['seqlen'], s['windows']) for s in b['segments']] == [
    (s['start_token'], s['seqlen'], s['windows']) for s in q['segments']]
result = dict(w16a16=b, w4a8=q, delta_ppl=q['ppl'] - b['ppl'],
              relative_ppl_increase_percent=(q['ppl'] / b['ppl'] - 1) * 100,
              delta_nll=q['nll'] - b['nll'])
with (root / 'summary.json').open('x') as output:
    json.dump(result, output, indent=2, allow_nan=False)
print(json.dumps({k: v for k, v in result.items() if k not in ('w16a16', 'w4a8')}, indent=2))
PY
