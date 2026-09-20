#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/paths.sh"
C_DIR=${C_DIR:-${PROJECT_ROOT}/runs/phase2/learned-sw-c-20260909.ByFYAM/C}
DOWN_SCALES_PATH=${DOWN_SCALES_PATH:-${PROJECT_ROOT}/runs/phase2/down-codebooks-c-20260909.6YRIty/results/down_scales.json}
RUN_DIR=$(mktemp -d "${PROJECT_ROOT}/runs/phase2/validation-diagnostics-c-20260913.XXXXXX")
echo "Diagnostics directory: ${RUN_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export OMP_NUM_THREADS=4
for mode in w4a16 w4a8_down_a16 w4a8; do
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 \
    "${PROJECT_ROOT}/scripts/phase2/validation_acceptance.py" \
    --accept-mode "$mode" --accept-output "${RUN_DIR}/${mode}" \
    --input_model "${MODEL_PATH}" --model_max_length 2048 \
    --per_device_eval_batch_size 1 --fp16 False --bf16 True \
    --k_bits 16 --v_bits 16 --seed 42 --report_to none \
    --output_dir "${RUN_DIR}/ptq-${mode}" \
    --w_bits 4 --a_bits 8 --w_groupsize -1 --no-w_asym --no-w_clip --no-w_rtn \
    --load_qmodel_path "${C_DIR}/rtn/w4_rtn_model.pt" \
    --optimized_rotation_path "${C_DIR}/rotation/R.bin" \
    --static_scale_path "${C_DIR}/rotation/quant_scales.pt" \
    --rotate --rotation_components r1_r2 --a_quant_mode rotation_static \
    --a_clip_ratio 1.0 --static_down_proj_fp16 \
    --accept-down-scales "${DOWN_SCALES_PATH}" \
    2>&1 | tee "${RUN_DIR}/${mode}.log"
done
"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
modes = ('w4a16', 'w4a8_down_a16', 'w4a8')
results = {mode: json.loads((root / mode / 'result.json').read_text()) for mode in modes}
reference = results[modes[0]]
for result in results.values():
    for key in ('token_count', 'predicted_tokens', 'split', 'window_length', 'weight_path', 'rotation_path'):
        assert result[key] == reference[key], key
    assert [(s['seqlen'], s['windows']) for s in result['segments']] == [(s['seqlen'], s['windows']) for s in reference['segments']]
    assert result['weights_match_checkpoint'] and result['activation_scales_unchanged']
delta = {f'{a}_to_{b}': {'delta_nll': results[b]['nll'] - results[a]['nll'],
                       'delta_ppl': results[b]['ppl'] - results[a]['ppl']}
         for a, b in zip(modes, modes[1:])}
with (root / 'summary.json').open('x') as output:
    json.dump({'results': results, 'conditional_deltas': delta}, output, indent=2, allow_nan=False)
print(json.dumps({mode: {k: results[mode][k] for k in ('ppl', 'nll')} for mode in modes}, indent=2))
print(json.dumps(delta, indent=2))
PY
