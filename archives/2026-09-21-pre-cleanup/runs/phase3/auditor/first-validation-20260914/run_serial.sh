set -u
cd /home/dongpeiyan/projects/rotation-quant
audit=runs/phase3/auditor/first-validation-20260914
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GIT_OPTIONAL_LOCKS=0
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint
export TMPDIR=/home/dongpeiyan/projects/rotation-quant/$audit/tmp
modes=("$@")
if [ ${#modes[@]} -eq 0 ]; then
    modes=(bf16 c-adam10)
fi
for mode in "${modes[@]}"; do
    date -Ins > "$audit/$mode.gpu-before.txt"
    nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu --format=csv >> "$audit/$mode.gpu-before.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >> "$audit/$mode.gpu-before.txt"
    printf '%s\n' "CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python $audit/replicate.py $mode" >> "$audit/commands.log"
    /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python "$audit/replicate.py" "$mode" > "$audit/$mode.log" 2>&1 &
    child=$!
    printf '%s\n' "$child" > "$audit/$mode.pid"
    printf '%s GPU1 PID%s\n' "$mode" "$child" >> "$audit/MAIN_THREAD_NOTICE.txt"
    wait "$child"
    result=$?
    printf '%s\n' "$result" > "$audit/$mode.exit"
done
date -Ins > "$audit/serial.completed"
