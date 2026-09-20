set -eu
cd /home/dongpeiyan/projects/rotation-quant
audit=runs/phase3/auditor/final-best-20260914
python=/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python
printf '%s\n' "wrapper PID=$$ GPU=$CUDA_VISIBLE_DEVICES" >> "$audit/MAIN_THREAD_NOTICE.txt"
for mode in bf16 final; do
    date -Ins > "$audit/$mode.gpu-before.txt"
    nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu --format=csv >> "$audit/$mode.gpu-before.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >> "$audit/$mode.gpu-before.txt"
    printf '%s\n' "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $python -u $audit/replicate.py $mode" >> "$audit/commands.log"
    "$python" -u "$audit/replicate.py" "$mode" > "$audit/$mode.log" 2>&1 &
    child=$!
    printf '%s\n' "$child" > "$audit/$mode.pid"
    printf '%s GPU%s PID%s\n' "$mode" "$CUDA_VISIBLE_DEVICES" "$child" >> "$audit/MAIN_THREAD_NOTICE.txt"
    set +e
    wait "$child"
    result=$?
    set -e
    printf '%s\n' "$result" > "$audit/$mode.exit"
    if [ "$result" -ne 0 ]; then
        printf '%s exit=%s\n' "$mode" "$result" > "$audit/serial.failed"
        exit "$result"
    fi
done
date -Ins > "$audit/serial.completed"
