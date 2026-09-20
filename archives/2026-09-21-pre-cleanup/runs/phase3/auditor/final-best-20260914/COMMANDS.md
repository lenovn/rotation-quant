# 最终审计实际命令与执行边界

GPU launcher（经 require_escalated；首次沙箱 NVML 查询不可见，不是评测失败）：

```bash
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python runs/phase3/auditor/final-best-20260914/launch.py
```

实际 tmux socket `rotation-quant-phase3`，独立 session `auditor-final-best-20260914`，wrapper PID3561101；完整 `exec env ... bash run_serial.sh`、环境和预启动全设备/进程快照在 `launch.json`。新建的是独立审计 session，不是新的训练任务或新 auditor。既存同 socket server 当时未运行，由已安装 tmux 正常启动，无安装/环境修改。

串行运行的两条实际 GPU 命令（全部环境见 launch.json，不仅 CUDA_VISIBLE_DEVICES）：

```bash
CUDA_VISIBLE_DEVICES=1 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -u runs/phase3/auditor/final-best-20260914/replicate.py bf16
CUDA_VISIBLE_DEVICES=1 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -u runs/phase3/auditor/final-best-20260914/replicate.py final
```

BF16 PID3561110，final PID3563303。各自启动前再次读取GPU和计算进程余量，见 `bf16.gpu-before.txt`、`final.gpu-before.txt`。`*.runtime.json` 的命令来自真实 `/proc/self/cmdline`；`*.modules.json` 保存实际导入源码路径。每次 stdout/stderr、PID、exit code独立保存；串行完成标志 `serial.completed`。tmux shell在两次正常结束后自然退出，不需要杀进程；日志与结果持久保留。

只读 CPU 证据扫描与小 JSON 汇总（无模型 forward；不是 pytest）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 GIT_OPTIONAL_LOCKS=0 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -u runs/phase3/auditor/final-best-20260914/audit_records.py > runs/phase3/auditor/final-best-20260914/records.log 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python runs/phase3/auditor/final-best-20260914/finalize.py > runs/phase3/auditor/final-best-20260914/finalize.log 2>&1
```

`audit_records.py` 仅复用上次 auditor 的纯CPU读取/比较辅助函数，不执行其main。它只mmap读取本次final/control保存数据及父链元数据，完整检查最终码/尺度对应；不实例化训练模型，不调用应用算法。`finalize.py` 只处理JSON/log，明确保留400步两个64点抽样不变的观察，不伪写所有权重每步抽样均变。

本次只产生这两次新完整GPU评测，无短烟测、无中间checkpoint GPU复跑、无训练/校准/搜索。原BF16及C10首审的方法沿用已纠正版本，不重新尝试旧的整体 `.to(bfloat16)` 构造。审计自己的所有写入限定本目录，应用、tests、Phase2和主协调文档不改。
