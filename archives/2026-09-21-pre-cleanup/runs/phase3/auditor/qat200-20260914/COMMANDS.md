# 本次 auditor 实际 CPU 命令

本目录脚本只导入标准库和 torch，CPU mmap 读取既有 tensor；不导入应用训练/评测模块、不创建模型、不 forward、不校准、不 pytest、不启动 GPU。以下第一条已实际成功执行，日志为 `audit.log`；第二条为小证据整理与父 D 张量定位，日志为 `finalize.log`。它们不是训练命令，也不是新的完整 PPL 测量。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 GIT_OPTIONAL_LOCKS=0 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -u runs/phase3/auditor/qat200-20260914/audit_evidence.py > runs/phase3/auditor/qat200-20260914/audit.log 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 GIT_OPTIONAL_LOCKS=0 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -u runs/phase3/auditor/qat200-20260914/finalize_evidence.py > runs/phase3/auditor/qat200-20260914/finalize.log 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 GIT_OPTIONAL_LOCKS=0 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -u runs/phase3/auditor/qat200-20260914/finalize_evidence.py > runs/phase3/auditor/qat200-20260914/finalize-recheck.log 2>&1
```

主执行历史训练完整 argv 另存 `evidence/actual_training_commands.txt`，各原 `*.launch.json` 中含 shell_command、tmux、实际 PID 和环境。本 auditor 没有执行这些训练命令。主执行恢复评测的完整 `python -c` 代码保存在参考组原 `.log` 和本目录证据副本的 `recovery_evaluation.json` 中，也没有由本 auditor 执行。

第三条只修正审计 AST 比较的分类：`main` 新增已声明 reference CLI 参数，不属于“loss/optimizer/eval 定义完全相同”的比较集合；仍完整记录在变更函数列表。保留最初 `finalize.log`，最终机器结论看 `findings.json` 和 `finalize-recheck.log`。另一次早期 CPU 元数据读取误用 recovery JSON 所在目录，纠正路径后才完成正式证据脚本；未触发 GPU/forward。

源码核查命令为 `git -C worktrees/SpinQuant-phase3-joint rev-parse HEAD`、`branch --show-current`、`status --short`、`diff --binary`，均设置 `GIT_OPTIONAL_LOCKS=0`。原始输出分别保存到 `evidence/head.txt`、`branch.txt`、`status.txt`、`dirty.patch`。无新 hash、环境变更或原结果覆写。
