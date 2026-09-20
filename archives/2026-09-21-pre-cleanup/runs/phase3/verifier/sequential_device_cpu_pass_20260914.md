# Sequential 冷载设备修正：限定 CPU / 源码 PASS

日期：2026-09-14。角色：独立 verifier；应用代码只读，未使用 GPU。

## 结论与范围

- **PASS（限定源码与 CPU 设备边界 mock）**：`sequential_postprocess.py:193` 初始父包、`:233` 最终输出包均调用 `load_static(path)`，不再强制 CPU。`postprocess.py:26` 的实际默认值为 `device="cuda"`，`:36` 使用 `.to(device)`，没有附加 dtype 转换。
- **2 passed，13.13 秒**；新增测试用例 **0**，仅更新并复跑既有 driver 的 round / SP2 两个参数分支，没有重跑此前 7 项或旧 suite。
- **NOT TESTED**：真实 CUDA 上 RoPE / position_ids 同设备、正式 selection / full validation 成功、GPU 精度与显存。此次结论不能替代下一次正式新名 GPU 运行，也不能把先前失败日志改称成功。

## 实际断言

测试：`tests/test_phase3_joint.py::test_sequential_driver_prefix_resume_data_cold_validation_and_provenance`，参数为 `round`、`sp2`。

1. 从真实 `load_static` 签名验证默认设备是 CUDA；driver loader spy 要求不传额外位置参数或关键字参数，因此显式 CPU 路径会失败。
2. 使用真实 tiny EvaluationModel、112 个实际量化 wrapper 与真实冻结包 reload。仅本测试将本地构建模型的参数设为 BF16，保留初始化时 FP32 `inv_freq`；并记录全部 RoPE buffer。这里模拟的是混合 dtype 状态，不声称重新验证生产 `from_pretrained` 文件加载。
3. 在 `Module.to` 边界记录真实请求的 `("cuda",)`，要求没有 dtype 等附加参数，然后映射到 CPU。每个分支的初始父包、恢复时父包、最终冷载包均记录到 CUDA 请求；全部 RoPE buffer 保持 FP32 且数值逐位相同，最终 validation mock 入口也断言顶层 RoPE 为 FP32。
4. 保留原 driver 测试的接受前缀、异常后恢复、完成目标不重做、训练选择与 validation 隔离、固定 SW / non-down SA / HP 以及 parent metadata 传播断言。round 的 PTQ 和 SP2 的 QAT provenance 均继续通过。

这验证了两行修正选用正确默认设备、未引入 RoPE BF16 强转；CPU mock 不验证 CUDA 内核或实际设备迁移效果。此前 sequential 的 CPU PASS 报告已将 GPU 设备迁移列为 NOT TESTED，保留原报告与失败历史，不回填 GPU 验收。

## 执行证据

在 `worktrees/SpinQuant-phase3-joint` 执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
tests/test_phase3_joint.py::test_sequential_driver_prefix_resume_data_cold_validation_and_provenance -q \
--basetemp=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/pytest-sequential-device-20260914 \
--junitxml=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-sequential-device-20260914.xml
```

- 日志：`runs/phase3/verifier/cpu-sequential-device-20260914.log`
- XML：`runs/phase3/verifier/cpu-sequential-device-20260914.xml`
- 结果：`2 passed, 2 warnings in 13.13s`；两条警告为既有 transformers 弃用提示。
- `GIT_OPTIONAL_LOCKS=0 git diff --check -- tests/test_phase3_joint.py`：退出码 0。
- 本次只修改上述既有测试与 verifier 证据；未修改应用、执行 GPU、安装或操作训练进程。
