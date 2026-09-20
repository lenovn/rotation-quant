# optimizer_scale_reference 极窄独立核验：PASS

时间：2026-09-14 02:31 +08:00。

**本轮仅执行 1 个定向 CPU 测试：1 passed，0 failed，9.06 秒，exit_code=0。** 没有重跑原 54 项，没有新 GPU 烟测，没有核验其他实验。

## 变更与检查范围

- 应用只读：`worktrees/SpinQuant-phase3-joint/experiments/phase3/run.py:34` 的 `make_optimizers`，新增读取 `args.optimizer_scale_reference` 中 `parameters` 的 scale mean 决定 Adam 绝对 LR；缺少该属性时 `getattr(..., None)` 保留旧路径。
- 被测 run.py 修改时间在执行前后均为 `2026-09-14 02:29:27.169337092 +08:00`；HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，实际包含 worktree 未提交源码，不以 HEAD 代替全部实现。
- 只新增 `tests/test_phase3_joint.py:714` 的 `test_optimizer_scale_reference_fixes_rates_without_loading_parameters`。复用既有 CPU 小模型 fixture 取得真实 96 SA、112 SW、16 SP2 模块名称/Parameter；保存测试专用 initial.pt，不读取或修改正式实验 initial.pt。

## 实际断言结果

1. 初始模型使用共同 reference 或缺省旧路径，224 个 scale `initial_lr` 完全一致。
2. 将三类全部 scale 乘以不同的 >=2 倍系数后，仍指定同一个 reference，224 个 `initial_lr` 与改变 scale 前逐项完全相同，且等于 `relative_scale_lr * reference_parameter.mean()`。
3. 同样改变 scale 后，不提供该属性或者显式设为 None，两者逐项完全相同，均使用当前 Parameter 的 mean；224 项均与固定 reference 路径不同，证明旧路径未被偷偷替换。
4. 所有 Adam `lr == initial_lr`，每组仍绑定当前模型的原 Parameter 对象；全部 241 个目标 Parameter 身份保持。
5. 构建 optimizer 不修改模型状态；改变后的 scale 不会被 reference 值覆盖；reference 文件字节不变。

结论：**本变更的上述 CPU 行为 PASS，无本项新增阻塞。** 不外推为已经运行的 mainC100 与其他 GPU 任务的实际逐步 LR 日志已完成比对，也不宣称初始化收益、等预算恢复训练或完整实验已经验收。

## 可复查证据

- 日志：`cpu-lr-reference-20260914.log`。
- JUnit：`cpu-lr-reference-20260914.xml`。
- 唯一告警为现有 Transformers deprecated API，不影响断言。

```bash
cd /home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_phase3_joint.py::test_optimizer_scale_reference_fixes_rates_without_loading_parameters \
  -q --basetemp=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/pytest-lr-reference-20260914 \
  --junitxml=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-lr-reference-20260914.xml
```

命令用 `set -o pipefail` 与 tee 留存 stdout/stderr。verifier 未改应用源码、环境、正式实验产物；没有 GPU 计算或进程操作。此次 targeted 核验结束，后续仅按新变更的具体风险复用本 verifier。
