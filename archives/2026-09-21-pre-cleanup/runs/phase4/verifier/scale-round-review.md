# Phase4 scale-round 独立数值审查

日期：2026-09-15。结论：**PASS（新增及受影响核心逻辑）**。这不是正式实验结果或最终成品 auditor 验收。

审查源码：`worktrees/SpinQuant-phase4-w4/experiments/phase4/scale_round.py`、`launch.py`、`tests/test_phase4_scale_round.py`；只追溯必要的 Phase3 `common.py`、`postprocess.py`、`sequential_postprocess.py`、`quantization.py` 与 `scripts/phase2/fixed_grid_rounding.py`。未修改主实现、未启动模型 forward 或 GPU、未重跑旧测试。

## 算法与数值

- `fit_scale` 24–40 行：SW 形状为 `[out_features, 1]`。令 H=XᵀX/N，分子 `sum((qH)*Wref)`、分母 `sum((qH)*q)` 正确等价于每输出通道 `<qX,WrefX>/<qX,qX>`。零能量、非正或非有限提议沿用旧尺度。
- 闭式解对应未舍入的 q·s；实现实际比较 `BF16(q·s)-Wref` 的行级重构二次型，仅接受下降行。没有将连续尺度目标下降冒充 BF16 加载后下降。该目标仍为 FP32 输出重构误差，未声称等于后续 BF16 网络前向或 NLL。
- `generate` 从父包 unpack 后的整数码接续，不重新按 Wref/SW 初始量化；改变 SW 后复用既有 `coordinate_round`。整数语义保持 [-8,7]、per-output-channel，既有 helper 对实际 BF16 反量化差分计算邻码收益。
- `reference_weights` 经原始模型/norm fusion/B100 state 加载后调用 `rotated_weight`，没有把已反量化 W4 当作未量化参考。当前父包链的参考是 B100，后处理无 R/D/主权重变更。这里核查来源与实际 Phase3 settings，不声称重新独立加载完整模型验证。
- `capture_module_inputs` hook 位于 `wrapper.module` 的 pre-hook，输入经过当前静态 SP2；整个候选生成阶段未 apply 权重，两臂使用同一未修改父包 X。固定非 down SA、SP2、R 和其余模块。

## 对照、选择与导出

- 两臂各两段相同请求坐标预算；A2 多两次尺度拟合及解释用 MSE 计算。实际停止步数、GPU 秒数需要按正式日志解释，不能称总计算量完全相同。
- 每模块最多两个候选端点，共享 heldout MSE 预筛和 8 个 train-window NLL；完整 validation 不在选码循环内。A2 的 scale-only 保存为诊断，不增加 A2 NLL 候选数。
- `score_candidate` 临时替换并回退当前模块；跨模块前缀仅在选定后接受。切换两臂时所有目标模块恢复原父码/SW，A1 冷载后的非目标参数和激活也继承原父，不存在观察到的 A1→A2 污染。
- `save_frozen(model, dict(records, **updates), ...)` 导出实际选择的 packed q 与 SW；成熟 loader 直接 `BF16(q*SW)`，不会覆盖学习尺度或重新 RTN。正式评测对实际导出包冷载执行。
- 无接受更新时引用的 A0 NLL `2.779600150933802`、PPL `16.112577060930427` 与本轮指定父包 `validation.json` 完全一致。此硬编码只对当前研究父包有效；如以后更换 `--parent`，必须相应改为该父包的真实评测来源。
- launcher 保留实际环境、独立 socket `rotation-quant-phase4`、显式 GPU、命令/PID/log 记录及旧产物保护，未改变成熟加载/量化/评测入口。没有运行 launcher 或操作他人进程。

## 验证证据

主执行报告的新测试 2 PASS 已查看其测试内容；独立 verifier 未重复整套测试，而在相同环境补充一个直接输出空间 oracle。实际命令使用 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.../worktrees/SpinQuant-phase4-w4 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python`，仅 CPU 小张量，约 5.93 秒。

32 个固定种子，每次 q 为 `[24,16]`，X 为 `[43,16]` BF16，以直接 `X @ q.T` 和 `X @ Wref.T` 独立计算正尺度提议，再显式计算 BF16 权重的输出误差：

```text
status: PASS
seeds: 32
valid_positive_bf16_worse_proposal_rows_rejected: 164
max_actual_row_mse_increase: 0.0
coordinate_fit_mse_before: 4.481971984660049e-07
coordinate_fit_mse_after: 4.481971984660049e-07
model_forwards: 0
gpu_used: False
```

164 个“连续正尺度提议有效，但 BF16 重构变差”的输出行全部精确回退旧 SW。末个小张量接续既有 helper 保持目标不增和合法码；该例没有需要接受的邻码更新，不把它当成新一轮成熟邻码算法验证。

未发现会污染本轮数值结论的新增核心错误。最终 PTQ/NLL 收益、A1 与 A2 的实际预算差异及最终包审计仍以正式运行结果为准。
