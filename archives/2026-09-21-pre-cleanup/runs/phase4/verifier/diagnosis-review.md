# Phase4 conditional restoration verifier

2026-09-15；独立 verifier；结论：**PASS（当前新增代码与指定 Phase3 PTQ 父包）**。

审查工作树：`worktrees/SpinQuant-phase4-w4`。范围为新 `experiments/phase4/diagnose.py`、`launch.py` 的 diagnose 入口，以及直接调用的成熟加载、参考构建和评测函数。没有修改实验源码，没有启动模型 CPU forward 或独立 GPU 复评；正式测量可并行执行。

| 核心问题 | 核查结果 |
|---|---|
| 每个 case 的权重起点 | `configure` 首先调用 `apply_records(model, records)`，所有父包矩阵由原 packed signed INT4 × 原 `[out,1]` SW 重建为模块 dtype，再覆盖指定 BF16 参考权重；恢复不累积。 |
| 激活实际旁路 | 逐个设置 backbone wrapper 的 `quantizer.bits`；all16 为全部 16，down16 仅 down 为 16，static 全部 8。`ActQuantWrapper.forward` 在 bits<16 才执行量化。输出量化器均为 16。SA/SP2 尺度未重估或写入。 |
| 参考来源及坐标 | `reference_weights` 使用原始 BF16 模型、norm fusion 和 B100 state 的 R1/R2；直接调用 `rotated_weight`，没有调用权重量化 forward。与 `frozen_model` 初始导出采用同一旋转和 dtype 顺序。当前父包来自该 B100 导出且无 D/恢复权重；既有 `inherited-sw-check.json` 的 NaN-aware 高精度边界相等记录支持父包 embedding/head/norm 沿用同一坐标。 |
| 恢复集合 | none/all/down/attention/gateup/down1/down_except1 分别为 0/112/16/64/32/1/15 个模块；attention 包含 q/k/v/o，gateup 不包含 down。 |
| 测量口径 | 直接调用现有 `full_validation`，该函数调用既有 validation_acceptance + evaluator；data_windows 固定核查 252852 输入 tokens 和 252728 targets。没有新 PPL driver、局部 PPL 平均或训练/调参。 |
| 启动及产物 | 新 task 正确映射 diagnose.py，仍用指定现有环境/CUDA_VISIBLE_DEVICES/tmux socket；输出目录 exist_ok=False，已有同名启动和日志拒绝覆盖。诊断仅输出源文件和 JSON，无包导出。 |

独立轻量测试：现有 rotation-quant-p0 Python，PYTHONDONTWRITEBYTECODE=1；创建 16 个小型 block、112 个 `ActQuantWrapper(Linear(4,2))`，使用真实 pack_int4、apply_records、SP2Quantizer、RotationStaticActQuantizer、configure。连续运行 all16:all → down16:down → static:none → all16:attention → all16:gateup → all16:down1 → all16:down_except1 → static:none。逐矩阵确认 q/SW 重置、恢复成员、input/output bits，以及激活 scale 完全不变。**PASS**。另以单个真实 down SP2 wrapper 确认 bits=16 的输出严格等于直接 BF16 linear，**PASS**。

解释边界：这些是当前父包下的条件恢复效应，不能把各组 NLL/PPL 改善相加当作唯一误差分解；旁路 SP2 后的 W4 可能保留针对 SP2 的补偿。诊断接口并不自动证明任意 --parent 与 --reference-state 坐标相同，本次 PASS 依赖指定 B100→当前 PTQ 父包来源；若换成带 D 或恢复训练的父包，必须重新判定参考来源，不能直接沿用该结论。当前没有需要阻塞正式测量的核心错误。

## 同 solver 位宽/粒度诊断增量审查

结论：**PASS**。审查新增 `rtn_reference`、`configure` 的 replacement 参数、case 第三字段和 `mlp` 选择组；未改变 launcher/evaluator。

- 原始未量化同 R 参考先移动到目标模块 GPU，再以 FP32 计算尺度、round/clip 和反量化，最终转回原 BF16 dtype。不存在从父包 W4 反量化权重再次量化的问题。
- `reshape(out, groups, width)` 仅沿输入通道连续分组；per-channel 时 width=in，group128 时 width=128。尺度为每行每组 absmax.clamp_min(1e-5)/(2^(bits-1)-1)。W4 范围 [-8,7]，W8 范围 [-128,127]，`torch.round` 为 ties-to-even；W4 per-channel 数学路径与已有 int4_codes + 原尺度初始化一致。
- 三个条件共用相同 RTN solver、相同原始参考、相同 MLP 48 个模块、相同 all-A16；没有额外校准、数据拟合或搜索预算。其它 64 个 attention 权重在每 case 开头由父 q/SW 重置并保留。
- 独立小张量测试使用异质行/组尺度的 BF16 3×256 矩阵；逐行逐组 Python 循环 oracle 与三个格式输出严格相同；W4 per-channel 与已有 int4_codes 严格相同；全零组保持零；MLP 选择精确为 48 个。**PASS**。
- 结果可以支持同 RTN 求解器下粒度/位宽的条件差异；与已优化 Phase3 父包的比较同时包含 solver/scale/code 来源差异，不能仅把 group128 相对父包的差异归因于粒度。该路径是诊断，无格式导出或最终方案变更。

## 范围截断与剩余网格误差增量审查

结论：**PASS（新增数学/条件逻辑静态审查）**。范围为 `error_components` 和 `remove_clipping/remove_grid` 两个 replacement 分支；复用主执行已通过的小张量恒等式/非对称端点检查，不重复此前测试和成熟加载评测检查。

令 P 为当前父包 q×SW 的实际 BF16 权重转 FP32，R 为同 R 未量化 BF16 参考转 FP32，B=clamp(R,-8SW,7SW)。实现定义 C=B-R、G=P-B；二者在浮点运算舍入精度内满足 P-R=C+G。SW 使用当前原记录 `[out,1]`，没有重估，正负端点与现有 signed INT4 约定一致。remove_clipping 写入 BF16(P-C)，remove_grid 写入 BF16(P-G)，后者在 FP32 运算精度内等于 BF16(B)。这两条都只修改已选择的 MLP 模块，case 起点仍是父包重置。

分量 SSE 统计同时记录 `2<C,G>` 和直接总 SSE，因此没有错误假设两个分量正交。`grid` 包含相对于截断参考的整数码决策、Phase3 优化以及 BF16 反量化舍入，并非纯最近舍入误差；脚本 settings 已明确这一点。回写 BF16 会增加最后一次舍入，不能声称部署权重的误差被数学上无限精度地消除。这是非部署诊断，不保存量化包，允许依据匹配完整 NLL 判断截断是否主导当前条件下的任务损失；不能把两次干预的任务收益相加，也不能由此证明某一种网格优化算法已经足够。
