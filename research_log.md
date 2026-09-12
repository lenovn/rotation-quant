# Research log

以下按日期保留当时的实验假设、结果与判断；“正在运行”“等待”等只表示记录当时的状态。当前状态见 [STATUS.md](STATUS.md)。Phase 1 和早期 smoke 产物已归档至 `delete/runs/`，旧脚本在 `delete/scripts/phase1/`。

## 2026-08-19 — Periodic static-scale SpinQuant

### 实验假设

目标是先得到面向 NPU 的全静态 rotation baseline：W4 symmetric per-channel、A8 symmetric per-tensor、KV16。SpinQuant 的 R1/R2 学习若直接在 W4A8 static fake-quant 网络上进行，并每 20 个 R-update steps 才重算一次 `S_W/S_A`，会比每次 forward 动态找 scale 更贴近最终 static 模型；最后基于最终 R 再 calibration 可避免导出过期 scale。后续精度模块在该 baseline 可信后再逐项加入。

论文依据：[SpinQuant](https://arxiv.org/abs/2405.16406) 冻结原始 W、用 Cayley SGD 训练 R1/R2 共 100 iterations；`SpinQuant_no_had` 只使用可吸收的 R1/R2。论文主结果通常在 rotation optimization 时使用 W16，附录则表明 learned rotation 可与 RTN 配合。本实验改为训练时直接 W4A8，并研究周期性 static scale。

### 最短验证计划

1. 实现 W4 per-channel scale 在 calibration 边界更新、区间内固定。
2. 实现 A8 per-tensor observer：initial calibration bypass A8；后续 calibration 用旧 `S_A` 跑 W4A8，同时在 A8 前观察。
3. 接入 calibration@0/20/40/60/80/100；只训练 R1/R2，KV16，关闭 R3/R4。
4. 先做小张量/小模型数值 sanity，确认旧 scale 在 recalibration forward 中确实生效且新 scale 只在轮末切换。
5. 再运行短程 GPU smoke experiment，直接检查 train loss、scale drift 和 R 正交误差；可信后再跑完整 100-step 实验。仅在需要呈现时画图。

### 当前状态

- 失败现象：首次单卡 smoke 在进入模型加载前报 `trying to initialize the default process group twice`。
- 原因判断：`HfArgumentParser` 构造 `TrainingArguments` 时已通过 Accelerate 初始化默认进程组；入口随后无条件再次调用 `dist.init_process_group()`。
- 修改内容：入口只在默认进程组尚未初始化时调用 `init_process_group`；量化实现已加入 rotation-static A8/W4 与周期 callback。
- 结果变化：2-step smoke 已跑通；21-step 边界实验正在运行，用于验证 step 20 周期重校准和 step 21 最终重校准。
- 是否保留：机制暂留，等待 21-step 数值决定。

### 21-step 边界实验与静态评测

- 实验假设：周期更新且最终重算的 SA/SW 可被离线 R1/R2 静态评测原样加载；短 calibration 先用于验证机制。
- 失败现象：首次静态 evaluator 未传显式 PREFILL phase；接通后，4×128 calibration 的 W4A8-static PPL 为 242.28。将 calibration 扩到 32×128 后平均 SA 从 0.07396 增到 0.11056，PPL 反而升到 414.24。
- 原因判断：不是 evaluator 或 A8 位宽本身。相同 8×2048 WikiText-2 子集：BF16 12.57、最终 R 的 W4A16 21.32、dynamic W4A8 21.54。退化集中在无 R4 的重尾 down_proj 输入；32×128 时 down_proj 平均 SA=0.551，其他投影约 0.01–0.04，最坏 down_proj SA=5.858。
- 修改内容：评测路径关闭 R3/R4、加载最终 SA/SW、显式使用 PREFILL；增加 calibration-only；仅作诊断地旁路 down_proj A8。
- 结果变化：旁路全部 down_proj 输入为 16-bit 后，其余保持 static W4A8，PPL 恢复到 21.45。21-step 训练前 5 步/后 5 步平均 loss 为 6.887/5.519；step20 后最终 step21 的 SA 仍漂移 3.52%，证明最终 calibration 必需。
- 是否保留：保留周期/最终 calibration、R/scale 导出和静态评测路径。全量 down_proj FP16 只保留为上界消融，不作为最终 NPU 方案；naive max-based 全 A8 记录为失败 baseline。

### down_proj calibration-only MSE diagnostic

- 实验假设：只替换 16 个 `down_proj` 的 static A8 scale，用 calibration activation 的逐元素 quant-dequant MSE 选择 clipping threshold，可能在不引入 A16 的情况下缓解重尾离群值；其余 96 个 SA、全部 SW 和 R 固定。
- 失败现象：32×128 calibration 下，16 层平均局部 MSE 降低 73.83%，WikiText-2 8×2048 PPL 由 shadow-MinMax 的 410.87 降至 163.07，但仍远差于 down_proj A16 上界 21.45。最坏的 layer 1 观测到 `max_abs=712`，MSE 搜索选择 `alpha=1.0`、`SA=5.6063`，局部 MSE 仅改善 0.39%，没有裁掉致命离群值。
- 原因判断：逐元素输入 MSE 按元素频率加权，不反映通道/层敏感度和误差经后续网络传播后的影响；平均局部 MSE 大幅下降不足以推出端到端 PPL 可信。shadow-MinMax 与旧 MinMax PPL 414.24 基本一致，说明收益来自 MSE clipping，不是“旧 W4A8 前向、A8 前 observer”这一收集协议的偶然变化。
- 修改内容：增加 calibration-only down_proj observer；recalibration 时保留旧 SA/SW 的 W4A8 前向，在每个 down_proj A8 前收集 BF16 输入；同一批输入同时导出 MSE scale 和 shadow-MinMax scale。没有更新 R、SW 或其他层 SA。
- 结果变化：MSE scale 平均 0.4021（旧值 0.5507），平均最佳 `alpha=0.3669`，平均 saturation 约 0.00209%；单变量核对确认只有 16 个 down_proj activation scale 改变。目标产物和逐层统计已保存，未生成图。
- 是否保留：保留 observer/诊断代码与结果，作为后续方法对照；不把纯 input-MSE scale 作为最终全静态 baseline，也不继续细调 alpha 网格。下一步应比较带层输出重构目标的 clipping，或把异常 down_proj 作为独立 NPU 精度模块处理。

## 2026-08-19 — W16A8 joint R+SA, followed by GPTQ W4

- 实验假设：Stage 1 在 W16、static A8 下用任务 loss 联合优化 R1/R2 和 96 个非 down_proj 的 per-tensor SA，能先把 rotation 与大部分静态激活网格对齐；Stage 2 固定 R/SA 后，对最终 rotated W 做 symmetric per-channel GPTQ W4。
- 最短计划：初始 calibration 后用 LSQ step-size gradient 学 SA，R 继续用 Cayley 更新；16 个 down_proj SA 保持 static 但不进梯度优化，训练后由独立后处理选择最终值；然后让 GPTQ 在已加载最终 A8 的前向上收集输入并评测 W4A8。
- 当前修改：已建立分支 `phase2/joint-r-sa-w16-gptq-w4`。正在接通可学习 SA、R/SA 双参数组以及 static-A8-aware GPTQ；不会在训练结束用全量 recalibration 覆盖学到的 SA。
- 失败现象：最初设置 `SA lr=1e-3` 时，1 step 后只有 6/96 个可学习 SA 在 float32 精度上变化，相对漂移仅 1.23e-8，等同于没有学习。
- 原因判断：LSQ step-size 梯度归一化再叠加 Trainer 的全局 gradient clipping，令 `1e-3` 的实际更新低于大部分 scale 的 float32 有效精度。
- 结果变化：改用 `SA lr=1.0` 后，1 step 有 88/96 个 SA 改变，最大绝对变化 2.79e-6；16/16 个 down_proj SA 精确不变，17 个 R 的最大正交误差 4.77e-7。2-step 真实训练已跑通，loss 6.754→5.854。
- 是否保留：保留 LSQ 与双参数组实现，并以 `SA lr=1.0` 作为下一轮短程默认值；100-step 和 Stage 2 等待明确 down_proj 在 Stage 1 中是固定还是周期无梯度更新。

### 2026-08-20 — 确认 down-A16 的两阶段实验

- 实验假设：Stage 1 用 W16、96 个非 down_proj static A8、16 个 down_proj A16 联合学习 R1/R2+SA；Stage 2 固定 R/SA，让 GPTQ 在相同 hybrid activation 前向上把所有 112 个投影（含 down_proj）量化为 symmetric per-channel W4。
- 失败现象：首次 GPTQ smoke 在直接调用 decoder layer 时缺少显式 quant phase，static A8 报 `phase context requires an explicit QuantPhase`。
- 原因判断：完整模型 forward 会解析 PREFILL/DECODE，但 GPTQ 为逐层收集 Hessian 而绕过了完整模型入口，因此必须由 GPTQ 显式声明 PREFILL。
- 修改内容：Stage 1 将 down_proj 设为真正 A16，不 calibration、不进 optimizer、不导出 SA；Stage 2 加载 96 个 SA 后保留 down-A16，并在 GPTQ 的两处逐层 forward 显式传 `QuantPhase.PREFILL`。长任务改用 tmux 托管。
- 结果变化：40 个相关数值测试通过。2-step smoke 导出 96 SA、0 SW，95/96 SA 发生更新，R 最大正交误差 3.58e-7；训练 loss 为 3.439/3.758。Stage 1 PPL 为 13.2549（2×2048）和 12.6286（8×2048）。GPTQ 2-sample smoke 得到 112 个 finite、4-bit、symmetric、per-channel quantizer，其中 16 个 down_proj；同口径 2×2048 PPL 为 19.7706。
- 是否保留：保留该两阶段路径。smoke 的 GPTQ calibration 仅 2 samples，PPL 只证明链路和方向，不作为正式精度结论；正式实验使用 100-step Stage 1 和更充分 GPTQ calibration。
- 正式运行：`w16a8-downa16-joint-r-sa-r12-100step-s42` 已在 tmux `rq_phase2_stage1_100` 用 GPU 0、7 启动；32×128 initial calibration 的 SA mean=0.03660，首步 loss=2.721，任务继续运行。

### 2026-08-22 — 正式 100-step Stage 1 与 128-sample GPTQ

- 实验假设：若 Stage 1 已用 W16、非 down_proj static A8、down_proj A16 联合优化最终 R/SA，则在完全固定 activation 策略后，Stage 2 的 PPL 变化可主要归因于 rotated W 的 GPTQ W4。
- 失败现象：最终 SA 相对初始 SA 的收益很小；同一最终 R、同一 8×2048 WikiText-2 评测中，PPL 只从 12.62413 降至 12.61457。GPTQ W4 后 PPL 升至 16.18434，比 W16 增加 3.56977（28.30%）。
- 原因判断：95/96 个 SA 虽发生变化，但 SA 向量相对漂移仅 9.36e-5，说明当前 100-step LSQ 没有显著改变 calibration 初始化；主要精度损失来自 W4，而不是 SA 是否使用训练前值。128-sample GPTQ 比 2-sample smoke 的 19.7706 明显更好，也确认过少 Hessian 样本会夸大 W4 退化。
- 修改内容：本轮未再修改源码；完成正式 100-step Stage 1、最终 SA/初始 SA 单变量对照，以及 128×2048 GPTQ calibration 和 8×2048 PPL 评测。正式 GPTQ 产物含 112 个 4-bit symmetric per-channel、groupsize=-1 quantizer，包括 16 个 down_proj，全部 scale/zero 为有限值。
- 结果变化：100-step train loss 前 5/后 5 步均值为 2.69004/2.51138；17 个 R 的最大正交误差为 7.15e-7。正式产物为 `runs/phase2/w16a8-downa16-joint-r-sa-r12-100step-s42/rotation/R.bin`、最终 `quant_scales.pt` 和 `gptq/w4_gptq_model.pt`。
- 是否保留：保留两阶段实现、正式 R/SA 和 128-sample GPTQ 产物；不把微小 PPL 改善解释为 SA 联合学习已有显著收益。下一步优先用同一 W4 配置做全 A16 activation diagnostic，区分纯 W4 误差与 W4/A8 交互，不先做冗余 sample sweep或画图。


## 2026-09-08 — 已有结果核对与文件整理

- 本次只读核对已有日志，未运行新实验。8 月 22 日的同一正式目录中，W16A16 PPL=12.58486，W4A16=16.06202；此前该目录记录的 hybrid A8/down-A16 结果分别为 W16=12.61457、W4=16.18434。全 A16 后 W4 损失仍明显，支持继续分析权重量化误差；不能据此直接断言 group32 能改善多少。
- 当前脚本默认目录 `w16a8-joint-r-sa-r12-s42` 是另一组运行，W16 hybrid PPL=12.63674、W4 hybrid=16.79897。分布分析默认消费这组 R.bin；不得与前一组结果合并。
- BF16 R1/R2 分布产物与 weight channel 分析文件保留在 `runs/distribution-atlas/llama32-1b-r12-bf16-wt2-s42/`。本轮未重新评估这些分析的数值正确性。
- 旧方案、Phase 1 和早期 smoke 已进入待删除目录；本文件保留失败原因和结果，供后续比较。原始九份根目录文档见 [归档](delete/root-docs-2026-09-08/)。
