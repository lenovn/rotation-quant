# Phase 3 B: 512-step cosine, fixed old 800 windows

## 本轮授权与假设

检验 B 联合优化是否仍能从更长预算获益。复用旧 B100 已完成结果，不重跑旧实验。
只比较固定探针与初始 SP2 导出包的完整 WikiText-2 validation；不接邻码搜索、范围后处理或 QAT，不用 test/C4 选择步数。
25/50 补评及 128/800 数据量消融暂缓。完整后处理＋QAT400 是出现明确收益后另行安排的候选比较，本任务不自动启动。

## 运行与输入

- 旧参照：`../route-b-adam-100-20260914a/`，100-step cosine 终态完整 validation PPL 17.11711490993314。
- 相同初始参数：`../common-init-20260914a/initial.pt`。启动前逐张量核对，其 241 个参数张量与旧 B checkpoint-0000/state.pt 完全相等。
- 新运行：512 optimizer updates，warmup 10；R SGDG/Cayley LR 1.5；scale Adam，相对初始 scale 的 LR 系数 0.001。
- 每步 8 个 microbatches，每窗 2048 tokens；只循环旧日志实际出现的不同梯度窗口，原顺序 0..799。
- 固定训练池 800 窗 / 1,638,400 tokens。512 步累计 4096 次窗口读取 / 8,388,608 输入 tokens，即该池的 5.12 遍。
- 固定 probe 为原 train split 窗口 1180..1183，共 8192 输入 tokens，始终排除在梯度训练外。
- 原始 BF16 权重冻结，学习 R、112 SW、96 非 down SA；训练 down 输入 A16。
- 精确命令、GPU、PID、tmux 会话见 `../route-b-adam-512-fixed800.launch.json`；实际设置见 `settings.json`；数据索引见 `data.json`。

## 检查点与导出

0 步仅记录相同固定探针。100、256、512 步保存 state.pt、固定探针和初始 static_w4a8.pt，并评测完整 validation。
每个检查点另建 EvaluationModel，保留该点已学 SW 和非 down SA；不重算或覆盖这些尺度。
down-SP2 校准数据固定为旧 seed42 所选 32 个 train 窗口的前 128 tokens（4096 tokens）。
规则固定为按每8行采样，按导出后 W4 的输出 MSE 对每个 down 搜索 33 个 coarse + 17 个 refined 候选；每个检查点重新得到自己的 down-SP2 尺度。
导出、校准和冷加载评测均不更新训练模型参数；训练随后继续使用原训练对象和优化器。

完整 validation 与旧口径相同：252852 输入 tokens、252728 预测 targets、123 个 2048 窗口及 948-token 尾窗，token 加权 NLL。
固定四窗 probe 使用 FP32-logit CE，训练配置 down-A16；完整 validation 沿用 BF16-logit evaluator，导出配置 down-SP2。两者不可直接相减作为转换损失。
这些是 BF16 fake-quant prefill 精度测量，不是手机 NPU 性能测量。

## 结果解释与状态

`COMPARISON.md` / `comparison.json` 随检查点评测更新。
旧 100-step 终态与新 512-step 日程的 100/256 中间点、新 512 终态分别标识；旧、新日程的学习率轨迹不同。
运行状态只认 `progress.json`、`training.jsonl`、各 checkpoint/validation.json；启动不表示训练或评测完成。
日志：`../route-b-adam-512-fixed800.log`。运行失败时保留 failure.json 与所有已有产物。

## 本轮验证证据

独立 verifier `/root/verify_b512` 明确 PASS：既有定向 CPU 测试 11 passed，新 `tests/test_phase3_budget_export.py` 4 passed。
新增测试执行实际训练循环的 512×8 次 CPU mock 微步，检查窗口循环、probe 排除、检查点触发、累计 tokens 和日程差异。
tiny 16 层 checkpoint 导出测试包括固定 probe、SP2 校准、保存与 cold reload；训练参数值/对象 identity/requires_grad/训练模式/down-A16 保持不变，SW/非 down SA 精确保留。
父进程真实重建数据，完整 metadata 与旧 data.json 相等，训练窗口恰为0..799，validation targets 为252728。
common.py 与 quantization.py 同旧 B 保存的源码快照一致。新数据选择及结果比较代码随运行保存于 source/。
CPU 验证用于实现检查，不替代正式 GPU 结果。

正式启动核对：PID 1565747 / physical GPU 6。已观察第1/512更新完成，step0固定probe NLL 3.600554883480072与旧B精确一致；首步8个microbatch losses及全部学习率与旧B首步精确一致，平均NLL 3.3947621881961823。此条只记录启动证据，最新进度以 progress.json 为准。
