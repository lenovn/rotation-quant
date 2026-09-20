# Phase5 独立架构验证

日期：2026-09-18。角色：独立 verifier；未修改应用代码。有效源码：`worktrees/SpinQuant-multimodel`。环境：`runs/phase5/env/bin/python`，torch 2.4.1+cu121、Transformers 4.51.3。

## 当前结论

**首次真实 Qwen checkpoint 架构接入独立 PASS；tiny CPU 与两卡 QAT 层放置 PASS。** 真实检查使用 GPU5 上24个固定token，无 CPU 全尺寸模型 forward，无完整 PPL 复跑。此 PASS 仅覆盖架构与导出新增风险，不声称正式 Joint100/QAT400 或最终 PPL 已完成。

独立文件：`worktrees/SpinQuant-multimodel/tests/test_phase5_architecture.py`。初次完整运行 6 passed / 1 failed。第一个失败是测试对已有禁用 output quantizer 的 NaN buffer 未指定 equal_nan，修正测试后继续暴露实际应用错误：`distill.prepare_student` 的 `isinstance(..., SP2Quantizer)` 未导入 `SP2Quantizer`，抛 `NameError`。主执行补导入后，仅重跑受影响测试，1 passed，10.99 秒。累计 7 项有 PASS 证据，没有因修复单一 import 重跑其余已通过项。

| 新增风险 | 已执行证据 |
|---|---|
| 原生 Qwen → 未量化适配 | tiny BF16、同 SDPA，logits 完全相等；显式 head clone，config tie_word_embeddings=false，调用 tie_weights 不重新绑定 |
| centering 与 fusion 分离 | 非单位 norm 权重，先单独测 embedding centering，再测同 centering 状态 norm fusion；Q/K Norm 参数及 eps 未改；训练入口 Q/K Norm 冻结 |
| R1 / GQA V/O R2 | 4 个 Q heads、2 个 KV heads、head_dim=8；同预处理未量化 logits 符合 BF16误差；R1 与每层 R2 均有有限非零梯度，checkpoint on/off 梯度完全一致 |
| 静态导出/冷加载 | 实际 W4 包保存、重新加载后 logits 完全相等；196等真实尺寸覆盖留待GPU；tiny全部状态（含禁用NaN buffer）不更新，Q/K Norm值恢复，解绑保留，QAT准备后norm冻结 |
| 28 层覆盖 | tiny宽度、28层结构，R=29，SW=196，SA=168，SP2=28；R3/R4在线Hadamard关闭 |
| 原生 teacher | tied embedding/head，参数全部冻结；CPU offload开关损失一致；真正GPU/CPU迁移留待GPU检查 |
| 旧 Llama 共享入口 | tiny Llama加载、norm fusion和R后未量化forward窄回归通过 |

Q/K Norm 位置另外核对本环境官方实现：`transformers/models/qwen3/modeling_qwen3.py:195-196` 使用真实 head_dim 和 config.rms_norm_eps；217-218 行 projection → per-head norm，222 行才 RoPE。适配只替换投影模块，没有替换该 attention forward。原实现计算精度与 RoPE 路径因此保留。

## 实际命令

```bash
PYTHONDONTWRITEBYTECODE=1 runs/phase5/env/bin/python -m pytest tests/test_phase5_architecture.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 runs/phase5/env/bin/python -m pytest tests/test_phase5_architecture.py::test_static_export_cold_load_qparams_and_qk_norms -q -p no:cacheprovider
```

上面 Python 路径实际使用绝对路径，cwd 为源码 worktree。只产生 PyTorch 2.4 CPU autocast deprecation warning，非数值失败。

## 真实 GPU 检查完成

脚本 `runs/phase5/verifier/architecture_gpu.py` 已在主执行分配的GPU5完成，checkpoint revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。24个输入token原文与token IDs保存在脚本/JSON。

| 检查 | 实测 |
|---|---|
| 原生BF16与仅架构适配 | logits max_abs=0，RMS=0 |
| embedding centering单独影响 | max_abs=0.31640625，RMS=0.0568190，相对RMS=0.0119687；明确属于历史预处理变化 |
| 相同centering下norm fusion | BF16 max_abs=0.5625，RMS=0.0538268，相对RMS=0.0113529 |
| 相同预处理下无量化R | BF16 max_abs=1.375，RMS=0.1198418，相对RMS=0.0252005；结合tiny代数/梯度检查，按BF16舍入判断，不要求bitwise |
| 实际W4/static A8 checkpoint backward | NLL=5.097129821777344；R=29、SW=196、SA=168、SP2=28；R1与所有28个R2梯度有限且非零。此步骤按照Joint100，down实际A16旁路 |
| 静态SP2包导出/冷加载 | 196权重覆盖；logits max_abs=0，真实qparams前向前后不更新，Q/K Norm数值保留，head解绑保持 |
| 原生tied teacher真正GPU/CPU offload | 短输入distillation forward/backward有限；teacher仍tied，body在CPU、head在CUDA；loss=9.343750953674316 |

首轮 tmux `phase5-qwen-architecture-verifier`，pane PID1350036 / Python PID1350038，完成原生/预处理/旋转后，因验证脚本将W设16做旁路却未在初始化前恢复4，触发既有W4-aware约束。属验证脚本设置错误，应用源码未因此修改。保留 `architecture_gpu.first_attempt.log/json` 与首轮原日志。修正脚本后以 `PHASE5_VERIFY_RESUME=1` 在 `phase5-qwen-architecture-verifier-resume`（pane PID1355791）仅续跑剩余量化梯度/导出/teacher检查，已结束PASS。日志 `architecture_gpu.resume.log`，总结果 `architecture_gpu.json`。

保留 `architecture_short_input_static.pt`；这是短输入验证包，不是正式训练/校准结果，不用于主表。实际GPU冷加载完全相等，未发生需放宽导出比较容差的情形。

尚未检查最终 seed42 包独立完整 PPL；按 handoff 待正式最终包，仅选择一次预定完整主指标复核。

## 两卡 QAT 层放置：独立 PASS

主执行分配 GPU5/6 后运行 `placement_gpu.py`，已完成并释放设备。4层 tiny Qwen、GQA、真实 static W4/INT8/SP2 QAT student，比较单卡与两卡（层0/1→GPU5，层2/3→GPU6，embedding/head/final norm→GPU5），均开启 non-reentrant checkpoint 重算。两者 NLL 都为 4.151698112487793；84个可学习 W/master、SW、SA、SP2 张量的梯度逐元素完全相等，最大绝对差0。证据 `placement_gpu.json`。这是层放置计算正确性的有限验证，不冒充完整模型24GiB显存可运行性或正式QAT完成。
