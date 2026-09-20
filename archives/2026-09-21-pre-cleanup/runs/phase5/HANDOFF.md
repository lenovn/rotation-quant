# Phase5 交接：Qwen3-1.7B 固定配方迁移与论文验收

更新：2026-09-18。项目根目录：`/home/dongpeiyan/projects/rotation-quant`。相对路径均以此为根。

依据：用户最新 handoff 附件，以及随后提供的 WikiText-2 test、C4 阶段归因实测汇总。本次仅整理文档，未登录服务器复核；新增结果按用户报告记录，接手时读取所列原始报告，不因此默认重跑。以下尚无新增实测证据的 Qwen 迁移状态为 **NOT TESTED**，如服务器已有进展，以实际记录更新。

## 1. 目标与执行范围

把 Llama-3.2-1B-Instruct 已接受的方法迁移到 `Qwen/Qwen3-1.7B`，验证同一算法及固定配方的跨家族效果。本轮只新增这一 Qwen checkpoint，不为 Qwen 重开最优路线搜索，不自动扩展模型名单。

默认主线：

```text
WikiText-2 train
→ Joint100（历史运行名 B100）
→ 校准 down 静态 SP2
→ down W4 邻码
→ SP2 范围收缩
→ down 邻码再适配
→ WikiText-only QAT400
→ 同一最终包接受 WikiText-2 test + 固定 C4 子集评测
```

迁移算法规则，不迁移 Llama 的 R、尺度、整数码、接受层号或 optimizer 状态。B100 是本阶段历史运行名，不与早期 A/B/C 实验中的 B 定义混用。最终方法包含 master weight 更新，不能称为全程冻结权重的纯 PTQ。

**本轮纳入：**必要架构适配、固定主线、匹配 INT8/SP2 与 PTQ/QAT 对照、代表模型的必要消融、关键 seed 稳定性，以及必要最终包的 WikiText-2 test/C4 验收。WikiText validation 继续用于开发和阶段分析。已有 Llama 结果按协议直接复用。

**不自动纳入：**新增模型尺寸、A/B/C 历史路线重比、LR/步数/初始化扫描、局部 D、Phase4 未胜出模块、全部中间 checkpoint 评测、下游 ACC 任务全集和原生 NPU 测试。后两项仍是论文后续工作，不由本轮 PPL 代替。

混合数据 QAT 是第 3.4 节的**可选数据消融**，目前不替换 WikiText-only 主线，不因本文存在该节就自动启动。

完成标准是可信、可比的真实结果，而不是达到某个预设 PPL。两个家族提供跨家族证据，不自动证明跨尺寸泛化或外部 SOTA；效果不足时如实报告，不一直调参直到进入 BF16＋1。

## 2. 继承依据与已完成结果

先读适用 `AGENTS.md`，再读本文及所列运行记录。根目录 `STATUS.md`、`SPEC.md`、`PLAN.md`、`LESSONS.md` 提供项目约束，但旧阶段进度不代替最新算法清单。具体配方以实际源码、启动参数与日志为准，不重扫全部历史。

### 2.1 Llama 的 WikiText-2 validation 结果

模型：`cache/models/llama-3.2-1b-instruct`。完整 validation 为 252852 个输入 tokens、252728 个预测 targets，123 个 2048-token 窗加 948-token 尾窗；窗口首 token 不计，按 targets 加权 NLL 后取 exp。

下表证据路径相对于 `runs/phase3/`：

| 阶段 | PPL | 证据 |
|---|---:|---|
| 原始 BF16 | 13.634657725643203 | `auditor/final-best-20260914/bf16.result.json` |
| Joint100 / B100 | 17.11711490993314 | `route-b-adam-100-20260914a/checkpoint-0100/validation.json` |
| down W4 邻码 | 16.507098542639632 | `seq-b100-down-round-20260914b/validation.json` |
| SP2 范围收缩 | 16.19254871508746 | `seq-b100-rounded-sp2-contract-20260914a/validation.json` |
| down 邻码再适配，即精确 PTQ 父包 | 16.112577060930427 | `seq-b100-sp2-refine-down-20260914a/validation.json` |
| QAT400 | 14.581654675328117 | `distill-b100-refined-ref-adam1e5-400-20260914a/checkpoint-0400/validation.json` |

关键产物同样位于 `runs/phase3/`：

- Joint100 同 R 浮点参考：`route-b-adam-100-20260914a/checkpoint-0100/state.pt`。
- 精确 PTQ 父包：`seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`。
- 最终包：`distill-b100-refined-ref-adam1e5-400-20260914a/checkpoint-0400/static_w4a8.pt`。
- 配方读取各 run 的 `settings.json`、`data.json`、`training.jsonl`；总结与历史审计见 `RESULTS.md`、`auditor/final-best-20260914/REPORT.md`，不寻找不存在的 `config.json`。

### 2.2 完整 WikiText-2 test：已完成

证据：`runs/phase3/WIKITEXT2_TEST_20260917.md`。

| 配置 | Test PPL |
|---|---:|
| 原始 BF16 | 13.16265 |
| 已选定 W4A8/SP2＋QAT400，KV16 | 14.15442 |

ΔPPL＝+0.99177；覆盖 **288934 个预测 targets，包含尾窗**。本次没有新训练、重校准或按 test 调参。以上为用户提供的显示精度，正式汇总读取原报告完整指标。

这组结果可用于论文公开 test 基准；validation 仍是开发选择证据。早期评测入口曾使用 test，不能宣称全项目首次盲测。“小于 BF16＋1”只描述此结果，不是所有模型或数据集的通过门槛。

### 2.3 固定 C4 子集阶段归因：已完成

新报告：`runs/phase3/C4_ATTRIBUTION_20260918.md`。历史 BF16/最终包记录：`runs/phase3/C4_RESULTS_20260915.md`。

全部使用同一既有 C4 validation 固定子集、**2096128 个预测 targets**。本轮没有使用 C4 进行校准、训练或调参。

| 配置 | C4 PPL | 结果来源 |
|---|---:|---|
| 原始 BF16 | 21.8434 | 历史复用 |
| B100＋匹配均匀静态 INT8 | 1330.1568 | 归因实验新测 |
| B100＋初始 SP2 | 29.1842 | 归因实验新测 |
| QAT400 的精确 PTQ 父包 | 27.4396 | 归因实验新测 |
| 最终 QAT400 | 26.9848 | 历史复用 |

历史完整数值为 BF16 21.843397624525274、最终包 26.98478050458106；其他行的正式精度读取新报告，不从四位小数反推。

**已支持的结论：**初始 SP2 → PTQ 父包，PPL 降 1.7446；父包 → QAT400，PPL 降 0.4548、NLL 降约 0.01671。两段变化均在 8/8 个评测分段上改善。后处理链具有跨语料收益；QAT400 相对精确父包也在修复 C4，只是幅度有限，不应再写成“尚不知蒸馏是否造成额外 C4 退化”。8 个分段不是 8 次独立训练，不代替 seed 稳定性。

**格式对照的边界：**两条 B100 对照共用 R、W4、SW、非 down SA，仅改变 down 格式并分别匹配校准范围。它支持当前严格静态 per-tensor 设置下的表示收益，不是外部强基线，也不是完整 Uniform-PTQ 或 Uniform-QAT400 的替代结果。

**仍未完成的关键比较：**同预算均匀 INT8＋QAT400；后处理优势经过同预算 QAT400 后是否仍保留的成组消融。最终 C4 仍比 BF16 高约 5.1414 PPL，不能声称接近无损；具体残余误差来源未由本表完全分解。

该 C4 子集不是全量 C4，也不是全项目未见过的盲测。接手时复用上述结果，不把已完成的精确父包评测重新列为待做实验。

### 2.4 证据范围与论文表述

当前证据均为 BF16 fake-quant、全序列 forward、`use_cache=False`、KV16。Llama 覆盖为 112 个 backbone W4 Linear、96 个非 down 静态 INT8 输入、16 个 down 静态 SP2 输入；embedding/head/norm/KV 保留高精度，R3/R4 与在线 Hadamard 关闭。A16 旁路指本实现的 BF16 浮点路径，不是 INT16。

论文可表述为：**在既定静态量化约束下，down 表示方式、离散后处理和恢复训练的收益不局限于 WikiText，但跨语料恢复仍不充分。**不能把收益全部归给蒸馏，也不能把一个 B100 内部对照包装成对全部均匀 INT8 方法的否定。

最终结果表与阶段机制表分开；算法 PPL 不代替下游 ACC、真实 decode 或原生 NPU 速度证据。“W4A8”限定上述 backbone Linear 的量化范围，不代表全图所有算子均 INT8。

### 2.5 源码继承

继承源：`worktrees/SpinQuant-phase3-joint`；分支：`phase3/joint-r-sw-sa-sp2`；历史记录 HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`。

**有效实现不只在 HEAD 中。**附件记录了 `optimize_rotation.py`、`train_utils/quant_linear.py`、`train_utils/rotation_calibration.py`、`utils/process_args.py`、`utils/quant_utils.py` 的 tracked 修改，以及未跟踪的 `experiments/` 和相关 tests。接手时核对当前状态，连同实际有效 dirty/untracked 实现一并继承，记录 revision、diff 和必要源码快照。

新工作区建议 `worktrees/SpinQuant-multimodel`，分支 `phase5/multimodel`。若已存在则先核对用途，不覆盖。保留旧源码、环境与产物；共用一套算法，仅参数化必要架构入口，不做无关重构。

## 3. 固定迁移配方

下面为配方摘要；缺项从第 2.1 节对应 run 与实现读取，不猜默认值。W4 整数边界、取整/截断及 SP2 码本、重构/范围定义均继承有效实现，不仅凭“INT4/SP2”名称重写量化器。

### 3.1 Joint100

实现：`experiments/phase3/run.py`、`common.py`。

| 项目 | 固定规则 |
|---|---|
| 初始化 | 首轮 seed42 随机符号 Hadamard；R1 按 hidden_size、每层 R2 按实际 head_dim 重新生成 |
| 可学习量 | R、非 down SA、SW；原始模型权重冻结，从第一步开始 W4-aware |
| down 输入 | 训练期间 A16；冻结导出后才校准 SP2，不声称已经联合训练 down SP2 |
| 训练量 | 100 次 optimizer updates；2048-token 窗；全局有效 batch 为 8 窗，共 1638400 token visits |
| 优化器 | R 用 SGDG，LR 1.5；尺度用 Adam，初始绝对 LR 为各尺度张量初值均值的 0.001 倍，eps=1e-12 |
| 调度 | warmup 10，100-step cosine |
| 初始校准 | WikiText train-only；32 个训练窗的前 128 tokens；SA 按原规则初始化，SW 由当前 R 下有效权重初始化 |

Qwen 重新 tokenize，生成自己的训练/校准索引并记录训练池大小及访问次数。**8 窗是所有 GPU 合计的有效 batch，不是每卡 8 窗。**并行方式不改变全局训练预算。

### 3.2 离散后处理

实现：`experiments/phase3/sequential_postprocess.py`。

```text
初始 SP2 包 → down 邻码 → SP2 收缩 → down 邻码再适配
```

使用 32 个完整 2048-token WikiText train 窗，按 24/8 分为 fit/selection，以真实 train-selection NLL 选择收益前缀。selection 属于训练 split，不是 WikiText validation/test，更不是 C4 评测子集。

邻码候选保留父候选及 512/2048/8192 坐标预算；SP2 收缩候选为 1、0.875、0.75、0.5、0.25、0.125。父包始终可选，不强行接受更差修改。每个模型重新处理其所有 down 层，不照抄 Llama 接受的层号、倍率或整数码。仅执行相应模式实际生效的候选，不扩大历史搜索。

按每层相同预算迁移，记录总搜索量与耗时；Qwen 层数不同，不能因步数相同宣称总 FLOPs 相同。保留阶段包与选择记录，不逐个内部候选做完整外部评测。

### 3.3 默认 WikiText-only QAT400

实现：`experiments/phase3/distill.py`。从本模型精确 PTQ 父包干净启动；teacher 为同一 checkpoint 的原生 BF16，未旋转、未 norm fusion，始终冻结。

| 项目 | 固定规则 |
|---|---|
| master 初始化 | 本模型 Joint100 的同 R 浮点参考，投影到实际父包 INT4 cell；沿用 `cell_reference_initialization` |
| 可学习量 | backbone FP32 master W、SW、非 down SA、SP2 尺度；R 已融合并固定，其他高精度模块冻结 |
| 训练量 | 400 updates；全局每步 8×2048 tokens，共 6553600 token visits |
| 优化器 | Adam，master W LR=1e-5，尺度相对 LR=0.001；其余参数读取原 settings |
| 调度与损失 | warmup 10，400-step cosine；T=1，0.9 KL(teacher∥student)＋0.1 CE |
| 数据 | 仅 WikiText-2 train；沿用 `data_start=800` 的索引/循环规则，按本 tokenizer 实际训练池执行并记录 |

100/200 是同一 400-step schedule 的中间状态，可用于恢复；不另做步数扫描。最终固定第 400 步，不根据 validation、test 或 C4 改挑中间点。

### 3.4 可选混合数据 QAT：不是本轮默认主线

仅在用户另行确定实施后启动，用于回答“仅改变恢复数据覆盖，能否改善同一模型在多种语料上的表现”。混合比例及采样规则尚未确定，不能把此前讨论中的举例当作已批准配方。

实施时遵守以下边界：

- 从与 WikiText-only 分支相同的精确 PTQ 父包干净启动；只改变 QAT 数据组成，不重跑 Joint100、校准和后处理。teacher、master 初始化、400 步、总 token visits、优化器和损失保持一致。
- 数据只取 WikiText-2 train 与 C4 train。实施前在 PLAN 固定比例、采样/拼接规则与全局预算，记录实际文本和 token 覆盖；不以 C4 评测分数扫描比例，不把原 WikiText 的 `data_start` 直接当作混合语料索引。
- 不使用任何评测子集作为蒸馏输入；teacher-only loss 也不构成例外。不能在 WikiText QAT400 后再追加 C4 QAT400，却标成与原方法等预算。
- 每种训练配方单独导出一份最终包；同一个包接受全部验收，不为 WikiText 与 C4 临时切换权重。结果分行报告，不挑不同包的最好单项拼成一行。
- 若将混合 QAT 升为正式主方法，主要 INT8 对照须匹配相同数据与恢复预算；Llama/Qwen 的默认与混合版本要分别标注，不能悄悄用不同训练数据支撑“固定配方跨家族迁移”。

混合配方使用 C4 train 后，C4 validation 仍可评测，但其解释变为见过该语料来源后的 held-out 表现，不再声称未用 C4 做量化优化的跨语料迁移。保留 WikiText-only 结果作为单独对照。

## 4. Qwen 最小适配与一次性正确性检查

目标为 `Qwen/Qwen3-1.7B`，不是 Base。记录实际 checkpoint/config/tokenizer revision。附件引用的配置为：28 层、hidden_size=2048、intermediate_size=6144、16 个 Q heads、8 个 KV heads、head_dim=128；attention 无 bias，原始 embedding/head tied。[S1]

原环境记录为 Transformers 4.44.2；附件引用的官方要求为支持 Qwen3 的 4.51.0 及之后版本。[S2] 检查已有兼容环境，必要时新建隔离环境，不原地升级旧实验环境；实际依赖及 API 以接手时读取的版本为准。

| 适配点 | 实施要求 |
|---|---|
| 模型与 tokenizer | `common.py`、训练模型类、`external_eval.py` 等按 architecture/config 路由；不固定 Llama 类、词表、token 数或缓存 |
| Q/K Norm | 保留 projection 后、RoPE 前的 per-head `q_norm/k_norm`，包括权重、eps、位置及原实现计算精度；不能当作层前 RMSNorm 向前融合，不全局将 norm weight 置 1 [S3] |
| R1/R2 | R1 改残差基底，R2 配对作用于 V/O；按实际 head_dim 与 GQA 处理；R3/R4 仍关闭，不新增 Q/K 输出旋转 |
| RoPE/attention | 保留 Qwen 的 theta、mask、position_embeddings 与 cache API；不继承 Llama RoPE scaling，不将应保留精度的 RoPE buffer 一刀切转 BF16 |
| tied embedding/head | 若 final-norm fusion 等使 head 与 embedding 不再相同，显式解绑；导出 config 与冷加载均保留解绑状态，避免 `tie_weights()` 覆盖。原始 BF16/teacher 保持原结构；实际模型大小计入解绑新增存储 [S1,S4] |
| 导出与蒸馏 | 保存/冷加载完整恢复 Q/K Norm、旋转权重与静态尺度；teacher/student 身份和词表一致，Q/K Norm 默认冻结 |

按每层 7 个 backbone Linear 推导，预期为 196 个 W4、168 个非 down INT8 输入、28 个 down SP2 输入，1＋28 张 R。它们是覆盖核对的预期值，不是实测 PASS；检查实际 forward 路径、评测时尺度不更新与高精度例外。

**架构适配、旋转等价性与历史预处理分开检查。**继承实现包含 embedding 去均值；原附件说明公开 SpinQuant 也有这一步，RMSNorm 本身不做去均值。[S3,S5] 先比较原生 BF16 与仅架构适配的未量化模型，再检查同一预处理状态下的无量化 R/fusion 和导出。历史 centering 若在主线中启用，单独记录其影响，不悄悄改配方，也不把差异吞进容差或交给 QAT 修复。

独立 verifier 首次接入新架构时检查：原生与适配后未量化一致性、R/导出冷加载一致性、Q/K Norm 与 tied 权重处理、真实量化覆盖及静态尺度。固定少量真实输入足以定位大部分新增风险，不给每个检查配完整 PPL。完整前向和正式评测用 GPU；CPU 只做轻量代数、状态和局部单元检查。按同 dtype/backend 的实际浮点误差判断，不要求 bitwise 相同。

旧 Llama 共享路径仅在被修改触及时做窄回归。QwenSpinQuant 不作为新底座；有需要才核对相关工程片段，架构以官方 Qwen3 为准，算法以本地 Phase3 为准。

## 5. 实验顺序与最小论文对照

### 5.1 执行顺序

先读取有效源码与最新实测报告，登记已完成结果及缺项；完成 Qwen 必要适配后：

```text
Qwen 原始 BF16
→ Joint100 + 初始 SP2
→ 三步后处理，得到精确 PTQ 父包
→ 默认 WikiText-only QAT400
→ validation/C4 阶段分析 + 必要最终包的 test/C4 验收
→ 匹配 INT8/SP2 对照、代表模型消融、关键 seed
```

模型包与评测节点按第 6.2 节执行。正式保留 BF16 身份、Joint100 参考、初始 SP2、精确 PTQ 父包和最终 QAT400；不要求保留每个被拒绝候选的大包或完整评测全部中间 checkpoint。

Llama 的完整 test 及 C4 归因已完成，按同协议复用。Qwen 第一轮补 B100 匹配 INT8/SP2 的 C4 配对，检查核心格式收益能否跨家族出现；不将全部历史诊断自动推广到每个 seed。

### 5.2 内部公平对照：INT8/SP2 × PTQ/QAT

两个家族以如下矩阵组织核心证据，已有合格结果直接复用：

| 配置 | down 输入 | 后处理 | 恢复训练 | 用途 |
|---|---|---|---|---|
| BF16 | BF16 | 无 | 无 | 原始模型参照 |
| Uniform-PTQ | 静态均匀 INT8 | 本格式下的匹配后处理 | 无 | 完整均匀静态 PTQ 对照 |
| SP2-PTQ | 静态 SP2 | 主线后处理 | 无 | 精确 PTQ 父包 |
| Uniform-QAT400 | 静态均匀 INT8 | 对应 INT8 父包 | 400 步 | 等恢复预算对照 |
| SP2-QAT400 | 静态 SP2 | 对应 SP2 父包 | 400 步 | 完整方法 |

两个分支从同一 seed 的 Joint100、尚未针对 down 格式适配的状态出发，各自校准 down 量化器，再在本格式下做权重适配、范围选择和恢复。匹配训练数据、候选预算、选择准则、初始化规则及全局训练预算；匹配规则不等于强迫两种量化器使用相同数值尺度。

INT8 必须获得合理且匹配的范围搜索机会；不能以未恢复的 INT8 对比恢复后的 SP2，将训练收益归给码本。**Llama 的 1330.1568 只填入 B100 阶段格式对照，不填成 Uniform-PTQ 或 Uniform-QAT400 的最终结果。**若完整对照实现尚缺，明确列缺项，不能伪称完成，也不阻塞先完成 SP2 主线。

独立外部方法与内部对照分开。在 PLAN 确定至多 1–2 个实际可运行的近邻基线，说明原生及适配设置；附件未确定其具体身份和可运行性，不能编造。W/A/KV、静态/动态、量化粒度、在线算子、数据和恢复预算均需披露，内部变体不冒称官方完整复现。

### 5.3 代表模型的必要消融

优先在 Llama 复用已有同协议、变量匹配的结果。仍需检查：

```text
同一 Joint100 + 初始 SP2 → QAT400
对比
同一 Joint100 + 完整离散后处理 → QAT400
```

两侧使用相同恢复配方及 master 初始化规则，回答“后处理收益在 QAT400 后是否仍然存在”。新 C4 结果已经证明蒸馏前后处理有收益，但不代替这一最终预算匹配的成组消融，也不证明三步各自都不可缺少。无需所有新模型重做；结果不理想时如实记录，不自动另开路线。

### 5.4 seed 稳定性

先完成 seed42 主线与核心矩阵，再对完整方法和最关键匹配对照做 42/43/44 三 seed 配对实验。辅助消融、内部候选及固定 BF16 不全部乘以三；额外 seed 不是第一轮迁移的前置条件。

从实际初始化/采样入口参数化 seed，记录变化的随机源，同 seed 方法之间配对。报告逐 seed 结果、均值±标准差及配对 ΔNLL；评测样本固定，不报最佳 seed 代替整体结果。同一包多次评分不是多 seed；只改变 R 初值时称 R 初始化稳定性，不泛称所有随机源稳健性。

## 6. 数据、指标与最终验收

### 6.1 数据职责

| 数据 | 角色 | 使用边界 |
|---|---|---|
| WikiText-2 train | Joint100、校准、后处理 fit/selection、默认 QAT400 | 唯一默认优化数据；内部 selection 仍来自 train |
| C4 train | 可选混合 QAT 的训练来源 | 仅第 3.4 节启用后使用，不进入默认主线 |
| WikiText-2 validation | 开发、诊断、必要消融 | 不更新参数；可以支持方法开发，不能称完全独立盲测 |
| WikiText-2 test | 固定方法及最终包的同语料 PPL 验收 | 本轮纳入；不用于候选范围、LR、步数或 checkpoint 选择 |
| 既有 C4 validation 固定子集 | 跨语料诊断及最终 PPL 报告 | 保留原样本和协议，不用于校准/蒸馏/候选选择，不按成绩换子集 |
| 固定下游任务 | 论文任务能力评测 | 后续单列任务与指标，本轮不自动扩展 |

**同一个模型、seed、方法和训练配方对应同一个最终权重包；该包接受全部预定最终评测。**不同 QAT 数据配方分行报告，不在不同数据集间暗换 checkpoint，不用多个包各自的最好结果拼成一行。

WikiText validation 已长期用于开发，test 早期也曾被项目使用，C4 子集已有诊断历史，均不包装为全项目从未见过。继续使用公开测试基准与固定 C4 子集，不因已看过成绩就自动重新抽样。若以后依据 C4 分数选择混合比例或方法，应承认其进一步承担开发选择作用，不再宣称该结果完全独立于方法选择。

### 6.2 评测节点与工作量

| 模型状态 | WikiText validation | WikiText test | 固定 C4 子集 |
|---|---|---|---|
| 原始 BF16 | 建立或复用开发参照 | 最终参照 | 最终参照 |
| B100＋匹配 INT8 / 初始 SP2 | 按机制分析需要，复用已有值 | 不默认测 | 首次核心格式配对，Llama 已完成 |
| 精确 PTQ 父包 | 恢复前阶段参照 | 列入正式 PTQ 对照时评测 | 恢复前阶段参照 |
| 最终 QAT400、主要匹配最终基线 | 开发汇总，已有结果复用 | 必须 | 必须 |
| 内部搜索候选、100/200 中间 checkpoint | 不要求逐一完整评测 | 不测 | 不测 |

进入最终主表的 BF16、完整方法及主要匹配对照，均使用固定包统一做完整 WikiText-2 test 与既有 C4 子集。主要 PTQ 对照如列入主表，也按此执行。新增验收不是把所有历史实验再测一遍。

第 400 步规则固定，评测不触发重新校准、optimizer 更新或逐任务选包。发现具体实现错误可修复并说明受影响结果；方法效果不足不是持续扩大搜索的授权。

### 6.3 多模型语料和指标口径

跨模型 C4 优先固定同一组原始文档，再分别 tokenize，沿用既有拼接/截断规则；不复用 Llama token IDs，不强迫 Qwen 得到 Llama 的精确 targets 数。记录每模型文本覆盖、输入 tokens、有效 targets 和尾窗。原始文档清单与协议从已有 C4 报告/数据记录读取，不另造一个“同名子集”。

WikiText 使用 `wikitext-2-raw-v1`；具体 revision、split、拼接与特殊 token 规则读取历史数据记录。附件记录训练按行无 BOS/EOS tokenize 后串接，validation 为双换行拼接；test 的确切实现继承第 2.2 节报告，不凭经验另换。普通语料 PPL 不套 chat template，不添加 thinking 提示。

每模型相对自身原始 BF16 报告 PPL、NLL、ΔNLL、PPL 比值和 targets：

```text
NLL = 所有有效预测目标的 loss 总和 / targets
PPL = exp(NLL)
ΔNLL = NLL_quant − NLL_BF16 = log(PPL_quant / PPL_BF16)
```

保留尾窗，窗口首 token 不计，不平均窗口 PPL。不同 tokenizer 的绝对 PPL 不直接排名；ΔNLL 仍是每 token 指标，不是完全消除 tokenizer 差异的跨模型统一尺度。[S6]

本轮继承现有指标实现，不趁迁移静默更换数值口径。历史 evaluator 用 BF16 logits CE、逐 token loss 转 FP32、分段 PPL 再取 log 按 targets 加权；训练/probe 的 FP32 logits CE 不混作同一比较。若以后统一为 FP32 logits CE 并直接累加 loss，需标新协议，仅重评必要最终包与 BF16，不重训、不将新旧数字混为同口径。

C4 的官方 split 名称按既有记录写为 validation；明确这是固定子集，不称全量或官方独立 test。默认 WikiText-only 路线未使用 C4 做量化优化；混合路线使用 C4 train 后须改写解释，样本与验收协议不因此更换。

### 6.4 论文后续评测边界

下游 ACC 的具体任务 ID、metric key（如 acc/acc_norm）、few-shot、模板、thinking 开关与 harness 版本尚需单列协议；本文不把举例中的任务名单当作已批准执行清单。同一最终包接受全部选定任务，不逐任务再训练并混报。

本轮属于 KV16、无 cache 的 fake-quant 全序列质量评测，不等同于真实自回归 decode。原生 NPU 的 prefill/decode、延迟、能耗、真实算子及软件/硬件码本一致性另行验收。下游 ACC 和硬件指标不能从 PPL 推算。

## 7. 验证、资源和交付

### 7.1 按风险验证，不重复保险流程

主执行负责实现、正式实验与修复；独立 verifier 检查新增量化/旋转/导出逻辑；auditor 检查实验身份、数据、量化开关、指标、历史复用与可比性。

新架构完成首次关键验证后，普通参数实验不重复整套 suite。首个最终 seed42 冷包做一次独立完整主 PPL 复核；复核集在 PLAN 固定，不同时机械重跑 validation、test、C4。BF16 已有同模型、同协议、同实现版本的可靠审计时不重复。首次完整结果与最终结果为同一包时合并审计。

其余 run/seed 以日志、配置、实际覆盖和异常检查为主，有具体异常才追加复测。已完成的 Llama test/C4 归因不因交接再跑。缺证据标未验证，不把“程序启动”“CPU 单测通过”或别的数据集 PASS 当作实验完成。

### 7.2 资源约定

按最新附件末尾启动语统一原正文与启动语的差异：**GPU4 已被用户报告故障，不调度；其余健康 GPU 按实际余量使用，不再固定为 1/3/5/6/7 白名单，也不限制同时使用的 GPU 数量。**启动前检查真实占用和显存，避免 OOM、干扰他人或抢占其进程；多卡仍保持全局有效 batch 与训练预算不变。

不恢复旧单卡、12GiB、占用比例或 GPU 总时长硬限制。长任务用 tmux，记录 session、PID、GPU、命令、日志与恢复方式；状态区分运行中、完成和失败。检查实际存储余量及 mergerfs 后端分支限制，保留必要恢复包，不擅自清理旧实验。

保留旧源码、环境和 Phase2/3/4 产物；不 reset/clean 覆盖、不 sudo、不终止他人进程、不自行提交或推送。准备模型和兼容依赖时优先复用已存在资源，必要时隔离环境。

### 7.3 文件与交付

本文建议落盘为 `runs/phase5/HANDOFF.md`；附带下载文件名为 `handoff.md`，落盘时统一为上述路径。

```text
worktrees/SpinQuant-multimodel/       共享实现与模型配置
runs/phase5/HANDOFF.md               本文
runs/phase5/PLAN.md                  固定配方、对照矩阵、数据/评测协议与 seed
runs/phase5/STATUS.md                进度、命令、结果、缺项与下一步
runs/phase5/summary.csv              模型/配方/seed/包/协议/指标/证据
runs/phase5/qwen3-1p7b/<run>/         参数、采样索引、日志及必要模型包
runs/phase5/verifier/                新架构与修改检查
runs/phase5/auditor/                 关键结果审计
```

每个 run 保存真实命令、模型/tokenizer/环境版本、数据 split/seed/采样记录、settings、Git revision＋dirty 源码证据、父包、最终包及指标。沿用已有记录，不另建 hash/contract/gate 系统。

summary 至少能区分：模型、seed、阶段、down 格式、QAT 数据配方、父包/评测包、数据集/split/子集身份、指标协议、PPL/NLL/targets、历史复用或新测、完成状态和证据路径。不同数据配方或 checkpoint 不合并。

交付包括真实的 Qwen 固定主线结果、必要最终 test/C4、核心匹配比较与剩余缺项。历史 Llama 引用原路径并标复用；失败、未完成或改配方如实记录。混合 QAT 未启用、下游 ACC 未执行、NPU 未验证均单列，不包装成全部完成。

## 8. 新会话启动语

本文不自行启动服务器任务。由用户在执行会话发送以下指令后实施：

```text
你是 Phase5 多模型主执行。先读项目适用 AGENTS.md 和：
/home/dongpeiyan/projects/rotation-quant/runs/phase5/HANDOFF.md。

我授权按 handoff 完成 Qwen/Qwen3-1.7B 的必要架构适配、独立验证和正式 GPU 实验；
可创建独立 worktree/分支和兼容环境，保留旧源码、环境与 Phase2/3/4 产物。

先复用已完成的 Llama WikiText test 与 C4 归因结果，再跑 Qwen seed42 固定主线：
Joint100 → down SP2 校准 → down W4 邻码 → SP2 收缩 → down 再适配 → WikiText-only QAT400。
落实 handoff 的核心匹配对照、代表模型消融与关键 seed，不重开历史搜索或自动增加模型。

validation 用于开发；必要最终包用同一份权重统一验收完整 WikiText-2 test 和既有固定 C4 子集。
不按测试集切换权重、重新校准或调参，不把 B100 的 INT8 格式对照冒充完整 Uniform-QAT400。
混合数据 QAT 仅保留为可选对照，本条指令不授权自动执行或替换默认主线。

GPU4 故障不使用，其余健康 GPU 按实际余量调度，不限制并行数量；避免 OOM 和影响他人，
保持全局 batch/token 预算。长任务用 tmux，验证只针对新增风险，不给每个实验加保险复跑。

建立独立 Phase5 goal，在当前执行会话推进，不额外设置 token/GPU 总时长预算。
结果写 runs/phase5，持续更新 STATUS/PLAN。客户端沿用用户指定的 GPT-6 Astra / high，
prompt 不代替实际客户端配置。不要自行提交推送或终止他人进程。

先报告有效源码、模型/环境选择与第一项具体改动，然后实际实施，不只重复给计划。
```

## 附：继承的官方参考来源

以下链接沿用用户附件，服务于架构和指标定义核对；本次文档整理未重新在线验证。服务器实验事实以第 2 节报告和各 run 记录为准，新 test/C4 结果来自用户提供的汇总，不虚构已读取报告全文。

- [S1] Qwen3-1.7B config：<https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json>
- [S2] Qwen3-1.7B 官方模型页：<https://huggingface.co/Qwen/Qwen3-1.7B>
- [S3] Transformers v4.51.3 Qwen3 实现：<https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py>
- [S4] 同版本 `tie_weights()`：<https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/modeling_utils.py>
- [S5] SpinQuant norm fusion：<https://github.com/facebookresearch/SpinQuant/blob/main/utils/fuse_norm_utils.py>
- [S6] Hugging Face PPL 说明：<https://huggingface.co/docs/transformers/en/perplexity>
