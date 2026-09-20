# 最新 C 实验 Atlas 与权重 per-channel 映射分析

结论：**部分权重 channel 有轻度厚尾，但本次没有发现应直接改用 SP2-W4 的 channel。保留当前 C RTN 权重及配套 R/SA/SW。** 权重分布与 down 输入激活的极端长尾是两个问题。

## 输入与 Atlas

- C 目录：`../../phase2/learned-sw-c-20260909.ByFYAM/C/`。
- R：`C/rotation/R.bin`；配套尺度：`C/rotation/quant_scales.pt`。
- 实际权重：`C/rtn/w4_rtn_model.pt`。使用 RTN，不是同目录 GPTQ。
- [BF16 Atlas](bf16/)：原始模型 fuse LayerNorm 后融入 C 的 R1/R2；W16/A16，无 R3/R4。
- [实际 RTN Atlas](rtn/)：经原 PTQ 入口重载完整 C RTN checkpoint 与 SA；W4/static non-down A8/down A16/KV16。激活在各输入量化器之前采集，包含上游量化误差传播。
- 两套均使用 WikiText-2 train、seed 42、128 个不重叠 2048-token 窗口、batch 1、prefill、use_cache=False。各含 112 个权重来源、64 个激活来源；每套另含 112 个 Linear、7 个投影族和全模型的 channel-score 图。
- RTN 重载 PPL **14.655250549316406**，精确复现源 C RTN 结果。见 [runtime.json](rtn/runtime.json)。

训练 C 只更新 R/SA/SW，原 BF16 模型参数冻结。因此观察量化前分布时从原模型与 C-R 重建；观察已交付权重时直接读取 RTN checkpoint，不将 RTN 权重二次旋转。

## 每个输出 channel 内是否长尾

统计对象为 112 个 backbone Linear 的 `W[out_channel, :]`，共 **376,832 个 channel、973,078,528 个权重元素**，不含 embedding/lm_head。channel 长度分别为 2048 或 8192。

| C-R BF16 指标 | 中位数 | P90 | P99 | 最大 |
| --- | ---: | ---: | ---: | ---: |
| T = max(abs(W)) / P99(abs(W)) | 1.4158 | 1.6291 | 1.8836 | 2.8647 |
| B = P99(abs(W)) / P90(abs(W)) | 1.5686 | 1.6377 | 1.7293 | 1.9582 |
| 中心四阶矩 / 方差²（非 excess kurtosis） | 3.0153 | 3.2237 | 3.6491 | 5.7845 |
| 绝对值最大的约 1% 元素占该行能量 | 8.3445% | 9.2323% | 10.6937% | 15.4125% |

多数 channel 的峰度接近 3，尾部比例温和；部分 o_proj 行更厚尾。最大峰度出现在 layer 9 o_proj，最大 T 出现在 layer 14 down_proj。没有 T>3 的行，但 **3 只是便于描述的阈值，不是长尾分布的统计判定标准**。有限权重样本不能证明某种渐近尾部分布。

因此不是“所有权重严格高斯/完全没有长尾”，而是没有发现类似 down 激活那种极端的主体与尾部范围冲突。不要把整个矩阵混合直方图、不同 channel 的尺度差异与 channel 内长尾混为一谈。

实际 RTN 的最大 T 为 3.5097；其离散化可改变 P99 和尾比，不能据此倒推 BF16 权重更长尾。

[实际 channel 直方图](channel_examples.png) 展示每个投影族 T 最大的行，以及 layer 1 down 的 T 中位行。横轴按该行均值/标准差归一化，纵轴为 log count。选择信息及原始行见 [channel_examples.json](channel_examples.json)、[channel_examples.pt](channel_examples.pt)。全部行统计见 [BF16 channel 数据](bf16/weight_channel_summary.pt) 和 [RTN channel 数据](rtn/weight_channel_summary.pt)。

## 同位宽映射对照

INT4 采用项目数值约定 `[-8,7] * alpha/7`（16 值）；SP2 按 [Mix and Match 论文 Eq. (8)](https://arxiv.org/pdf/2012.04240) 使用 sign/2/1 编码，非负数值为 `{0, 1/8, 1/4, 1/2, 5/8, 3/4, 1}`，去重后 **13 个 signed 数值**。二者编码预算均为 4 bit，数值数量不同。原 A8 激活实验的 SP2 码本未用于本次 W4 对照。

从同一 C-R BF16 权重出发，每种格式每个输出 channel 独立拟合 alpha，以该行权重 MSE 为目标。相同搜索预算：alpha/max 从 2^-3 到 2^1 的 33 个对数粗搜点，再在最优邻域细搜 17 点；FP32 投影后转 BF16。所有行粗搜最优点均未落在两端。有限搜索不等于数学上的全局最优，也不包含格式专用重训或 GPTQ。

NMSE 定义为 `sum((Q(W)-W)^2)/sum(W^2)`，按实际元素数汇总 SSE/能量，不平均各行 NMSE。

| 权重族 | 当前 C RTN NMSE | MSE 拟合 INT4 NMSE | MSE 拟合 SP2 NMSE |
| --- | ---: | ---: | ---: |
| q_proj | 2.2511% | 1.1718% | 1.8877% |
| k_proj | 2.2407% | 1.1683% | 1.8831% |
| v_proj | 2.2445% | 1.1684% | 1.8836% |
| o_proj | 2.9448% | 1.4176% | 2.1989% |
| gate_proj | 2.2524% | 1.1715% | 1.8884% |
| up_proj | 2.2553% | 1.1733% | 1.8894% |
| down_proj | 3.0200% | 1.2590% | 1.9985% |
| 全部 | **2.7536%** | **1.2476%** | **1.9840%** |

**376,832 个 channel 中，SP2 的权重 MSE 胜过拟合 INT4 的数量为 0。** 相对当前任务学习得到的 RTN，SP2 的原始权重误差虽然更小，但它同时换了尺度，不能将此认作格式本身的收益。

逐模块数据：[mapping_summary.json](mapping_summary.json)；逐行 MSE、alpha、峰度及能量：[mapping_channels.pt](mapping_channels.pt)。

## 端到端检查

固定 C-R、C-SA、非 down static A8、down A16、KV16；每种候选均从同一 BF16 重建，替换全部 112 个权重矩阵。embedding、norm、lm_head 与原 C RTN 相同。WikiText-2 test 8×2048，batch 1，prefill；test loss 不用于 alpha 搜索。

| 权重处理 | PPL |
| --- | ---: |
| 原 C RTN，保留任务学习 SW | **14.65525** |
| INT4，按每行权重 MSE 重拟合 alpha | 17.12465 |
| SP2-W4，按每行权重 MSE 拟合 alpha | 20.17590 |

结果见 [ppl_mappings.json](ppl_mappings.json)，日志见 [ppl_mappings.log](ppl_mappings.log)。**更低的权重 MSE 不保证更低 PPL**；重拟合 INT4 已经劣于原 C，SP2 又劣于该 INT4 对照。R/SA/SW 是联合学习得到的配套结果，离开其任务目标单独优化权重重建，会改变模型误差的方向和传播。本轮只证明固定当前 C 后直接映射替换无收益，不排除另做格式专用训练的可能。

## 与 down 激活的区别及建议

实际 C RTN Atlas 中，layer 1 down 输入 max(abs)=**1080**、P99.99(abs)=**0.439453125**；其主体与极端值差距远大于上述权重行。旧 BF16 Atlas 的同位置在本轮 C-R 中仍为 max=1168、P99.99=0.427734375。

当前证据支持：

1. **保留 C RTN 的权重格式及 SW，不因直方图有尾巴就换 SP2。** 本轮没有找到可由权重 MSE 支持的 SP2 channel 子集。
2. 不把“按权重 MSE 重拟合 INT4 更准”当成下一步直接优化方案；本轮 PPL 已否定其直接替换收益。
3. 非均匀映射研究继续集中于 down 输入激活的极端范围冲突。已有 A8 SP2 结果是激活侧证据，不能转用为权重 SP2 的理由。

以上均为 GPU BF16 fake-quant 精度与分布测量，不含原生 SP2 packing/kernel、Snapdragon 延迟或 decode/KV 量化验收。论文的 FPGA 加速结果不能当成本项目手机后端收益。

## 复现脚本

- BF16 Atlas：现有 `worktrees/SpinQuant-distribution-experiment/experiments/distribution_atlas/collect.py`，显式传入上述 C-R 和新的 `--output-dir`；绘图使用同目录 `plot.py`。
- [analyze_weights.py](analyze_weights.py)：重建 BF16、读取 RTN、统计全部 channel、拟合两类码本。
- [collect_rtn.py](collect_rtn.py)：使用原 PTQ 入口、完整 C 权重/R/SA 参数，重载并生成实际轨迹 Atlas。参数与 [run_mapping_ppl.sh](run_mapping_ppl.sh) 一致，更换脚本及输出路径。
- [evaluate_mappings.py](evaluate_mappings.py)、[run_mapping_ppl.sh](run_mapping_ppl.sh)：固定 C 的端到端映射对照。
- [plot_channel_examples.py](plot_channel_examples.py)：确定性选择并画实际 channel 直方图。

脚本中的 OUT 为脚本所在目录。复跑时复制分析脚本到新目录并相应修改 shell 中的输出路径，避免覆盖本次产物；先产生 mapping_channels.pt 再执行映射 PPL。全部绘图和统计只新增在本实验目录，原 R、尺度、checkpoint 和 Atlas 脚本保持不变。
