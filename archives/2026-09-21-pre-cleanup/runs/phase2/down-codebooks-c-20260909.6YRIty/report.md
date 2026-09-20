# C RTN：down 输入 A8 码本第一轮实验

本轮实际完成：INT8、PoT、SP2 公平尺度校准；相同输入下逐层损伤；全部 down 同时替换的 PPL。
独立验收见 [verification.md](verification.md)。原始结果见 [summary.json](results/summary.json)，逐层长表见 [local_damage.csv](local_damage.csv)。

## 权重选择与固定项

用户要求 C 优于 B 时改用 C。同一 WikiText-2 test 8×2048、非 down-A8/down-A16 配置下，已有结果为：

| 来源 | B PPL | C PPL |
| --- | ---: | ---: |
| GPTQ | 15.15180779 | 15.31429577 |
| RTN | 15.02653885 | **14.65525055** |

因此选择 C 的 **RTN** 权重及其配套 R、非 down SA。C 的 GPTQ 不优于 B。
权重：`/home/dongpeiyan/projects/rotation-quant/runs/phase2/learned-sw-c-20260909.ByFYAM/C/rtn/w4_rtn_model.pt`。
R：`/home/dongpeiyan/projects/rotation-quant/runs/phase2/learned-sw-c-20260909.ByFYAM/C/rotation/R.bin`。
SA：`/home/dongpeiyan/projects/rotation-quant/runs/phase2/learned-sw-c-20260909.ByFYAM/C/rotation/quant_scales.pt`。

本轮重载后 down-A16 PPL 精确复现 14.6552505493。96 个非 down SA 与 C 导出值一致，实验结束后仍一致；不重新量化权重、不更新 R、SA 或 SW。

## 全部 down 同时替换

测试口径：WikiText-2 test 前 8×2048，batch 1，prefill，KV16。PPL 越低越好。

| down 输入格式 | PPL | 相对 down-A16 的 PPL 增量 | NLL 增量 |
| --- | ---: | ---: | ---: |
| down_a16 | 14.65525055 | +0.00000000 | +0.00000000 |
| int8 | 42.35678101 | +27.70153046 | +1.06132986 |
| pot | 16.85359573 | +2.19834518 | +0.13976536 |
| sp2 | 15.79659176 | +1.14134121 | +0.07499553 |

在本轮校准配置下，SP2 是三种 A8 候选中最好的，显著优于均匀 INT8；但相对 down-A16 仍增加约 1.14134 PPL，尚未达到参考精度。

## 相同输入下的单层损伤

输入均取 C RTN、非 down-A8、down-A16 的同一参考轨迹；候选量化误差不传播到后续层。
采用独立 WikiText-2 validation 前 8×2048 的全部 token，表中为 `||(Q_BF16(X)-X) W4^T||² / ||X W4^T||²`（FP32 线性响应），以百分比显示。层编号从 0 开始。

| Layer | INT8 输出 NMSE (%) | PoT 输出 NMSE (%) | SP2 输出 NMSE (%) | 最低输出 MSE |
| --- | ---: | ---: | ---: | --- |
| 0 | 10.31745 | 2.48804 | 8.14257 | pot |
| 1 | 18.13555 | 21.80715 | 11.77431 | sp2 |
| 2 | 1.07636 | 3.21002 | 0.45245 | sp2 |
| 3 | 0.67699 | 3.33119 | 0.41895 | sp2 |
| 4 | 0.76603 | 3.34704 | 0.42166 | sp2 |
| 5 | 0.52913 | 3.48932 | 0.44107 | sp2 |
| 6 | 0.32911 | 3.47765 | 0.37959 | int8 |
| 7 | 0.41004 | 3.62858 | 0.40139 | sp2 |
| 8 | 0.72573 | 3.63393 | 0.42941 | sp2 |
| 9 | 0.88975 | 3.73962 | 0.47071 | sp2 |
| 10 | 1.26116 | 3.61784 | 0.50807 | sp2 |
| 11 | 1.26740 | 3.59795 | 0.54230 | sp2 |
| 12 | 1.42743 | 3.70555 | 0.57118 | sp2 |
| 13 | 1.01623 | 3.51248 | 0.47642 | sp2 |
| 14 | 1.31033 | 2.82931 | 0.71321 | sp2 |
| 15 | 4.81662 | 1.57643 | 1.07271 | sp2 |

SP2 在 14/16 层最优；PoT 在 layer 0 最优，INT8 在 layer 6 最优。逐层最佳格式的混合配置尚未评测，不能由单层最优推定其 PPL。

Layer 1 的均匀 INT8 alpha=462.816210，validation 非零输入被量化为零的比例为 99.999933%。这显示出范围与主体精度的冲突；尚不能将整体 PPL 损失归因于该层，仍需单层替换/恢复的 PPL 对照。

## 公平性与限制

- 所有格式共享 32 个不重叠 train 2048-token 窗口，每窗口抽样 64 个完整 token 向量（共 2048 行）；相同层内三种格式输入完全相同。完整窗口最大值确定搜索范围。
- 每格式每层 33 个粗搜点加 17 个细搜点，统一输出 MSE 目标；各自选择最优 alpha。alpha 搜索范围为完整校准输入 max/2^16 到 2max。校准不使用 validation 或 test loss。
- PoT layer 13、15 选中 2max 上界；结果不是更宽搜索范围下的 PoT 最优证明。
- 校准使用抽样 token 行，可能遗漏影响大的稀有 token；validation 局部指标使用全部 token。单层误差与最终 PPL 不是等价指标。
- INT8 保留项目 [-128,127] 共 256 数值；PoT 255 唯一数值；SP2 依论文式 (8) 采用 sign/4/3 编码，存在重复表示，仅 187 个唯一数值。没有用 APoT 代替 SP2。
- FP32 构建和投影，量化输出转 BF16 进入现有 GEMM；这些是 fake-quant 精度结果，不是原生 SP2/PoT 后端实现或手机延迟结果。

## 复现与代码

运行 `bash scripts/phase2/42_run_down_codebooks_c_local.sh`，每次使用新的输出目录。
算法来源、适配差异及数据设置见 [down_codebooks.md](../../../scripts/phase2/down_codebooks.md)。
校准逐点结果：`results/calibration_layer_00.json` 至 `results/calibration_layer_15.json`；独立下游尺度：`results/down_scales.json` 和 `results/down_scales.pt`；逐层原始指标：`results/local_damage.json`。
