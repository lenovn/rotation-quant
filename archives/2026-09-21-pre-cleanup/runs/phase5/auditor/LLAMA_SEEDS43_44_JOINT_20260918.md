# Llama seeds43/44 Joint100 配对身份审计

2026-09-18。**限定审计 PASS：两组初始化、Joint100 和匹配初始 INT8 包均已完成。** 本次只读 CPU 核对新增 seed 的原始 data/state/训练日志、包和校准记录；未运行 GPU、PPL、tokenize、模型重新初始化或架构测试，也未访问在途后处理产物。机器证据见 `LLAMA_SEEDS43_44_JOINT_20260918.json`。未将线索文件 `seed_initialization_43_44.json` 当作证明替代原始证据。

## Seed 实际入口与初始化加载

检查路径为 `runs/phase5/llama32-1b/{init-s43,init-s44,joint100-s43,joint100-s44,uniform-initial-s43,uniform-initial-s44}`。六个 run 的 progress 都为 completed，无 failure.json；正式记录均使用旧 Llama 环境 Transformers4.44.2 和相同本地 Llama 模型。

- 各 init/Joint/Uniform settings 的实际 source seed 均为对应43或44；init `initial.pt` 的 seed metadata、Joint `data.json` 也一致。
- 分别按 `torch.randperm(1180, generator=manual_seed(seed))[:32]` 独立生成校准索引，与各 seed 原始 data.json 的32个索引逐项相同。43/44索引不同；索引全在 WikiText train 可用窗口内。
- 各 seed 的 **241个初始参数**与自身 Joint `checkpoint-0000/state.pt` 全部逐字节相同，且 checkpoint-0 metadata 指向自身 `init-s{seed}/initial.pt`，证明实际加载了对应初始状态。
- 两组各 **17个初始 R** 均与历史 seed42 的 `runs/phase3/common-init-20260914a/initial.pt` 不同；seed43与44的17个 R 也全部彼此不同。各组全部17个 R 又在 Joint100 后发生改变。
- 初始化、Joint 和 Uniform 保存的 run/common/architecture/quantization/uniform_baseline 源码与已审 seed42 版本逐字节一致。seed入口仍是既有环境 seed→`set_seed(SEED)`/独立校准 generator，不是新增算法分支。

以上是原始参数、独立索引生成和启动/源码记录的联合证据；没有重建一次随机初始化来声称完整的 RNG 轨迹复现。两组训练文本相同，只改变所规定的随机初始化与校准采样 seed。

## 固定训练预算与 validation 来源

每组 `training.jsonl` 都恰有更新1–100，无缺失或重复；每步8个微批，2048输入/2047预测目标，实际窗口索引依次0–799。每组预算为 **1,638,400输入 token visits /1,637,600预测目标**，global batch8。两组均为 route B、100步 schedule、无 resume，父初始状态路径正确。

每一步损失和梯度范数均有限，记录17个 R、96个非down SA、112个 SW 梯度，并在这些组中有实际参数变化。down SP2 在 route B 训练时无梯度/更新，随后在冻结模型上做既有 train-only 校准；不能把这100步称为直接训练 SP2 scale。原始数据记录保持2,435,022训练 tokens、1,188完整窗，并排除末8窗作为训练循环候选范围外的保留窗口。

| Seed | 原始 validation PPL | 原始 validation NLL | Targets | 原始来源 |
|---|---:|---:|---:|---|
| 43 | 17.32269586594728 | 2.8520175414808633 | 252728 | joint100-s43/checkpoint-0100/validation.json |
| 44 | 17.117556465258705 | 2.8401046306937046 | 252728 | joint100-s44/checkpoint-0100/validation.json |

两组分段 PPL→log→targets加权 NLL→exp 的重算与原始值精确一致；summary 的官方 `meta-llama/Llama-3.2-1B-Instruct`、seed、`B100-initial-SP2`、精确包路径和 PPL/NLL 均对应正确。这里只接受初始包 validation，不把它当后处理或 QAT 最终结果。

## 每个 seed 内的 SP2/INT8 匹配身份

各 `uniform-initial-s{seed}/static_w4a8.pt` 都是完整物化的初始 INT8 包，精确父包为同 seed `joint100-s{seed}/checkpoint-0100/static_w4a8.pt`。每个 seed 内直接比较得到：

- **112组 W4 packed码、SW及其他权重记录字段逐字节相同**。
- **96个非down SA相同**，模型 config 相同。
- **74个高精度状态张量逐字节相同**，包括 embedding/head、层norm和RoPE；相同的未使用 NaN占位缓冲按原始字节比较。
- SP2初始包的实际激活格式是 **96 INT8 +16 SP2**；Uniform初始包是 **112 INT8**。仅16个 down 激活格式及对应校准范围改变。

冻结包中的 R 已融合到权重与高精度状态中，没有独立在线 R 参数；上述全部权重和高精度状态相同证明每个 seed 内共享同一旋转基底，而不是重新优化了另一套 INT8 R。

两种格式均使用同 seed 的32×128 train校准窗口，共4096输入 tokens。每层 capture 有512采样行，16层的 full_absmax、采样行数与 SP2记录精确相同。两种格式各层都实际记录 **33粗搜 +17细搜 =50候选**；粗搜 log2倍率范围[-16,1]、围绕自身粗搜最优点的细搜规则、选中最小 output-MSE 候选，以及最终 INT8 scale 与选中值均通过检查。各 Uniform校准前/后的112个父量化器状态保持相同。

capture数组未保存，因此这里不冒称逐元素比较重新捕获的激活；证据是相同源码、索引、父模型不变量，以及每层精确相同的捕获计数/边界。没有为验证这些信息重新 capture 或校准。

该 PASS 仅覆盖 seeds43/44 的初始与配对身份。当前后处理、PTQ外部验收、QAT400和三seed汇总均不在本次范围内，也没有以 seed42结果补齐缺失阶段。
