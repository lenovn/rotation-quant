# Phase6 联合训练内部闭环

2026-09-21 已授权执行。从原始预训练权重重新初始化 R 和尺度，不延续 B100；原始权重冻结。R1/R2、W4 per-channel SW、非 down INT8 和 down INT16 per-tensor SA 联合训练。浮点 fakequant 质量实验，KV BF16，不声称 NPU 原生性能。

- 模型：Qwen3-0.6B/1.7B、Llama3.2-1B/3B-Instruct；seed42。
- 固定 WikiText2 train 前800个2048窗口，循环，global batch8；从这800窗中seed42抽32个完整窗口初始化校准。validation 只选预算；test/C4和下游任务不参与训练或选择。
- R SGDG/Cayley LR1.5；尺度 Adam 初始scale均值乘0.001、eps1e-12；warmup10、cosine horizon512；纯token NLL，梯度范数裁剪1。
- Llama1B full joint先512步，0/64/128/192/256/320/384/448/512评测完整validation。选最早 >=128 且至少两个后续检查点、后续最佳PPL相对改善<=0.2%的T；否则512。0.2%是预先规定的实用容差，不是论文通用阈值或统计不显著。
- 后续三个模型及Llama两个训练消融跑T，仍使用horizon512余弦前缀。Llama主线直接复用T检查点。
- 用户在执行中再次明确确认：保持cosine512，在T截断，不改cosine T，也不重新训练。为并行加速，其余训练可先跑共同曲线前128步并暂停；参数、SGDG/Adam状态和随机状态连续恢复，训练窗口次序保持不变。
- 执行调整：按用户提出的小模型优先建议，Qwen0.6B从已有128步连续恢复至512，提前提供完整validation曲线和候选预算；不重跑前128步，不改cosine512。现有Llama1B/3B继续并行。0.6B曲线不能单独证明其他模型已收敛；当前自动选择器仍依据既定Llama1B规则，获得0.6B曲线后结合跨模型validation检查预算适用性，任何进一步选步规则调整须显式记录，不用test/C4选择。
- Llama消融：无优化复用主线step0；仅R训练且末尾重校准SW/SA；完整联合；固定down-SA且其余组联合。后两组不覆盖学到的尺度。
- 用户进一步授权Qwen1.7B从128步并行续训至256步，保留cosine512及优化器/随机状态；这是提前补充该模型的validation曲线，不代表最终统一预算已定。最终T大于256时继续恢复，小于等于256时复用相应检查点。
- Llama1B主线320步validation PPL=15.10456203551094，相对256步仍改善0.7684%，既定规则已排除128/192/256候选。因此将固定down-SA和仅R两个消融从128并行续至320（GPU2/6），提前完成最终预算必需的前缀；最终T未定，仍保持cosine512，T>320时继续恢复。
- 四主线及相应BF16、Llama消融评测：WikiText2完整test、既有固定C4 validation子集、BoolQ/PIQA/SIQA/HellaSwag/WinoGrande/ARC-E/ARC-C/OBQA 0-shot及等权Avg。复用完全匹配结果；不加MMLU/外部方法/多seed。
- 用户补充：新路线精度/能力结果须与既有SP2路线对照。复用Llama1B、Qwen1.7B的蒸馏前后两包原始结果；较大差距先排查真INT16、尺度/权重导出、冷加载、评测口径，再区分训练预算和方法差异。旧最终SP2含后处理与蒸馏，不标成等预算消融；不因此擅自追加蒸馏或用测试分数选择配置。
- 长实验tmux，保存日志/中间检查点。GPU每卡总显存<90%；可共享余量足够的卡，优先空闲卡并观察计算负载。不得结束他人任务。
- 实现：scripts/phase6/joint.py；独立verifier检查真INT16前向/梯度、冻结权重、旋转和导出冷加载。保留旧代码和已完成Phase6精度诊断。

状态：11项实现CPU测试及3项恢复/预算测试PASS；真实GPU三臂首步更新范围、冷加载一致性已独立PASS。Llama1B主线、两个消融及两个Qwen模型训练已tmux启动；Llama3B从公开ModelScope源下载后自动启动。主线实际目录llama1b-joint512-s42-r1，旧llama1b-joint512-s42仅为修复设备检查前的初始化记录，初始参数/validation已复用。后台调度schedule.py会记录已死亡但未完成任务，主agent处理异常；不能将等待或源码测试当作完整实验完成。
