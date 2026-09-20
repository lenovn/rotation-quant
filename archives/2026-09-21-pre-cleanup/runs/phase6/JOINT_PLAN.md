# Phase6 联合训练内部闭环

2026-09-21 已授权执行。从原始预训练权重重新初始化 R 和尺度，不延续 B100；原始权重冻结。R1/R2、W4 per-channel SW、非 down INT8 和 down INT16 per-tensor SA 联合训练。浮点 fakequant 质量实验，KV BF16，不声称 NPU 原生性能。

- 模型：Qwen3-0.6B/1.7B、Llama3.2-1B/3B-Instruct；seed42。
- 固定 WikiText2 train 前800个2048窗口，循环，global batch8；从这800窗中seed42抽32个完整窗口初始化校准。validation 只选预算；test/C4和下游任务不参与训练或选择。
- R SGDG/Cayley LR1.5；尺度 Adam 初始scale均值乘0.001、eps1e-12；warmup10、cosine horizon512；纯token NLL，梯度范数裁剪1。
- Llama1B full joint先512步，0/64/128/192/256/320/384/448/512评测完整validation。选最早 >=128 且至少两个后续检查点、后续最佳PPL相对改善<=0.2%的T；否则512。0.2%是预先规定的实用容差，不是论文通用阈值或统计不显著。
- 后续三个模型及Llama两个训练消融跑T，仍使用horizon512余弦前缀。Llama主线直接复用T检查点。
- Llama消融：无优化复用主线step0；仅R训练且末尾重校准SW/SA；完整联合；固定down-SA且其余组联合。后两组不覆盖学到的尺度。
- 四主线及相应BF16、Llama消融评测：WikiText2完整test、既有固定C4 validation子集、BoolQ/PIQA/SIQA/HellaSwag/WinoGrande/ARC-E/ARC-C/OBQA 0-shot及等权Avg。复用完全匹配结果；不加MMLU/外部方法/多seed。
- 长实验tmux，保存日志/中间检查点。GPU每卡总显存<90%；可共享余量足够的卡，优先空闲卡并观察计算负载。不得结束他人任务。
- 实现：scripts/phase6/joint.py；独立verifier检查真INT16前向/梯度、冻结权重、旋转和导出冷加载。保留旧代码和已完成Phase6精度诊断。

状态：实现和独立测试进行中，尚无新训练结果。
