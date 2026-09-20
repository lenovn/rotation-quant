# Llama seeds43/44 后处理及 QAT 启动合并审核（2026-09-18）

结论：限定 PASS。四条已完成后处理链、逐层选择、恢复跨目录记录和 QAT 精确父包一致。QAT 仍在训练；不据此宣称400步完成或最终统计通过。

引用已通过的 [Joint/匹配初始化审计](LLAMA_SEEDS43_44_JOINT_20260918.md) 和 [seed43恢复审计](LLAMA43_READAPT_RESUME_20260918.md)，未重复检查大权重或 optimizer。

| seed | 方法 | 精确 QAT 父目录 | validation PPL | NLL |
|---|---|---|---:|---:|
| 43 | SP2 | `sp2-down-readapt-s43-r4` | 16.2149972123967 | 2.7859365678441694 |
| 43 | INT8 | `uniform-down-readapt-s43` | 45.19500691896424 | 3.8109866143417275 |
| 44 | SP2 | `sp2-down-readapt-s44` | 16.147560805984856 | 2.781769004576782 |
| 44 | INT8 | `uniform-down-readapt-s44` | 41.44517090665076 | 3.724371370732677 |

12个完成阶段均覆盖16层 down：round → range → readapt，逐级 parent 精确接续；本 seed Joint100 reference 一致。每个 seed 六阶段数据完全相同：32个2048训练完整窗，24拟合/8选择，49152/16384行、16376选择 targets。所有逐层 selected NLL 等于候选中最小训练 NLL，层间和阶段间选择 NLL 精确相接；不使用 validation 选候选。

范围收缩每层均精确六档 `[1,.875,.75,.5,.25,.125]`，共96候选/阶段。round/readapt上限 `[512,2048,8192]` 加父候选0：SP2各阶段64候选；Uniform43首轮和Uniform44首轮/再适配各62，其余64。三个提前收敛条目为 Uniform43 首轮 layer1 `[0,41]`、Uniform44 首轮 layer1 `[0,28]`、Uniform44 再适配 layer1 `[0,33]`，属于此前已审的继承坐标算法收敛行为，不能写成每层都执行满8192。

seed43 SP2 再适配合并旧目录 `sp2-down-readapt-s43` 的索引0–3及 `sp2-down-readapt-s43-r4` 的索引4–15，16层无重复/缺失；新 result.completed_targets 完整，new_module_results 恰为后12层。最终包、summary evaluation_package 和 QAT parent 均使用 **-r4** 目录。旧任务索引4未提交计算的重复消耗沿用恢复报告说明。

12份 validation 格式描述分别为112 W4 / 96 INT8 +16 SP2或112 W4 /112 INT8；本次只核对记录，不重新读取大包做格式张量验证。summary四行精确使用官方模型名、seed43/44、SP2-PTQ-parent或Uniform-PTQ-parent、validation、252728 targets；PPL/NLL、父包、评测包和原始证据路径逐项一致。这些是开发父包记录，不进入QAT最终结果。

四个 QAT launch/settings 仅输出目录、各自父包和本 seed reference 不同，其他参数完全一致：WikiText-only、400步/global8、2048长度、data_start800、Adam weight_lr1e-5、relative_scale_lr0.001、warmup10、temperature1、CE权重0.1，原生未旋转冻结BF16 teacher，target_ppl=None，仅400步validation。全部正式旧Python/Transformers4.44.2，计划有效输入6553600、预测6550400 token visits。每项仅读前3条及文件尾最新完整training记录，实际每步8窗×2047 targets，窗序号按1180训练窗循环，与本seed WikiText缓存一致；未扫描完整训练或读取optimizer。QAT data.calibration_length=128为既有校准元数据，实际训练日志为2048窗，不是128训练长度。

| QAT run | 启动 PID | GPU | 本次抽样 steps |
|---|---:|---|---|
| `sp2-qat400-s43` | 3925847 | [0] | [1, 2, 3, 100] |
| `uniform-qat400-s43` | 3893776 | [1] | [1, 2, 3, 139] |
| `sp2-qat400-s44` | 3984664 | [3] | [1, 2, 3, 43] |
| `uniform-qat400-s44` | 3984670 | [7] | [1, 2, 3, 43] |

没有发现需修应用或summary的差异。无GPU/PPL/架构suite/复tokenize，无应用或产物修改。完整路径、逐层证据目录和精确数字见 [机器报告](LLAMA_SEEDS43_44_POSTPROCESS_20260918.json)。
