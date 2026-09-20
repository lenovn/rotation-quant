# Llama Uniform 两个中间阶段审计

2026-09-18。**只读产物审计 PASS；不是最终 Uniform-PTQ/QAT 验收。** 未启动测试、GPU或PPL复跑，未等待第三步。

| 阶段 | train-selection NLL 起点→选定 | validation PPL | validation NLL |
| --- | --- | ---: | ---: |
| uniform-down-round-s42 | 6.445073606790942 → 5.994491577236984 | 722.8870870577282 | 6.583253037151634 |
| uniform-range-s42 | 5.994491577236984 → 3.549447552560451 | 44.19421129635554 | 3.7885938143610356 |

两者均 completed、无 failure，16 个 down 层全部完成。validation 均为 252852 inputs / 252728 targets，分段重算 NLL/PPL 与记录精确一致。

核对结果：

- 父包链为 `uniform-initial-s42 → uniform-down-round-s42 → uniform-range-s42`，settings/result/保存包 metadata 一致。首包继承历史 B100 与匹配 INT8 overlay；邻码浮点参考为历史 Joint100 同R `checkpoint-0100/state.pt`，范围阶段不需要参考权重。
- 两阶段从相同历史32个 **WikiText train** 索引取完整2048窗，24 fit /8 selection，49152/16384行；选择依据16376个train targets的NLL。逐层保存的选定值确为已评分候选中最低NLL，前缀首尾连续。源码先完成train-selection再运行完整validation；没有test/C4用于选择的入口。
- 保存包实际均为112个 W4 records及112个 `format=int8` 输入量化记录，包含全部16个down。只读mmap核对确认所有W4 scales、非down整数码/SA保持不变；首轮邻码全部输入scale保持不变；范围阶段全部整数码保持不变，down尺度与选中倍率一致。不是SP2包装后错误标成INT8。
- 邻码声明预算为 parent/512/2048/8192，15层保留4个快照；layer1因下面的继承停止分支只保留0/40，共62个候选快照。接受11层。round模式settings中历史alpha扩张默认值不被执行，不构成额外范围搜索。
- 范围阶段显式传入 `1/.875/.75/.5/.25/.125`，每层恰好6条记录，共96条（包含父候选）；接受8层。没有增加候选或借用validation选范围。

**layer1 的 `selected_step=40` 是继承算法提前收敛的快照标签。** `source/fixed_grid_rounding.py` 与当前公共实现及历史 `seq-b100-down-round-20260914b/source/fixed_grid_rounding.py` 逐字节相同。第62–64行在最佳列增益 `gains[j] >= -1e-15` 时先 `snapshot(step)` 再 break；40不在milestones，因而只有这个分支能生成该记录。它表示第40轮检测不到超过阈值的可改善列，保留前39轮后的状态；不是新加40步搜索预算或人为少给INT8候选。该快照随后仍通过heldout MSE条件与真实train-selection NLL接受，选定NLL从6.399005847232703降为6.177020119016253。日志未单独保存该轮gain数值，终止原因由实际快照源码与唯一分支对应确认，没有重新计算梯度。

summary两条新行数值、父包、评测包、split=validation、down_format=INT8均匹配。最初model别名 `llama32-1b` 已由主执行统一为 `meta-llama/Llama-3.2-1B-Instruct`，本次再次读取已确认修正。两条阶段名分别为uniform-down-round/uniform-range，不能填成完成的Uniform-PTQ或Uniform-QAT400最终行。

机器核对记录：`uniform_postprocess_checks_20260918.json`；只读脚本：`check_uniform_postprocess.py`。脚本首次将范围阶段也误要求浮点reference，修正这项审计假设后通过；应用/实验无需修复或重跑。

## 追加：完整 Uniform-PTQ seed42 验收

**最终 PTQ 证据审计 PASS。Uniform-QAT 尚不属于此结论。** 第三阶段 `uniform-down-readapt-s42` 与 test/C4 均 completed、无 failure；只读审计，无新测试/GPU/PPL复跑。

同一最终包 `runs/phase5/llama32-1b/uniform-down-readapt-s42/static_w4a8.pt` 的三项结果：

| 数据 | PPL | NLL | targets |
| --- | ---: | ---: | ---: |
| WikiText validation | 41.5470707013954 | 3.7268270182480743 | 252728 |
| WikiText test | 38.47028685874723 | 3.6498861734233596 | 288934 |
| 固定 C4 validation 子集 | 81.13316772229233 | 4.396091850662515 | 2096128 |

- 第三阶段仍为32个相同train窗、24/8 fit/selection、parent/512/2048/8192预算及同Joint100浮点参考；父包为 `uniform-range-s42`。16层全部完成；layer1快照0/109来自同一继承收敛分支，其他15层均0/512/2048/8192。train-selection NLL从3.549447552560451降至3.484595846334799，逐层仍取候选最小值。
- 最终包实际为112个W4及112个INT8输入。第三阶段所有activation scales/W4 scales与父包精确相等，非down整数码不变。
- validation源码快照先保存该包、再冷加载该包评分。外部test/C4的settings、result和model身份均指向此同一包，无临时overlay。外部两次量化器before/after全部112个字典、每个tensor精确相等；两次评测的起始状态相互相同，全部input scale等于最终包记录。KV16、use_cache=False和FP32 RoPE保持。
- test tokens与历史Llama BF16/QAT test逐元素相等，metadata和分段目标边界一致；C4使用原固定token文件，逐元素/metadata/分段口径与历史BF16、SP2-PTQ及QAT400一致。validation复用已核对历史数据metadata及冷加载执行路径，本次未重新tokenize validation。
- 三项分段NLL/PPL重算精确相等，summary的三个 `uniform-ptq` 行（validation/test/C4）与raw指标、同包路径、INT8格式、无QAT数据身份一致。没有把初始B100 INT8格式诊断当完整PTQ。

新增机器证据：`uniform_ptq_final_checks_20260918.json`；只读脚本：`check_uniform_ptq_final.py`。这完成了Llama seed42完整Uniform-PTQ这一项；不能据此标记Uniform-QAT400或整体Phase5完成。
