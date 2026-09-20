# Qwen seeds43/44 后处理合并审计（2026-09-18）

结论：**12 阶段及12条开发summary记录全部限定 PASS**，无待完成或待登记项。初始化与Joint匹配引用 [既有审计](QWEN_SEEDS43_44_JOINT_20260918.md)。

| seed | 方法 | 阶段 | 层数 | 实际候选数 | validation PPL | NLL |
|---|---|---|---:|---:|---:|---:|
| 43 | sp2 | down-round | 28 | 112 | 15.910624656710956 | 2.766987103471548 |
| 43 | sp2 | range | 28 | 168 | 15.47834976229269 | 2.739442257975861 |
| 43 | sp2 | down-readapt | 28 | 112 | 14.985970044814936 | 2.707114433062164 |
| 43 | uniform | down-round | 28 | 110 | 20.170829327393548 | 3.0042374679970743 |
| 43 | uniform | range | 28 | 168 | 18.364192021237717 | 2.9104026827273044 |
| 43 | uniform | down-readapt | 28 | 110 | 18.22110026571323 | 2.9025802778083905 |
| 44 | sp2 | down-round | 28 | 112 | 16.05017436240812 | 2.775719713218456 |
| 44 | sp2 | range | 28 | 168 | 15.763760989180486 | 2.757713697404855 |
| 44 | sp2 | down-readapt | 28 | 112 | 14.958610912101744 | 2.705287114766046 |
| 44 | uniform | down-round | 28 | 110 | 19.60780237716365 | 2.975927567490997 |
| 44 | uniform | range | 28 | 168 | 18.817133157121283 | 2.9347677929641733 |
| 44 | uniform | down-readapt | 28 | 110 | 18.423304287656563 | 2.913616400605703 |

每个seed两格式各阶段使用完全相同32个2048完整训练窗，24拟合/8选择：49152/16384行、16376选择targets。全部settings指向本seed Joint100参考；round→range→readapt的parent逐级接续，已完成包metadata保留精确父包metadata。range不读取FP参考作拟合，此处reference只是同seed provenance。

已完成阶段均逐项验证28层覆盖、逐层previous/selected训练NLL连续、父候选以及在已评分候选中选择最小训练NLL；跨阶段initial NLL精确接续上阶段selected NLL。round/readapt上限512/2048/8192，父候选0保留；range每层六档1/.875/.75/.5/.25/.125，共168候选/阶段。不存在基于validation选候选的记录。

round/readapt继承训练heldout-MSE前筛：候选heldout_mse不低于父候选时train_nll=null、不评测选择NLL；每一null均逐条验证此条件。其余候选再以8训练窗NLL决定接受。因此候选生成总数不等于实际NLL调用数，不能把null误称失败或漏测。

实际提前收敛条目（沿用此前已审坐标算法；不能称所有层均执行满8192）：
- `uniform-down-round-s43` / `model.layers.2.mlp.down_proj`：[0, 13]。
- `uniform-down-readapt-s43` / `model.layers.2.mlp.down_proj`：[0, 13]。
- `uniform-down-round-s44` / `model.layers.2.mlp.down_proj`：[0, 13]。
- `uniform-down-readapt-s44` / `model.layers.2.mlp.down_proj`：[0, 13]。

对已完成包进行CPU mmap有限字段检查：每包196 W4；SP2分支168 INT8+28 SP2，Uniform分支196 INT8，格式和config延续父包。所有SW不变；非down SA不变；round/readapt全部SA不变；range的down尺度精确对应选中alpha和格式的存储规则。本次没有重复全W-code或高精度张量比较，不扩大为新的完整状态审计。

所有已完成validation分段按log(PPL)×targets重算与原始NLL/PPL精确相同，262208targets；12条summary行的官方模型名、seed、stage、父包/评测包路径、PPL/NLL/targets逐项相同。down-round/range与PTQ-parent均为开发记录，不是QAT最终结果或外部验收。

pending阶段只读settings/data与progress快照，不检查在途包或宣称28层完成；详细快照保存在JSON。正式Qwen环境Transformers4.51.3，与Llama旧4.44.2不同。未GPU/PPL/架构suite/复tokenize，无optimizer读取，无应用/summary/实验产物修改。机器证据：[QWEN_SEEDS43_44_POSTPROCESS_20260918.json](QWEN_SEEDS43_44_POSTPROCESS_20260918.json)。

补充核对：seed43的SP2-PTQ-parent及Uniform-PTQ-parent两行已登记，精确匹配原始validation、父包与评测包。已清除summary pending；本次只补核这两行，未重审12阶段，保留heldout-MSE前筛及early-convergence证据。
