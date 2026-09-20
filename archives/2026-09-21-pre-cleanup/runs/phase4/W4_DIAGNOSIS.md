# 当前 Phase3 PTQ 父包 W4 掉点成因诊断

日期：2026-09-15。16项新GPU完整WikiText2 validation；原父包静态A8/SP2结果复用。

## 结论及证据强度

当前主要剩余误差定位到 **MLP 的 W4 离散表示误差**（整数码、共享行尺度格点及BF16反量化的组合）；不是普遍超范围截断，也不是仅layer1 down。属于对当前父包、当前R、已检验量化规则和数据的因果干预结论，不是对所有W4方法的最优性证明。

1. 同R W16/all-A16 NLL=2.613853954849，当前W4/all-A16 NLL=2.744512915240，静态父包NLL=2.779600150934。先去激活量化后仍有 78.83% 的NLL差距；比例限定于这个干预顺序，不是唯一可加分解。
2. 恢复MLP全部48矩阵后PPL=14.00216485，去掉上述W4条件差距的 80.59%。down、gate/up分别有明显影响；注意力也有剩余误差。恢复layer1 down只由15.5570到15.3995，其余down恢复到14.8263；旧单层主导证据不适用于当前父包。
3. 从原q/SW直接去掉MLP截断分量，PPL15.5601，与父包15.5570持平；去掉格点分量，PPL13.9989，接近恢复MLP到W16的14.0022。这是任务干预证据，支持范围截断不是主项。13.9989与14.0022的小差值不用于宣称保留截断更好。
4. 当前MLP超出[-8SW,7SW]范围的参考权重为 101388/805306368（0.01259%）；截断SSE只占总权重SSE约0.00204%。这是对当前所学SW的针对性统计，不重做历史Atlas；SSE本身不作为任务归因。
5. 同一RTN求解器、同一未量化参考：MLP W4 per-channel 16.4540，W4 group128 16.2742，W8 per-channel 14.0226，MLP W16 14.0022。支持在本组条件下位宽降低造成的离散近似是强影响，而group128改善有限。不能排除group32、更小分组或优化分组尺度的收益。三种RTN之间可比较粒度/位宽；与现有已优化父包15.5570的比较含求解变化。
6. 固定非down INT8，SP2条件NLL代价：原父包0.032204079，恢复全部down权重后0.037118168，差0.004914089。存在交互；恢复down也改变后续输入，因此这个差值不能单独证明整数码专门学到了SP2补偿。

## 完整结果

| 条件 | NLL | PPL | 产物目录 |
|---|---:|---:|---|
| 原PTQ父包静态A8/SP2（复用） | 2.779600150934 | 16.112577061 | ../phase3/seq-b100-sp2-refine-down-20260914a |
| all16:none | 2.744512915240 | 15.557034492 | diag-activation-20260915a |
| down16:none | 2.747396072297 | 15.601952588 | diag-activation-20260915a |
| all16:all | 2.613853954849 | 13.651562103 | diag-activation-20260915a |
| all16:down | 2.686289921350 | 14.677121507 | diag-families-20260915a |
| all16:attention | 2.713902355379 | 15.088039674 | diag-families-20260915a |
| all16:gateup | 2.694081213725 | 14.791921893 | diag-families-20260915a |
| all16:mlp:rtn4c | 2.800570961709 | 16.454038716 | diag-mlp-format-20260915a |
| all16:mlp:rtn4g128 | 2.789579545532 | 16.274175813 | diag-mlp-format-20260915a |
| all16:mlp:rtn8c | 2.640669879414 | 14.022593909 | diag-mlp-format-20260915a |
| all16:down1 | 2.734338045536 | 15.399546263 | diag-mlp-local-interaction-20260915a |
| all16:down_except1 | 2.696401072794 | 14.826276901 | diag-mlp-local-interaction-20260915a |
| all16:mlp | 2.639211949921 | 14.002164852 | diag-mlp-local-interaction-20260915a |
| static:down | 2.726035130155 | 15.272214477 | diag-mlp-local-interaction-20260915a |
| all16:mlp:remove_clipping | 2.744710457704 | 15.560107970 | diag-mlp-error-components-20260915a |
| all16:mlp:remove_grid | 2.638977908340 | 13.998888146 | diag-mlp-error-components-20260915a |
| down16:down | 2.688916962315 | 14.715729597 | diag-down-interaction-20260915a |

case格式：激活:作用组:替换方法。all16=所有backbone输入A16，down16=非down静态INT8/down-A16，static=既有INT8/SP2。未指替换方法则恢复同R未量化BF16权重。attention64矩阵、gateup32、down16、MLP48。所有其他权重/尺度/高精度部分不变。

## 参考、误差构造与评测

父包：`runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`。参考：`runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/state.pt`中的R；由原BF16模型重新norm fusion后旋转得到未量化权重，与父包高精度部分匹配。没有将W4反量化权重当参考，没有重估原父SW。

每个case先apply_records恢复112个父q/SW，再设置实际wrapper输入bits及输出bits=16，最后只覆盖指定组。所有生成权重在GPU上，CPU只负责小张量/序列化/文档。使用既有phase3.common.full_validation→phase2.validation_acceptance→utils.eval_utils.evaluator；252852输入、252728预测targets、123×2048+948尾窗，按预测token加权NLL后exp。BF16 logits CE、loss转FP32沿用既有实现，不平均窗口PPL。

RTN保持既有absmax/maxcode尺度初始化约定及torch.round ties-to-even、signed clip；W4 [-8,7]，W8 [-128,127]。group128沿输入轴、每输出行分别分组。无训练或搜索，三种RTN工作量为同一矩阵集合的一次投影；时间差不作为算法收益。

分量诊断：C=clamp(Wref,-8SW,7SW)，e_clip=C-Wref，e_grid=Wparent-C；分别从Wparent减去一项再转BF16。原q/SW固定，grid包含既有优化整数码及BF16反量化残差，不能等同纯RTN舍入。权重FP32误差可按这一定义分解，NLL和PPL恢复效应不可相加。两种反事实均不是合法W4部署包。

WikiText2 validation已长期查看，本次也根据完成的诊断选择后续问题，不称盲测。没有将validation用于选码、调参、训练或多seed搜索。没有新C4或恢复训练；因此不声称外部泛化改善或恢复收益。

## 对算法路线的含义

保持Phase3最终PTQ及既有恢复冠军。当前未发现加载/坐标/码范围错误；没有产出可采纳的新部署方法，禁止将W8/group128/高精度恢复诊断当最终方案。

已有Phase3输入Gram/固定SW邻码优化是有效的，不能重新命名为新模块。此前Phase4局部SW/码及GuidedQuant g1小范围候选未改进任务，不能推出所有量化求解或任务敏感目标都无效。本轮也未证明“局部重构目标与后续非线性失配”是唯一主因，未证明W4已达不可改进下限。

研究依据： [AdaRound原论文](https://proceedings.mlr.press/v119/nagel20a.html) 区分任务相关舍入与单权重近似；[GPTQ原论文](https://arxiv.org/abs/2210.17323) 对应已有输入二阶线性重构。若继续改算法，优先在这些任务感知离散重构思路上针对MLP检验真正改变目标的最小改动，而非继续追逐范围截断、重复group128或堆在线补偿。此处只是后续方向，不将尚未测的非线性放大机制当已证实结论。

## 命令、资源与审查

源码仅Phase4 `experiments/phase4/diagnose.py` 与 `launch.py`；每run保存实际执行diagnose.py副本与source_record（HEAD/branch/实际模块路径）。继承依据沿用inheritance.json/diff/status，不新增hash体系。未改旧Phase3或根协调文档，未提交/推送/清理。

| run | GPU | PID | session |
|---|---:|---:|---|
| diag-activation-20260915a | 5 | 970404 | phase4-diag-activation-20260915a |
| diag-families-20260915a | 7 | 971190 | phase4-diag-families-20260915a |
| diag-mlp-format-20260915a | 5 | 986965 | phase4-diag-mlp-format-20260915a |
| diag-mlp-local-interaction-20260915a | 7 | 987767 | phase4-diag-mlp-local-interaction-20260915a |
| diag-mlp-error-components-20260915a | 0 | 997465 | phase4-diag-mlp-error-components-20260915a |
| diag-down-interaction-20260915a | 5 | 1001011 | phase4-diag-down-interaction-20260915a |

真实命令、资源快照和日志在同名.launch.json/.log；socket=rotation-quant-phase4。所有任务自然完成。仅小JSON/源码/日志，没有新模型导出；用户给定28.66GiB初始真实余量不被聚合df覆盖，无安装、清理或磁盘不足事件。

轻量测试：父q/SW重置与激活切换；RTN轴/组/符号约定；误差分量恒等式。独立verifier限定PASS见 `verifier/diagnosis-review.md`；最终auditor报告 `auditor/DIAGNOSIS_AUDIT.md`。无新部署胜者，不额外冷载复评旧冠军。
