2026-09-15：按用户缩小范围，仅生成layer1 down通道图，见[layer1-down-channels-20260915a/README.md](layer1-down-channels-20260915a/README.md)。2048输出行×8192权重；保留原q/SW、同R未量化参考，无forward/PPL/训练/模型导出。两图及统计JSON已完成，独立verifier PASS。权重RMSE最大行867与tail最大行1649不同，不外推为任务敏感通道。

# 最新：逐层MLP/up/gate/down正式定位完成

2026-09-15：65项新完整GPU测量完成，复用layer1 down与匹配既有对照。全家族条件恢复收益 down > up > gate；整体MLP layer1和15近乎并列，layer10次之，9/11/12/13也敏感。单矩阵layer1 down最突出；layer10/11 up、layer13 gate不可忽略。所有排名限定当前父包同R全A16，恢复效应不可相加、不直接外推最终SP2。未训练/导出新模型。

结果：[完整分析](mlp-layer-analysis-20260915a/ANALYSIS.md)、[16层表](mlp-layer-analysis-20260915a/TABLE.md)、[热力图](mlp-layer-analysis-20260915a/layers.png)。独立verifier PASS；auditor已核对全部65项及汇总PASS，正式报告auditor/MLP_LAYER_AUDIT.md。下面保留历史记录。

## 当前W4成因诊断完成（2026-09-15）

16项新完整GPU测量表明：当前主要剩余误差在MLP的W4离散表示，非范围截断或单layer1 down主导。关闭激活后PPL15.5570，同R W16参考13.6516；恢复MLP14.0022；只去截断15.5601，只去格点误差13.9989。匹配RTN：W4 per-channel16.4540，group12816.2742，W8 per-channel14.0226。全部精确结果、交互、条件归因边界及命令见 [W4_DIAGNOSIS.md](W4_DIAGNOSIS.md)、w4-diagnosis-summary.json。新增方法未获任务收益，保留原方法；这不证明W4不可改进。

## 2026-09-15 当前W4成因诊断（进行中）

本轮从原PTQ父包开始，所有干预均先恢复完整父q/SW，未量化权重由B100 state保存R与原BF16重建。GPU5 activation批PID970404，GPU7 family批PID971190；命令/session见对应.launch.json。只有条件恢复，未优化任何参数，未保存模型。轻量小张量测试已确认逐case重置、per-row广播与输入量化bits切换。

首项完整结果：W4/all-A16 PPL15.557034492007398，恢复全部down/all-A16 PPL14.677121507296714。当前父包静态A8/SP2 PPL16.112577060930427复用。需等待同R W16参考与其他组，暂不下主因结论。

# Phase4 结果

以下首表为继承证据；Phase4新正式PPL与判断见后续A/B结果段。

| 包/条件 | 完整WikiText2 PPL |
| --- | ---: |
| 原BF16 | 13.634657725643203 |
| Phase2固定C W4/all-A16 | 16.231066669209774 |
| 同C W4/non-down INT8/down-A16 | 16.292812357698388 |
| 同C W4/non-down INT8/down-SP2 | 17.64239997588965 |
| Phase3 B100 | 17.11711490993314 |
| B100→down整数码→SP2收缩→再修正down（A0） | 16.112577060930427 |
| A0干净起点400步恢复（WLR1e-5） | 14.581654675328117 |

主要父包 `runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`；共同未量化参考 `runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/state.pt`。A0没有接受D。上述C条件效应不能用于分解Phase3父包。

已有Atlas覆盖112矩阵/376832输出通道，max_abs/P99_abs median约1.4158、P99约1.8836、max约2.8647；不重扫，不据此声称W4最优。旧短测RTN14.65525、MSE-INT4拟合17.12465、weight-SP2 20.17590，不用纯weight MSE替代任务选择。历史GPTQ/all-A16 down恢复敏感性仅作线索，不作为当前排名。

既有C4固定validation子集（2096128 targets）：原BF16 21.843397624525274，恢复冠军26.98478050458106；缺同父恢复前对照，不归因于蒸馏或SP2。来源为Phase3 STATUS/RESULTS/PHASE2_POSTPROCESS_COMPARISON/C4_RESULTS_20260915与Phase2 validation-diagnostics报告。

## A首轮正式GPU搜索（完成）

GPU0/PID2800949，进程正常完成；算法/选择182.47秒，峰值allocated7.287GiB（总进程含加载更长）。每臂3模块×2×2048=12288坐标更新。A1优化共67.0334秒，A2共68.0861秒，额外约1.6%，另共享输入捕获及13次8窗train选择。所有12候选heldout局部误差下降但训练NLL高于父2.6472935144882825，零接受。此时返回的16.1125771明确为复用父，非Phase4完整新测。

为完成方向成品全评测，预先选两臂训练分最接近父包的layer8/cycle2，重建此前未保存码并分别冷载全评测：`a-layer8-final-20260915a`，GPU0/PID2820170。额外重建只1模块/两臂，不隐藏成本；该运行明确diagnostic_best_rejected，不冒称训练已接受。

## A最终完整validation

`a-layer8-final-20260915a` GPU0/PID2820170正常完成。重建出的两个cycle2 train NLL与首轮精确一致。

| 成品 | NLL | PPL | 相对A0 PPL |
| --- | ---: | ---: | ---: |
| A0复用 | 2.779600150933802 | 16.112577060930427 | 0 |
| A1继续邻码 | 2.779605266359144 | 16.112659483826263 | +0.000082422895836 |
| A2 SW/邻码交替 | 2.7803035383838837 | 16.123914432238468 | +0.011337371308041 |

两项新完整测量均252728 targets、0未评分尾tokens。A2相对A1也退步0.01125495；不采纳。已证实局部SW可修正且交替改善局部heldout，但没有本轮任务收益；不能说已解决W4条件损失，也不能否定COMQ或所有SW优化。无新竞争胜者，不自动追加A16/C4/蒸馏或重审旧冠军。完整新增两包约3.85GiB，保留。

记录说明：`a-layer8-final-20260915a/settings.json` 的budget说明字符串沿用“3 modules”，实际arguments、日志、局部结果均只处理layer8一个模块；以实际执行为准。保留原始settings，不改写历史记录。

## B最终完整validation

`b-guided-g1-layer8-20260915a`，GPU7/PID2843696正常完成。新完整 **NLL2.780084071227797/PPL16.12037615087552/252728 targets**，比父PPL高0.00779909，比A1高0.00771667，不采纳。选择cycle1（2048坐标）的train NLL2.649110330984922，高于父2.6472935144882825；cycle2加权局部目标更低，但train略更差。

梯度32窗6.4589秒，邻码22.2677秒（实际共4096坐标生成两个端点）；进程内含加载/选择/完整评测114.8638秒，peak allocated7.8572GiB。saliency每窗最后token为0，共32个0，符合没有末位预测target；fit均值归一后median0.5795、P99 7.2087、max47.3048，有效fit行数约13985.9/49152，目标确实重新加权而非乘全局常数。加权heldout从0.0001277691降至所选0.0001265332，普通heldout从0.0001412973升至0.0001442442。说明代理目标已被优化，尚未转化为最终任务收益。

无已接受的新赢家：本轮不新增C4、不做下游恢复训练、A16诊断或多seed，不复跑旧冠军。未执行项目的结果明确为NOT TESTED，不把旧冠军结果当新方法恢复成绩。

## 本轮决策与五个问题

1. **剩余机制证据**：112 SW实证未随后处理更新，三个down模块确有尺度可降低重构误差的方向；但A局部heldout优势没有正式NLL收益。B改变任务敏感性加权后局部目标降而任务仍不降，支持“局部代理目标不足以保证任务收益”，不能定位全部剩余W4损失的唯一来源。
2. **具体方法差异**：A为COMQ论文式(10)的SW闭式更新加既有邻码接续，A1同码扫描预算；B为GuidedQuant g=1加权Gram加同一固定SW邻码。均有明确论文依据和适配范围，不声称新通用算法，不以更多计算冒充机制收益。
3. **收益归属**：没有可靠PTQ收益。因此本轮不能声称改善权重条件损失、SP2补偿或恢复效率；也没有匹配粒度对照，不能归因为解决per-channel粒度问题。恢复/SP2旁路/C4交互均未由本轮测量。
4. **是否纳入或替代**：不纳入，不替代已接受Phase3步骤。原PTQ 16.1125771、原恢复冠军14.5816547继续保留；新增源码与失败成品作为研究证据，不接入最终导出主线。
5. **负结果边界**：A覆盖layer1/8/11局部搜索及预选layer8成品，B覆盖layer8的g=1量化父STE梯度代理。这不是COMQ/GuidedQuant整体复现，也不是所有SW优化、后续RMSNorm直接重构、更多模块或其他数据的否定。未发现可支撑继续加补偿项或400步恢复的证据，故本轮收束，不盲目堆模块/搜索。

validation长期参与项目开发；本轮layer8范围依据train选择预定，B方向设计依据A的局部/train不一致，实施前A完整结果亦已可见，因此不称盲测。本次仍只有静态格式的GPU BF16 fake-quant prefill证据，无原生整数kernel/NPU/decode/延迟或任务准确率结论。

最终独立auditor **限定PASS**，见`auditor/FINAL_AUDIT.md`与`artifacts.json`。这是执行证据与“不接受新方法”结论的审计，不是方法有效性PASS。三包实际q/SW、固定激活/高精度、完整targets与NLL汇总匹配；0独立GPU复跑，0恢复训练。
