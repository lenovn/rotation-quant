# Phase5 handoff 交付覆盖限定复核

## 当前最终结论：本轮授权交付要求已覆盖（2026-09-18T09:52:10.313258+00:00）

**最终独立覆盖核对 PASS。HANDOFF 与当前用户授权所要求的 Phase5 研究执行、必要结果及独立证据现已齐备，没有仍待执行的授权内实验或未关闭的结果审计缺项。** 本结论是在最后Uniform43固定C4结果登记，以及`auditor/QWEN_QAT400_S43_20260918.md/.json`完成最终PASS后作出；该JSON的pending_external_runs与pending_summary_evidence均为空，包含已独立核对的Qwen18值三seed统计。Llama三seed此前PASS继续复用。

以下09:47:53和06:18:34 UTC两段原文均保留作历史快照；其中“等待”“未完成”等措辞不再代表当前结论。

### 最终交付覆盖

| 授权要求 | 最终状态与证据定位 |
|---|---|
| 必要Qwen架构适配、有效dirty源码继承与旧环境/产物保留 | 完成。继承清单/status/diff、独立`ARCHITECTURE_REPORT`、tiny/真实GPU/两卡placement结果齐备；原生Qwen与未量化适配、Q/K Norm、GQA R、显式解绑、冷加载、静态覆盖/重算梯度均有证据。旧`worktrees/SpinQuant-phase3-joint`、`repos/SpinQuant`、旧`rotation-quant-p0`与隔离`runs/phase5/env`仍存在；旧Phase2/3/4来源继续被实际结果引用。本次仅核对保留路径与原证据，不把路径存在扩称全目录逐字节未变。 |
| Qwen seed42固定主线、两家族完整核心矩阵 | 完成。Joint100→校准→round/range/readapt→WikiText-only QAT400与两格式匹配对照均有实际源、命令、包、metrics及独立审计；B100格式对照没有冒充Uniform-QAT400。seed42矩阵及首次格式C4证据沿用下方初次审查。 |
| 首个最终冷包一次独立完整test | 完成。`verifier/QWEN_SEED42_FINAL_TEST_20260918.*`，PPL14.151216160997155、298931targets含尾窗，与主执行一致；没有因收尾重复validation/C4或其他seed。 |
| Llama代表消融与两家族42/43/44稳定性 | 完成。消融报告和Llama三seedPASS保留；Qwen seed44及seed43最终报告均PASS。各模型每个seed的两方法训练/参考/预算配对，各最终包三split齐备；各模型18个原始QAT指标已独立核对。不是最佳seed选择，也不把同包复评分当多seed。 |
| 同一最终包统一test/C4、数据和预算边界 | 完成。直接检查两模型×两方法×三seed的12组QAT记录，每组恰好validation/test/C4三行且只有一个evaluation_package；必要seed42 PTQ主表已在前次核对。新Qwen43/44四项result均completed_steps=400、target_reached=false，命令固定400/global8/原优化与损失规则、精确父包/同seedJoint参考；独立审计核对实际400日志与6553600输入/6550400预测visits。无test/C4校准、overlay、训练或候选选择。固定C4文档/各tokenizer文本覆盖和尾窗协议报告保留。 |
| summary与结果表 | 完成。当前summary.csv共99条COMPLETED记录，含20个字段：模型/seed/stage/format/qat_data/父包/评测包、dataset/split/subset/子集身份/协议、PPL/NLL/targets/input/tail、复用或新测、完成状态和原始路径；适用字段齐全，BF16等不适用父包/seed留空不构成缺失。模型仅授权的Llama与Qwen，QAT数据仅WikiText-only。新增12项Qwen43/44原始JSON与CSV的PPL/NLL/input/targets及包逐项一致；CORE_RESULTS已包含完整两家族三seed统计。 |
| 唯一外部近邻候选 | 按授权层级完成。SpinQuant本地适配有限smoke PASS，实际1更新/1校准窗、112矩阵GPTQ/group32与动态A8前向。A8先于GPTQ、在线Hadamard、embedding/head高精度等差异明确披露；它只作为可运行候选，不伪称官方完整性能复现或替代内部Uniform。 |
| 失败恢复、资源与范围披露 | 完成记录。Llama直接QAT消融25步迁移、Llama43 readapt已保存4层前缀恢复及独立验证脚本setup失败均保留原记录；见下述具体边界。100份Phase5 launch GPU列表均无GPU4，保存tmux/真实PID/命令。未扩展模型、搜索或启用mixed；原生ACC/NPU/decode不在本轮结果范围。 |

### 关闭的最后缺项与数值核对

本次只增查前次缺少的Qwen43/44四个最终run及12项原始JSON，没有重做已PASS架构、旧矩阵、GPU/PPL、suite或完整optimizer检查。四份第400步包及八份外部JSON均已存在，四训练run均无failure.json；每项test/C4原始结果的package、实际格式、calibration/training/candidate_selection=false与activation_scales_unchanged=true均匹配。逐张量前后状态、实际IDs与固定数据的证明继续引用独立auditor，不把本次小JSON检查冒充重做张量审计。

最后`qwen3-1p7b/uniform-qat400-c4-s43/result.json`为**PPL27.45911062612704/NLL3.3126980118374294/2096128targets**，同`uniform-qat400-s43/checkpoint-0400/static_w4a8.pt`，已登记summary。auditor核实196INT8状态前后及跨test位级相同、固定IDs/metadata/1024窗、分段与summary；旧test summary pending同时关闭。

| Qwen三seed数据 | SP2 PPL均值±样本SD | Uniform PPL均值±样本SD | 配对ΔNLL均值±样本SD |
|---|---:|---:|---:|
| WikiText validation | 14.808379±0.005118 | 15.707236±0.049887 | −0.058925±0.003262 |
| 完整WikiText test | 14.135739±0.026576 | 15.069851±0.064991 | −0.063985±0.006118 |
| 固定C4 | 24.153178±0.103593 | 27.405296±0.277549 | −0.126292±0.012910 |

本次另作轻量算术复核，与CORE及auditor显示精度一致；样本SD用ddof=1，配对ΔNLL为每seed SP2−Uniform后求均值/SD，未混入B100/PTQ/消融或挑最好seed。既有Llama三seed统计与本次Qwen统计都只支持指定模型/固定配方/数据和R、校准初始化随机源的描述性稳定性，不泛称所有随机源或跨尺寸泛化。

### 保留的证据限制、恢复成本和范围外工作

- QAT validation没有独立input IDs/quantizer before-after快照；已有metadata、分段、计数及包来源。teacher身份/冻结由源码、设置与参数覆盖支持，没有独立teacher全张量快照。最终test/C4的实际IDs、量化状态及同包证据已审，不需要为这些明示限制追加保险复跑。
- Llama代表消融在WikiText上完整后处理后QAT更好，在C4上初始SP2直接QAT略好；两包各自整行报告，没有按split换权重。两家族C4的保留文本字符范围不同，不能把跨tokenizer绝对PPL作同文本排名。
- `INITIAL_SP2_ABLATION_20260918.md`明确旧1–25更新和恢复26–400连续，迁移中未提交的工作被丢弃，6553600为有效更新输入预算而非设备物理总处理量。`LLAMA43_READAPT_RESUME_20260918.md`保留未知原因退出和原layer4至少384坐标迭代的重做成本；新-r4从已保存layers0–3之后恢复，正式QAT使用恢复完成精确父包。不能把恢复说成零额外算力或猜测OOM。独立smoke/架构脚本初始调用错误亦保留且仅补未完成检查。
- 混合数据QAT未授权启用，仍未执行；下游ACC、原生NPU、decode/KV低比特及延迟能耗不是本轮完成项。外部候选完整100步/128窗/正式性能和CLI Trainer/DDP也未验证。这些范围外工作不能写成已完成，也不形成当前已授权Phase5交付的待跑清单。
- 本次复核没有改主执行STATUS/PLAN/summary；主执行可据最终证据完成其收尾记录和goal状态。客户端模型配置不由prompt或本报告更改，不宣称在此审计中验证过客户端配置。

**当前未闭合的授权内研究执行/结果证据项：无。** 本报告证明交付覆盖与证据一致性，不替代论文全科学审稿、正式发布审批或硬件验收；未新增gate/contract、实验预算、模型或方法。

---

## 最终覆盖更新：新增证据核对中（2026-09-18 09:47:53 UTC）

前次缺少的四个Qwen seed43/44 QAT400均已完成，四份checkpoint-0400包和八份最终test/C4原始JSON现均存在；各包身份、400步、固定参数命令和新增外部指标已核对。最后Uniform43 C4为PPL27.45911062612704/NLL3.3126980118374294，2096128targets。**此快照仍等待该结果summary登记、QWEN_QAT400_S43最终审计及Qwen三seed统计审计，暂不声明整体完成。**

不依赖最后C4的覆盖已更新：summary具备handoff要求的20个字段，模型仅Llama/Qwen两个、QAT数据标签仅WikiText-only的两种等义写法；100份launch记录均未使用GPU4。旧SpinQuant/Phase3工作区、旧环境和Phase5隔离环境路径均保留，继承清单/status/diff及各run源码快照仍可追溯。本次未改应用、STATUS/PLAN/summary或任何实验产物，未运行GPU/PPL/suite或读取完整optimizer。

以下保留首次覆盖核对原文，属于14:18:34北京时间历史快照，其“未完成”清单不代表当前进度。最终结论将在本节后补齐。

## 历史快照：2026-09-18 06:18:34 UTC

结论：**已宣称完成的本轮检查项有对应真实证据；Phase5 整体尚未完成，剩余主要工作是 Qwen seed43/44 四个最终 QAT400 及各自同包 test/C4 验收、三seed汇总。** 未发现需要把已完成 seed42 核心矩阵、Llama 消融/三seed或首次冷包复核改记为失败的证据缺口。

核查快照：2026-09-18 06:18:34 UTC（北京时间14:18:34）。这是已有产物的只读覆盖核对，不是新增gate、验收协议或实验。按 `ccf-integrity-auditor` 的 quick claim/numeric audit 执行；除本报告外未写入其他文件，没有GPU、PPL、模型forward、新suite、重新tokenize或完整optimizer读取。运行状态取保存的progress与文件存在性，本次未重新审计宿主进程存活。

## 要求与证据对应

相对路径均以 `runs/phase5` 为根；原历史证据保留其项目根相对路径。

| Handoff 要求 | 覆盖结论 | 关键证据及本次追溯范围 |
|---|---|---|
| §2.5有效源码继承、§4必要Qwen架构适配 | 已覆盖 | `source_inheritance/inheritance.json`登记源/目标、完整HEAD与dirty/untracked文件，旁有原status/diff；`verifier/ARCHITECTURE_REPORT.md`及`architecture_gpu.json`为真实GPU独立PASS，另有tiny测试与两卡placement证据。核对实际报告/JSON存在且PASS；此前真实原生适配零差、Q/K Norm、GQA R1/R2、分离centering、checkpoint梯度、196/168/28覆盖和冷加载检查没有被扩称原生NPU结果。 |
| §7.1首个最终seed42包一次独立完整主PPL | 已覆盖 | `verifier/QWEN_SEED42_FINAL_TEST_20260918.md/.json`及`qwen3-1p7b/sp2-qat400-test-independent-s42/result.json`。固定WikiText test已单独冷进程执行，PPL14.151216160997155，与主执行全值/分段相同；298931targets含70-token尾窗，196量化器状态不变。没有把该复核扩展为validation/C4重跑。 |
| §2.2/2.3 Llama已有test/C4归因复用 | 已覆盖 | `auditor/INHERITED_EVIDENCE_20260918.md`、`inherited_checks.json`、summary原路径。重新读取历史BF16、精确PTQ、最终SP2-QAT400原始JSON及B100 C4两行，数值与summary一致。历史最终test14.154416405154963/C4 26.98478050458106共用原包；1330.1567504736056仍标B100格式对照，没有充当完整Uniform结果。 |
| §3/5.1 Qwen seed42固定主线及首次B100格式C4配对 | 已覆盖 | `auditor/QWEN_JOINT_FORMAT_20260918.*`、`QWEN_POSTPROCESS_20260918.*`、`QWEN_SP2_QAT400_20260918.*`与external审计。实际两个QAT `.launch.json`均指定各自精确readapt父包、同Joint100参考、400/global8/Adam1e-5/relative .001/T1/CE.1/data_start800。B100 C4原结果为SP2 27.18323240302331、INT8 33.33932032786121，独立保留机制行。 |
| §5.2 两家族seed42完整INT8/SP2×PTQ/QAT矩阵 | 已覆盖 | Qwen `QWEN_PTQ_ACCEPTANCE_20260918.json`、`QWEN_SP2_QAT400_EXTERNAL_20260918.json`、`QWEN_UNIFORM_QAT400_20260918.md/.json`；Llama `UNIFORM_POSTPROCESS_20260918.md`、`UNIFORM_QAT400_20260918.md`及历史SP2。直接核对主表各原始指标与包存在性、必要test/C4逐方法同包。后者包含本轮补测的Llama `sp2-ptq-test-s42/result.json`，PPL15.560178512473657。 |
| §5.3 Llama代表模型：初始SP2直接QAT400 vs完整后处理QAT400 | 已覆盖，结论须分数据集 | `auditor/INITIAL_SP2_ABLATION_20260918.md`、training/external checks；直接读取`initial-sp2-qat400-test-s42/result.json`与C4原结果，核对恢复run `.launch.json`明确resume25、初始SP2父包及历史同R参考。完成1–25/26–400的连续性已有独立审计，本次未再读optimizer。WikiText完整方法更好，C4直接QAT更好，不能声称消融支持全语料后处理优势。 |
| §5.4 Llama完整方法/关键对照42/43/44配对 | 已覆盖 | `LLAMA_*QAT400_S43_20260918.*`、`LLAMA_QAT400_S44_20260918.*`和各最终原始结果。直接读取三seed两方法三split共18行并复算均值、样本SD、配对ΔNLL，吻合CORE_RESULTS；没有混入B100、PTQ、代表消融或最佳seed。seed/初始化与恢复链沿用JOINT/POSTPROCESS限定审计。 |
| §5.2唯一外部候选的实际可运行性及差异披露 | 有限可运行性已覆盖 | `verifier/external-spinquant-smoke/{EXECUTION.md,result.json,completion_audit.json}`为PASS：实际1次2048-token仅R更新、112矩阵GPTQ W4/group32、动态A8有限forward，原脚本/命令/失败记录保留。只能登记本地适配核心callable可运行；A8先于GPTQ、embedding/head高精度等差异已披露。100步/128窗、CLI Trainer/DDP、正式外部PPL没有执行，也未被主表或PLAN宣称完成。 |
| §6.3跨模型固定C4原文与tokenizer口径 | 已覆盖 | `auditor/QWEN_DATA_METRIC_20260918.md`、`C4_TEXT_COVERAGE_20260918.md/.json`。4480篇原文保序，各自tokenize并按既有2097152-token前缀截断。文本覆盖差异、Llama旧offset限制已有明确记录，不能把相同targets当作相同保留字符范围或跨tokenizer绝对PPL排名。 |

## 本次原始数值抽核

直接读取45条BF16/完整PTQ/完整QAT/代表消融/Llama三seed原始JSON，PPL、NLL与targets均匹配summary，所有引用量化包存在；另查4条Llama/Qwen B100 C4格式对照。所有应成对的test/C4主表行逐方法/seed指向同一包。未读取大权重或optimizer；包内容/冻结状态的结论追溯已有独立审计，不冒称本次再次逐张量验证。

Qwen seed42已完成矩阵的test/C4 PPL：

| 方法 | WikiText test | 固定C4 |
|---|---:|---:|
| BF16 | 16.715764347250076 | 23.136144236064272 |
| SP2-PTQ | 14.302163393485294 | 24.903332773203157 |
| Uniform-PTQ | 16.78537625006989 | 29.322131492633147 |
| SP2-QAT400 | 14.151216160997155 | 24.24703443394153 |
| Uniform-QAT400 | 15.055184987698569 | 27.10478088408928 |

Llama三seed重算：test SP2 14.185554491144666±0.027205375043345085，Uniform17.942991787146553±0.6860106729543555，配对ΔNLL均值−0.23449619898061025；C4 SP2 26.71310234407695±0.24332731456792694，Uniform41.031980903288094±3.6528158355555718，配对ΔNLL均值−0.42666350969817507。SD为ddof=1，ΔNLL为SP2减Uniform；结果按实际R/校准初始化随机源解释，不泛称所有数据shuffle随机性均已验证。

## 当前具体未完成项

四个Qwen额外seed最终包在快照时均不存在，8份必要最终test/C4 `result.json`也均不存在；这与PLAN/STATUS保留未完成状态一致。

| Qwen run | 保存进度快照 | 第400步包/最终result | 必要test/C4 |
|---|---|---|---|
| `sp2-qat400-s43` | distillation-training，step25/400，PID282520，GPU7/3 | 均未产生 | 均未完成 |
| `uniform-qat400-s43` | run目录尚不存在，尚未启动 | 均未产生 | 均未完成 |
| `sp2-qat400-s44` | distillation-training，step58/400，PID242215，GPU1/0 | 均未产生 | 均未完成 |
| `uniform-qat400-s44` | distillation-training，step37/400，PID270807，GPU2/5 | 均未产生 | 均未完成 |

因此剩余交付是这四个固定400步训练与最终包、各自同包完整test/既有固定C4验收，以及结果齐备后的Qwen三seed逐值/均值±SD/配对ΔNLL与对应已有证据审计汇总。43/44各自Joint100/匹配初始化和12个后处理阶段已有PASS，不重新列为待重做。无需再对额外seed机械重复首次架构suite或独立完整test复核。

## 证据限制与声明边界

1. **已明示的validation证据限制（非隐藏失败）：** `auditor/QWEN_UNIFORM_QAT400_20260918.md`、`LLAMA_QAT400_S44_20260918.md`等指出QAT validation没有单独保存输入IDs或量化器before/after快照；实际有最终包来源、metadata/targets/segments和数值。不能写成“三split均做了独立逐张量输入/状态比对”。test/C4的快照与同包检查已覆盖本轮正式外部验收，不由此要求重跑validation。
2. **代表消融的反向C4结果必须保留：** `initial-sp2-qat400-c4-s42/result.json`为26.730584258553062，历史完整方法`runs/phase3/c4-best-fixed-20260915a/result.json`为26.98478050458106。完整方法在test略好不推出在C4也好；现有消融报告已如实写明，没有证据支持把这两包按split择优拼成一行。
3. **旧审计是时间快照：** `auditor/INHERITED_EVIDENCE_20260918.md`曾列Llama SP2-PTQ test、完整Uniform、消融和额外seed为missing，`EXTERNAL_BASELINE_SCOPE_20260918.md`曾列源码就绪但未运行；后续原始结果及当前PLAN已闭合相关项。保留旧报告不构成当前产物缺失，交付时引用当前报告，避免把早期missing重新当待做。
4. **非本轮缺口：** 外部方法完整性能复现、混合数据QAT、下游ACC、原生NPU/decode没有完成；前者本轮只要求候选范围与实际有限可运行性，后三者不在本次授权默认实验范围。它们不能冒称完成，也不应据此自动新增实验或阻塞既定Phase5交付。

严重性结论：没有发现已完成主表的重大证据缺口；Qwen43/44最终训练/验收是明确且预期的未完成工作，整体goal仍须保持未完成。下一执行责任继续归主执行；本报告不新增实验、gate或contract。
