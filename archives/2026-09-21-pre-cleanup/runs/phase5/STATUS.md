# Phase5 状态

2026-09-18：**本轮授权范围已完成，独立结果审计与最终 handoff 覆盖复核 PASS。** 两家族 seed42 核心矩阵、Llama 代表消融、两家族 seed42/43/44 配对、必要最终包完整 WikiText-2 test 与既有固定 C4 验收均已齐备。没有仍需执行的本轮实验或结果审计。

| 工作 | 当前状态 | 证据/位置 |
|---|---|---|
| 有效源码继承及 Qwen 架构适配 | 完成；真实 Qwen、窄 Llama 回归、两卡放置独立 PASS | source_inheritance/；verifier/ARCHITECTURE_REPORT.md |
| Llama 历史 test/C4 归因 | 合格原结果复用，审计 PASS，未重复评分 | auditor/INHERITED_EVIDENCE_20260918.md |
| Qwen seed42 固定主线 | Joint100→初始SP2→邻码→范围收缩→再适配→WikiText-only QAT400，完成 | qwen3-1p7b/；summary.csv |
| 两家族完整核心对照 | BF16、SP2/Uniform × PTQ/QAT400 同包验收全部完成 | CORE_RESULTS.md；auditor/ |
| 首个 Qwen 最终冷包独立完整 test | 一次复核 PASS，含尾窗与实际静态状态 | verifier/QWEN_SEED42_FINAL_TEST_20260918.md |
| Llama 代表消融 | 初始SP2直接QAT400与完整后处理QAT400成组验收完成，审计 PASS | auditor/INITIAL_SP2_ABLATION_20260918.md |
| 两家族 seed42/43/44 配对 | 两格式各三split全部完成，逐seed、均值±样本SD、配对ΔNLL独立 PASS | CORE_RESULTS.md；auditor/LLAMA_QAT400_S44_20260918.md；auditor/QWEN_QAT400_S43_20260918.md |
| 唯一外部近邻候选 | 本地SpinQuant有限smoke可运行性 PASS；不是正式性能复现 | verifier/external-spinquant-smoke/EXECUTION.md |
| 最终交付覆盖 | 全部授权内要求有对应实际证据，无未闭合项 | verifier/HANDOFF_COVERAGE_REVIEW_20260918.md |

三seed最终 PPL，均值±样本标准差（ddof=1）：

| 模型 | 方法 | 完整 WikiText-2 test | 固定 C4 |
|---|---|---:|---:|
| Llama-3.2-1B-Instruct | SP2-QAT400 | 14.185554 ± 0.027205 | 26.713102 ± 0.243327 |
| Llama-3.2-1B-Instruct | Uniform-QAT400 | 17.942992 ± 0.686011 | 41.031981 ± 3.652816 |
| Qwen3-1.7B | SP2-QAT400 | 14.135739 ± 0.026576 | 24.153178 ± 0.103593 |
| Qwen3-1.7B | Uniform-QAT400 | 15.069851 ± 0.064991 | 27.405296 ± 0.277549 |

完整精度、validation、相对本模型BF16指标、逐seed配对ΔNLL与每条原始路径见 [CORE_RESULTS.md](CORE_RESULTS.md) 和 [summary.csv](summary.csv)。主执行最终逐项读取99条完成记录，原始PPL/NLL/input/targets及包存在性均一致；主表19组模型/方法/seed共57项指标，每组三split使用同一包。最后训练和评测进程均已自然退出，Phase5 tmux socket已无运行会话；100份启动记录均未使用GPU4。

结果解释与交付边界：这是BF16 fake-quant、静态backbone W4A8、KV16、无cache的prefill质量证据，不是原生内核或手机速度结果。跨tokenizer绝对PPL不直接排名；Qwen WikiText低于原始BF16包含训练适应收益，不能全归因于量化。Llama代表消融在WikiText支持完整后处理，但C4上初始SP2直接QAT略好，未按split切换包。QAT validation未另存独立IDs/量化状态快照的证据限制，以及异常恢复的额外物理计算成本，均保留在独立报告中。

混合数据QAT、下游ACC、原生NPU/decode及外部方法完整性能复现未执行，不列为本次授权内待办。旧源码、环境和Phase2/3/4产物保留；未提交推送，未终止他人进程。客户端模型配置未由本执行更改，也不以prompt代替客户端配置。

以下为保留的执行记录，早期“待完成”表述按上表更新。

- 已读项目AGENTS、根文档与handoff；将原小写handoff.md复制为HANDOFF.md，保留原文件。
- 已创建 `phase5/multimodel` 工作树，继承22个有效dirty/untracked源码文件。revision、原status/diff与文件清单在source_inheritance/。
- 已核实 Llama 原报告：test BF16 13.162650325300651、QAT400 14.154416405154963；C4 BF16 21.843397624525274、B100 INT8 1330.1567504736056、B100 SP2 29.18423776720624、PTQ父包27.43961641336637、QAT400 26.98478050458106。独立auditor正在核对原JSON，以上不标为本轮新测。
- 现有环境Transformers4.44.2/4.46.3，均不支持原生Qwen3；准备隔离4.51.3环境。
- 宿主GPU可见；GPU4故障排除。其他卡均有他人进程，启动前重新核实实际余量。
- 存储：/mnt/home1可用67GiB、低于mergerfs常见100GiB阈值；/mnt/home2可用约1.5TiB。新写入依赖后者，未清理旧文件。
- 应用适配、独立验证、Qwen实验、完整Uniform对照、代表消融与额外seed均未完成；不将准备工作计为PASS。

下一步：下载指定Qwen模型/建立环境，参数化模型和seed入口，接入官方Qwen3旋转路径并独立验证。

## 实施更新

- 独立历史证据审计已完成：13条分段NLL/PPL重算一致；7条外部run已completed、无failure；test/C4的QAT400为同包。报告 `auditor/INHERITED_EVIDENCE_20260918.md`，已将完整精度继承行写入 `summary.csv`。
- 隔离环境 `env` 已安装并验证导入 Transformers4.51.3/tokenizers0.21.4，复用旧torch2.4.1+cu121；旧环境未改。
- 官方Qwen下载进行中，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，tmux socket `rotation-quant-phase5` / session `phase5-model` / PID1295198；不占GPU。
- 已实施：官方Qwen架构入口、Q/K Norm保留、显式head解绑、R1/R2接入、模型/seed参数化、真实计数、原始C4文档重新tokenize入口、完整INT8包格式保存/冷加载、共享范围后处理、固定400默认不提前停。
- 独立验证正在进行；发现新增QAT分支缺SP2Quantizer导入并已修复，待独立复测结论。不能把tiny架构测试当成真实模型GPU PASS。
- Qwen QAT新增按层两卡放置，以容纳FP32 master/grad/Adam；不复制样本、不改变全局batch，独立跨卡梯度测试待完成。
- 正式Qwen训练尚未启动。代表消融/完整Uniform/extra seeds仍待主线后执行。

## 首批Qwen实测

- 官方模型下载完整；C4既有4480原文保序重编码2191766 tokens，沿用前缀2097152，2096128 targets，未新增文档。
- BF16完整test **16.715764347250076** / NLL2.816352247049716 / 298931 targets，已completed，GPU6/PID1352859，`qwen3-1p7b/bf16-test-s42/result.json`。
- BF16完整validation **17.7149122731184** / NLL2.8744067861808946 / 262208 targets，已completed，GPU6/PID1360225，`qwen3-1p7b/bf16-validation-s42/result.json`。
- BF16 C4运行中，GPU0/PID1355800，tmux `phase5-bf16-c4-s42`，run `qwen3-1p7b/bf16-c4-s42`。
- Uniform独立CPU限定PASS：13项；真实tiny两卡层放置限定PASS：84参数梯度max差0。报告在auditor/UNIFORM_CPU_20260918.md与verifier/ARCHITECTURE_REPORT.md。
- 实际Qwen短输入验证日志已PASS，等待独立verifier正式结论；其首轮脚本W4开关错误已单独留失败记录并仅续跑未完成项，不是应用故障。

- 实际Qwen首次独立验证最终PASS，冷加载max差0，真实29张R梯度有限非零，Q/K Norm冻结/保留，teacher offload通过；报告 `verifier/ARCHITECTURE_REPORT.md`。
- 正式seed42初始化已启动：GPU5/PID1367237/socket rotation-quant-phase5/session phase5-init-s42；run `qwen3-1p7b/init-s42`。训练语料2517232tokens、1229完整窗；末240tokens按历史规则舍去，最后8窗保留为probe。
- BF16 C4已完成：PPL23.136144236064272 / NLL3.1413960802266425 / 2096128 targets。三项BF16参照均已写summary.csv，不再重复测量。

- Qwen seed42正式初始化已完成，训练pool1221窗；initial.pt与initial_static.pt保留，后者不是Joint100包。下一步Joint100正式启动参数为100updates/global8/warmup10/cosine100/RLR1.5/尺度Adam相对0.001，只在100做完整validation。
- Llama实际数据窄回归发现AutoTokenizer多BOS，已恢复按家族使用历史LlamaTokenizerFast；修复后train/calibration/validation metadata与历史Joint100 data.json完全相等，证据verifier/llama_tokenizer_regression.json。没有影响Qwen已完成结果。

## 正式主线及配对工作运行中

- Qwen Joint100 seed42：GPU5/PID1387607/session phase5-joint100-s42；run qwen3-1p7b/joint100-s42。
- Llama初始SP2直接QAT400消融：GPU6/PID1389835/session phase5-initial-sp2-qat400-s42；run llama32-1b/initial-sp2-qat400-s42；复用历史Joint100初始SP2与同R参考，旧4.44.2环境。第1步已完成，实际peak16.18GiB，尚非最终结果。
- Llama Uniform初始完整包已由历史已验收overlay物化完成，GPU3/PID1390227已退出；run llama32-1b/uniform-initial-s42。未重新校准，后续完整PTQ/QAT未完成。
- Uniform down邻码后处理开始，范围/再适配/QAT待后续；所有正式候选仍只用train-selection。

- Uniform down邻码：GPU3/PID1398552/session phase5-uniform-down-round-s42；候选parent/512/2048/8192，32 train窗24/8选择，运行中。
- 正在补Llama SP2-PTQ完整test这一真实缺项，固定历史精确PTQ父包，不重复已完成validation/C4；GPU0，run llama32-1b/sp2-ptq-test-s42。

- Llama SP2-PTQ完整test已完成：15.560178512473657 / NLL2.7447149912084505 / 288934 targets；固定scale不变，已入summary。

- GPU6他人进程显存出现约5GiB短时峰值，本QAT实测约19GiB设备占用，已在第25步现有resume完整保存后仅向已验证本人PID1389835发SIGINT；migration.json记录原因、命令、步数。旧目录KeyboardInterrupt是主动迁移记录，不是算法失败。GPU0以run `initial-sp2-qat400-s42-r25` 恢复同一400步schedule/optimizer/data_start；不重做前25次更新。

- 迁移恢复已实证：r25 training.jsonl首行为step26，当前已到27；旧run前25次更新没有重做，仍按原400步schedule推进。

- Llama Uniform首轮down邻码完成：train-selection NLL6.445073606790942→5.994491577236984；完整validation PPL722.8870870577282/NLL6.583253037151634，252728 targets，已入summary。这是中间阶段，不是完整Uniform-PTQ/QAT400。范围收缩已接续GPU3/PID1470547，run `llama32-1b/uniform-range-s42`，固定候选1/.875/.75/.5/.25/.125。

- Llama Uniform范围收缩完成：train-selection NLL5.994491577236984→3.549447552560451；完整validation PPL44.19421129635554/NLL3.7885938143610356，252728 targets，已入summary。最后一轮down邻码再适配已启动GPU3/PID1500386，run `llama32-1b/uniform-down-readapt-s42`；仍同Joint100参考、512/2048/8192预算，未完成前不计完整Uniform-PTQ。

- 独立Uniform后处理审计PASS（仅已完成首轮邻码+范围）：`auditor/UNIFORM_POSTPROCESS_20260918.md`。两个实际包均112 INT8输入；邻码仅down codes变化，范围仅down scales变化；非down codes/SA固定，32 train窗24/8选择与候选预算核实。layer1 step40是继承算法无负gain的提前收敛，不是新增候选。两行summary模型名已统一官方名，数值未改。

- Llama完整Uniform-PTQ三阶段完成：validation PPL41.5470707013954/NLL3.7268270182480743，252728 targets；train-selection NLL3.549447552560451→3.484595846334799。精确包`llama32-1b/uniform-down-readapt-s42/static_w4a8.pt`。Uniform-QAT400已启动GPU2/PID1574860/run `uniform-qat400-s42`，同WikiText-only预算。该PTQ包完整test已启动GPU3/PID1577135/run `uniform-ptq-test-s42`，随后同包固定C4；无重新校准。

- Llama Uniform-PTQ完整test完成：PPL38.47028685874723/NLL3.6498861734233596，288934 targets，112实际INT8输入，量化状态前后不变，已入summary。同包C4已启动GPU3/PID1585200/run `uniform-ptq-c4-s42`。Uniform-QAT400已实际完成前3步，peak18.4238944054GiB，无失败。

- Llama Uniform-PTQ同包C4完成：PPL81.13316772229233/NLL4.396091850662515，2096128 targets，112实际INT8输入，量化状态前后不变，已入summary。validation/test/C4均为同一`uniform-down-readapt-s42/static_w4a8.pt`。这是完整Uniform-PTQ，仍不代替运行中的Uniform-QAT400。

- 完整Llama Uniform-PTQ seed42最终独立只读审计PASS：同包三split、112INT8、量化状态、历史token/分段对齐及summary均一致；详见auditor/UNIFORM_POSTPROCESS_20260918.md最终节和uniform_ptq_final_checks_20260918.json。
- 已生成`CORE_RESULTS.md`，由`summary.csv`直接汇总，B100与中间候选不进入完整方法表；seed43/44与在途QAT明确未完成，不预先计算三seed统计。

- `CORE_RESULTS.md`渲染与新增BF16相对指标独立限定审阅PASS（`auditor/SUMMARY_RENDER_REVIEW_20260918.md`）：当前15条核心记录的PPL/NLL/Delta NLL/PPL比值/targets均从summary正确计算；缺seed不汇总，三seed统计仍未完成。

- 2026-09-18T03:32:49+08:00 实际进程等待核对：qwen3-1p7b/joint100-s42 PID1387607 step36；llama32-1b/uniform-qat400-s42 PID1574860 step136；llama32-1b/initial-sp2-qat400-s42-r25 PID1426248 step324。三进程仍存活，均无当前failure；最终QAT验收待固定400完成。

- Llama代表消融初始SP2直接QAT400完成：validation PPL14.756133058231322/NLL2.6916587969227876，252728 targets，112W4/96INT8/16SP2。旧run steps1–25与r25 run steps26–400恰好连续400步，不早停。最终包`initial-sp2-qat400-s42-r25/checkpoint-0400/static_w4a8.pt`已入summary；完整test启动GPU0/PID1812163/run `initial-sp2-qat400-test-s42`，随后同包C4。

- 消融完整test完成：PPL14.320253706327318/NLL2.6616748782978554，288934 targets，量化状态未变，已入summary。相同第400步包C4启动GPU0/PID1821547/run `initial-sp2-qat400-c4-s42`。独立训练审计初核400有效更新连续；迁移前存在未提交的部分计算，6553600仅指有效优化更新的token预算，不称实际设备累计处理量。

- Llama代表消融同包C4完成：PPL26.730584258553062/NLL3.285808387695454，2096128 targets，96INT8/16SP2，量化状态不变，已入summary。完整后处理再QAT400的历史C4为26.98478050458106，因此此seed的C4反而是直接QAT略好；validation/test则完整后处理更好。按固定方法分行保留，不据测试结果换主线包。训练独立审计PASS见auditor/INITIAL_SP2_ABLATION_20260918.md。

- 代表消融最终独立审计PASS：`auditor/INITIAL_SP2_ABLATION_20260918.md`和`initial_sp2_external_checks_20260918.json`。同一checkpoint0400的三split、test/C4原始IDs/metadata/分段、全部量化state及summary一致。完整后处理减直接QAT的Delta NLL：validation -0.011894587390206102；test -0.011648189467362347；C4 +0.009464634363233415。该必要消融已完成，不推广为三语料一致改善，不改固定主线。

- Llama完整Uniform-QAT400固定400步完成：validation PPL18.104944766122504/NLL2.8961850924602586，252728 targets，112W4/112静态INT8。training.jsonl恰好1–400，未提前停。固定`uniform-qat400-s42/checkpoint-0400/static_w4a8.pt`并行验收：test GPU2/PID2029776/run `uniform-qat400-test-s42`，C4 GPU0/PID2031344/run `uniform-qat400-c4-s42`。

- Uniform-QAT400完整test完成：PPL17.51898842630044/NLL2.863285345616469，288934 targets，量化state不变，已入summary。同包C4仍在运行。

- Uniform-QAT400同包C4完成：PPL38.77759462968858/NLL3.6578426218384275，2096128 targets，112实际INT8，量化状态不变，已入summary。Llama seed42核心INT8/SP2×PTQ/QAT四项及必要代表消融测量已齐；Uniform-QAT外部最终独立审计待收尾。Qwen主线与seed43/44仍未完成，不能关闭Phase5。

- Llama完整Uniform-QAT400同包外部审计PASS：`auditor/UNIFORM_QAT400_20260918.md`及`uniform_qat400_external_checks_20260918.json`。三split同包、112INT8无overlay、冻结state、历史输入IDs/metadata、分段指标与summary一致。seed42 SP2减Uniform的Delta NLL：validation -0.21642088292767703；test -0.213258656785976；C4 -0.36256959977974024。Llama seed42核心矩阵正式闭合；三seed仅1/3，未汇总。

- 2026-09-18T05:08:36+08:00 Qwen Joint100实际进程等待核对：GPU5/PID1387607仍存活，已到80/100，无failure。Llama seed42核心矩阵与代表消融均复用已完成结果；Qwen下游仍等待同一Joint100完整包。

- C4源文本覆盖已独立补齐：`auditor/C4_TEXT_COVERAGE_20260918.md`及同名JSON。相同4480篇源文档、2097152输入tokens/2096128预测targets；Llama覆盖4382篇完整文档及下一篇996/4569字符，拼接前缀9840845字符；Qwen覆盖4325篇完整文档及下一篇40167/67339字符，拼接前缀9659562字符。两个边界均未切Unicode字符。Llama旧tokenizer原offset不可靠，边界由同环境缓存前缀无清理解码与源文本逐字相等定位；缓存与metadata未修改。跨tokenizer绝对PPL不作为模型排名。
- 最新进度核对：Qwen Joint100 seed42已完成91/100更新，progress仍为training、无failure；等待固定100及其导出/SP2校准/validation完成后接续后处理。

- 为补齐handoff §5.2“实际可运行”而非仅源码存在的证据，独立verifier启动唯一外部SpinQuant本地适配候选的小规模smoke：1次真实2048-token仅R更新、单训练窗GPTQ W4/group32、动态A8及finite forward；不追加正式100步或PPL、不增加模型/方法。初次GPU2/PID2245736在检查脚本处理参数tuple时退出，未进入模型阶段；失败记录保留，修正检查脚本后续跑未完成项，应用源码不改。目录`verifier/external-spinquant-smoke`。Qwen主线当前95/100，未受影响。

- 外部唯一候选本地适配SpinQuant有限smoke独立PASS：`verifier/external-spinquant-smoke/EXECUTION.md`、`result.json`、`completion_audit.json`。实际1次2048-token仅R更新（17个R有限非零梯度及实际变动）→112矩阵GPTQ W4/group32/单校准窗→112动态A8 wrapper有限前向，16在线down Hadamard；数据仅WikiText train。两次检查脚本准备错误保留，仅复用已更新R续跑未完成PTQ，无重复R更新。实际Python PID2258826已退出、GPU2释放；应用源码未改。仅证明本地核心调用路径可运行，不是100步/128窗、CLI Trainer/DDP或正式外部PPL，不入核心结果表。Qwen主线99/100，无failure。

- Qwen Joint100 seed42正式completed：100updates/global8，训练耗时及导出合计13351.280987806036秒，初始SP2 validation PPL16.340940643824144/NLL2.793673654716446，262208 targets，196W4/168INT8/28SP2。`joint100-s42/checkpoint-0100/{state.pt,static_w4a8.pt,sp2_calibration.json,validation.json}`已保存、无failure，validation已入summary。
- 接续Qwen固定主线：down邻码`sp2-down-round-s42`，GPU2/PID2301066；匹配INT8 train-only校准/完整包`uniform-initial-s42`，GPU0/PID2302293；初始SP2固定C4`b100-sp2-c4-s42`，GPU5/PID2304883。均由tmux承载，完整命令在各.launch.json。INT8导出后接同预算三阶段后处理及初始INT8 C4格式配对；不以B100结果代替最终Uniform。

- Qwen匹配INT8初始化`uniform-initial-s42`已completed：28 down层×50候选，24.398999647935852秒，实际capture与初始SP2一致、原量化state不变，完整INT8包已导出。Uniform down邻码接续GPU0/PID2308657/run `uniform-down-round-s42`，同Joint参考和parent/512/2048/8192候选；初始INT8 C4 GPU3/PID2310163/run `b100-int8-c4-s42`。SP2邻码GPU2/PID2301066及初始SP2 C4 GPU5/PID2304883仍在运行。独立auditor开始核对首次Qwen实际Joint/匹配格式证据，不做GPU复跑。

- Qwen初始SP2 C4 `b100-sp2-c4-s42`已completed：PPL27.18323240302331/NLL3.3026003274437667，2096128 targets，168INT8/28SP2、量化状态不变；已入summary的B100-initial-SP2归因行。不是精确SP2-PTQ或QAT400最终结果。匹配INT8 C4仍在运行。

- Qwen初始INT8 C4 `b100-int8-c4-s42`已completed：PPL33.33932032786121/NLL3.5067374910279026，2096128 targets，196INT8、量化state不变；已入summary。首次初始格式C4配对齐：SP2减INT8 Delta NLL=-0.2041371635841358。仅支持本次同Joint100/匹配校准的初始格式差异，不替代完整Uniform-PTQ/QAT400；两支后处理仍在执行。

- 首次Qwen实际Joint100/初始格式配对独立审计PASS：`auditor/QWEN_JOINT_FORMAT_20260918.md`及JSON。更新恰1–100/global8，1638400输入token visits；seed42校准索引独立复算一致。两包196W4/SW记录、154高精度张量（含56Q/Knorm）、168非downSA逐字节相同，仅28down格式/范围变化；28层均50候选且capture bounds/counts匹配。两C4同缓存/metadata/分段、196量化state冻结、summary与原JSON一致。未保存capture数组，不声称其逐元素实测一致。

- Qwen SP2首轮down邻码`sp2-down-round-s42`已completed：28层，train-selection NLL2.8335653562108885→2.8035783728731345，validation PPL15.930924848715014/NLL2.768262179280969，262208 targets；已入summary。固定收缩接续`sp2-range-s42`，GPU2/PID2377170，倍率1/.875/.75/.5/.25/.125，同Joint100参考，未改候选。

- Qwen Uniform首轮down邻码`uniform-down-round-s42`已completed：28层，train-selection NLL3.0333426037460383→3.0204473088695813，validation PPL19.521916498772708/NLL2.9715377574550645，262208 targets；已入summary。同六倍率收缩接续`uniform-range-s42`，GPU0/PID2381502。当前两支范围运行中，下一步各自down再适配；未进入QAT400。

- Qwen SP2范围收缩`sp2-range-s42`已completed：train-selection NLL2.8035783728731345→2.766265582072881，validation PPL15.384419587171699/NLL2.7333552821716736，262208 targets；已入summary。最后down邻码再适配`sp2-down-readapt-s42`接续GPU2/PID2401790，同参考和512/2048/8192预算。

- Qwen Uniform范围收缩`uniform-range-s42`已completed：train-selection NLL3.0204473088695813→2.9303037690963833，validation PPL17.887933411651215/NLL2.884126374450734，262208 targets；已入summary。最后down邻码再适配`uniform-down-readapt-s42`接续GPU0/PID2404048，同Joint100参考和512/2048/8192预算。SP2再适配GPU2/PID2401790也已启动；精确PTQ/QAT400尚未完成。

- Qwen首轮邻码/范围四个已完成阶段独立限定审计PASS：`auditor/QWEN_POSTPROCESS_20260918.md`及JSON。同seed32完整train窗24fit/8selection、28层覆盖及父候选/预算/贪心prefix核对一致。round仅down码改变（SP2 13层、INT8 7层），range仅down尺度收缩（SP2 8层、INT8 11层）；所有154高精度张量含56Q/Knorm及非down状态不变。Uniform layer2的[0,14]来自继承收敛退出，终止gain未保存、不冒称复算。validation/summary精确匹配；在途再适配与最终PTQ尚未纳入该限定PASS。

- Qwen两支down再适配均completed，精确PTQ包为各`*-down-readapt-s42/static_w4a8.pt`。SP2 train-selection NLL2.766265582072881→2.7368355058754466，validation PPL14.935424095235582/NLL2.7037358473302344；Uniform train-selection NLL2.9303037690963833→2.907194164907816，validation PPL17.491760837320335/NLL2.861729960767992；各262208targets，已入summary的SP2-PTQ/Uniform-PTQ正式阶段行。
- Qwen WikiText-only SP2-QAT400已启动：GPU2+5/PID2477067/run `sp2-qat400-s42`，按层两卡、全局8×2048、400固定、同Joint100参考、T1/.9KL+.1CE/Adam W1e-5/尺度相对.001/data_start800，checkpoints100/200/400，完整validation仅400，不早停。启动尚不等于完成首步。
- 两份精确PTQ完整test并行启动：SP2 GPU0/PID2479254/run `sp2-ptq-test-s42`，Uniform GPU3/PID2480190/run `uniform-ptq-test-s42`。每份test完成后原包接固定C4，无重校准；GPU0/3验收完成后接完整Uniform-QAT400同预算。

- Qwen SP2-QAT400已实际完成4/400更新，主卡峰值allocated14.99091100692749GiB，无failure；两卡全局预算保持不变。精确PTQ完整test均completed：SP2 PPL14.302163393485294/NLL2.6604108120809626，Uniform PPL16.78537625006989/NLL2.8205080460324137；各298931targets，实际格式168INT8+28SP2/196INT8，量化state冻结，均已入summary。
- 对应同包固定C4接续：SP2 GPU0/PID2485935/run `sp2-ptq-c4-s42`，Uniform GPU3/PID2486982/run `uniform-ptq-c4-s42`。完成后GPU0/3接匹配Uniform-QAT400，参数与主线一致且各用本格式精确父包。

- Qwen两支最终down再适配独立限定审计PASS，`auditor/QWEN_POSTPROCESS_20260918.md`追加最终节及`qwen_readapt_checks_20260918.json`；仅down码变化（SP2 7层、Uniform12层），尺度/非down码/154高精度张量含56QKnorm不变，预算及validation/summary一致。此前四阶段未重复审计。
- Qwen精确PTQ同包C4均completed：SP2 PPL24.903332773203157/NLL3.215001640827423，Uniform PPL29.322131492633147/NLL3.3783425719420435；各2096128targets，实际168INT8+28SP2/196INT8，量化state未变，已入summary。两包validation/test/C4均齐，外部最终独立审计待补。
- 匹配Qwen Uniform-QAT400已启动：GPU0+3/PID2508742/run `uniform-qat400-s42`，完整INT8精确父包、同Joint参考、WikiText-only、固定400、global8×2048，参数与SP2-QAT一致。SP2-QAT已提交第25次更新并写首次恢复文件；不是停滞/失败，不重启。当前两支QAT均未完成。

- Qwen两份精确PTQ最终外部独立审计PASS：`auditor/QWEN_POSTPROCESS_20260918.md`最终节和`QWEN_PTQ_ACCEPTANCE_20260918.json`。各方法三split同包，外部196量化state冻结、同方法test/C4初态相等，实际格式及输入IDs/metadata/分段对齐各自BF16，summary六行精确一致。SP2减Uniform Delta NLL：validation -0.15799411343775738/test -0.16009723395145103/C4 -0.16334093111462034。validation未独立保存IDs/量化快照，审计依据冷加载源码、metadata及分段，不冒称对应tensor比较。QAT400/关键seed仍未完成。

- 2026-09-18T07:07:57+08:00 Qwen QAT运行节点：SP2已提交并保存第100步恢复状态，`sp2-qat400-s42/checkpoint-0100/training_probe.json`存在、无中间完整validation；继续同400步schedule。Uniform已提交75步、正在保存恢复状态。宿主两PID2477067/2508742存活，无failure；GPU0/2各约6.3/8.0GiB空余。此记录不是QAT完成或最终PPL。

- 2026-09-18T07:16:19+08:00 Qwen两支QAT均跨过100步：Uniform已保存100步恢复状态及checkpoint-0100/training_probe.json，并继续到101；SP2已到134。两支没有中间完整validation/test/C4，无failure，仍固定400步最终包；后续seed未提前启动。

- 2026-09-18T07:34:51+08:00 Qwen SP2-QAT已保存第200步恢复状态及checkpoint-0200/training_probe.json，并继续至202；Uniform训练至171。无中间完整validation或外部评测，无failure；仍执行同一400步schedule，未据probe改选包。

- 2026-09-18T07:44:30+08:00 Qwen两支QAT均跨过200步中间节点：SP2至244，Uniform已保存200步恢复状态及checkpoint-0200/training_probe.json并继续到202；两PID2477067/2508742存活、无failure。未做中间完整validation或外部评测；最终400及关键seed仍待完成。


### Qwen seed42 QAT 进度：SP2 已完成第 300 步更新

记录时间：2026-09-18T08:01:33+08:00（Asia/Shanghai）。SP2-QAT400 的 training.jsonl 已记录 step300，当前正在保存该步续跑状态；Uniform-QAT400 已到 step260。两支尚未产生最终 result.json/checkpoint-0400，均无 failure.json；前次主机检查 PID2477067、2508742 均存活。维持原两卡放置（SP2 GPU2+5、Uniform GPU0+3）及全局8×2048预算，不更改400步终点。仅为运行进度，不记入最终PPL矩阵；最终同包test/C4与首个Qwen主线包独立test复核仍待完成。


### Qwen SP2-QAT400：训练更新完成，导出和验收仍在进行

记录时间：2026-09-18T08:33:21+08:00。`qwen3-1p7b/sp2-qat400-s42/training.jsonl` 已连续记录 step1–400，objective/CE/KL/梯度范数均有限；第400步累计训练输入token visits为6553600。PID2477067 正在保存最终续跑状态，此刻最终static包、validation和result尚未完成，不能记为最终PPL验收完成。Uniform-QAT400继续运行（最新step363）。固定400后才分别导出、记录validation并衔接各自同包完整test/C4，独立主线test复核仍待执行。


### Qwen SP2-QAT400 已完成，最终同包 test/C4 已启动

时间：2026-09-18T08:39:01+08:00。`qwen3-1p7b/sp2-qat400-s42/result.json` 记录 completed_steps=400、target_reached=false；旧训练PID2477067已退出。固定最终包 `checkpoint-0400/static_w4a8.pt` 的完整validation PPL=14.813941218960903、NLL=2.6955687116448495，262208 targets，已登记summary/CORE_RESULTS。实际覆盖196 W4 / 168静态INT8 / 28静态SP2，BF16 KV16 prefill。训练更新预算6553600输入token visits；正式训练完成不代表外部验收完成。

同一个最终包已启动两个tmux任务：`sp2-qat400-test-s42`，PID2988651/GPU2；`sp2-qat400-c4-s42`，PID2990965/GPU5。后者使用既有`qwen3-1p7b/c4-data/input_tokens.pt`，未重新校准。独立auditor开始核对本次实际400步及最终包证据；独立verifier首次完整test复核仍待主执行test完成后进行。Uniform-QAT400仍在运行。


### Qwen SP2-QAT400 同包三项评测完成；首次独立完整test PASS

固定400步包：`qwen3-1p7b/sp2-qat400-s42/checkpoint-0400/static_w4a8.pt`。validation PPL14.813941218960903/NLL2.6955687116448495；完整test PPL14.151216160997155/NLL2.649800568169599（298931targets）；固定C4 PPL24.24703443394153/NLL3.1882943185214367（2096128targets）。全部同包，test/C4尺度前后不变，均已登记summary.csv/CORE_RESULTS.md。test和C4正式PID已正常退出。

独立完整test仅执行一次，见`verifier/QWEN_SEED42_FINAL_TEST_20260918.md`及JSON；整体和3分段与主执行完全相同，146整窗+70token尾窗（69targets），196量化器状态不变、196W4/168INT8/28SP2覆盖及56QK norm/head解绑核对PASS，PID3000641已退出。独立QAT训练/validation审计见`auditor/QWEN_SP2_QAT400_20260918.md`；400连续更新、Adam全部step400、154高精度张量冻结PASS，test/C4联合审计正在补齐。Uniform已记录第400步更新，最终导出/validation尚在进行，不记作完成。


### Qwen Uniform-QAT400 已完成；同包test/C4验收启动

`uniform-qat400-s42/result.json`记录completed_steps=400、target_reached=false；训练PID2508742已退出。第400步包`checkpoint-0400/static_w4a8.pt`完整validation PPL15.715716054138214/NLL2.7546612342218633、262208targets，实际196W4/196静态INT8，已登记summary及CORE_RESULTS。这是真实完整Uniform-QAT400，不是B100格式overlay。

同一个最终包的完整test已在`uniform-qat400-test-s42`启动（GPU0/PID3025467），固定C4在`uniform-qat400-c4-s42`启动（GPU3/PID3027074）；均tmux、固定输入、无校准或训练。独立auditor继续核对实际400步、最终包和同包三项评测；seed43/44仍在seed42核心矩阵完成后执行。


### Seed42核心矩阵完成，进入43/44配对实验

Uniform最终test PPL15.055184987698569/NLL2.7117224493174277；C4 PPL27.10478088408928/NLL3.2997101287131727。独立auditor同包三split/196INT8静态状态/固定输入与指标/summary核对PASS，见`QWEN_UNIFORM_QAT400_20260918.md`和`QWEN_UNIFORM_QAT400_EXTERNAL_20260918.json`。Qwen QAT SP2减Uniform的Delta NLL为validation−0.059092522577013806、test−0.06192188114782882、C4−0.11141581019173596。结论仅限seed42，未生成三seed均值或标准差。

下一阶段固定为两模型各自seed43/44的完整方法与Uniform-QAT400配对：每个模型/seed重新初始化R和校准索引，共用该seed Joint100，再分别格式校准、三步后处理、WikiText-only QAT400及最终同包test/C4。采用已验证源码和原预算；不追加BF16、代表消融、B100 C4诊断或PTQ外部主表评测。新run使用init-s43/init-s44及joint100-s43/joint100-s44，GPU按真实余量选择。


### Seed43/44 四项初始化实际启动

均为tmux socket `rotation-quant-phase5`，命令/环境/源码快照见各run同目录外`.launch.json`与run内settings/source。实际启动如下：

| 模型/seed | GPU | PID | RUN_DIR（相对runs/phase5） |
|---|---:|---:|---|
| Qwen43 | 2 | 3068390 | qwen3-1p7b/init-s43 |
| Qwen44 | 5 | 3070256 | qwen3-1p7b/init-s44 |
| Llama43 | 0 | 3071859 | llama32-1b/init-s43 |
| Llama44 | 3 | 3073100 | llama32-1b/init-s44 |

实际模型加载/初始尺度校准已开始；没有将启动视为完成。Qwen用隔离4.51.3环境，Llama用旧4.44.2环境。各完成initial.pt后接相同seed Joint100，global8×2048/100updates/warmup10/cosine100/RLR1.5/尺度Adam相对0.001。此次无应用源码修改、无新suite/GPU保险复测。


### Seed43/44 初始化完成，四项 Joint100 实际运行

四份initial.pt均完成，saved seed与data seed实际为43或44；32个校准索引与该seed Generator结果一致，Qwen29张R/Llama17张R在43和44之间均不同。轻量CPU证据见`seed_initialization_43_44.json`；未重复架构或完整PPL检查。

| 模型/seed | GPU | Joint100 PID | RUN_DIR（相对runs/phase5） |
|---|---:|---:|---|
| Qwen43 | 2 | 3089096 | qwen3-1p7b/joint100-s43 |
| Qwen44 | 5 | 3098418 | qwen3-1p7b/joint100-s44 |
| Llama43 | 0 | 3090518 | llama32-1b/joint100-s43 |
| Llama44 | 3 | 3100833 | llama32-1b/joint100-s44 |

以上均tmux、100updates/global8×2048/warmup10/cosine100/RLR1.5/尺度Adam相对0.001。每项完成checkpoint-0100后共享给本seed两格式分支；额外seed暂无最终PPL。初始化旧PID已由完成状态替代，不把这些新任务启动当作完成。


- 四项Joint100首个实际更新均已发生：Qwen43/44当前step2/1、Llama43/44 step2/1；每步cumulative_train_tokens增加16384，当前minibatch NLL有限、无failure.json。首步峰值分配约Qwen5.304GiB、Llama4.392GiB（仅此时观察，不代替全程峰值）。四个PID已由主机ps确认存活。后续按固定100步完成校准与导出，当前并无seed43/44最终质量结果。


### 关键seed Joint100持续运行

2026-09-18T10:56:41+08:00。四个已记录PID经主机ps确认均存活；无failure，GPU0/2/3/5各有17GiB以上余量。

- qwen3-1p7b seed43：52/100，连续更新/有限minibatch NLL/累计token预算检查=True。
- qwen3-1p7b seed44：50/100，连续更新/有限minibatch NLL/累计token预算检查=True。
- llama32-1b seed43：81/100，连续更新/有限minibatch NLL/累计token预算检查=True。
- llama32-1b seed44：74/100，连续更新/有限minibatch NLL/累计token预算检查=True。

尚未到各自100步导出；后续按完成顺序接格式校准与三步后处理，不改变固定配方。


### Llama seed43 Joint100完成，进入两格式分支

`llama32-1b/joint100-s43` completed100，无failure；checkpoint-0100/{state.pt,static_w4a8.pt,sp2_calibration.json,validation.json}已完成。初始SP2 validation PPL17.32269586594728/NLL2.8520175414808633、252728targets，已按B100-initial-SP2登记summary；这是开发阶段值，不是最终seed结果。SP2校准已由Joint100自动执行，不重复。

后续tmux：`sp2-down-round-s43` GPU0/PID3697102，父包为该Joint100初始SP2，同R参考为checkpoint-0100/state.pt，候选parent/512/2048/8192；`uniform-initial-s43` GPU1/PID3698944，使用同一Joint100 data.json及SP2校准记录，执行匹配INT8校准并导出完整包，随后进入本格式三步后处理。其余Llama44、Qwen43/44继续原Joint100。


- Llama43 `uniform-initial-s43` 已completed（18.81479156005662秒），完整INT8包已导出。接续 `uniform-down-round-s43` GPU1/PID3709002，tmux同名session；parent为uniform-initial-s43/static_w4a8.pt，reference为本seed Joint100 state.pt，milestones512/2048/8192。SP2 down邻码GPU0/PID3697102同时运行；Llama44和Qwen43/44仍继续各自Joint100。


### Llama seed44 Joint100完成，两格式分支启动

`joint100-s44` completed100，无failure；初始SP2 validation PPL17.117556465258705/NLL2.8401046306937046、252728targets，已按B100-initial-SP2记summary。SP2校准及checkpoint-0100包/同R参考已自然导出，不重复校准。

新tmux：`sp2-down-round-s44` GPU3/PID3766558；`uniform-initial-s44` GPU7/PID3768360。两支父状态共用本seed Joint100，INT8读取本seed相同校准索引与SP2校准记录。GPU7启动前有约19GiB可用显存，校准及后处理按实际余量安排；后续QAT需要重新评估显存，不能直接套用此时余量。seed43两支邻码仍继续，Qwen43/44仍原Joint100。


### Llama43首轮邻码完成，两格式范围收缩运行中

SP2：train-selection NLL2.827238092585334→2.776798308614551，validation PPL16.614797384228233/NLL2.81029370698217；Uniform：train-selection NLL6.892943425322137→6.468229704699745，validation PPL945.6594870086818/NLL6.851882553946269。两项中间值已登记summary，不作为最终Uniform/PTQ或QAT结果。

接续`sp2-range-s43` GPU0/PID3785225、`uniform-range-s43` GPU1/PID3790349；均显式alpha-factors1/.875/.75/.5/.25/.125，train-only选择，之后仍需down再适配与QAT400。

Llama44匹配INT8校准已完成（25.831584536936134秒），`uniform-down-round-s44`已接续GPU7/PID3780034，SP2首轮邻码仍GPU3/PID3766558。两Qwen原Joint100继续。


### Llama43两格式范围收缩完成，接续down再适配

SP2范围选择：train-selection NLL2.776798308614551→2.7589552842354874；validation PPL16.216045719451984/NLL2.786001228549522。Uniform范围选择：train-selection NLL6.468229704699745→3.698355611448475；validation PPL48.92127282760269/NLL3.890212329033289。均已按中间阶段写summary，未在test/C4上选择或调整候选。

接续`sp2-down-readapt-s43` GPU0/PID3816753、`uniform-down-readapt-s43` GPU1/PID3819641，各以自身range包为父、共享本seed Joint100浮点参考，parent/512/2048/8192候选。完成后才从两份精确父包分别干净启动WikiText-only QAT400。

新增seed身份限定审计已PASS：`auditor/LLAMA_SEEDS43_44_JOINT_20260918.md`及JSON；实际采样/100updates/global8/各seed两格式权重和非down状态匹配，无新增GPU复测。seed44两支首轮邻码及Qwen43/44 Joint100继续运行。


### Llama43 SP2 再适配异常退出及前缀恢复；Llama44 接续范围收缩

2026-09-18约11:52，`sp2-down-readapt-s43` PID3816753日志停在layer4坐标384；随后主机ps与tmux确认原进程/session均不存在，无failure.json、无最终包。不能把该任务记为暂停中或completed；退出原因未明，当前账号不能读取内核日志，未归因于用户操作或OOM。执行器未发送暂停/终止命令。其余运行任务的PID/进度持续，Llama44首轮round自然completed。

旧run的prefix.pt可读，完成layers0–3、16个down尺度、updates为空、selection NLL2.7589552842354874，与原父包一致；旧目录/日志/选择记录保留。使用现有恢复入口启动`sp2-down-readapt-s43-r4` GPU0/PID3869795，`--resume-prefix .../sp2-down-readapt-s43/prefix.pt`，同seed43、原range父包和Joint100参考、原512/2048/8192候选。从layer4继续；未完成层的部分计算需重做，不将其声称为零重复算力。独立auditor正在核对这次恢复，不增加GPU复测。最终父包将取恢复run完成包（若无任何收益则按既有规则保留原父包）。

Llama44 SP2首轮round validation PPL16.54402434818432/NLL2.8060249700478757；Uniform PPL546.933827494308/NLL6.304327821601918，均为中间阶段值，已记summary。接续`sp2-range-s44` GPU3/PID3869801、`uniform-range-s44` GPU7/PID3869807；显式六档收缩，随后各自down再适配。Qwen43/44原Joint100及Llama43 Uniform原再适配继续；未改变正式QAT400预算。


### Llama43 Uniform 进入 QAT400；Llama44 两格式进入再适配

`uniform-down-readapt-s43` completed，validation PPL45.19500691896424/NLL3.8109866143417275；精确PTQ父包已导出，按Uniform-PTQ-parent登记开发值，不新增额外seed的PTQ test/C4。接续`uniform-qat400-s43`，GPU1/PID3893776，tmux；本seed同R参考joint100-s43/checkpoint-0100/state.pt，原seed42同配方，400updates/global8×2048/warmup10/cosine400/masterAdam1e-5/scale相对.001/T1/CE.1/data_start800/WikiText-only/teacher-body offload；resume25，checkpoint100/200/400，完整validation仅400。未提前停止或按评测选权重。

Llama44两格式range completed：SP2 validation PPL16.237405902912396/NLL2.787317586932191；Uniform PPL45.938361575446926/NLL3.827300532181054。两项中间值登记summary。接续`sp2-down-readapt-s44` GPU3/PID3893782及`uniform-down-readapt-s44` GPU7/PID3895353；各自range父包、本seedJoint100参考、原round候选预算。Llama43 SP2恢复run仍在运行，原完整前缀已实际恢复到layer4后继续，不重复完成的前4层。


- Llama43 SP2前缀恢复限定独立审计PASS：`auditor/LLAMA43_READAPT_RESUME_20260918.md/.json`；新旧data完全相同、六份相关源码相同、原4层前缀和16尺度正确恢复，实际从layer4继续。完整逐层记录需合并旧run的0–3和恢复run后续层。旧layer4至少384坐标迭代及fit/capture是丢弃重做计算，有效候选未扩大；中断原因仍未定。
- Llama43 `uniform-qat400-s43` 已发生实际训练更新（核查时step2），无failure；GPU1显存约19415MiB/24564MiB。Qwen43/44原Joint100核查时85/81，SP2恢复至layer11，Llama44两支再适配均实际运行；六PID均由宿主ps确认，未仅凭progress文件判断存活。


### Llama43 SP2恢复完成，两格式均进入QAT400

`sp2-down-readapt-s43-r4` completed，16个completed_targets齐全；旧run四层加恢复run十二层组成完整逐层记录。train-selection NLL2.7589552842354874→2.756980167181222；validation PPL16.2149972123967/NLL2.7859365678441694。精确父包为恢复目录`static_w4a8.pt`，开发值按SP2-PTQ-parent记录summary，不新增额外seed PTQ外部评测。独立恢复核对PASS已记录；异常旧目录保持不动。

接续`sp2-qat400-s43` GPU0/PID3925847，tmux；parent=上述恢复最终包，reference=joint100-s43/checkpoint-0100/state.pt。完整命令除parent/output外复制匹配Uniform-s43的400步固定参数：WikiText-only/global8×2048/data_start800/Adam1e-5/尺度相对.001/T1/CE.1/warmup10/cosine400/offload-teacher-body/resume25/checkpoints100,200,400/完整validation仅400。GPU0启动前2304MiB/24564MiB；运行峰值仍持续观察。Uniform-s43已保存第25步恢复点且继续训练。Llama44两格式再适配、Qwen43/44 Joint100继续原进程。

- Llama43 SP2-QAT首个实际更新检查已完成：step4，连续4步、每步8窗、累计tokens=step×16384、objective有限；GPU0实际21688MiB/24564MiB，峰值allocated18.423856GiB。同期Uniform-QAT43 step43，Llama44再适配约8/16层，Qwen43/44 Joint100 step88/85，六个当前PID均经宿主ps确认存活，无failure。


### Llama44精确父包完成，四条额外seed QAT均已启动

Llama44 SP2再适配completed16层，train-selection NLL2.745155764548289→2.737697194267978，validation PPL16.147560805984856/NLL2.781769004576782；Uniform再适配completed16层，train-selection NLL3.6882648128986584→3.589305168146256，validation PPL41.44517090665076/NLL3.724371370732677。精确父包分别为`sp2-down-readapt-s44/static_w4a8.pt`及`uniform-down-readapt-s44/static_w4a8.pt`；开发值记SP2/Uniform-PTQ-parent，不新增额外seed PTQ外部验收。

接续`sp2-qat400-s44` GPU3/PID3984664及`uniform-qat400-s44` GPU7/PID3984670，均tmux、各自精确父包及joint100-s44同R参考，复制已运行seed43的完整固定QAT400参数（global8×2048、WikiText-only、masterAdam1e-5、尺度.001、data_start800、warmup10/cosine400、T1/CE.1、offload teacher、resume25、checkpoint100/200/400、完整validation仅400）。GPU3/7启动前实际占用1232/1916MiB，24GiB卡、已测单卡QAT峰值allocated约18.424GiB；继续观察真实余量。

Llama43/44四条QAT均已启动但均未完成正式400步。Qwen43/44仍原Joint100；训练中间指标不替代最终test/C4。

- Llama44新QAT两支已实际完成step4：每步8窗、累计tokens=step×16384、objective有限，无failure；GPU3/7实际20024/20668MiB（均24564MiB卡）。同次宿主ps确认六项当前任务全部存活。Llama43 SP2/Uniform分别63/100步（100为中间节点，仍继续固定400）；Qwen43/44 Joint100为93/90步。


- Llama43/44后处理与QAT启动合并限定审计PASS：`auditor/LLAMA_SEEDS43_44_POSTPROCESS_20260918.md/.json`。12阶段全部16层、同seed32训练窗24/8、parent/reference及逐层NLL接续；43恢复前后记录合并无漏层，summary/QAT正确指向-r4精确包。三处Uniform inherited坐标早收敛预算如实记录，其余遵循固定候选上限；四项QAT参数匹配400/global8/WikiText-only，当前只确认启动/抽样更新，不是QAT最终验收。没有GPU/PPL复跑。核查时Qwen43/44 Joint100为98/94，六PID均宿主存活。


### Qwen43 Joint100完成，进入两格式分支

`qwen3-1p7b/joint100-s43` completed；100条连续训练更新、每步16384输入token访问检查通过。checkpoint-0100/{state.pt,static_w4a8.pt,sp2_calibration.json,validation.json,reload_check.json}已导出，无failure。初始SP2 validation PPL16.04792947249727/NLL2.7755798364253557，按B100-initial-SP2开发行登记summary，不作为最终seed结果或新增B100外部评测。SP2自动校准直接复用。

接续tmux：`sp2-down-round-s43` GPU2/PID4075184，parent=该Joint100静态包、reference=同checkpoint state.pt、round候选parent/512/2048/8192；`uniform-initial-s43` GPU2/PID4075190，读取该Joint100相同data.json与SP2校准记录，导出匹配INT8完整包后接本格式round。

调度依据：GPU2在旧Joint100退出后实际20MiB/24564MiB；seed42实测SP2/Uniform round最大reserved8.2324/8.0449GiB、Uniform校准3.9707GiB，两项同卡可留余量。优先在本任务释放的GPU并行，GPU6当前他人计算繁忙不追加负担；不设总并行数上限。Qwen44仍原Joint100，Llama四QAT继续。新任务实际峰值持续观察。

- Qwen43匹配INT8校准`uniform-initial-s43`已completed，原PID4075190自然退出；`uniform-down-round-s43` GPU2/PID4080976已接续，parent为该校准完整包、reference为joint100-s43/checkpoint-0100/state.pt、原round预算。启动前GPU2占用8917MiB/24564MiB，SP2邻码PID4075184正在运行。Qwen44同次核查97/100，原PID3098418存活。


### Qwen44 Joint100完成，两个额外seed均进入两格式分支

`joint100-s44` completed，100条连续更新与每步16384输入token visits检查通过，checkpoint-0100下完整state/static/SP2 calibration/validation/reload记录已导出。初始SP2 validation PPL16.316429520957282/NLL2.7921725462471594，开发行按B100-initial-SP2登记，不新增B100外部评测。原PID3098418自然退出。

接续tmux：`sp2-down-round-s44` GPU5/PID4109819、`uniform-initial-s44` GPU5/PID4109825，均同seed44 Joint100父状态、同校准indices，SP2 round使用state.pt同R参考及原候选预算。GPU5启动前569MiB/24564MiB；按已测峰值同卡并行，校准完成后再接Uniform round。Qwen43两支round继续GPU2，Llama43/44四QAT继续；当前没有任何额外seed的最终QAT400外部结果。

- Qwen44匹配INT8校准`uniform-initial-s44`已completed，原PID4109825自然退出；`uniform-down-round-s44` GPU5/PID4116221已接续，以该完整INT8包为parent、joint100-s44/state为同R参考。启动前GPU5实际9672MiB/24564MiB，SP2 round PID4109819仍运行。至此Qwen43/44四条格式round均已启动；独立auditor正合并核对两seed Joint100/匹配初始化证据，不增加GPU/PPL复跑。


- Qwen43/44 Joint100/初始格式匹配独立限定审计PASS：`auditor/QWEN_SEEDS43_44_JOINT_20260918.md/.json`。每seed421个初始化参数=checkpoint0位级一致；实际32采样索引匹配seed；42/43/44两两29个R均不同，43/44 Joint后R均更新。两份100步global8、1638400输入/1637600预测visits正确。每seed两格式196W4/SW、168非downSA、154高精度含56QKnorm位级一致，实际168INT8+28SP2对196INT8；28层50候选/512rows/4096train tokens及最终尺度匹配，父quantizer在校准前后不变。validation聚合及summary一致，未GPU/PPL复跑。当前后处理/QAT仍运行，未把该PASS扩大为最终验收。
- Qwen四条round现均已实际进入搜索；GPU2/5同卡并行核查占用15816/18671MiB（均24564MiB卡），无failure。Llama四QAT继续，八个原PID均经宿主ps确认存活。


### Qwen43两格式首轮邻码完成，范围收缩已接续

两支completed28层，无failure。SP2 train-selection NLL2.69476843058009→2.6684236642399277，validation PPL15.910624656710956/NLL2.766987103471548；Uniform selection2.94840331490133→2.9412856828638385，validation PPL20.170829327393548/NLL3.0042374679970743。两项中间阶段值已登记summary，未新增test/C4。

原PID4075184/4080976自然退出，GPU2实际565MiB/24564MiB后接续`sp2-range-s43` PID64021及`uniform-range-s43` PID64027，均GPU2/tmux。各自round包为parent，joint100-s43/state为reference，显式alpha-factors1/.875/.75/.5/.25/.125、28层down、train-only选择；随后各自down再适配。Qwen44两支round及Llama四QAT继续。


### Qwen44首轮邻码及Qwen43范围收缩完成，后续阶段已接续

Qwen44两支round均completed28层：SP2 selection NLL2.7819858067461727→2.7567715166986013，validation PPL16.05017436240812/NLL2.775719713218456；Uniform selection2.987788166725654→2.9802183416111188，validation PPL19.60780237716365/NLL2.975927567490997。接续GPU5 `sp2-range-s44` PID109090、`uniform-range-s44` PID113524，均各自round父包、本seedJoint100参考、显式六档1/.875/.75/.5/.25/.125。

Qwen43两支range均completed28层：SP2 selection2.6684236642399277→2.6431519825772964，validation PPL15.47834976229269/NLL2.739442257975861；Uniform selection2.9412856828638385→2.847781625433573，validation PPL18.364192021237717/NLL2.9104026827273044。接续GPU2 `sp2-down-readapt-s43` PID113530、`uniform-down-readapt-s43` PID113536，各自range父包、Joint100同R参考，原parent/512/2048/8192预算。

以上四项中间开发值已记summary；不新增内部候选test/C4。旧父阶段PID均自然退出，启动前GPU2为20MiB/24564MiB，GPU5已有SP2 range仍有余量。所有新任务tmux，不修改应用源码或配方。同期Llama43 Uniform-QAT已接近400步，其余QAT继续；最终外部验收尚未开始。


### 首个额外seed QAT400完成：Llama43 Uniform，已启动同包外部验收

`llama32-1b/uniform-qat400-s43` completed，400条更新连续，实际每步8微批，6553600输入/6550400预测token visits，objective全部有限。最终validation PPL18.143936807329858/NLL2.898336444698109；固定第400步包`checkpoint-0400/static_w4a8.pt`已导出，按Uniform-QAT400/WikiText-only/seed43登记summary。不是按validation选择的中间包；独立最终状态审计仍待同包外部结果齐备后合并。

旧训练PID3893776自然退出、GPU1实际20MiB/24564MiB后启动两项tmux验收：完整WikiText-2 test `uniform-qat400-test-s43` PID123828；既有固定C4 `uniform-qat400-c4-s43` PID123834，均GPU1、同一上述第400步静态包。测试入口和C4 tokens均复用seed42协议，无重新校准/optimizer更新/格式overlay。此次仅必要最终验收，不重复首次独立完整test。

同次接续的Qwen43再适配和Qwen44范围收缩仍在执行；其余Llama三条QAT继续固定400步。三seed配对统计仍未齐。

- Llama43 Uniform-QAT400完整WikiText-2 test已completed：PPL17.575531211766275/NLL2.866507662660506，288934预测targets，已记正式summary。原test PID123828自然退出；同一checkpoint-0400包的固定C4 PID123834经宿主ps确认仍运行，尚无C4结果，不将两项启动视为均完成。

- Llama43 Uniform-QAT400同包固定C4已completed：PPL39.07188227779196/NLL3.66540308496798、2096128预测targets；已记summary并更新CORE_RESULTS。该最终包validation/test/C4三项均完成，独立auditor正核对完整400步、真实静态覆盖、冻结状态和三split同包/固定输入；不新增GPU复跑。seed43 SP2与seed44两支尚未完成全部验收，三seed配对统计仍未齐。


### Qwen44范围收缩完成，四条Qwen额外seed均进入再适配

Qwen44 SP2 range completed，selection NLL2.7567715166986013→2.7328470146278048，validation PPL15.763760989180486/NLL2.757713697404855；Uniform range completed，selection2.9802183416111188→2.9438710303420788，validation PPL18.817133157121283/NLL2.9347677929641733。两项开发值已记summary。

接续`sp2-down-readapt-s44` GPU1/PID156812，使用Llama43 Uniform最终评测完成后释放的GPU1（启动前20MiB），以及`uniform-down-readapt-s44` GPU5/PID160186（启动前569MiB）。各自本格式range父包、本seedJoint100同R参考，原parent/512/2048/8192候选，全28层down；均tmux。Qwen43两支再适配继续GPU2，四支完成后才进入各自固定QAT400。

- Llama43 Uniform-QAT400最终独立限定审计PASS：`auditor/LLAMA_UNIFORM_QAT400_S43_20260918.md/.json`。实际400连续更新与3200微批预算正确；最终112W4/112INT8，全部W4/SW/SA实际变化、74高精度张量冻结。test/C4同包量化状态逐张量前后不变、实际输入IDs/metadata与历史BF16及完整SP2一致、尾窗/分段聚合/summary三项指标一致。validation无单独input IDs或before/after快照，报告明确该证据限制。未GPU/PPL复跑；其余未完成seed不据此记PASS。


### Llama43 SP2-QAT400完成，同包test/C4已启动

`sp2-qat400-s43` completed，无failure；400条实际更新连续、global8、6553600输入/6550400预测token visits、objective全部有限。QAT阶段没有中断恢复，父包仍精确指向此前恢复完成的`sp2-down-readapt-s43-r4/static_w4a8.pt`；最终validation PPL14.66907305636532/NLL2.685741403816717，已登记SP2-QAT400/WikiText-only/seed43开发值。第400步静态包已完成，不挑中间checkpoint。

旧训练PID3925847自然退出、GPU0实际1757MiB/24564MiB后启动同包tmux验收：`sp2-qat400-test-s43` PID175632、`sp2-qat400-c4-s43` PID175638，均GPU0、同`sp2-qat400-s43/checkpoint-0400/static_w4a8.pt`。沿用完整test入口/既有固定C4 token文件，无重校准、overlay或optimizer更新。独立最终证据核对待结果齐备，不重复首次独立完整test。

- Llama43 SP2-QAT400完整test已completed：PPL14.20472174646267/NLL2.653574426692097、288934预测targets，已登记正式summary。test PID175632自然退出，同包C4 PID175638继续运行；不重复完整test。
- GPU7曾短时观测24078MiB/24564MiB；随后逐PID复核显示本QAT19390MiB、另一进程614MiB，占用回落约20.0GB，当前恢复点已正常保存、进程继续。未发送暂停/终止信号或实际迁移；继续按真实余量监测，不把短时显存波动归因为已发生OOM。

- Llama43 SP2-QAT400同包C4已completed：PPL26.639323032142762/NLL3.2823884336429527、2096128targets，已登记summary并更新CORE_RESULTS。seed43 SP2/Uniform的test与C4配对齐备：SP2 test14.20472174646267/C4 26.639323032142762；Uniform test17.575531211766275/C4 39.07188227779196。独立SP2最终审计进行中，引用已完成的父包恢复审计；不重跑PPL。三seed配对统计仍等待seed44。

- Llama43 SP2-QAT400最终独立审计PASS：`auditor/LLAMA_SP2_QAT400_S43_20260918.md/.json`。QAT自身400连续更新无恢复，精确父包为已恢复完成的sp2-down-readapt-s43-r4；实际112W4+96INT8+16SP2、74高精度张量不变。与Uniform43训练窗序列/预算匹配，test/C4实际IDs与量化器前后状态一致，三split指标和summary精度一致；validation无独立IDs/状态快照的限制保留。未GPU复跑。seed43配对ΔNLL（SP2−Uniform）validation −0.21259504088139192、test −0.21293323596840885、C4 −0.38301465132502743。

### 首个Qwen额外seed进入QAT400

Qwen44 SP2 down再适配completed28层，selection NLL2.7328470146278048→2.6820040097972013，validation PPL14.958610912101744/NLL2.705287114766046。开发值登记SP2-PTQ-parent，不新增PTQ外部评测。原PID156812自然退出后，GPU1实际20MiB、GPU0实际1757MiB；以该精确父包和joint100-s44同R参考启动`sp2-qat400-s44`，PID242215，GPU1/0，tmux。沿用seed42双卡放置、固定400步/global8/data_start800/WikiText-only/原生BF16 teacher与原优化器参数，未恢复/换配方。其他三支再适配仍运行，完成后分别接固定QAT400。

### Llama44两格式QAT400完成，最终test/C4验收中

两支均400连续更新、global8、6553600输入/6550400预测visits，objective有限；SP2 validation PPL14.604525985565553/NLL2.6813314796858645，Uniform19.286803756216127/NLL2.9594211188506145。已登记summary。原训练PID3984664/3984670自然退出后，GPU3约635MiB、GPU7约4451MiB；同各自checkpoint-0400包启动SP2 test PID249209/C4 PID249234（GPU3），Uniform test PID249260/C4 PID249383（GPU7），均tmux、无重校准或训练。最终独立审计待外部结果齐备合并。

Qwen44 Uniform再适配completed28层，selection NLL2.9438710303420788→2.9216023079614875，validation PPL18.423304287656563/NLL2.913616400605703，登记Uniform-PTQ-parent开发值。GPU5已释放，等待合适双卡余量接续固定QAT400；Qwen43两支再适配仍GPU2，Qwen44 SP2-QAT仍GPU1/0。

- Llama44两格式完整test均completed并登记：SP2 PPL14.197525321816363/NLL2.653067676308368，Uniform PPL18.734455723372943/NLL2.930364380495814，均288934targets含尾窗。原test PID249209/249260自然退出；两支同包C4 PID249234/249383经宿主ps确认仍运行。三seedtest行已齐，但整体统计/最终审计等待C4。

### Qwen额外seed后处理全部完成，seed44两格式QAT运行

Qwen43两支再适配completed28层：SP2 selection NLL2.6431519825772964→2.608141962339313，validation PPL14.985970044814936/NLL2.707114433062164；Uniform selection2.847781625433573→2.826277790346091，validation PPL18.22110026571323/NLL2.9025802778083905。两份精确父包已导出，开发值登记*PTQ-parent，不新增额外seed PTQ外部评测。原PID113530/113536自然退出，GPU2约20MiB。独立auditor合并核对43/44共12个后处理阶段，未GPU复跑。

接续Qwen44 `uniform-qat400-s44` PID270807，GPU2/5，tmux；启动前GPU5约1450MiB，沿用seed42全部QAT参数，仅改本seed父包/参考/输出，双卡global8、WikiText-only固定400。Qwen44 SP2 PID242215继续GPU1/0，Qwen43两份父包等待可用双卡。Llama44两支同包C4 PID249234/249383仍运行GPU3/7。

### Llama三seed最终测量齐备；Qwen三个QAT运行

Llama44两项C4均completed并登记：SP2 PPL26.515203495507034/NLL3.277718285291192，Uniform PPL45.24646580238375/NLL3.8121245632809497，各2096128targets。与各自val/test共用第400步包。Llama三seed汇总已生成CORE_RESULTS：test SP2 14.185554±0.027205、Uniform17.942992±0.686011；C4 SP2 26.713102±0.243327、Uniform41.031981±3.652816（样本SD）；配对ΔNLL均值test−0.234496/C4−0.426664。seed44最终独立合并审计及统计核验进行中，不将汇总字段检查当作完整审计。

Qwen43/44十二后处理阶段和十二条开发summary独立限定PASS：`auditor/QWEN_SEEDS43_44_POSTPROCESS_20260918.md/.json`。逐层28覆盖/父包/同seed参考/32train窗24fit8selection/尺度保持与范围选择均已核对；Uniform四个round/readapt的layer2在13坐标早收敛，部分候选由既有训练heldout-MSE前筛跳过NLL（null），不误报失败或全部执行满8192。不重复GPU/PPL或全W-code/HP状态比较。

原Llama C4 PID249234/249383自然退出，GPU7实际1526MiB、GPU3实际4458MiB；以Qwen43 SP2精确父包和本seed Joint100参考接续`sp2-qat400-s43` PID282520，GPU7/3，tmux，原双卡400预算。当前Qwen44 SP2 PID242215 GPU1/0、44 Uniform PID270807 GPU2/5、43 SP2 PID282520 GPU7/3运行；43 Uniform父包已完成，等待一组双卡释放，不抢占他人或使用GPU4。

- Llama44两格式最终证据及Llama42/43/44统计独立PASS：`auditor/LLAMA_QAT400_S44_20260918.md/.json`。400连续更新/global8/实际窗序列与优化规则、各自精确父包、同Joint参考、最终实际格式、74高精度张量冻结、test/C4量化状态前后及跨split相同、IDs/metadata/尾窗覆盖和六行summary已核对。18条原始指标及ddof1统计与CORE_RESULTS一致，未混入消融/中间行或取最佳seed。validation缺独立IDs/状态快照限制保留。Llama所需实验与独立证据已齐；Phase5仍等待Qwen额外seed四条QAT及最终验收，不提前关闭。

- 独立handoff交付覆盖限定复核完成：`verifier/HANDOFF_COVERAGE_REVIEW_20260918.md`。45条核心/消融/Llama三seed原始指标及4条B100 C4抽核一致，必要test/C4同包，已完成架构/首次冷包test/历史复用/外部候选有限smoke均有对应证据。未发现已完成主表重大证据缺口；明确剩Qwen43/44四个固定QAT400、八项同包最终test/C4、三seed统计及最终审计。保留validation无独立ID/量化状态快照和代表消融C4初始直接QAT略优的边界，未新增GPU/PPL/测试或门禁。报告中的运行步数为14:18:34快照，不替代实时进度。

### Qwen额外seed首个QAT400完成，最后一条训练已接续

Qwen44 SP2-QAT400 completed：连续400更新、global8、6553600输入/6550400预测token visits、objective全部有限；最终validation PPL14.803867869217726/NLL2.6948884891407743，固定checkpoint-0400静态包完成，精确父包sp2-down-readapt-s44/static_w4a8.pt。已登记SP2-QAT400/WikiText-only/seed44并更新CORE_RESULTS；不挑中间包。

原PID242215完成退出，GPU1约20MiB、GPU0约1757MiB；接续最后一条`uniform-qat400-s43` PID811495，GPU1/0，tmux，精确uniform-down-readapt-s43父包、本seed Joint100参考，沿用seed42双卡/global8/固定400全部参数。Qwen44 Uniform PID270807、43 SP2 PID282520仍保持原进程。

GPU6启动前18MiB/24564MiB、0%利用率，按seed42每项约6GiB reserved实测并行两项最终同包验收：`sp2-qat400-test-s44` PID811501与`sp2-qat400-c4-s44` PID811507，均GPU6/tmux、同sp2-qat400-s44/checkpoint-0400/static_w4a8.pt。完整test入口、既有Qwen固定C4输入，无重校准/overlay/训练。其启动不等于完成，最终审计待结果齐备。

### Qwen44两格式QAT400完成，最终同包验收中

Uniform44 completed且原PID270807退出，连续400/global8/6553600输入/6550400预测visits、objective有限。最终validation PPL15.752338332216512/NLL2.756988819789281，精确uniform-down-readapt-s44父包；已登记Uniform-QAT400/WikiText-only/seed44。GPU2/5释放后分别启动同包完整test `uniform-qat400-test-s44` PID827890（GPU2）与固定C4 `uniform-qat400-c4-s44` PID827896（GPU5），均tmux、同checkpoint-0400包，无校准/overlay/训练。

SP2-44完整test已completed并登记：PPL14.105051810881225/NLL2.6465330176752335，298931targets含尾窗。原test PID811501完成退出，同包C4 PID811507仍GPU6。独立auditor已完成SP2训练/validation限定PASS（QWEN_QAT400_S44_20260918.md/.json阶段快照），其余证据仍pending，不扩称最终完成；接续合并Uniform与外部结果。Qwen43 SP2 PID282520 GPU7/3、Uniform PID811495 GPU1/0继续固定400。

- Qwen44 SP2同包C4已completed：PPL24.17047498768119/NLL3.1851318464743503、2096128targets；其val/test/C4全部完成并登记。Uniform44完整test已completed：PPL15.140921674493724/NLL2.7174011229393273、298931targets含尾窗，已登记；同包C4 PID827896仍GPU5。已更新CORE_RESULTS，三seed统计仍待43最终结果。Qwen43 SP2与Uniform原训练PID282520/811495均存活，未重启。

- Qwen44 Uniform同包C4已completed并登记：PPL27.651997752899558/NLL3.319697976819438、2096128targets，原PID827896退出。至此seed44两方法各自同包val/test/C4全部齐备；SP2 test14.105051810881225/C424.17047498768119，Uniform test15.140921674493724/C427.651997752899558。两训练/validation、SP2 test/C4及Uniform test已独立PASS，最后Uniform C4补审中。Qwen三seed完整配对现2/3，不报整体统计；余Qwen43两条固定训练及最终验收。

- Qwen44两格式全部最终证据独立PASS：`auditor/QWEN_QAT400_S44_20260918.md/.json`。两支连续400/global8、同窗序列/W-LR/Joint参考；实际196W4+168INT8+28SP2或196INT8，各196W4/SW/SA实际变化，154高精度张量含56QKnorm位级冻结。test/C4唯一最终包、196量化器before/after及跨split状态、实际固定IDs/metadata/尾窗、分段聚合与summary全部匹配；validation快照限制保留。seed44配对ΔNLL（SP2−Uniform）validation−0.06210033064850684、test−0.07086810526409382、C4−0.13456613034508758。无GPU复跑，不扩称三seed已齐；余43两训练及其同包验收。


- Qwen43 SP2-QAT400已completed，连续400步/global8，6553600输入token visits、6550400预测targets；第400步完整validation PPL14.807326864507779（NLL2.6951221166817696），已登记summary/CORE。该包同一权重的完整test GPU6/PID884898、固定C4 GPU2/PID884904已通过tmux启动（sp2-qat400-test-s43、sp2-qat400-c4-s43）。独立auditor接续seed43最终证据，Uniform43仍原PID811495/GPU1,0固定400训练；未重启或改预算。

- Qwen43 SP2第400步同包完整test已completed并登记：PPL14.150947756398566、NLL2.6497816010970188、298931预测targets含尾窗；独立auditor已核对训练/validation/test实际证据PASS，test summary登记待补核，C4仍PID884904/GPU2原评测中、Uniform仍固定400训练。阶段报告auditor/QWEN_QAT400_S43_20260918.md/.json，未将pending记为完成。

- Qwen43 SP2同一第400步包三split测量已齐：validation14.807326864507779/test14.150947756398566/C4 24.042025123401675；C4 NLL3.179803345861205，2096128预测targets，原评测completed后自然退出，已登记summary/CORE。独立auditor仅补C4及外部summary关联；不复跑已完成测量。最后Uniform43原PID811495/GPU1,0继续固定400，最近87步；完整配对三seed统计仍等待此分支最终验收。

- Qwen43 SP2最终训练及同包validation/test/C4独立审计完整PASS，主执行已读auditor/QWEN_QAT400_S43_20260918.md/.json结论与覆盖。新增C4固定IDs/metadata、196量化状态前后及跨test比较、最终包尺度、全部分段和两external summary核对一致，旧test登记pending已关闭；validation未另存IDs/前后快照的限制保留。剩余Uniform43与完整配对统计不提前记完成，已PASS部分不复跑。

- 最后Uniform43原QAT进程PID811495（GPU1/0）已实核推进至300/400，无failure记录；仍按原global8与固定400预算执行，未重启。除此分支最终validation/test/C4、相应独立审计及Qwen三seed完整配对统计/最终handoff覆盖外，无新增待跑实验。此前Qwen43 SP2及Qwen44全部最终证据PASS继续复用。

- 最后Uniform43已completed400次更新，无resume，global8、6553600输入/6550400预测token visits核对通过；最终validation PPL15.65365239674437/NLL2.750704269740956，已登记summary/CORE。该checkpoint-0400/static_w4a8.pt同包完整test GPU1/PID1221890、固定C4 GPU2/PID1221896已由tmux启动（uniform-qat400-test-s43、uniform-qat400-c4-s43）。所有必要训练均已结束；独立auditor接续最后Uniform43与Qwen三seed统计，外部尚未完成不提前关闭goal。

- Uniform43同包完整test已completed并登记：PPL15.013445852436345/NLL2.708946189746898/298931targets含尾窗；独立审计训练/validation/test已PASS（test summary新登记待补核）。当前唯一正式GPU测量为原C4进程PID1221896/GPU2；完成后补三seed全量统计及最后审计。主执行已逐条读取登记最后Uniform43 test前97条完成summary的原始JSON，PPL/NLL/input/targets及包存在性均无不一致；100份Phase5 launch记录均未使用GPU4。

- 最后一项Uniform43固定C4已completed并登记：PPL27.45911062612704/NLL3.3126980118374294/2096128targets，与本方法test/validation共用第400步包。原GPU2评测进程已自然退出，全部必要GPU测量已完成。summary/CORE现已包含两家族42/43/44完整配对，样本SD采用ddof1；独立auditor补最后C4/18值统计、verifier更新最终handoff覆盖，审计完成前不关闭goal。

- 最终收尾：主执行已读取最后Qwen43结果/三seed统计PASS和最终HANDOFF覆盖PASS；STATUS/PLAN当前状态更新为本轮授权范围完成，保留所有历史快照和恢复记录。
