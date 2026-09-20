# Phase 2 后处理收益与 Phase 3 当前搜索差异

本记录是2026-09-14重读历史JSON和两版源码的分析，不是重跑Phase2或新增GPU测量。

## 实际累计路径

以下均为同一完整WikiText2 validation协议、252728预测targets。每个父路径来自对应`loop_settings.json`，分数来自`loop_results.json`。表中是顺序累计收益，不是三个独立同起点消融，也不外推收益可加。

| 步骤 | 完整PPL | 证据目录（runs/phase2下） |
| --- | ---: | --- |
| 原C W4/non-down INT8/down SP2 | 17.64239997588965 | 后续结果的reference_ppl；validation-acceptance-c-20260909.9WVDue |
| 第一轮逐层down W4码优化 | 17.134821523344698 | down-d-search-20260913.k0nDFs |
| 逐层SP2范围扩张 | 16.701812082653035 | down-d-search-20260913.VVciEd |
| 边界最优层继续扩SP2范围 | 16.508818358818367 | down-d-search-20260913.89KzK6 |
| layer1固定SW的INT4邻码优化 | 16.161902488363097 | down-d-search-20260913.9Zycd9 |
| 继续逐层down邻码 | 16.122226577621436 | down-d-search-20260913.Ds4vSf |
| 第一批非down邻码 | 16.11180256293439 | down-d-search-20260913.JVI83z |
| 第二批非down邻码 | 16.11105493245762 | down-d-search-20260913.qvjyDe |
| 局部D及其声明SW规则 | 16.05715342622255 | down-d-search-20260913.7U1l7n / selected_d |

最后D的匹配I+同SW规则对照为 **16.118253028142952**，D相对该对照净收益 **0.061099601920402** PPL；不能把17.6424→16.0572的全部改善归因于D。D前原父16.1111与I规则16.1183也不是相同配置。

## 为什么能改善

- 固定SW的离散码优化直接改变INT4整数，不等同于再算MinMax或只学连续尺度；在实际量化输入分布上减少层输出误差，并用真实train NLL拒绝有害候选。权重逐元素最近舍入并不保证实际输入下的输出误差或任务NLL最优。
- SP2范围改变覆盖区间与各量化格点。历史output-MSE选出的alpha不保证端到端NLL最优；Phase2固定码/SW/非down SA逐层按真实train NLL选alpha，保留已接受前缀。layer0在第二轮相对父alpha选8倍，结合第一轮为相对原SP2的16倍。这是实测选择，不凭大范围普遍更好作推论。
- 配对离线D在未量化SwiGLU函数中抵消，但改变W4/SP2看到的数值分布与离散误差。该特定layer1/channel1417候选在匹配SW规则控制下获益；不是所有D、通道或层都获益。

## 当前Phase3不是等价复现

| 项目 | Phase2已见收益路径 | Phase3首轮post-b100实现 |
| --- | --- | --- |
| 候选数据 | 32个完整2048-token train窗，24/8分割；49152 fit行、16384 heldout行 | 32窗各前128 tokens，再每8行抽样；384 fit行、128 heldout行 |
| W4坐标检查点 | 0/512/2048/8192 | 0/8/32/128 |
| W4选择 | 逐层局部MSE预筛，再真实8窗train NLL；保留收益前缀，后续传播随前缀更新 | 父模型一次采样，各模块独立按MSE选，再只测top4/top16组合的四窗probe |
| SP2选择 | 单层候选用真实8窗train NLL决策，保留已接受前缀；边界最优继续扩 | 单层按局部MSE选出六层修改，合成一组后做四窗probe |
| 底座 | Phase2 C有已学习R0/SA0历史，再做100步W4-aware训练 | Phase3三主路线从共同未学习Hadamard R启动，100步后选择B |

源码证据：`scripts/phase2/{precision_loop.py,neighbor_rounding_loop.py,sp2_alpha_loop.py,non_down_rounding.py}`及历史`loop_settings.json`；当前`worktrees/SpinQuant-phase3-joint/experiments/phase3/postprocess.py::{capture_inputs,rounding_candidates,range_candidates}`、`common.py::data_windows`和各`post-b100-*/settings.json`。

因此Phase3现有W4/SP2负结果只支持“这两组较小搜索生成的组合未改善probe”，不能支持“Phase2方法在新底座无效”“联合优化已吸收全部后处理收益”或“PTQ潜力已耗尽”。新R/尺度底座也不同，但没有相匹配的实验，不能将差异单独归因给它。

## 调整下一步

先在保留的Phase3 B100 PTQ父包上补齐较充分的完整窗、逐层真实train-NLL选择及固定SW邻码搜索；W4→SP2→后续邻码/局部D沿收益前缀继续，不把三个独立同父组合直接相加。保留I/0-step、原量化范围和未接受候选证据，完整validation仍按原协议。

只修改Phase3隔离源码，新算法选择/回滚路径需独立窄项verifier；不重跑Phase2、三条R路线训练或不必要的保险测试。此工作可用小日志和最终packed包推进，不需新增约10.9GiB的全权重Adam恢复文件；存储门槛仍约束后续长蒸馏，但不再据此宣称所有主goal工作均无可执行步骤。已有蒸馏15.5789222和原Phase2产物完整保留，不将尚未补测的PTQ收益写成事实。

## 后续新实测与可比性边界

主线新完整结果已经落盘：Phase3 B100 **17.11711490993314→顺序W4 16.507098542639632→SP2收缩16.19254871508746→继续down邻码16.112577060930427**。SP2扩张/o_proj的训练选择收益未转化为完整validation收益，均没有更新最佳。新匹配局部D的五个强度均未胜未改变父包的训练选择NLL，故未进入完整I/D validation；不能把16.1125771称为已采用D后的新实测分数。

Phase2 D前最后接受包为 **16.11105493245762**（`down-d-search-20260913.qvjyDe/loop_results.json`），与当前Phase3 D前/未接受D包仅差 **0.001522128472807** PPL。Phase2最终D为16.05715342622255，与当前Phase3差 **0.0554236347078785**。因此数值上不是Phase3后处理全面失效，而是两条有限搜索在D前到达很接近的水平，Phase2随后接受了D；Phase3新D相对I规则改善，但仍不如未改变父包，不能强加到当前模型。

更深的因果解释尚未由单变量实验识别。两者R/SW/SA底座不同，Phase2非down采用历史敏感性选择的late-MLP/QKV两批，Phase3本轮优先16个o_proj；Phase2 SP2接受扩张、Phase3本轮接受收缩。固定SW网格、残余误差和可用离散候选均会随底座改变，起点PPL更好不保证另一个受限搜索后的排名不变；但不能把“联合训练已吸收后处理收益”等假说写成已测机制。

另有明确数据差异：`scripts/phase2/down_codebook_experiment.py::text_windows`先将train文本双换行拼接再tokenize，在全部完整窗中以torch seed42抽32窗；`experiments/phase3/common.py::data_windows`按原始行无BOS/EOS tokenize再拼接IDs，从排除末8窗的训练池抽32窗。Phase2本次历史校准索引开头205/888/943，Phase3为182/546/792，8窗选择集合也不同；同seed与相同24/8、2048长度不等于相同token数据。因此这是同完整validation口径的路线结果比较，**不是已控制校准tokens/候选顺序后的纯R或联合训练因果消融**。两边完整validation仍均252728 targets、相同尾窗/NLL协议，这一差异不否定各自完整PPL的真实性。
