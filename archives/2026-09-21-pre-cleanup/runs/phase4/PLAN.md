2026-09-15：按用户缩小范围，仅生成layer1 down通道图，见[layer1-down-channels-20260915a/README.md](layer1-down-channels-20260915a/README.md)。2048输出行×8192权重；保留原q/SW、同R未量化参考，无forward/PPL/训练/模型导出。两图及统计JSON已完成，独立verifier PASS。权重RMSE最大行867与tail最大行1649不同，不外推为任务敏感通道。

逐层定位执行完成：65项新测量+既有复用，结果见mlp-layer-analysis-20260915a。没有把高精度恢复当同格式优化收益；下一步研究对象依据任务敏感性选择，仍需单独证明优化可行。

# 2026-09-15 逐层MLP与投影定位（进行中）

用户明确授权深入全部16个layer的MLP及up/gate/down。固定原Phase3 PTQ父包q/SW、B100同R参考、全A16、成熟完整WikiText2 validation；每case独立从原父包重置。新增16个单层MLP整体恢复、16个单层up、16个单层gate、15个单层down（layer1复用旧完整结果），另测全家族up及gate。共65项新测量，复用此前全down/gateup/MLP与W4/W16参照。该范围由本轮用户明确要求，不继承旧全扫描禁令作为阻塞。

预先记录：按完整NLL降低量排序（正值为恢复收益），不按权重MSE。MLP联合干预与三个单投影恢复效应可能相互抵消、不可相加；保留正负效果，不把近零差异当稳健排名。必要时仅补关键层两投影联合恢复解释交互。已有同协议结果不重复。层编号使用源码0..15（第1..16层）。不训练、不选码、不写模型包；新文件预计仅MiB级，存储不足暂停。独立verifier只审新增模块选择，最终auditor审成品表格与结论。

执行完成：6批16项新完整GPU诊断，独立verifier及auditor均PASS。当前阶段定位到MLP离散表示误差及条件交互；没有证明W4全局最优，没有新部署方案。精确结果与边界见W4_DIAGNOSIS.md；后续算法研究不冒充本轮已实现成果。

## 诊断执行中的追加对照

首批完整评测表明MLP两组均有较大条件恢复效应，因此新增同R、同absmax/RTN求解的MLP W4 per-channel、W4 group128、W8 per-channel三项；它们之间用于位宽/粒度比较，相对既有父包则同时改变了求解过程，不能混称粒度收益。MLP联合W16恢复提供同条件参照。只确认layer1 down及其余down，不扩展全层扫描。

为直接检验范围截断解释，令 C=clamp(Wref,-8SW,7SW)，保持原父包Wq与SW，计算 e_clip=C-Wref、e_grid=Wq-C。两项分别从Wq减去，再BF16回写，复用完整validation。grid包含优化整数码、SW格点及BF16反量化误差，不仅是RTN；局部SSE保留交叉项，任务效应不作可加分解。均为非部署的高精度反事实。

原父包/恢复down权重 × down-A16/down-SP2四格固定非down INT8。原父包SP2结果复用，唯一缺格补测。研究中已查看WikiText2 validation并据此选择下一诊断，明确不是盲测；没有用其选码或训练。所有诊断只写小型JSON，未增加checkpoint。论文依据：AdaRound https://proceedings.mlr.press/v119/nagel20a.html 提醒任务感知整数码决策不同于逐权重近似；GPTQ https://arxiv.org/abs/2210.17323 的输入二阶重构在现有Phase3已覆盖。本轮为成因干预，不重命名为新方法。

# Phase4 研究计划

## 已核对的 Phase3 实现

| 问题 | 实际实现与证据 |
| --- | --- |
| 输入 X | sequential_postprocess.capture_module_inputs 在 wrapper.module pre-hook 捕获：已过静态 SP2 的实际 down 输入；32×2048 train窗，24 fit/8 heldout，49152/16384行。 |
| 参考 | postprocess.reference_weights 用原始 BF16 模型、norm fusion、B100 state 的 R1/R2，调用 rotated_weight 得到未量化 BF16 权重；目标 W_ref X_q，与候选共享 X_q，非独立 FP 教师轨迹。父包无 D、无主权重恢复训练。 |
| 目标与接受 | fixed_grid_rounding.coordinate_round 优化 ||(BF16(q*SW)-W_ref) X_q||²，使用 H=X_q^T X_q/N，±1码搜索与解析梯度更新；heldout MSE预筛后按8个train窗NLL接受前缀。已有输入感知/Gram/邻码，不作为新方法。 |
| 更新变量与 SW | 后处理只改码或SP2范围。SW最后由B100联合R/96SA/112SW、down-A16的100步学习；冻结导出沿用learned SW。SP2收缩及最后down邻码均明确固定SW。源码证明没做该组合的SW拟合，但尚不证明有显著可修正任务损失。 |

## 方向 A：最终条件下 SW 与码匹配

假设：固定最终SP2与整数码后存在可减少输出重构误差的逐输出通道尺度方向，并可转化为正式NLL收益。

差异：固定 q，拟合 s_i=<q_i X,Wref_i X>/<q_i X,q_i X>，保持正尺度；按实际BF16反量化后的fit误差逐行回退无改善项。再调用原有邻码，不另造训练框架。

首轮范围 layer1/8/11 down，依据最终父包最近一次接受日志。A0复用PTQ父完整PPL16.112577060930427。A1每模块原邻码2段×2048坐标；A2相同2段×2048坐标，在各段前拟合SW。每模块两臂共用从未修改父包捕获的X、同Wref、fit/heldout；避免两臂前缀改变后再捕获造成输入来源差异。每臂最多两个邻码候选，A2另保留尺度单独结果作解释，不为它默认全量评测。

每臂从父模型出发按相同模块顺序用8个train窗NLL选择；两臂实际候选数据与码扫描预算相同，额外尺度拟合/Gram/矩阵运算和GPU秒数单列，不声称总FLOPs严格相等。最终两组成品各完整validation；无接受改变则复用A0。局部下降但NLL不降只否定本轮任务转化，不宣称解决W4条件损失或否定机制。

若有竞争力：原/新×down-A16/SP2最小消融，再按结论决定匹配恢复（精确继承旧400步run）和固定C4成品比较。无依据不启动400步补救。

## 方向 B 的进入条件

若A或既有日志支持局部目标与任务不一致，再考虑真正经过后续RMSNorm/非线性的最小重构；仅加相同残差是数学等价，不作为新方法。先隔离目标变化，不同时改教师输入、更新范围或SP2。

完整评测复用既有 full_validation：252852输入、252728 targets、123×2048+948尾窗、token加权NLL。validation长期参与开发，不称盲测。最终格式始终112 signed INT4[-8,7] per-output-channel、96静态INT8、16静态SP2，R1/R2融合、R3/R4关闭、BF16其他/KV16、GPU fake-quant prefill。

## 相关原始研究

AdaRound将舍入决策转为局部重构软松弛：[原论文](https://proceedings.mlr.press/v119/nagel20a.html)。GPTQ采用近似二阶信息：[原论文](https://arxiv.org/abs/2210.17323)。本轮不将输入感知、二阶信息或自适应舍入重新命名为创新；检验的是现有最终组合中遗漏的SW更新及增量收益。

原实现核查：[GPTQ原作者实现](https://github.com/IST-DASLab/gptq/blob/main/gptq.py) 通过输入累积H并在fasterquant传播量化误差；本项目复用的邻码是固定码网格上的另一求解器，不将其混称为新GPTQ。

## 2026-09-15 用户方法取向与直接论文依据

后续优先从明确论文机制出发，结合当前模型实测做最小适配；避免临时补偿叠加、通用性弱的复杂方案。论文出处不替代匹配实测，不要求凑新模块。

A轮更直接对应 [COMQ原论文](https://arxiv.org/html/2403.07134v3) 第3.2节式(8)/(10)、Algorithm2：[作者quant.py](https://raw.githubusercontent.com/AozhongZhang/COMQ/main/quant.py)。标准输出重构目标中，整数码和逐输出通道尺度交替求解，SW闭式正是本轮公式。不是新通用算法，也不是完整复现COMQ：原文用FP模型特征、自己的整数坐标闭式与顺序/初始化；本轮从已优化父q/SW接续，输入为当前静态SP2真实输出，保持[-8,7]，复用Phase3 ±1贪心邻码，并检查BF16反量化误差。仅检验这些部署约束下增加尺度匹配的增量。

方向B若需要，先核查已有重构/任务敏感性论文再实施，不凭包装更大函数声称块级方法。检索已发现 GuidedQuant（ICML2025，原作者 https://github.com/snu-mllab/GuidedQuant ）直接研究局部重构对最终损失敏感性差异；尚未选用或实现，不引入其非均匀权重码本。

## B：按GuidedQuant检验任务敏感性目标（2026-09-15）

A实测局部与任务不一致支持检验目标缺口。采用 [GuidedQuant原论文](https://arxiv.org/html/2505.07004v2) 第3.2节g=1（其W+A设置）与[官方实现](https://github.com/snu-mllab/GuidedQuant/blob/main/spin_quant/eval_utils/gptq_guided_utils.py)的任务梯度加权Gram。B不扩函数包围范围，而利用经过后续归一化/非线性/注意力的最终CE梯度衡量局部误差。仍是代理目标，不等同真实任务Hessian或保证改善。

相对A1唯一优化目标变化：对同一个实际父X的每个token乘sqrt(mean_output_channel(grad_CE²)/fit_mean)，然后复用原邻码。固定layer8/SW/SP2/reference和2×2048坐标。g=1是offline目标聚合，不改变权重per-output-channel部署粒度。

适配差异公开：梯度从当前冻结量化父包而非论文的原FP起点取得；采用当前INT8 STE和Phase3 SP2ScaleSTE，仅让梯度通过离散激活，forward仍完全相同；目标WrefX、输入X仍与A1一致。全部模型参数冻结，32窗只做梯度测量，无optimizer/teacher蒸馏，无恢复训练。额外32次完整窗反向的GPU成本单列，不声称与A1总成本相同。固定2个端点用既有8train窗NLL选择一个成品，再完整validation；即使父更好也报告该成品，不把拒绝伪装接受。

如果加权目标充分下降但正式NLL仍不改善，只说明当前g=1/parent-STE敏感性近似未转化收益。若梯度非有限/全零或局部目标不下降，则属于实现/优化未成功，先定位而非否定机制。范围不扩全112扫描，不添加非均匀权重码本，不盲搜g或正则。

论文/源码交叉核查补充：COMQ公开`quant.py`当前版本在24–28行初始化delta/bit_code，此后循环只更新Q，未看到论文式(10)的尺度重估。因此本轮尺度公式依据原论文，不声称直接照搬公开代码里的交替更新；公开实现与论文描述的这一差异已保留。GuidedQuant官方`gptq_guided_utils.py`则确实在add_batch中按saliency累积加权H，并按输出组调用GPTQ；本轮借用目标、保留项目原邻码求解器。
# 当前W4成因诊断（2026-09-15追加任务）

先测当前父包W4/all-A16、W4/非down INT8/down-A16及同R W16/all-A16；复用当前父包静态SP2完整结果。再在all-A16下分别恢复down、attention、gate/up三组权重。检验主要损失是否在权重量化及哪组模块，恢复效应不可相加。仅在结果支持时做少量定位和匹配求解器的粒度诊断，不重做全矩阵敏感性扫描。诊断不用validation调码，不另写评测器，不保存大模型副本。旧A/B研究不作为当前W4主因结论。
