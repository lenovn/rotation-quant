2026-09-15：按用户缩小范围，仅生成layer1 down通道图，见[layer1-down-channels-20260915a/README.md](layer1-down-channels-20260915a/README.md)。2048输出行×8192权重；保留原q/SW、同R未量化参考，无forward/PPL/训练/模型导出。两图及统计JSON已完成，独立verifier PASS。权重RMSE最大行867与tail最大行1649不同，不外推为任务敏感通道。

# 最新：逐层MLP/up/gate/down正式定位完成

2026-09-15：65项新完整GPU测量完成，复用layer1 down与匹配既有对照。全家族条件恢复收益 down > up > gate；整体MLP layer1和15近乎并列，layer10次之，9/11/12/13也敏感。单矩阵layer1 down最突出；layer10/11 up、layer13 gate不可忽略。所有排名限定当前父包同R全A16，恢复效应不可相加、不直接外推最终SP2。未训练/导出新模型。

结果：[完整分析](mlp-layer-analysis-20260915a/ANALYSIS.md)、[16层表](mlp-layer-analysis-20260915a/TABLE.md)、[热力图](mlp-layer-analysis-20260915a/layers.png)。独立verifier PASS；auditor已核对全部65项及汇总PASS，正式报告auditor/MLP_LAYER_AUDIT.md。下面保留历史记录。

# 2026-09-15 逐层MLP与投影定位（进行中）

用户明确授权深入全部16个layer的MLP及up/gate/down。固定原Phase3 PTQ父包q/SW、B100同R参考、全A16、成熟完整WikiText2 validation；每case独立从原父包重置。新增16个单层MLP整体恢复、16个单层up、16个单层gate、15个单层down（layer1复用旧完整结果），另测全家族up及gate。共65项新测量，复用此前全down/gateup/MLP与W4/W16参照。该范围由本轮用户明确要求，不继承旧全扫描禁令作为阻塞。

预先记录：按完整NLL降低量排序（正值为恢复收益），不按权重MSE。MLP联合干预与三个单投影恢复效应可能相互抵消、不可相加；保留正负效果，不把近零差异当稳健排名。必要时仅补关键层两投影联合恢复解释交互。已有同协议结果不重复。层编号使用源码0..15（第1..16层）。不训练、不选码、不写模型包；新文件预计仅MiB级，存储不足暂停。独立verifier只审新增模块选择，最终auditor审成品表格与结论。

# 最新：当前W4成因诊断完成，独立审计PASS

2026-09-15 新诊断goal：16项完整GPU测量全部完成；主要定位MLP离散表示误差，范围截断不是主项，旧layer1单层主导不成立。完整结论及边界见 [W4_DIAGNOSIS.md](W4_DIAGNOSIS.md)。独立verifier全部增量PASS；最终auditor PASS，见 auditor/DIAGNOSIS_AUDIT.md。没有新部署候选，无新恢复/C4、无大包输出。下面保留此前状态时间线。

# Phase4 当前状态

2026-09-15 新任务进行中：用户要求找出**当前**W4掉点原因，已实际创建新的诊断goal。此前A/B候选失败不能代替成因诊断，也不能证明per-channel粒度是主因。正在补当前PTQ父包的all-A16/down-A16旁路、同R全权重恢复及三组权重恢复；全部调用现有完整validation入口，仅存小型JSON，不导出模型。历史完成记录保留如下。

2026-09-15：两轮正式研究及三项新成品完整WikiText2 validation已完成，**没有新胜者，保留Phase3最终算法**。最终独立auditor已完成限定PASS：`auditor/FINAL_AUDIT.md`。本轮研究、正式评测、独立审查与记录全部完成；原生Phase4 goal已实际标记complete（约26分31秒），Phase3 goal未恢复。

| 配置 | 完整PPL |
| --- | ---: |
| 原PTQ父A0（复用） | 16.112577060930427 |
| A1继续固定SW邻码 | 16.112659483826263 |
| A2 COMQ思路SW/邻码 | 16.123914432238468 |
| B GuidedQuant g=1目标/固定SW邻码 | 16.12037615087552 |

三项新完整测量均252852输入、252728预测targets、123×2048+948尾窗、token加权NLL。局部重构确有改善，但未带来正式任务收益；没有额外恢复训练/C4/新A16旁路/多seed。结果与限定结论见 RESULTS.md，论文与适配差异见 PLAN.md。

源码`worktrees/SpinQuant-phase4-w4`，分支`phase4/w4-per-channel`；从Phase3 HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`加5个tracked修改/14个untracked文件继承。证据`inheritance.json`/`inheritance.diff`/`inheritance-status.txt`。新增源码只在Phase4工作树，根协调文档/Phase3保留，不提交推送。

主父包`runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`，未量化参考`runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/state.pt`。`inherited-sw-check.json`证明112个SW与B100学习终点完全一致，非量化权重/buffer亦一致（允许同位未用NaN哨兵）。原恢复冠军14.581654675328117保留，不重训/复评旧结果。

GPU均通过实际环境审批后在tmux专用socket`rotation-quant-phase4`运行：A搜索GPU0/PID2800949；A layer8最终GPU0/PID2820170；B GPU7/PID2843696。真实命令/资源/session在各`.launch.json`，日志在各`.log`；全部自然完成，无他人进程操作。新增2+2项轻量数学测试PASS；独立verifier分别限定PASS，见`verifier/`。最终auditor只读核查，无新胜者不加GPU复跑。

用户指定的真实可新增空间约28.66GiB（每分支100GB限制），df约222GB不作写入预算。未安装或改环境，未清理任何旧产物；若空间不足立即暂停。维持客户端实际模型设置，没有自行切换模型。

最终审计确认三个包只有layer8整数码改变，A2另改layer8 SW；所有激活参数及高精度参数与父包相同，导出与选中记录精确一致，完整NLL算术吻合。无新GPU复评。最终新增产物约5.9GiB，未遇空间不足；`inheritance-final-check.json`再次确认Phase3 diff/status与继承19源码文件均未改变。
