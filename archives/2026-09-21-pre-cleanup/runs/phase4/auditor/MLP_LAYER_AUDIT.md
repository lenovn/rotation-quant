# MLP 逐层及投影诊断独立审计

2026-09-15；独立 auditor。**PASS：65 项新增完整 GPU 测量、1 项匹配复用，形成 16×4 个逐层条件恢复结果及两项新增全家族结果。** 本次只审新选择器、实际实验和汇总；复用上一轮成熟路径审计，无新增模型 CPU forward、GPU 复评或训练。

## 实际运行与新增范围

| run（runs/phase4） | GPU | launch PID | 新完整 cases |
|---|---:|---:|---:|
| diag-layer-mlp-20260915a | 5 | 1080049 | 18 |
| diag-layer-down-20260915a | 7 | 1080055 | 15 |
| diag-layer-up-20260915a | 0 | 1080061 | 16 |
| diag-layer-gate-20260915a | 1 | 1080067 | 16 |

四批 `.launch.json` 的完整命令、CUDA_VISIBLE_DEVICES、tmux socket/session、GPU headroom、输出/日志路径齐备；实际 source/settings 的父包、参考、分支/HEAD、case 列表与命令一致。全部 complete 列表与 results 一致。socket=`rotation-quant-phase4`，源分支 `phase4/w4-per-channel`，HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`。GPU和PID来自真实launch记录，本审计不把它们冒充独立复查的存活进程。

父包始终 `runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`；同R未量化参考来自 `runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/state.pt`。每case重置全部原q/SW，all-A16、output A16，其他权重/高精度边界固定。未重估原SW、SA或SP2。

`selected` 新增 `projection@layer`，独立verifier的 `verifier/mlp-layer-review.md` 已穷举80种合法选择并检查非法输入。auditor另逐项核对65个结果中实际 `restored` 完整名称集合：MLP@k精确3矩阵，up/gate/down@k各1，全up/gate各16；无layer1→10/11混选。全部112 input_bits均16，replacement=reference、components为空。

各批保存的diagnose.py中 configure/run/rtn_reference/error_components 的AST与上轮实际快照相同，仅选择器扩展。继承来源和坐标系沿用 `DIAGNOSIS_AUDIT.md` 的限定PASS；本轮未修改成熟helper/evaluator。实际新源码是Phase4 diagnose选择器及report_mlp_layers.py；没有新部署包、训练checkpoint或大缓存导出。

## 评测与汇总核查

- 所有65个evaluation字典逐一等于原始日志RESULT；token_count=252852、predicted_tokens=252728、123×2048(251781 targets)+948尾窗(947 targets)、unscored_tail_tokens=0。独立重算两个segment的token加权NLL及exp后PPL，全部与JSON相等。保留既有BF16 logits CE后转FP32精度，未换评测driver。
- layer1-down精确复用先前 `diag-mlp-local-interaction-20260915a/results.json::all16:down1` 并映射到down@1；同父包/同R/同all-A16/同16层模型/同数据。其余63逐层格与全up、全gate两项共65项新测。旧全down、gateup、MLP和all-A16父/参考按来源复用，不冒充新测。
- 独立复算summary.json全部64个格的NLL/PPL/ΔNLL、排序、joint_minus_sum，核对layers.csv逐值相等。ΔNLL定义始终父all-A16 NLL减恢复后NLL，正值表示收益；未相加PPL。
- 查看TABLE.md及layers.png：16行层号0..15、4列MLP/up/gate/down、色标正负方向和标注与summary一致；没有错行错列或归一化造成的排序混淆。SVG和PNG均由同一个数值矩阵导出。

## 核查后的主要数值

| 全家族恢复（16层） | PPL | 条件ΔNLL |
|---|---:|---:|
| down | 14.677121507297 | 0.058222993890 |
| up | 15.068328325061 | 0.031917836089 |
| gate | 15.242441255564 | 0.020429190437 |
| gateup | 14.791921893127 | 0.050431701515 |
| mlp | 14.002164851643 | 0.105300965319 |

| 排名对象 | 数值靠前层号 | 边界 |
|---|---|---|
| 整层MLP恢复 | 1:0.01077030，15:0.01073088，10:0.00980551；9/13/11/12约0.0082–0.0088 | 1和15仅差0.00003942 NLL，不宣称稳健第一 |
| down恢复 | 1:0.01017487，15:0.00557999，10:0.00475629 | layer1是当前最强单down干预，不等于它解释全模型主要损失 |
| up恢复 | 10:0.00357159，11:0.00320546，15:0.00285909 | 近邻顺序仅描述本次数据 |
| gate恢复 | 13:0.00313865，12:0.00221922 | layer13 gate大于up；全家族顺序不代表每层同序 |

所有层的down单项恢复数值均高于同层up/gate，但layer3/5/11/13差距较小，不将这些近邻差距表述为确定机制。全家族down>up>gate是当前all-A16条件下测量结果，并非依据量化误差MSE给出的推断。

## 结论边界

逐层/投影恢复的边际收益不是可加误差归因；整MLP改三张矩阵，不能把它与一张矩阵的raw收益比较后声称单位参数更敏感。joint_minus_sum只记录特定干预非加性，不单独证明某个非线性机制。全家族与逐层的ΔNLL同样不能累加解释总损失。

这份结果只定位当前PTQ父包、当前R和all-A16下的敏感性；静态INT8/SP2条件排名可能因输入与补偿交互变化。不能将恢复到BF16获得的收益等同于W4约束内可实现的算法收益，不能据此断言哪种SW/码优化必成功。未检验全部微小差值的随机性，不包装为盲测、下游准确率或外部C4普适结论。

无需要阻塞本轮结果交付的核心异常。用户本轮明确要求逐层/投影细化，65项测量属于该新增研究范围，未复做旧初始化、旧Atlas或旧完整模型基线。

最终解释文档核查：已阅读 `mlp-layer-analysis-20260915a/ANALYSIS.md` 与 STATUS/RESULTS/PLAN 最新段落。全家族排序、layer1/15近乎并列、layer10随后、单矩阵layer1 down最突出但非全模型单层主导、局部投影近邻限制均与实测一致。明确限定all-A16排名、敏感性不等于W4优化可实现收益，不外推SP2/C4。结论表述 **PASS**。
