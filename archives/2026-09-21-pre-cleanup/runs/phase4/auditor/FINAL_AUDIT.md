# Phase4 A/B 最终实验独立审计

2026-09-15。结论：**限定 PASS：现有证据支持“本轮 A、B 成品均未胜 Phase3 PTQ 父包，保留原方法”，没有发现会污染这一结论的核心错误。** 不是新方法有效性 PASS；没有独立 GPU 复评、重训或重审旧冠军。

主执行、新增核心逻辑 verifier 与本 auditor 分工独立。本 auditor 只读源码、启动记录、日志和张量产物；只在 `runs/phase4/auditor/` 新增本报告、轻量检查脚本和 JSON，不写旧工作树，不建立 hash/contract，不构建模型或执行 CPU/GPU 模型 forward。

## 真实执行与来源

| 实验 | GPU / PID | 真实入口与结果 |
| --- | --- | --- |
| `a-sw-code-matched-20260915a` | 0 / 2800949 | `experiments/phase4/scale_round.py`；layer1/8/11，各臂 2×2048 坐标，12 个 train 候选均拒绝，返回 A0 明示 `reused_parent=true` |
| `a-layer8-final-20260915a` | 0 / 2820170 | 同入口，显式 `--targets model.layers.8.mlp.down_proj --evaluate-best-rejected`；重建 A1/A2 cycle2，分别保存后冷载全评测 |
| `b-guided-g1-layer8-20260915a` | 7 / 2843696 | `experiments/phase4/guided_round.py`；layer8，2×2048 坐标，另 32 窗 CE 梯度捕获；cycle1 按 train NLL 选定但 `train_accepted=false`，保存后冷载全评测 |

三项真实命令、Python 环境、GPU 启动快照、专用 `rotation-quant-phase4` tmux socket/session、PID、log/output 都见各自同名 `.launch.json`；完成事件与 PID/GPU 一致。环境为 `rotation-quant-p0`，PyTorch 2.4.1+cu121，4090。没有以新冒烟或旧基线重跑替代实验。

当前源码 live HEAD 为 `24918316ed594848d4de797c356b120f2a4ee0f3`，分支 `phase4/w4-per-channel`。Git dirty 为继承的 5 个 tracked 修改，加 experiments/ 和相关 tests；并非仅 checkout HEAD。`artifacts.json` 的 live 文件逐字节比较确认：5 个 tracked 修改、14 个继承新增文件均与 Phase3 对应文件相等，Phase3 当前 binary diff 与 `inheritance.diff` 相等。成熟外部 helper 为 `scripts/phase2/fixed_grid_rounding.py` 与 `validation_acceptance.py`，实际路径由源码解析。

最终 A/B source snapshots 与当前新源码逐字节相同。A 第一轮 source 比当前少后加的 `--evaluate-best-rejected` 诊断导出分支；已直接 diff 核查，仅此功能和参数声明变化，原候选生成未变。首次 auditor 自动比较因假定所有历史 snapshot 都等于当前源码而停止；确认这个预期源码演进后修正 auditor 记录逻辑并完成检查，未改实验源码或结果。

父包统一为 `runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`；未量化参考统一为 `route-b-adam-100-20260914a/checkpoint-0100/state.pt`。Phase3 parent metadata 链是 B100→down 邻码→SP2 收缩→down 邻码，没有接受 D/恢复主权重。`reference_weights` 从原始 BF16 模型开始，融合 norm，加载 B100 parameters/R1/R2 后取 `rotated_weight`；没有把已反量化 W4 当未量化参考。此处核对了源码与产物来源链，没有另外重复整个参考模型 forward。

## 核心算法与实际边界

| 核查项 | A | B |
| --- | --- | --- |
| 实际输入 | 未改父模型 wrapper.module pre-hook，静态 SP2 后的 X | 同一路径、同父、同窗口与顺序 |
| 参考目标 | Wref×同 X，而非独立教师轨迹 | 同 Wref×X，误差按 token saliency 加权 |
| 新差异 | A2 固定 q 的逐输出通道 SW 闭式拟合，然后复用 Phase3 邻码；A1 仅继续邻码 | 仅把 Gram 改为 Xᵀdiag(saliency/fit_mean)X/N，继续原邻码 |
| 变量 | A1 q；A2 q/SW | 仅 q；所有 SW 固定 |
| 固定项 | R、非目标模块、全部非 down SA、SP2 格式/范围、高精度参数 | 同左 |

A 的 SW 是 `[out,1]`，闭式分子/分母与输出最小二乘一致；实际以 BF16(q·SW) 行重构误差验收，不接受因 BF16 舍入变差的行。原 helper 使用真实 BF16 反量化邻码差分，码保持 [-8,7]。固定最终码/SP2 下存在可下降的尺度方向，但“原 SW 未达到此局部目标最优”不能直接升级成“可恢复的最终任务损失”。这是 COMQ 论文启发下的受约束适配，不能声称完整复现 COMQ 或原创了码/尺度交替思想。

B 的目标 leaf 为 down 输出，模型参数冻结，CE 梯度经过真实后续网络和 INT8/临时 SP2 STE。前向仍为原 SP2，hooks 最后移除；loss×1000 的公因子由 fit mean 归一消除（BF16 舍入仍是估计限制）。每 token 对输出通道取梯度平方均值，故同一 token 的所有输出方向共享一个权重；不等于完整 Hessian，也不是直接优化后续非线性子块重构。`saliency.pt` 实查有限非负，恰好每窗最后位置为零，其余行均正；符合因果 next-token 无该末位 target 的结构。

B 使用 `token_nll` 的 FP32 logits CE 梯度代理；最终 selection/validation 继承 BF16 logits CE evaluator。该精度差异在 verifier 报告中已明示，应保留为代理目标限制，不能称敏感度是正式 evaluator 离散损失的精确梯度。

## 匹配预算与选择

32 个固定 train/calibration 窗，每窗 2048；前 24 窗 49152 行 fit，后 8 窗 16384 行 heldout 与 train 选择，来源 indices 与 Phase3 一致。验证集不在码或尺度内循环。它长期参与路线判断，不是盲测。

A 首轮每臂实际 3×2×2048=12288 坐标；优化 A1 67.0334 秒，A2 68.0861 秒，A2 的额外尺度计算如实收费。成品重建实际仅 1 个模块，每臂 4096 坐标；A1 22.3936 秒，A2 22.6604 秒。重建费用是额外研究成本，不能隐去。该成品 run 的 settings budget 静态字符串仍写 3 modules，arguments/日志/RESULTS 已明确实际 1 module；不影响实际执行或结论，不覆盖历史 settings。

B 同 1 模块、同 q/SW 起点、同数据、同两个端点和每端点坐标预算；仅局部度量改变，但额外 saliency 32 窗反向为 6.4589 秒，邻码 22.2677 秒。因此是匹配码搜索预算，不是总 FLOPs 或总 GPU 成本完全相等。A1 的 heldout 预筛在本次两端点均通过，所以两者实际都有两次同 train NLL 选择机会。

## 产物与完整评测

只读 CPU mmap 张量比较 `check_artifacts.py` → `artifacts.json` **PASS**：三成品均 112 张 W4，96 个静态 INT8 输入和 16 个静态 SP2 输入；与父包相比，只有 layer8 down packed q 改变，只有 A2 的 layer8 SW 改变。全部 activation/high_precision 逐元素相同（同位 NaN 哨兵视为相同），非目标权重记录相同，导出目标 q/SW 精确等于所选 updates/candidate record。未重新 RTN、未覆盖 learned SW，未留下在线补偿模块。R1/R2 仍融合，wrapper 默认关闭在线 Hadamard，R3/R4 未接入，BF16 embedding/head/norm、KV16 与 use_cache=False 保持成熟路径。

三个新包在正式运行中均使用成熟 `load_static` 冷载后调用 `full_validation`。auditor 未再做第二次 GPU 复评，因为不存在拟接受新胜者。

| 条件 | 完整 NLL | 完整 PPL | ΔPPL 对 A0 |
| --- | ---: | ---: | ---: |
| A0，既有父包结果复用 | 2.779600150933802 | 16.112577060930427 | 0 |
| A1，继续邻码 cycle2 | 2.779605266359144 | 16.112659483826263 | +0.000082422895836 |
| A2，SW/邻码交替 cycle2 | 2.7803035383838837 | 16.123914432238468 | +0.011337371308041 |
| B，g1 敏感性目标 cycle1 | 2.780084071227797 | 16.12037615087552 | +0.007799089945093 |

三个新 validation JSON 均为 252852 输入，123×2048 的 251781 targets 加 948 尾窗的 947 targets，共 252728，未评分尾 token=0。auditor 独立用各段 NLL×targets 聚合并 exp，与 JSON 精确相同；没有平均窗口 PPL。成熟 evaluator 精度是 BF16 logits CE、逐 token loss 转 FP32、从原 evaluator float32 PPL 取 log 后按段 targets 加权，未改成熟口径。A0 值与指定 Phase3 父包的真实 validation.json 一致。

## 可成立和不可成立的结论

A2 layer8 heldout MSE 从父 0.0001412973 降至 0.0001294530，也低于 A1 的 0.0001340939，但正式 NLL 更差；不是 SW 优化未运行或局部优化完全失败，是本轮局部改善没有转化为任务收益。B 的加权 fit/heldout 下降，但 uniform heldout 和 train/validation NLL 变差，说明当前 g1/量化父 STE 代理没有成功改善任务。

没有可靠 PTQ 增益或明确恢复互补性证据，保留 Phase3 合理；无需为这三个失败成品自动补全 A16、C4、400 步恢复或多种子矩阵。A1 的 +0.0000824 PPL 只应描述为近乎持平、未测到收益，不值得声明稳健退化。A/B 较小差值同样不作显著性或普适性宣称。

本轮没有匹配粒度实验，不能宣称解决 per-channel 粒度损失；没有 A16 消融，不能把局部收益分成纯权重改善与 SP2 补偿；没有新增恢复，不能声称恢复交互收益。不能否定 COMQ、GuidedQuant 或全部非线性块目标，更不能把 Phase2 固定 C 的条件数值用于分解当前 Phase3 包。

## 审计命令与限制

```text
GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python /home/dongpeiyan/projects/rotation-quant/runs/phase4/auditor/check_artifacts.py > /home/dongpeiyan/projects/rotation-quant/runs/phase4/auditor/artifacts.json
```

这是已有保存张量比较和结果算术核对，模型 forward=0、GPU=0。相关论文的学术对应关系依据本轮主执行核查及实际公式；auditor 没有重复外网文献检索，不将此报告称为论文独立复现。原始旧冠军与 C4 沿用已有证据，不在本次审计范围。
