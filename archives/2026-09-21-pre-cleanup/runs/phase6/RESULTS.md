# Phase 6：down 静态 INT16 精度比较

2026-09-21。三项新增完整 WikiText-2 validation 评测已完成；复用历史校准统计，没有重新训练、重新校准、后处理或 QAT。

**结果：在固定 B100 与本次 4096-token train 校准范围下，INT16 大幅避免 INT8 退化，但相对浮点旁路仍有小幅可测损失。**

| down 输入 | PPL | NLL | ΔPPL vs 浮点 | ΔNLL vs 浮点 | PPL 相对变化 |
|---|---:|---:|---:|---:|---:|
| float | 16.1341967763 | 2.7809410428 | +0.0000000000 | +0.0000000000 | +0.000000% |
| int8 | 1528.1961277976 | 7.3318433174 | +1512.0619310213 | +4.5509022746 | +9371.783126% |
| int16 | 16.2769545990 | 2.7897502791 | +0.1427578227 | +0.0088092363 | +0.884815% |

## 能说明什么

- 支持继续探索 down 静态 INT16：它比共同范围的 INT8 保留了明显更多精度；相对 down 浮点，PPL 增加约0.143（0.885%），不能称无损。
- 这个结果比较的是同一模型的输入量化精度。不能由此判定 down-SA 是否需要梯度学习，也没有比较 Adam 与 SGD。
- 共同范围的均匀 INT8 退化，不意味着所有 INT8 校准、训练或粒度方案都不行。当前仅16个 down 改精度，权重与非down激活固定。
- 量化后回到BF16的fake-quant不等于真实INT4×INT16内核；仍未验证手机NPU精度、速度、KV量化或decode。

## 共同实验口径

- 父包：`../phase3/route-b-adam-100-20260914a/checkpoint-0100/static_w4a8.pt`。同一旧B100，无后处理或蒸馏。
- 112个W4 per-output-channel权重、96个非down static INT8 per-tensor激活保持；R1/R2已融合，关闭在线R3/R4；KV BF16、use_cache=False。
- 校准只复用32×128个WikiText train输入的历史每层absmax（2.5625–816）。INT8与INT16分别除127、32767；零点为0，signed integer grid。没有读取validation选择范围。
- 复用全部down浮点旁路的并行捕获统计，沿用历史v_proj列主序捕获布局；不是新整数轨迹下级联重新校准。来源与交叉核验路径见[计划](PLAN.md)。
- 完整WikiText-2 validation：252852输入tokens、252728 targets，123×2048+948尾窗。沿用历史BF16 logits CE与分段token加权NLL/PPL。
- INT8/INT16均通过同一个显式Linear pre-hook，FP32舍入/截断后返回BF16，绕过旧bits=16的浮点旁路语义。

## 执行与复用

- 新增：3项完整GPU评测。没有同口径旧分数，所以三项均补测。
- 复用：B100父包、16层历史校准统计、原数据索引、原evaluator；不重复校准和已完成实验。
- 旧B100 SP2 validation PPL17.11711490993314只是背景，不列入本次位宽主表。旧searched INT8 C4与Phase5后处理/QAT结果口径不同，未混用。
- GPU0 / PID550536 / tmux socket `rotation-quant-phase6`。三项原始分数、分段数据、真实量化调用/越界/数值变化统计见各臂result.json。
- 状态：全部completed，无failure.json；独立CPU窄测7项PASS，最终结果审计见[验证报告](verifier/REPORT.md)。

## 原始证据

- [浮点结果](b100-down-precision-20260921/float/result.json)
- [INT8结果](b100-down-precision-20260921/int8/result.json)
- [INT16结果](b100-down-precision-20260921/int16/result.json)
- [机器可读汇总](summary.json)
- [启动记录](b100-down-precision-20260921.launch.json)
- [运行日志](b100-down-precision-20260921.log)
- [实验入口](../../scripts/phase6/down_precision.py)
- [独立验证报告](verifier/REPORT.md)
