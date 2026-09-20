# 独立验证报告

结论：**PASS**。本次仅验证用户新授权的 layer 1 整块 BF16/A16 恢复诊断。独立 verifier 未运行 GPU，未修改实验代码或既有结果；本文件为授权新增记录。

## 实现与协议检查

- CPU 读取真实 C checkpoint，确认 `model.layers.1` 精确包含 q/k/v/o、gate/up/down 七个 Linear，其余 105 个 Linear 保留旧 C W4。
- 七个恢复权重来自本次加载原 BF16 权重并融合同一 C R1/R2 后、W4 量化前的 `fp`；整个 layer 1 的七个输入量化器设为 A16。其余 90 个非 down 输入保留 C SA，15 个 down 输入保留旧 SP2 alpha。不运行校准、搜索或重新量化。
- 审查发现并修正了边界检查的 wrapper 双别名问题：backbone 的 `.weight` 与 `.module.weight` 均须排除出高精度边界对比；lm_head 的两个键仍须保留。最终结果记录 36 个边界张量，包括 embedding、33 个 norm 权重及 lm_head 两个键。
- 实际 16 层 tiny BF16 Llama CPU 集成检查通过：完整执行 mixed 构造及后验冻结检查，仅模拟 evaluator 返回值；精确恢复 7 个权重、保留 105 个权重，输入位宽及 SP2 类型符合预期，边界双别名检查通过。该检查不替代真实 GPU 数值执行。
- 本次 GPU 日志和结果显示正常完成；`mixed_result.json` 的 `frozen_checks_pass=true`。实现评分前后逐项核对恢复权重与 fp、其余权重与 C checkpoint、全部输入量化器 buffer 不变及输入位宽正确。

## 独立结果复核

将本次结果与 `runs/phase2/validation-diagnostics-c-20260913.fs4I9m/w4a8/result.json` 的旧 C + SP2 控制逐项比较：输入 252852 tokens，预测 252728 tokens；所有 segment 的起点、长度、窗口数、预测 token 数一致，均为 validation、KV16、use_cache=False。独立按预测 token 数加权重算 NLL，并重算与旧控制的差值，全部相符。

| 指标 | 旧 C + SP2 | layer 1 BF16/A16 | 差值 |
| --- | ---: | ---: | ---: |
| PPL | 17.64239997588965 | 16.472544418121483 | -1.169855557768166 |
| NLL | 2.8703050943792365 | 2.8016950203037343 | -0.06861007407550224 |

改善已在这一次 matched full validation 上得到证据。这是混合精度诊断，不是 all-W4A8 结果；同时恢复了 layer 1 的权重与激活，不能将全部改善单独归因于其中一项。

## 预算

- 本次完整 validation 1 次，是原 9 次之外的新授权单次诊断。
- 本次耗时 51.55963035702007 秒，prior 558.0677173670265 秒，累计 **609.6273477240466 秒**，低于 1800 秒。
- 本次 allocator 峰值 4.376953125 GiB；2942 MiB 进程显存仅为采样峰值，不能称为全时真实峰值。未观察到超限记录。
