# Phase4 guided-round 独立数值审查

日期：2026-09-15。结论：**PASS（新增及受影响核心逻辑）**，没有发现会污染当前正式运行的数值错误。尚不代表方法收益或最终成品验收。

范围：`experiments/phase4/guided_round.py`，受影响的 `scale_round.py --evaluate-best-rejected`，必要追溯 Phase3 `SP2ScaleSTE`、`SP2Quantizer`、`token_nll`、INT8 `RotationStaticActQuantizer`。未修改主实现；仅在本 verifier 目录写 toy oracle 和报告；无完整模型 forward、训练或 GPU。

## 梯度来源与边界

- target 是 down Linear 输出，而非输入或量化码。forward hook 返回 `result.detach().requires_grad_(True)`；所有模型参数已被冻结。`autograd.grad` 只求该输出 leaf 的梯度，不更新 FP 权重、R、SW、SA 或 SP2。
- 梯度来自当前量化父模型的 `token_nll`（FP32 logits CE），经过真实后续网络以及现有非 down INT8 STE；不是未量化教师轨迹，也不应称为离散量化算子的精确梯度。
- 冻结 SP2 原 forward 为 `sp2_project`；临时 hook 使用 `SP2ScaleSTE.apply`，其 forward 仍调用相同 `sp2_project`，backward 为已有 clipped input STE。尺度为不求梯度的 buffer，所有 hooks 在 finally 移除。它提供后缀敏感性的代理，未改变成品 forward。
- 将 loss 乘 1000 在理想算术下只给均方梯度乘 10^6，随后 fit mean 归一会消除此公因子；其实际用途是降低 BF16 梯度下溢，不能称改变任务权重。由于 BF16 舍入，仍属于梯度估计。

## 数据、目标与对照

- saliency 按 calibration 窗口顺序、每窗 token 行顺序拼接；`mean(-1)` 仅平均输出通道。与 `capture_module_inputs` 的同父实际 SP2 X 行顺序一致，前 49152 行 fit、后 16384 行 heldout。
- `weighted_inputs` 乘 `sqrt(saliency / fit_mean)`；因参考和量化输出均为该输入上的线性映射，所得重构目标等价于 `sum_t saliency_t * ||(Wq-Wref)X_t||² / fit_mean`，Gram 为 Xᵀdiag(saliency/fit_mean)X/N。heldout 使用相同 fit normalizer，没有使用 heldout 重估规范化。
- `generate(..., "A1", ...)` 关闭 SW 拟合，只从 parent q 接续原邻码；除输入加权改变目标外，Wref、SW、SP2、作用模块和 2×2048 请求坐标预算保持原设定。
- 额外 32 个全窗反传成本已在 settings 明示，capture 秒数单独记录；不能把该实验称为与 A1 总 GPU 成本严格相等。A1 的实际匹配还依赖其日志中确实评估了同两个端点，主执行已报告第一轮候选均通过局部 MSE 筛选。
- 此实现是逐 token 标量敏感性加权；同一 token 的所有输出通道共享一个权重。它不等于完整输出 Hessian，也没有直接优化后续子块的非线性重构。失败只能约束该 g=1/STE 代理及当前优化条件，不能否定所有目标与任务不匹配的解释。

## 选择与导出

- 两个端点都使用既有 8 个 train-window NLL 选择，完整 validation 只评一次选定包；`train_accepted` 明确区分诊断成品与训练开发接受。
- 最佳候选 packed q 与原 SW 传入既有 save/load，冷载后使用成熟 full_validation；没有将优化后 BF16 权重再次 RTN，没有新在线模块。
- A 方向新增 `--evaluate-best-rejected` 只在无接受更新时选择已计算 NLL 的候选，标记 `diagnostic_best_rejected=True`，未伪装成 train 接受。当前运行 12 个候选已有有效 NLL，路径可执行；未来若所有候选都被 heldout 筛掉，该选项下 `min(candidates)` 会因空列表失败，届时应跳过诊断导出，不把异常当作算法失败。这不影响当前已有候选的运行。

## 独立 toy capture

脚本：`runs/phase4/verifier/guided-capture-oracle.py`。在现有 rotation-quant-p0 环境以 CPU 小张量执行，约 6.35 秒。测试只模拟两层 4×4 Linear/SP2 后缀，不跑 Llama。将 `.cuda()` 临时映射为 CPU，使用显式后缀 `SP2ScaleSTE` 梯度作为独立 oracle。

```text
status: PASS
rows: 14
nonzero_rows: 14
forward_bitwise_equal: True
explicit_suffix_gradient_bitwise_equal: True
frozen_parameters: True
hooks_removed: True
full_model_forwards: 0
gpu_used: False
```

检查捕获时 forward 与原冻结 SP2 逐位相同、saliency 与显式后缀梯度逐位相同、窗口/token 排列一致、参数不求梯度且无累积 grad、hooks 移除后 forward 不变。该测试验证新增捕获逻辑，不替代正式模型运行或论文复现结论。
