# 独立验证：PASS

独立 verifier 未运行 GPU。审查 `fixed_grid_rounding.py`、`precision_loop.run_rounding` 并运行 2 项 CPU 单测通过；额外 20 个随机 seed 的 BF16 数值检查逐坐标对比解析代价 `2*delta*G + delta^2*Hjj` 与直接重算输出误差差值，全部通过，接受更新的 fit 目标单调不增。输入权重与 scale 不被改写。

采集对象是旧 C+SP2 实际量化器输出；32 个固定 train 窗口各取 64 fit 行和不重叠的 64 heldout 行，共各 2048 行。heldout 仍来自相同 train 窗口，不是独立文档或 validation。目标是固定 SP2 输入上的 weight-only 局部输出重建，不是整个 MLP 的误差。

独立读取冻结 `rounding_weights.pt`：唯一目标为 `model.layers.1.mlp.down_proj`，形状 2048×8192，SW 与原 C 逐 tensor 一致；解包 codes 范围为 [-8,7]，相对原 C 恰有 482403 个 code 改变。选择的 512 步确为记录的 heldout MSE 最小项；未引入高精度恢复。

完整 validation 与原 C+SP2 对照的 252852 输入 tokens、252728 预测 tokens 及所有分段相同。冻结检查通过，独立重算 token-weighted NLL 与差值相符：PPL **17.395098280145866**，NLL **2.8561884584957267**，delta PPL **-0.2473016957437828**，delta NLL **-0.014116635883509865**。

本轮固定网格舍入的端到端收益成立；局部误差降幅不能被解释为同幅端到端改善。本次仅 1 次完整 validation，未把混精度归因结果当作此次模型精度。
