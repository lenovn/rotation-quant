# Layer 1 down channel plots verifier

2026-09-15。独立增量审查：**PASS**。

范围：`worktrees/SpinQuant-phase4-w4/experiments/phase4/plot_down_channels.py` 和 launcher 的 down-channels 入口。没有重审成熟模型评测，没有模型 CPU forward/GPU 重评，也没有修改应用源码。

- 目标固定为 `model.layers.1.mlp.down_proj`。`build_training_model(reference_state)` 从原始 BF16 模型和 norm fusion 构建参考，加载 B100 R，遍历 `rotated_linears` 但仅对该 target 调用既有 `rotated_weight`；down 使用既有 transpose 方向。这里仍需加载训练模型以取得原始权重及旋转，实际仅生成一张目标旋转权重，无模型 forward。
- 当前父包的该记录通过原 `unpack_int4` 和 `dequant_record(..., bfloat16)` 读取；SW 未重估，q 未更新。参考与父包来源的适用边界沿用本阶段已核对的 B100→当前 PTQ 父包。
- 权重为 `[output,input]`，所有 max/P99/RMS/MSE/zero-code/clipping 统计均沿 dim=1；SW 从 `[out,1]` 取每个输出行一个值。relative RMSE 为 `sqrt(mean((Wq-Wref)^2)/mean(Wref^2))`，不是相对每个元素，也不是按输出通道数归一化。截断条件是严格超出原 `[-8SW,7SW]`，端点本身不计截断。
- 选行规则明确为 relative RMSE 排序中位行、最大 max/P99 行、最大 relative RMSE 行，属于解释性选择，不能称为随机代表样本。三个规则可能选到同一行，仍应按真实选择报告。
- 行图横轴为原始 Wref/SW；参考直方图除以该行输入数和 bin width，表示密度。存储整数码的计数除以输入数、以 width=1 绘制，表示单位宽度码格的概率质量，两者面积均归一为 1。存储码展示并非声称 BF16 反量化值除 SW 后无限精度地等于整数。
- 统一绘图 limit 为选中各行 `abs(Wref/SW)` 最大值向上取整且至少为 10；参考分布不因固定 INT4 范围被截掉，码柱 [-8.5,7.5] 也全部覆盖。JSON 同时记录每行 reference_outside_plot，允许产物核查完整覆盖。
- launcher 正确映射 plot_down_channels.py，沿用已有隔离运行目录与 GPU 环境。产物是两组 PNG/SVG 以及小 JSON/源码记录；无模型权重导出或量化参数更新。

独立轻量检查：两行四列、不同 SW、正负两侧越界和端点相等的人工张量。核对 relative RMSE 解析值、逐行截断/零码比例、max_abs；两行 histogram 各完整计入所有元素且密度积分为 1，码计数总和正确。**PASS**。未重复主执行已完成的其它测试。

解释边界：这是当前坐标系下的权重分布与量化残差图，不含输入激活、输出重构或任务梯度，不能仅由某行长尾/相对误差大断言该通道主导 PPL 掉点。
