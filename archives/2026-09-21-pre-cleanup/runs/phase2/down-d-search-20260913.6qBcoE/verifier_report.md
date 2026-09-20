# 独立验证：PASS

独立 verifier 未运行 GPU。审查 `rounding_family.py`：每层在真实冻结 SP2 输出处截获训练行，使用原 C 固定 SW 独立进行 floor/ceil 舍入；按 heldout 局部误差选步后更新该层 down 权重，再真实传播选中模型给下一层。未更新 SA、SP2 alpha 或非 down 权重。

CPU 实际 16 层 tiny BF16 Llama 完整执行截获、舍入、选中后传播、16 权重打包重载与最终 configure；原 FP 权重保持不变，全部流程通过。该 CPU 集成使用 32×128 短窗，每窗仍为 64 fit 行加不重叠 64 heldout 行；真实运行设置是 32×2048，每个集合均 2048 行。坐标代价与 BF16 数值已在上一轮独立验证。

独立读取真实 `rounding_weights.pt`，确认：

- 恰好包含 16 层 down_proj，全部形状为 2048×8192。
- 16 个 SW 与原 C 逐 tensor 一致，解包整数范围全部在 [-8,7]。
- 每层选择 512 步，均是对应记录中的 heldout MSE 最小项；各层 fit MSE 单调不增。
- 相对原 C 共 5223514 个 code 改变，各层实际解包计数与所选 trial 的 changed_codes 完全一致。

最终结果与旧 C+SP2 控制的输入 252852 tokens、预测 252728 tokens、全部分段及 KV16/use_cache=False 协议一致。独立重算 token-weighted NLL 与差值相符，冻结权重/量化器检查通过，没有高精度 W/A 恢复：

- PPL：**17.546664783380706**。
- NLL：**2.8648638910188193**。
- 相对旧 C delta NLL：**-0.005441203360417202**。

该模型优于旧 C 的 17.6424，但弱于只优化 down1 的 17.3951。16 层局部 heldout 改善不能解释为可加和的端到端收益，应保留这一不利比较。

本次完整 validation 1 次，运行 95.13678088499 秒，allocator 峰值 3.994140625 GiB；进程显存 2816 MiB 为采样值。用户已取消 GPU 时间上限，显存约束保留。
