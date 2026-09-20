# 独立验证：PASS

独立 verifier 未运行 GPU。3 项 CPU 单测通过；另用 10 个随机 seed 的任意 BF16 参考输出，逐坐标比较解析代价与直接重算误差差值，通过。实际 tiny 两层 BF16 Llama 完整 run_rounding 中额外挂独立 hooks，确认 SP2 fit/heldout 输入及 BF16 MLP 参考输出逐行逐位相符，窗口顺序和每窗前 64/后 64 行没有错位。

新目标使用相同 MLP 输入、相同 R 的原 BF16 gate/up/down 参考。候选只改变 down1 的 floor/ceil 舍入，原 C SW/SA/SP2 alpha 不变。局部候选输出用 FP32 矩阵乘计算，未包含最终 BF16 GEMM 输出舍入；它是以完整 MLP 参考为目标的解析局部误差，并非与运行时 BF16 输出误差完全相同。

独立读取真实 payload：target 为 layer 1 down_proj，SW 与原 C 逐 tensor 一致，选择 512 步确为记录中的最小 heldout MSE；相对原 C 实际改变 490430 个 code，与记录一致。

最终完整 validation 与旧 C+SP2 的 252852 输入 tokens、252728 预测 tokens、全部分段、KV16/use_cache=False 一致；冻结检查通过，无高精度恢复。独立重算 token-weighted NLL 相符：PPL **17.40639276660993**，NLL **2.856837539125371**，delta NLL **-0.013467555253865449**。

该模型优于原 C，但略弱于前一轮 weight-only 目标的 PPL 17.395098280145866；更完整的局部参考没有在这次最终评测上产生进一步收益。
