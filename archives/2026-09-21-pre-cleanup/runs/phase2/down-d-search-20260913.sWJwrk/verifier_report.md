# 独立验证：PASS

独立 verifier 未运行 GPU。直接提取并执行当前源码的 full-coverage collect hook，以 32 个可追踪的 2048-token 窗口检查：前 24 窗全部 49152 行进入 fit，后 8 窗全部 16384 行进入 heldout，没有共享窗口或 token，采集计数恰为 32。

真实 settings 核查通过：24/8 个 fit/heldout 窗口索引不交叠，合并后与原 32 个校准窗口索引完全一致；候选步数为 0/512/2048/8192，仍使用真实冻结 SP2 输入、weight-only 局部重建目标和原 C 固定 SW/SA/alpha。

独立读取真实 payload：唯一目标 layer 1 down_proj，SW 与 C 逐 tensor 相同；选择 8192 步确为记录的最小 heldout MSE，相对 C 实际改变 4765850 个 code，与 trial 记录一致。

最终完整 validation 与旧 C+SP2 相同的 252852 输入 tokens、252728 预测 tokens、全部分段及 KV16/use_cache=False 协议一致；冻结检查通过，没有高精度恢复。独立重算 token-weighted NLL 相符：

- PPL：**17.156840423438286**。
- NLL：**2.8423969525985595**。
- 相对旧 C delta NLL：**-0.02790814178067702**。

该候选优于此前 down1 小样本舍入的 17.3951，但仍未达到 BF16+1 目标。本次同时改变了训练覆盖和迭代步数，不能只归因于其中一项。

本次 1 次完整 validation，耗时 82.23379650700372 秒，allocator 峰值 4.869140625 GiB；记录的进程显存 5448 MiB 是采样峰值。
