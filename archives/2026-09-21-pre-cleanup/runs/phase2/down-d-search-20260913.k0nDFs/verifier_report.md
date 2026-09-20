# 独立验证：PASS

独立 verifier 未运行 GPU。源码审查确认 24 个完整 train 窗口用于局部 fit，8 个不重叠完整 train 窗口用于局部筛选及真实 next-token NLL；原 C SW/SA/SP2 alpha 固定。每层候选只覆盖当前 down，先前层保留已选 prefix，未来层保留 C。选择后显式恢复最佳权重并重新传播该层输出。

CPU 实际 tiny 三层 BF16 模型状态跟踪检查通过：模拟每层 2048 步最佳、最后 8192 步更差，确认基线只评一次，后层 trial0 复用前层最佳 NLL；评估中没有 prefix 或未来层污染，最终传播和重载恢复的是最佳候选而非最后候选。

独立读取真实 16 层 JSON 与 payload，确认：

- 每层 trial0 的 NLL 精确等于前层所选 NLL；每层选择均为实际评分候选中的最小 NLL。
- 共 49 次 heldout-train 评分：原基线 1 次，16×3 候选 48 次；这些不是完整 validation 次数。
- 接受 layer 1/4/8 的 8192 步、layer 10 的 2048 步、layer 11 的 512 步。其余 11 层解包 codes 与原 C 逐位相等。
- 16 个 SW 与原 C 逐 tensor 相同，所有 codes 范围合法；实际共改变 9472584 个 code，与所选 trial 逐层计数一致。
- 最终所选 heldout-train NLL 为 2.8015499986647328。

最终完整 validation 与原 C+SP2 的 252852 输入 tokens、252728 预测 tokens、所有分段及 KV16/use_cache=False 一致；冻结检查通过，无高精度恢复。独立重算 token-weighted NLL 相符：

- PPL：**17.134821523344698**。
- NLL：**2.8411127393601405**。
- 相对旧 C delta NLL：**-0.029192355019096006**。

最终结果略优于只优化 down1 且扩大 coverage 的 17.156840423438286；改善幅度有限，未达到 BF16+1。heldout-train 被反复用于选层选步，不能将其改善作为独立泛化证据；本轮 validation 为冻结候选的端到端检查，整个持续 loop 的 validation 也已参与方法选择。

本次完整 validation 1 次，耗时 593.3596961749718 秒，allocator 峰值 5.91796875 GiB；进程显存 6522 MiB 为采样峰值。
