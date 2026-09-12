# 仍适用的经验

## 实验归因

- 同配置比较优先；不同 RUN_NAME 的 R、SA 和 PPL 不可混用。当前默认 R 与早期正式 100-step R 分属两个目录。
- GPTQ 求解方法和 per-channel/group32 粒度分别描述。粒度比较从 BF16 和同一 R 重新量化，不能从 W4 checkpoint 再量化。
- down_proj 重尾输入会让 naive static A8 出现严重退化；输入逐元素 MSE 改善不保证端到端 PPL 改善，须看同口径评测。
- SA 有非零梯度不代表实际可见更新；检查实际 scale 变化。历史 lr=1e-3 的有效更新曾接近 float32 分辨率，不能仅看 optimizer 配置。
- R4 包含权重变换和在线 activation Hadamard，关闭时须成对处理。

## 保留路径的语义

- 旧 periodic calibration 以 optimizer update 计数；末步 final calibration 只执行一次。初始观察 bypass A8，后续在旧 SA forward 中观察，新 scale 仅在整轮完成后统一替换。SW 从有效 rotated W 计算。
- 当前 learnable-SA 路径不能在训练末尾用旧 periodic 方案覆盖学到的 scale。
- legacy `a_groupsize=-1` 的动态 token-wise 语义不能被重新解释为 static per-tensor。
- static quantizer 需真正接到 wrapper；phase/coverage/顶层初始化前校验与下游校验分别解决不同问题，不能因某层单测通过就删掉其他检查。
- cache-derived phase、checkpoint 重算和显式 PREFILL 的 GPTQ 逐层 forward 都须保持一致。prefill PPL 不验证 decode 或 KV cache。
- PyTorch fullgraph/contextlib 诊断失败不等于 mllm/QNN 失败，也不能用 GPU fake-quant 结果声称手机 NPU 已验证。

## 工具与证据

- Git 分支、真实源码、启动参数和日志优先于旧状态文档；源码 Git 操作明确指定仓库。
- 环境、Python、pytest 和依赖先核对；未收集测试不能报告代码 FAIL，历史 PASS 不能当作本轮 PASS。
- 获准测试时使用 `PYTHONDONTWRITEBYTECODE=1` 和 pytest `-p no:cacheprovider`，避免污染工作树。
- rg 不存在时用 grep/find，不为此安装工具。沙箱中的 GPU/tmux/进程可见性限制不是实验失败证据。
- 归档保留原始内容与路径对应关系；恢复前确认目标未被新文件占用。归档脚本仍使用旧输入输出路径，复现前需恢复所依赖的目录。

更细的历史排错记录见 [旧版 LESSONS](delete/root-docs-2026-09-08/LESSONS.md)，实现约束见 [SPEC.md](SPEC.md)。
