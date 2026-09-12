# ADR 0027: Cache-derived dual-phase propagation

- Status: Superseded by ADR 0028
- Date: 2026-08-11

## Context

M001 已实现 prefill/decode 各自独立的 frozen activation qparams，但模型尚未从真实 cache 进度解析 phase，也未将 phase 传到 decoder layer。两份 Llama modeling 文件同时存在普通调用与 gradient-checkpointing 重算路径。

Transformers 4.44.2 的 DynamicCache 可从张量 shape 直接返回既有长度，而 StaticCache 的 `get_seq_length()` 会扫描预分配 cache 的 occupancy 并执行 reduction。后者不适合进入目标静态图。当前 PTQ 与 rotation 入口还强制 `use_cache=False`，所以本切片不能证明真实 decode 执行。

## Decision

1. dual-phase static profile 保持规范方案；不切换为 single-static。
2. phase 表示调用开始时是否已有 KV token：既有长度 0 为 prefill，大于 0 为 decode。
3. DynamicCache/eager 路径使用 `Cache.get_seq_length()` 作为主事实源，并与显式 `cache_position` 交叉校验。
4. StaticCache/compiled 路径避免调用 occupancy-reduction `get_seq_length()`，改用已验证、非空且连续的 `cache_position[0]` 表示既有长度。预分配 cache shape 不构成 decode 证据。
5. 禁止从输入 shape、`past_key_values is None` 或手工布尔标志推断 phase；矛盾或无效元数据必须 fail closed。
6. phase 显式传给每个 decoder layer，包括 gradient-checkpointing 参数。decoder layer 在自身执行体内建立 phase context，使重算路径恢复相同 phase。
7. M002 实现范围仅限新增 `utils/quant_phase.py`、修改训练/评测两份 Llama modeling 文件，以及新增 `tests/test_quant_phase.py`。

## Consequences

- phase 语义集中于共享工具，训练/评测模型必须镜像接线。
- StaticCache 静态图避免数据相关 occupancy reduction，但调用方必须提供可信且严格校验的 `cache_position`。
- wrapper 以后可以通过 fail-closed context 获取 phase，不需要自行猜测。
- M002 不接入 wrapper、K/V、CLI、校准或 evaluator；它的完成不能作为真实逐 token decode、quantized cache 或手机 NPU 执行证据。
- 后续改变 single-static/dual-phase 策略，或改变 StaticCache 的事实来源，必须通过新的 ADR 覆盖本决策。
