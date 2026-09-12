# rotation-quant 协作规则

适用于 `/home/dongpeiyan/projects/rotation-quant` 整棵目录树。用户当前任务中的明确授权优先；历史计划和归档中的授权不自动延续到新任务。

## 阅读与事实来源

开始工作先读 [STATUS.md](STATUS.md)、[SPEC.md](SPEC.md)、[PLAN.md](PLAN.md) 和 [LESSONS.md](LESSONS.md)，需要实验历史时读 [research_log.md](research_log.md)。以真实源码、启动参数、Git 状态和实验日志核对关键事实。

- 主源码仓库：`repos/SpinQuant`；FHT：`repos/fast-hadamard-transform`。源码 Git 操作须明确指向对应仓库。
- 独立分布实验：`worktrees/SpinQuant-distribution-experiment`。共享目录中的既有未提交工作必须保留。
- 根目录协调文档由 coordinator 更新；历史术语、决策和工作日志已合并，不再要求读取根目录 CONTEXT/DECISIONS/WORKLOG。
- 旧版文档在 [归档目录](delete/root-docs-2026-09-08/)，更早方案在 [旧方案目录](delete/older_plan/2026-08-11-dual-phase-static/)。归档用于追溯，不决定当前执行顺序。

## 共享服务器操作边界

- 每次修改前说明具体文件、目的和副作用；写入、测试、安装、删除、CPU/GPU 实验、提交和推送须有当前任务的明确授权。已授权范围内不重复索要许可；只读核查可直接进行。
- 每次执行一个清晰步骤，完成后报告证据；设计同意、实现完成和验证通过分别记录。
- 不使用 sudo。不得擅自创建/切换分支、修改环境、覆盖已有实验产物或他人工作。
- Phase 0 审计分支与 worktree、FHT 仓库、Phase 0 证据及既有 `R.bin` 保持只读；用户明确授权的归档移动不改变文件内容。
- 应用代码实现与独立验证由不同角色承担；没有独立 verifier 的明确 PASS，不关闭应用代码里程碑。文档整理不属于应用代码实现。
- 本轮文件整理授权不包括运行新实验、测试、提交或推送。

## 保留的实现与证据要求

- M001–M004 已有代码、测试、legacy 兼容路径及安全措施保留；旧方案归档不授权删除它们。历史验收记录见归档，不当作当前重新测试结果。
- dual-phase static 模式保留冻结 qparams、cache-derived phase、精确 coverage、缺参/不一致 fail-closed，以及顶层资源初始化前校验和下游二次校验。
- 相关既有约束见 [SPEC.md](SPEC.md) 和 [ADR](docs/adr/)；更改这些能力时回查归档中的具体接口和边界，不以精简文档为由绕过。
- `torch.compile(fullgraph=True)` 的生成器 context manager 限制是非阻塞诊断，不能替代或冒充后续 mllm/QNN 验收。
- 不把 dynamic A8、KV16、浮点 fake-quant cache 或 GPU PPL 描述为 strict-static 手机 NPU 结果。
- 默认不新增 hash、冻结 contract、baseline 或 gate。只有能明确指出具体失败场景并解释 Git、版本号、主键、事务、唯一约束、类型及普通测试为何不足时才考虑；门禁只放在不可逆、跨系统、安全或正式发布边界。已有安全措施不得为了简化而删除，前置检查不得挤掉真实执行或测量。
