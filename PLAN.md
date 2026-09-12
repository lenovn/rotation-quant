# 当前计划

更新日期：2026-09-08。

本轮任务是整理文件和文档，未授权新算法实现或 CPU/GPU 实验。

## 本轮已完成

- 旧方案、Phase 1 脚本与产物、四组 Phase 2 smoke 已移入 `delete/`，保留原目录结构。
- 九份根目录 Markdown 原版已归档；根目录保留协作规则、当前说明、计划、状态、经验与研究记录六份文档。
- 旧 CONTEXT 的有效术语及 DECISIONS 的有效约束合并到 SPEC/AGENTS；WORKLOG 的历史内容归档，当前整理记录见 STATUS 与归档清单。

## 后续研究候选，尚未启动

优先明确同配置对照，再决定是否实现或运行新实验：分别分析 W4 相比 W16 的误差，以及 W4 per-channel 相比 group32 的粒度误差。若采用 module-family 混合 groupsize，需要先明确参数接口与实现范围；当前启动脚本只传统一 `w_groupsize=-1`，不应把候选接口写成已实现功能。

新增实验须使用明确的 RUN_NAME、同一 BF16/R 和相同数据及 activation 口径。现有两组 Phase 2 结果不能拼接成同一次实验，见 [STATUS.md](STATUS.md)。这段候选方向不构成启动授权。

旧里程碑、periodic recalibration 和 dual-phase 后续实施顺序仅用于追溯，见 [原计划](delete/root-docs-2026-09-08/PLAN.md)。
