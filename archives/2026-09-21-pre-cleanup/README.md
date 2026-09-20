# 2026-09-21 源码与实验数据归档

用户确认当前主路线为 Phase 6 down 静态 INT16；旧 SP2 和蒸馏仅保留数据对照。用户授权先推送原 GitHub 仓库，再清理，且不得影响 Phase 6。

## 保存范围

- 外层仓库继续按已有方式保存 repos/ 与 worktrees/ 的普通源码文件，不改动嵌套 Git 分支和工作树。source_revisions.json 记录备份时的实际来源。
- 本目录 runs/ 是原位置同名文本文件的快照，保留指标 JSON/CSV、训练 JSONL、配置、日志、源码快照和独立审计记录。retained_records.json 列出本次复制的文件。
- 原始文件仍留在 runs/；C4 原始 documents.jsonl 文档、逐样本 samples_*/generation_batches* 输出、二进制评测输入、图表、张量统计也继续留在本地，不在本次 Git 文本归档范围内。
- Phase 6 仍在开展，本归档只是时点快照，不将正在进行的工作标记为完成。

## 清理边界

cleanup_candidates.json 是具体候选清单。只有远端提交确认存在后，才执行删除并记录实际结果。保留所有环境、预训练模型、Phase 6 文件、源码、B100 父包及校准统计。

旧蒸馏 resume.pt 与静态模型包、Phase 5 SP2 后处理静态包及 prefix.pt 将被清理；Git 未引用临时 pack 和重新可下载的安装缓存也列入。执行时仍被进程使用或已发生变化的文件跳过。删除权重后，既有指标可查，但复跑和精确续训需要重新生成模型。

本任务不修改算法，不重跑训练或评测；Git 提交不代表新的算法验收。
