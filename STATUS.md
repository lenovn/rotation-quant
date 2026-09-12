# 当前状态

核对日期：2026-09-08。依据本地 Git、当前启动脚本和已有日志；本轮没有重跑实验或测试。

## 源码与入口

- `repos/SpinQuant`：`phase2/joint-r-sa-w16-gptq-w4`，HEAD `2491831`，本轮核查工作树干净。
- `worktrees/SpinQuant-distribution-experiment`：已有未跟踪 `experiments/`，本轮保留原状。
- 当前流程与默认设置见 [SPEC.md](SPEC.md)；研究历史见 [research_log.md](research_log.md)。

## 保留的 Phase 2 结果

以下均为已有 `wikitext2_8x2048.log` 中的 PPL，四舍五入至五位小数。表中同名精度不代表跨运行完全相同的配置。

| 运行目录（位于 runs/phase2/） | 评测 | PPL |
| --- | --- | ---: |
| w16a8-downa16-joint-r-sa-r12-100step-s42 | W16A16 | 12.58486 |
| 同上 | W16/static A8，down-A16 | 12.61457 |
| 同上 | W4A16 | 16.06202 |
| 同上 | GPTQ W4/static A8，down-A16 | 16.18434 |
| w16a8-downa16-joint-r-sa-r12-100step-s42-initial-sa-eval | 最终 R、初始 SA 的 W16 对照 | 12.62413 |
| w16a8-joint-r-sa-r12-s42 | W16/static A8，down-A16 | 12.63674 |
| 同上 | GPTQ W4/static A8，down-A16 | 16.79897 |

前一组证据分别位于该目录的 `eval-w16a16/`、`eval-w16a8-downa16/`、`eval-w4a16/`、`gptq/`；后一组位于其 `eval-w16a8-downa16/`、`gptq/`。分布分析默认引用后一组的 R.bin；两组 R/SA/结果均保留，不能混用。

## 分布分析产物

`runs/distribution-atlas/llama32-1b-r12-bf16-wt2-s42/` 保留权重/激活 JSON、112 张 weights PNG、64 张 activations PNG，以及 `weight_channel_score_summary.json`、`weight_channel_summary.pt` 和 `weight_channel_scores/`。本轮只核对文件存在和数量，不把文件存在当作新数值验收。

## 整理记录

- 第一批：7 个目录、81 个文件（13 个 Markdown），约 5.75 GiB，移至 `delete/`，尚未删除。
- 第二批：九份根目录 Markdown 原版移至 `delete/root-docs-2026-09-08/`；重建六份当前文档，根目录不再保留 CONTEXT/DECISIONS/WORKLOG。
- Phase 0、正式 Phase 2、分布产物、源码和缓存保留；未启动 CPU/GPU 实验、测试、安装、提交或推送。
- 详细路径及恢复方式见 [整理清单](delete/README.md)。

M001–M004 的历史验收不在本轮重述为当前 PASS；详情见 [历史状态](delete/root-docs-2026-09-08/STATUS.md)。
