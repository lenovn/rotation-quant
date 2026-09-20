# FIRON 四指标评测报告

用户于2026-09-20将当前范围调整为 WikiText-2、C4-subset、Zero-shot Avg. (8)、MMLU。Qwen W4A16不纳入，也不列为待补。
状态：63/63项选择题任务完成（七组模型，每组八项zero-shot及MMLU）；14项PPL齐备。此次范围调整仅复用既有结果，未启动GPU评测。

## 主结果

PPL越低越好，其余为百分比、越高越好。C4-subset为既有固定1024窗口英文validation子集。

| Model | Method | W–A format | WikiText-2 ↓ | C4-subset ↓ | Zero-shot Avg. (8) ↑ | MMLU ↑ |
|---|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | Original BF16 | W16A16 | 16.72 | 23.14 | 58.22 | 60.15 |
| Qwen3-1.7B | FIRON (QAT400, seed42) | W4A8 (INT8/SP2) | 14.15 | 24.25 | 56.37 | 54.40 |
| Qwen3-1.7B | FIRON pre-distillation (seed42) | W4A8 (INT8/SP2) | 14.30 | 24.90 | 54.17 | 45.77 |
| Llama-3.2-1B-Instruct | Original BF16 | W16A16 | 13.16 | 21.84 | 55.28 | 45.54 |
| Llama-3.2-1B-Instruct | FIRON (QAT400, seed42) | W4A8 (INT8/SP2) | 14.15 | 26.98 | 52.20 | 37.86 |
| Llama-3.2-1B-Instruct | FIRON pre-distillation (seed42) | W4A8 (INT8/SP2) | 15.56 | 27.44 | 50.99 | 33.73 |
| Llama-3.2-1B-Instruct | Historical Phase2 GPTQ | W4A16 | 15.62 | 26.65 | 51.29 | 37.32 |

## 能力保留与蒸馏作用

### Qwen3-1.7B

相对BF16：八项平均 58.22 → 56.37（-1.85 个百分点）；MMLU 60.15 → 54.40（-5.75 个百分点）。

蒸馏前父包与最终FIRON比较：

| 指标 | 父包 | 最终FIRON | 变化 |
|---|---:|---:|---:|
| WikiText-2 | 14.30 | 14.15 | -0.15 |
| C4-subset | 24.90 | 24.25 | -0.66 |
| Zero-shot Avg. (8) | 54.17 | 56.37 | +2.20 |
| MMLU | 45.77 | 54.40 | +8.63 |

八项平均改善不代表逐项改善：相对父包，arc_easy -0.34 个百分点。

### Llama-3.2-1B-Instruct

相对BF16：八项平均 55.28 → 52.20（-3.08 个百分点）；MMLU 45.54 → 37.86（-7.68 个百分点）。

蒸馏前父包与最终FIRON比较：

| 指标 | 父包 | 最终FIRON | 变化 |
|---|---:|---:|---:|
| WikiText-2 | 15.56 | 14.15 | -1.41 |
| C4-subset | 27.44 | 26.98 | -0.45 |
| Zero-shot Avg. (8) | 50.99 | 52.20 | +1.21 |
| MMLU | 33.73 | 37.86 | +4.12 |

八项平均改善不代表逐项改善：相对父包，social_iqa -0.41、winogrande -0.08 个百分点。

## 结论与解释范围

- 两种模型的蒸馏均同时改善这四项汇总指标：WikiText-2/C4-subset PPL下降，八项平均和MMLU上升。
- 相对BF16，最终FIRON的八项平均下降1.85/3.08个百分点，MMLU下降5.75/7.68个百分点（Qwen/Llama）。这些结果支持部分能力保留与蒸馏恢复，不能表述为知识无损。
- 当前四指标不覆盖自由生成推理和指令遵循，不据此声称这些能力得到保留。MMLU采用5-shot候选答案似然评分，不生成thinking过程。
- 历史Llama W4A16使用Phase2 GPTQ配方，不是最终FIRON的单变量激活位宽消融。
- 本轮为GPU fake-quant质量评测；FIRON主干W4、静态INT8/SP2输入，embedding/head/norm保持高精度，不能推断手机原生整数内核性能。

## 产物与追溯

- 主表：main_results.csv/.tex；含caption及附录的论文入口：paper_tables.tex。
- 八项明细与相对BF16差值：appendix_zero_shot.csv/.tex；四指标差值：metric_deltas.csv/.tex；逐任务来源：appendix_details.csv。
- 协议：PROTOCOL.md；模型路径：models.json；复用PPL来源：ppl_reuse.json；历史W4A16新增PPL原文：llama/w4a16/ppl-*。
- 原始评分、提示及量化记录：各模型/方法/任务目录下results.json、samples_*.jsonl、settings.json、quantization.json。源码与环境：source/、environment.txt。
- 评测入口：scripts/capability_eval/README.md。调度器当前只含八项zero-shot和MMLU；成功结果不重复执行。

## 范围调整记录

此前已完成GSM8K和IFEval；按用户最新要求，不再安排这些评测，也不纳入当前主表或结论。原始结果保留；调整前完整表格、报告、协议及入口快照保存在archive/full-scope-before-user-reduction/。此次删列来自用户明确缩减范围，既有结果未删除。

## 未完成项

无。当前约定范围全部完成。
