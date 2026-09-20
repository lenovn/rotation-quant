# FIRON 模型能力补充评测报告

状态：七组可用模型的77项能力任务及14项PPL全部齐备。
Qwen W4A16 未找到已有包，整行待补；未重新训练、选择checkpoint、重新校准或复现外部方法。

## 主结果

PPL越低越好，其余为百分比、越高越好。C4是既有固定validation子集，不是全量C4。GSM8K单元格为 **strict / flexible extraction**；两项来自同一次生成，统一报告，不择优选列。

| Model | Method | W–A format | WikiText-2 | C4 | Zero-shot Avg. (8) | MMLU | GSM8K | IFEval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | Original BF16 | W16A16 | 16.72 | 23.14 | 58.22 | 60.15 | 4.70 / 61.56 | 67.28 |
| Qwen3-1.7B | FIRON (QAT400, seed42) | W4A8 (INT8/SP2) | 14.15 | 24.25 | 56.37 | 54.40 | 11.30 / 42.23 | 42.70 |
| Qwen3-1.7B | FIRON pre-distillation (seed42) | W4A8 (INT8/SP2) | 14.30 | 24.90 | 54.17 | 45.77 | 18.88 / 31.31 | 37.34 |
| Qwen3-1.7B | W4A16 control (unavailable) | W4A16 | — | — | — | — | — | — |
| Llama-3.2-1B-Instruct | Original BF16 | W16A16 | 13.16 | 21.84 | 55.28 | 45.54 | 10.61 / 23.20 | 48.80 |
| Llama-3.2-1B-Instruct | FIRON (QAT400, seed42) | W4A8 (INT8/SP2) | 14.15 | 26.98 | 52.20 | 37.86 | 1.74 / 4.55 | 28.28 |
| Llama-3.2-1B-Instruct | FIRON pre-distillation (seed42) | W4A8 (INT8/SP2) | 15.56 | 27.44 | 50.99 | 33.73 | 0.53 / 2.27 | 31.05 |
| Llama-3.2-1B-Instruct | Historical Phase2 GPTQ | W4A16 | 15.62 | 26.65 | 51.29 | 37.32 | 0.15 / 0.76 | 26.80 |

## 能力保留与蒸馏作用

### Qwen3-1.7B

相对BF16：八项平均 58.22 → 56.37（-1.85 个百分点）；MMLU 60.15 → 54.40（-5.75 个百分点）。
GSM8K strict 4.70 → 11.30；flexible 61.56 → 42.23。IFEval prompt strict 67.28 → 42.70（-24.58 个百分点）。

蒸馏前精确父包 → 最终QAT400，同包分别接受全部评测：

| 指标 | 父包 | 最终FIRON | 变化 |
|---|---:|---:|---:|
| WikiText-2 | 14.30 | 14.15 | -0.15 |
| C4 | 24.90 | 24.25 | -0.66 |
| Zero-shot Avg. (8) | 54.17 | 56.37 | +2.20 |
| MMLU | 45.77 | 54.40 | +8.63 |
| GSM8K | 18.88 | 11.30 | -7.58 |
| IFEval | 37.34 | 42.70 | +5.36 |
| GSM8K flexible | 31.31 | 42.23 | +10.92 |

蒸馏并非逐项改善：相对父包，arc_easy -0.34 个百分点；八项平均的改善不能替代这些单项结果。

### Llama-3.2-1B-Instruct

相对BF16：八项平均 55.28 → 52.20（-3.08 个百分点）；MMLU 45.54 → 37.86（-7.68 个百分点）。
GSM8K strict 10.61 → 1.74；flexible 23.20 → 4.55。IFEval prompt strict 48.80 → 28.28（-20.52 个百分点）。

蒸馏前精确父包 → 最终QAT400，同包分别接受全部评测：

| 指标 | 父包 | 最终FIRON | 变化 |
|---|---:|---:|---:|
| WikiText-2 | 15.56 | 14.15 | -1.41 |
| C4 | 27.44 | 26.98 | -0.45 |
| Zero-shot Avg. (8) | 50.99 | 52.20 | +1.21 |
| MMLU | 33.73 | 37.86 | +4.12 |
| GSM8K | 0.53 | 1.74 | +1.21 |
| IFEval | 31.05 | 28.28 | -2.77 |
| GSM8K flexible | 2.27 | 4.55 | +2.27 |

蒸馏并非逐项改善：相对父包，social_iqa -0.41、winogrande -0.08 个百分点；八项平均的改善不能替代这些单项结果。

## 总体判断

在本轮固定协议下，八项zero-shot平均下降较小（Qwen 1.85、Llama 3.08个百分点），说明这些选择题任务上的能力保留相对较好；MMLU下降更大（5.75、7.68个百分点）。两种最终FIRON的GSM8K flexible及IFEval均低于BF16，不能声称推理和指令遵循无损。
两个模型的蒸馏都同时改善了WikiText-2/C4 PPL、八项平均、MMLU和GSM8K flexible，但改善不覆盖所有指标：Qwen GSM8K strict下降，Llama IFEval也下降。生成分数还受到既定长度上限和答案格式影响，尤其应结合下方预算耗尽比例解读，而非把全部差距归因为知识遗忘或纯算术错误。

## 生成预算统计

下表只统计成功尝试。耗尽表示达到生成上限且未发现EOS或任务停止串，不把batch padding算作有效输出；它提示协议下的长度敏感性，不单独证明推理成败。

| Model | Method | Task | Samples | Budget exhausted (%) |
|---|---|---|---:|---:|
| llama | bf16 | gsm8k_cot | 1319 | 5.69 |
| llama | bf16 | ifeval | 541 | 7.95 |
| llama | firon | gsm8k_cot | 1319 | 6.52 |
| llama | firon | ifeval | 541 | 33.27 |
| llama | parent | gsm8k_cot | 1319 | 42.08 |
| llama | parent | ifeval | 541 | 32.53 |
| llama | w4a16 | gsm8k_cot | 1319 | 21.08 |
| llama | w4a16 | ifeval | 541 | 15.53 |
| qwen | bf16 | gsm8k_cot | 1319 | 22.97 |
| qwen | bf16 | ifeval | 541 | 6.28 |
| qwen | firon | gsm8k_cot | 1319 | 43.67 |
| qwen | firon | ifeval | 541 | 63.96 |
| qwen | parent | gsm8k_cot | 1319 | 8.79 |
| qwen | parent | ifeval | 541 | 65.43 |

## 解释边界

- 八项平均与MMLU支持蒸馏有下游恢复作用，但最终包仍不等于高精度模型；不能把PPL接近直接表述为知识、推理或指令能力无损。
- MMLU是选项条件概率评分，没有生成thinking过程。Qwen的生成统一enable_thinking=False，BF16与量化匹配，因此MMLU组间差距不能归因于thinking开关不同。未做thinking开关消融，不推断其绝对贡献。
- GSM8K采用官方8个CoT示例加模型聊天模板、single-user turn、greedy、256新token上限。此方式受harness支持，但不是原始completion或multiturn协议。部分模型回答示例题、使用非strict答案措辞或触及生成上限，因此不能把所有低分都归因于纯算术能力。官方strict和flexible均完整保留，未因分数改变提示或选择提取器。
- 格式影响的可追溯例子：Qwen BF16 的 GSM8K doc_id=1 输出 `The answer is **3**.`，目标为3；strict提取返回invalid，flexible提取为3并判对。原文及两种评分在 qwen/bf16/gsm8k_cot/samples_gsm8k_cot.jsonl。该例说明严格分数包含答案格式敏感性，不能将其全部解释为解题能力。
- IFEval为官方541条提示、prompt-level strict accuracy、1280新token上限；另有官方loose及instruction-level指标供追溯。所有生成的预算耗尽标记见generation_lengths.csv：排除EOS/stop后才记为耗尽，不能将batch padding当实际生成长度。
- 下游评测使用既定seed42，无新训练或seed选择；本表不提供新训练seed方差。历史Phase2 W4A16来自单独配方，不能用于断言仅A8造成全部差距。
- 这是BF16 GPU fake-quant执行，backbone W4、静态INT8/SP2输入、KV16；并非原生INT4/INT8内核或手机延迟评测。

## 产物与追溯

- 主表：main_results.csv / main_results.tex；可直接插入论文的含caption版本：paper_tables.tex。main_results_numeric.csv保留原定strict数值列供机器处理。
- 附录：appendix_zero_shot.csv及两个LaTeX明细表；appendix_details.csv；metric_deltas.csv/.tex；generation_all_metrics.csv。
- 完整协议：PROTOCOL.md。模型和包：models.json。12项历史PPL精确复用：ppl_reuse.json；W4A16新增完整test/C4位于llama/w4a16/ppl-*。
- 每任务：settings.json、launch-N.json、run-N.log、results.json、samples_*.jsonl、quantization.json；生成原文位于settings指定generation_log。有效源码及环境：source/、environment.txt。
- 独立验证：verifier/。Llama padding适配失败和Qwen批调度的完整记录：RECOVERY.md。原失败/部分输出均保留，最终统计只使用成功尝试。
- 入口：scripts/capability_eval/run.py、schedule.py、summarize.py、generation_summary.py、report.py。成功结果不会被调度器重复执行。

## 未完成项

Qwen W4A16：未定位已有模型包。本轮不新建/搜索该对照，WikiText-2、C4及全部11项能力任务均标为待补。
