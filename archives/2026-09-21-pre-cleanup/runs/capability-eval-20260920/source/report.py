"""Report the four metrics in the user's revised evaluation scope."""
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/capability-eval-20260920'
summary = json.loads((RUN / 'summary_status.json').read_text())
rows = summary['rows']
zero = list(csv.DictReader((RUN / 'appendix_zero_shot.csv').open()))
missing = json.loads((RUN / 'incomplete.json').read_text())
metrics = ['WikiText-2', 'C4-subset', 'Zero-shot Avg. (8)', 'MMLU']
def fmt(value):
    return '未完成' if value is None else f'{value:.2f}'
def delta(a, b):
    return '未完成' if a is None or b is None else f'{a-b:+.2f}'
lines = ['# FIRON 四指标评测报告', '',
    '用户于2026-09-20将当前范围调整为 WikiText-2、C4-subset、Zero-shot Avg. (8)、MMLU。用户进一步明确Qwen W4A16采用原始BF16直接GPTQ量化基线，只用训练集做一次校准。',
    f"状态：{summary['completed']}/{summary['requested']}项选择题任务完成（{len(rows)}组模型，每组八项zero-shot及MMLU）；{sum(row[k] is not None for row in rows for k in ['WikiText-2','C4-subset'])}/{2*len(rows)}项PPL齐备。仅新增Qwen W4A16四指标评测，其余结果复用。", '',
    '## 主结果', '',
    'PPL越低越好，其余为百分比、越高越好。C4-subset为既有固定1024窗口英文validation子集。', '',
    '| Model | Method | W–A format | WikiText-2 ↓ | C4-subset ↓ | Zero-shot Avg. (8) ↑ | MMLU ↑ |',
    '|---|---|---|---:|---:|---:|---:|']
for row in rows:
    lines.append('| ' + ' | '.join([row['Model'], row['Method'], row['W–A format']] + [fmt(row[k]) for k in metrics]) + ' |')
lines += ['', '## 能力保留与蒸馏作用', '']
for model in ['Qwen3-1.7B', 'Llama-3.2-1B-Instruct']:
    own = [x for x in rows if x['Model'] == model]
    bf = next(x for x in own if x['Method'] == 'Original BF16')
    final = next(x for x in own if x['Method'] == 'FIRON (QAT400, seed42)')
    parent = next(x for x in own if x['Method'] == 'FIRON pre-distillation (seed42)')
    lines += [f'### {model}', '',
        f"相对BF16：八项平均 {fmt(bf['Zero-shot Avg. (8)'])} → {fmt(final['Zero-shot Avg. (8)'])}（{delta(final['Zero-shot Avg. (8)'], bf['Zero-shot Avg. (8)'])} 个百分点）；MMLU {fmt(bf['MMLU'])} → {fmt(final['MMLU'])}（{delta(final['MMLU'], bf['MMLU'])} 个百分点）。", '',
        '蒸馏前父包与最终FIRON比较：', '',
        '| 指标 | 父包 | 最终FIRON | 变化 |', '|---|---:|---:|---:|']
    for key in metrics:
        lines.append(f'| {key} | {fmt(parent[key])} | {fmt(final[key])} | {delta(final[key], parent[key])} |')
    zf = next(x for x in zero if x['Model'] == model and x['Method'] == final['Method'])
    zp = next(x for x in zero if x['Model'] == model and x['Method'] == parent['Method'])
    drops = [f'{k} {float(zf[k])-float(zp[k]):+.2f}' for k in zf if k not in ['Model', 'Method'] and ' delta' not in k and zf[k] and zp[k] and float(zf[k]) < float(zp[k])]
    if drops:
        lines += ['', '八项平均改善不代表逐项改善：相对父包，' + '、'.join(drops) + ' 个百分点。']
    if model == 'Qwen3-1.7B':
        control = next(x for x in own if x['Method'] == 'GPTQ (original BF16)')
        lines += ['', f"相对独立GPTQ基线，最终FIRON的八项平均变化 {delta(final['Zero-shot Avg. (8)'], control['Zero-shot Avg. (8)'])} 个百分点，MMLU变化 {delta(final['MMLU'], control['MMLU'])} 个百分点。这是完整量化配方的比较，不能将差距归因为A8/A16位宽本身；FIRON还包含既有优化与蒸馏。"]
    lines += ['']
lines += ['## 结论与解释范围', '',
    '- 两种模型的蒸馏均同时改善这四项汇总指标：WikiText-2/C4-subset PPL下降，八项平均和MMLU上升。',
    '- 相对BF16，最终FIRON的八项平均下降1.85/3.08个百分点，MMLU下降5.75/7.68个百分点（Qwen/Llama）。这些结果支持部分能力保留与蒸馏恢复，不能表述为知识无损。',
    '- 当前四指标不覆盖自由生成推理和指令遵循，不据此声称这些能力得到保留。MMLU采用5-shot候选答案似然评分，不生成thinking过程。',
    '- Qwen W4A16是原始BF16→GPTQ独立基线：196个主干Linear、per-output-channel尺度[out_features,1]、INT4码[-8,7]、零点0；WikiText-2 train固定128×2048 token、seed42，一次校准。groupsize=-1、damp=0.01、blocksize=128、无act-order/权重裁剪搜索，无旋转学习、蒸馏或评测数据调参。',
    '- 历史Llama W4A16使用Phase2 GPTQ配方，不是最终FIRON的单变量激活位宽消融；Qwen为原始权重GPTQ，Llama为历史配方，二者方法标签分别保留。',
    '- 本轮为GPU fake-quant质量评测；FIRON主干W4、静态INT8/SP2输入，embedding/head/norm保持高精度，不能推断手机原生整数内核性能。', '',
    '## 产物与追溯', '',
    '- 主表：main_results.csv/.tex；含caption及附录的论文入口：paper_tables.tex。',
    '- 八项明细与相对BF16差值：appendix_zero_shot.csv/.tex；四指标差值：metric_deltas.csv/.tex；逐任务来源：appendix_details.csv。',
    '- 协议：PROTOCOL.md；模型路径：models.json；复用PPL来源：ppl_reuse.json；历史W4A16新增PPL原文：llama/w4a16/ppl-*。',
    '- 原始评分、提示及量化记录：各模型/方法/任务目录下results.json、samples_*.jsonl、settings.json、quantization.json。源码与环境：source/、environment.txt。',
    '- 评测入口：scripts/capability_eval/README.md。调度器当前只含八项zero-shot和MMLU；成功结果不重复执行。',
    '- 新增GPTQ的构建、模型格点和结果独立审核：verifier/QWEN_GPTQ_REVIEW.md及qwen_gptq_results_audit.json。早期范围审核保留为历史记录。', '',
    '## 范围调整记录', '',
    '此前已完成GSM8K和IFEval；按用户最新要求，不再安排这些评测，也不纳入当前主表或结论。原始结果保留；调整前完整表格、报告、协议及入口快照保存在archive/full-scope-before-user-reduction/。此次删列来自用户明确缩减范围，既有结果未删除。之前误将Qwen W4A16移出范围，随后又先构造了FIRON/A16消融。用户明确选择原始BF16→GPTQ基线后，消融的已有原始结果整体存于qwen/firon_a16_ablation/，不混入当前主表；中间报告位于archive/。', '',
    '## 未完成项', '',
    '无。当前约定范围全部完成。' if not missing else '见incomplete.json。']
(RUN / 'REPORT.md').write_text('\n'.join(lines) + '\n')
