"""Human-readable experiment report from the complete, non-selective result tables."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/capability-eval-20260920'
summary=json.loads((RUN/'summary_status.json').read_text())
rows=summary['rows']; paper=list(csv.DictReader((RUN/'main_results.csv').open()))
extra=list(csv.DictReader((RUN/'generation_all_metrics.csv').open())) if (RUN/'generation_all_metrics.csv').exists() else []
lengths=list(csv.DictReader((RUN/'generation_lengths.csv').open())) if (RUN/'generation_lengths.csv').exists() else []
zero=list(csv.DictReader((RUN/'appendix_zero_shot.csv').open()))
def fmt(v):return '未完成' if v is None else f'{v:.2f}'
def diff(a,b):return '未完成' if a is None or b is None else f'{a-b:+.2f}'
def flex(model,method):
 values=[x for x in extra if x['Model']==model and x['Method']==method and x['Metric']=='exact_match,flexible-extract']
 return float(values[0]['Score']) if values else None
complete=summary['completed']==77
lines=['# FIRON 模型能力补充评测报告','',
 '状态：'+('七组可用模型的77项能力任务及14项PPL全部齐备。' if complete else f"评测进行中：已完成 {summary['completed']}/77 项可用模型能力任务。"),
 'Qwen W4A16 未找到已有包，整行待补；未重新训练、选择checkpoint、重新校准或复现外部方法。','',
 '## 主结果','',
 'PPL越低越好，其余为百分比、越高越好。C4是既有固定validation子集，不是全量C4。GSM8K单元格为 **strict / flexible extraction**；两项来自同一次生成，统一报告，不择优选列。','',
 '| '+' | '.join(paper[0])+' |','| '+' | '.join(['---']*len(paper[0]))+' |']
for row in paper:
 vals=[]
 for k,v in row.items():
  if not v:vals.append('—')
  elif k in ['WikiText-2','C4','Zero-shot Avg. (8)','MMLU','IFEval']:vals.append(f'{float(v):.2f}')
  else:vals.append(v)
 lines.append('| '+' | '.join(vals)+' |')
lines+=['','## 能力保留与蒸馏作用','']
for model in ['Qwen3-1.7B','Llama-3.2-1B-Instruct']:
 own=[x for x in rows if x['Model']==model]
 bf=next(x for x in own if x['Method']=='Original BF16')
 final=next(x for x in own if x['Method']=='FIRON (QAT400, seed42)')
 parent=next(x for x in own if x['Method']=='FIRON pre-distillation (seed42)')
 lines += [f'### {model}','',
  f"相对BF16：八项平均 {fmt(bf['Zero-shot Avg. (8)'])} → {fmt(final['Zero-shot Avg. (8)'])}（{diff(final['Zero-shot Avg. (8)'],bf['Zero-shot Avg. (8)'])} 个百分点）；MMLU {fmt(bf['MMLU'])} → {fmt(final['MMLU'])}（{diff(final['MMLU'],bf['MMLU'])} 个百分点）。",
  f"GSM8K strict {fmt(bf['GSM8K'])} → {fmt(final['GSM8K'])}；flexible {fmt(flex(model,bf['Method']))} → {fmt(flex(model,final['Method']))}。IFEval prompt strict {fmt(bf['IFEval'])} → {fmt(final['IFEval'])}（{diff(final['IFEval'],bf['IFEval'])} 个百分点）。",'',
  '蒸馏前精确父包 → 最终QAT400，同包分别接受全部评测：','',
  '| 指标 | 父包 | 最终FIRON | 变化 |','|---|---:|---:|---:|']
 for key in ['WikiText-2','C4','Zero-shot Avg. (8)','MMLU','GSM8K','IFEval']:
  lines.append(f"| {key} | {fmt(parent[key])} | {fmt(final[key])} | {diff(final[key],parent[key])} |")
 a,b=flex(model,parent['Method']),flex(model,final['Method'])
 lines += [f'| GSM8K flexible | {fmt(a)} | {fmt(b)} | {diff(b,a)} |','']
 zf=next(x for x in zero if x['Model']==model and x['Method']==final['Method'])
 zp=next(x for x in zero if x['Model']==model and x['Method']==parent['Method'])
 drops=[f"{key} {float(zf[key])-float(zp[key]):+.2f}" for key in zf if key not in ['Model','Method'] and ' delta' not in key and zf[key] and zp[key] and float(zf[key])<float(zp[key])]
 if drops:lines += ['蒸馏并非逐项改善：相对父包，'+ '、'.join(drops)+' 个百分点；八项平均的改善不能替代这些单项结果。','']
lines += ['## 总体判断','',
 '在本轮固定协议下，八项zero-shot平均下降较小（Qwen 1.85、Llama 3.08个百分点），说明这些选择题任务上的能力保留相对较好；MMLU下降更大（5.75、7.68个百分点）。两种最终FIRON的GSM8K flexible及IFEval均低于BF16，不能声称推理和指令遵循无损。',
 '两个模型的蒸馏都同时改善了WikiText-2/C4 PPL、八项平均、MMLU和GSM8K flexible，但改善不覆盖所有指标：Qwen GSM8K strict下降，Llama IFEval也下降。生成分数还受到既定长度上限和答案格式影响，尤其应结合下方预算耗尽比例解读，而非把全部差距归因为知识遗忘或纯算术错误。','',
 '## 生成预算统计','',
 '下表只统计成功尝试。耗尽表示达到生成上限且未发现EOS或任务停止串，不把batch padding算作有效输出；它提示协议下的长度敏感性，不单独证明推理成败。','',
 '| Model | Method | Task | Samples | Budget exhausted (%) |',
 '|---|---|---|---:|---:|']
for item in lengths:
 lines.append(f"| {item['model']} | {item['method']} | {item['task']} | {item['samples']} | {float(item['budget_exhausted_percent']):.2f} |")
lines += ['', '## 解释边界','',
 '- 八项平均与MMLU支持蒸馏有下游恢复作用，但最终包仍不等于高精度模型；不能把PPL接近直接表述为知识、推理或指令能力无损。',
 '- MMLU是选项条件概率评分，没有生成thinking过程。Qwen的生成统一enable_thinking=False，BF16与量化匹配，因此MMLU组间差距不能归因于thinking开关不同。未做thinking开关消融，不推断其绝对贡献。',
 '- GSM8K采用官方8个CoT示例加模型聊天模板、single-user turn、greedy、256新token上限。此方式受harness支持，但不是原始completion或multiturn协议。部分模型回答示例题、使用非strict答案措辞或触及生成上限，因此不能把所有低分都归因于纯算术能力。官方strict和flexible均完整保留，未因分数改变提示或选择提取器。',
 '- 格式影响的可追溯例子：Qwen BF16 的 GSM8K doc_id=1 输出 `The answer is **3**.`，目标为3；strict提取返回invalid，flexible提取为3并判对。原文及两种评分在 qwen/bf16/gsm8k_cot/samples_gsm8k_cot.jsonl。该例说明严格分数包含答案格式敏感性，不能将其全部解释为解题能力。',
 '- IFEval为官方541条提示、prompt-level strict accuracy、1280新token上限；另有官方loose及instruction-level指标供追溯。所有生成的预算耗尽标记见generation_lengths.csv：排除EOS/stop后才记为耗尽，不能将batch padding当实际生成长度。',
 '- 下游评测使用既定seed42，无新训练或seed选择；本表不提供新训练seed方差。历史Phase2 W4A16来自单独配方，不能用于断言仅A8造成全部差距。',
 '- 这是BF16 GPU fake-quant执行，backbone W4、静态INT8/SP2输入、KV16；并非原生INT4/INT8内核或手机延迟评测。',
 '', '## 产物与追溯','',
 '- 主表：main_results.csv / main_results.tex；可直接插入论文的含caption版本：paper_tables.tex。main_results_numeric.csv保留原定strict数值列供机器处理。',
 '- 附录：appendix_zero_shot.csv及两个LaTeX明细表；appendix_details.csv；metric_deltas.csv/.tex；generation_all_metrics.csv。',
 '- 完整协议：PROTOCOL.md。模型和包：models.json。12项历史PPL精确复用：ppl_reuse.json；W4A16新增完整test/C4位于llama/w4a16/ppl-*。',
 '- 每任务：settings.json、launch-N.json、run-N.log、results.json、samples_*.jsonl、quantization.json；生成原文位于settings指定generation_log。有效源码及环境：source/、environment.txt。',
 '- 独立验证：verifier/。Llama padding适配失败和Qwen批调度的完整记录：RECOVERY.md。原失败/部分输出均保留，最终统计只使用成功尝试。',
 '- 入口：scripts/capability_eval/run.py、schedule.py、summarize.py、generation_summary.py、report.py。成功结果不会被调度器重复执行。',
 '', '## 未完成项','',
 'Qwen W4A16：未定位已有模型包。本轮不新建/搜索该对照，WikiText-2、C4及全部11项能力任务均标为待补。']
if not complete:lines+=['','其余运行中/失败任务逐项见incomplete.json，当前表不是最终交付。']
(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
