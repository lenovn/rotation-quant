"""Regenerate paper tables from raw completed results; never fill missing cells."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/capability-eval-20260920'
TASK_METRICS={'boolq':'acc,none','piqa':'acc_norm,none','social_iqa':'acc,none','hellaswag':'acc_norm,none','winogrande':'acc,none','arc_easy':'acc_norm,none','arc_challenge':'acc_norm,none','openbookqa':'acc_norm,none','mmlu':'acc,none','gsm8k_cot':'exact_match,strict-match','ifeval':'prompt_level_strict_acc,none'}
NAMES={'qwen':'Qwen3-1.7B','llama':'Llama-3.2-1B-Instruct'}
METHODS={'bf16':'Original BF16','firon':'FIRON (QAT400, seed42)','parent':'FIRON pre-distillation (seed42)','w4a16':'Historical Phase2 GPTQ'}
ppl=json.loads((RUN/'ppl_reuse.json').read_text())
models=json.loads((RUN/'models.json').read_text())
records=[]; details=[]; missing=[]
for model in NAMES:
 for method in METHODS:
  row={'Model':NAMES[model],'Method':('W4A16 control (unavailable)' if model=='qwen' and method=='w4a16' else METHODS[method]),'W–A format':'W16A16' if method=='bf16' else 'W4A16' if method=='w4a16' else 'W4A8 (INT8/SP2)'}
  for dataset,col in [('Salesforce/wikitext','WikiText-2'),('allenai/c4','C4')]:
   hits=[x for x in ppl if x['model_key']==model and x['method']==method and x['dataset']==dataset]
   if hits:row[col]=float(hits[0]['ppl'])
   else:
    path=RUN/model/method/('ppl-test' if col=='WikiText-2' else 'ppl-c4')/'result.json'
    row[col]=json.loads(path.read_text())['ppl'] if path.exists() else None
  for col in ['WikiText-2','C4']:
   if row[col] is None:
    missing.append(dict(model=model,method=method,task=col,reason='No existing matching model package found' if method not in models[model] else 'PPL measurement pending; historical short-window results excluded'))
  scores={}
  for task,metric in TASK_METRICS.items():
   base=RUN/model/method/task; result=base/'results.json'
   value=None;status='pending'
   if result.exists():
    data=json.loads(result.read_text()); values=data.get('results',{}).get(task,data.get('groups',{}).get(task,{}))
    value=values.get(metric)
    if value is not None: value*=100;status='completed'
    else:status='missing requested metric'
   elif (base/'failure.json').exists():status=json.loads((base/'failure.json').read_text())['message']
   elif method not in models[model]:status='No existing matching model package found'
   if value is None:missing.append(dict(model=model,method=method,task=task,reason=status))
   scores[task]=value
   details.append(dict(Model=NAMES[model],Method=row['Method'],Task=task,Metric=metric,Score=value,Delta_vs_BF16_pp=None,Status=status,Source=str(result) if result.exists() else ''))
  eight=[scores[t] for t in list(TASK_METRICS)[:8]]
  row['Zero-shot Avg. (8)']=sum(eight)/8 if all(x is not None for x in eight) else None
  row.update(MMLU=scores['mmlu'],GSM8K=scores['gsm8k_cot'],IFEval=scores['ifeval'])
  records.append(row)
for d in details:
 ref=next(x for x in details if x['Model']==d['Model'] and x['Method']==METHODS['bf16'] and x['Task']==d['Task'])
 if ref['Score'] is not None and d['Score'] is not None:d['Delta_vs_BF16_pp']=d['Score']-ref['Score']
def csvwrite(name,rows):
 with (RUN/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def esc(s):return str(s).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%').replace('–','--')
def latex(name,rows):
 columns=list(rows[0]);lines=[r'\begin{tabular}{'+'l'*sum(not isinstance(rows[0][c], (int,float)) and c in ['Model','Method','W–A format'] for c in columns)+'r'*sum(c not in ['Model','Method','W–A format'] for c in columns)+'}',r'\toprule',' & '.join(esc(c) for c in columns)+r' \\',r'\midrule']
 for row in rows:
  lines.append(' & '.join('--' if v is None else f'{v:.2f}' if isinstance(v,float) else esc(v) for v in row.values())+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}'];(RUN/name).write_text('\n'.join(lines)+'\n')
# Keep numeric primary metrics for machine use and deltas; show both official GSM extractors in paper table.
csvwrite('main_results_numeric.csv',records)
paper_records=[]
for row in records:
 display=dict(row)
 model=next(k for k,v in NAMES.items() if v==row['Model'])
 method=next((k for k,v in METHODS.items() if v==row['Method']), 'w4a16')
 path=RUN/model/method/'gsm8k_cot/results.json'
 if path.exists():
  values=json.loads(path.read_text())['results']['gsm8k_cot']
  flexible=values.get('exact_match,flexible-extract')
  if row['GSM8K'] is not None and flexible is not None:
   display['GSM8K']=f"{row['GSM8K']:.2f} / {100*flexible:.2f}"
 paper_records.append(display)
csvwrite('main_results.csv',paper_records);latex('main_results.tex',paper_records)
csvwrite('appendix_details.csv',details)
wide=[]
for row in records:
 w={'Model':row['Model'],'Method':row['Method']}
 for task in list(TASK_METRICS)[:8]:
  d=next(x for x in details if x['Model']==row['Model'] and x['Method']==row['Method'] and x['Task']==task)
  w[task]=d['Score'];w[task+' delta (pp)']=d['Delta_vs_BF16_pp']
 wide.append(w)
csvwrite('appendix_zero_shot.csv',wide)
score_rows=[{k:v for k,v in row.items() if ' delta (pp)' not in k} for row in wide]
delta_rows=[{k.replace(' delta (pp)',''):v for k,v in row.items() if k in ['Model','Method'] or ' delta (pp)' in k} for row in wide]
latex('appendix_zero_shot_scores.tex',score_rows)
latex('appendix_zero_shot_deltas.tex',delta_rows)
(RUN/'appendix_zero_shot.tex').write_text('% Scores in percent; second table gives percentage-point differences vs BF16.\n'+(RUN/'appendix_zero_shot_scores.tex').read_text()+'\n\\par\\medskip\n'+(RUN/'appendix_zero_shot_deltas.tex').read_text())
deltas=[]
for row in records:
 ref=next(x for x in records if x['Model']==row['Model'] and x['Method']==METHODS['bf16'])
 d={'Model':row['Model'],'Method':row['Method']}
 for key in list(row)[3:]:d[key+' delta']=row[key]-ref[key] if row[key] is not None and ref[key] is not None else None
 deltas.append(d)
csvwrite('metric_deltas.csv',deltas);latex('metric_deltas.tex',deltas)
generation_extra=[]
for result in sorted(RUN.glob('*/*/*/results.json')):
 task=result.parent.name
 if task not in ['gsm8k_cot','ifeval']:continue
 model=result.parts[-4];method=result.parts[-3]
 data=json.loads(result.read_text())['results'][task]
 refpath=RUN/model/'bf16'/task/'results.json'
 reference=json.loads(refpath.read_text())['results'][task] if refpath.exists() else {}
 for metric,value in data.items():
  if ',' in metric and 'stderr' not in metric and isinstance(value,(int,float)):
   generation_extra.append(dict(Model=NAMES[model],Method=METHODS[method],Task=task,Metric=metric,Score=100*value,Delta_vs_BF16_pp=100*(value-reference[metric]) if metric in reference else None,Source=str(result)))
if generation_extra:
 csvwrite('generation_all_metrics.csv',generation_extra)
 labels={'GSM8K strict':'exact_match,strict-match','GSM8K flexible':'exact_match,flexible-extract','IFEval prompt strict':'prompt_level_strict_acc,none','IFEval prompt loose':'prompt_level_loose_acc,none','IFEval instruction strict':'inst_level_strict_acc,none','IFEval instruction loose':'inst_level_loose_acc,none'}
 gs=[];gd=[]
 for row in records:
  score={k:row[k] for k in ['Model','Method']};delta=dict(score)
  for label,metric in labels.items():
   hit=next((x for x in generation_extra if x['Model']==row['Model'] and x['Method']==row['Method'] and x['Metric']==metric),None)
   score[label]=hit['Score'] if hit else None;delta[label]=hit['Delta_vs_BF16_pp'] if hit else None
  gs.append(score);gd.append(delta)
 latex('generation_scores.tex',gs);latex('generation_deltas.tex',gd)
(RUN/'incomplete.json').write_text(json.dumps(missing,indent=2)+'\n')
status=dict(completed=sum(d['Score'] is not None for d in details),requested=len(details),rows=records)
(RUN/'summary_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
