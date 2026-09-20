import csv,json,math
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
NAMES={'qwen':'Qwen3-1.7B','llama':'Llama-3.2-1B-Instruct'}
METHODS={'bf16':'Original BF16','firon':'FIRON (QAT400, seed42)','parent':'FIRON pre-distillation (seed42)','w4a16':'Historical Phase2 GPTQ'}
models=json.loads((RUN/'models.json').read_text())
table=list(csv.DictReader((RUN/'main_results.csv').open()))
report={'subjects':[],'aggregates':[]}
for model in NAMES:
 ref={}
 for method in models[model]:
  base=RUN/model/method/'mmlu'; result=json.loads((base/'results.json').read_text())
  files=sorted(base.glob('samples_mmlu_*.jsonl')); correct=0; total=0
  for file in files:
   task=file.stem[len('samples_'):]; rows=[json.loads(s) for s in file.open()]
   actual=[(s['doc_id'],s['arguments'],s['target']) for s in rows]
   if method=='bf16':ref[task]=actual
   n=len(rows);hits=sum(s['acc'] for s in rows);correct+=hits;total+=n
   checks=dict(full_count=n==result['n-samples'][task]['original']==result['n-samples'][task]['effective'],unique_ids=len({s['doc_id'] for s in rows})==n,
      matched_actual_prompts=actual==ref[task],five_shot=result['n-shot'][task]==5 and result['configs'][task]['num_fewshot']==5,
      six_answer_markers=all(s['arguments'][0][0].count('\nAnswer:')==6 for s in rows),
      sample_accuracy=math.isclose(hits/n,result['results'][task]['acc,none'],abs_tol=1e-12))
   report['subjects'].append(dict(model=model,method=method,task=task,n=n,checks=checks))
  q=json.loads((base/'quantization.json').read_text());expected=(196 if model=='qwen' else 112) if method in ['firon','parent'] else 0
  main=next(r for r in table if r['Model']==NAMES[model] and r['Method']==METHODS[method]);accuracy=correct/total
  checks=dict(subject_count=len(files)==57,total_count=total==14042,weighted_result=math.isclose(accuracy,result['results']['mmlu']['acc,none'],abs_tol=1e-12),
     main_table=math.isclose(accuracy*100,float(main['MMLU']),abs_tol=1e-10),scales_unchanged=q['scales_unchanged'],
     quantizer_coverage=q['backbone_linears']==len(q['quantizer_calls'])==expected and all(c.get('prefill',0)>0 for c in q['quantizer_calls'].values()))
  report['aggregates'].append(dict(model=model,method=method,n=total,accuracy_percent=accuracy*100,checks=checks))
report['pass']=all(all(x['checks'].values()) for x in report['subjects']+report['aggregates'])
(RUN/'verifier/mmlu_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='subjects'},indent=2))
print('subject_failure_count',sum(not all(x['checks'].values()) for x in report['subjects']))
