import csv,json,math
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
TASKS={'boolq':('acc',3270),'piqa':('acc_norm',1838),'social_iqa':('acc',1954),'hellaswag':('acc_norm',10042),'winogrande':('acc',1267),'arc_easy':('acc_norm',2376),'arc_challenge':('acc_norm',1172),'openbookqa':('acc_norm',500)}
NAMES={'qwen':'Qwen3-1.7B','llama':'Llama-3.2-1B-Instruct'}
METHODS={'bf16':'Original BF16','firon':'FIRON (QAT400, seed42)','parent':'FIRON pre-distillation (seed42)','w4a16':'Historical Phase2 GPTQ'}
registry=json.loads((RUN/'models.json').read_text())
main=list(csv.DictReader((RUN/'main_results.csv').open()))
appendix=list(csv.DictReader((RUN/'appendix_zero_shot.csv').open()))
report={'checks':[],'averages':[],'missing':[]}
scores={}
for model in NAMES:
 for task,(metric,count) in TASKS.items():
  ref=None
  for method in registry[model]:
   base=RUN/model/method/task
   if not (base/'results.json').exists():
    report['missing'].append(f'{model}/{method}/{task}');continue
   raw=json.loads((base/'results.json').read_text())
   samples=[json.loads(s) for s in (base/f'samples_{task}.jsonl').open()]
   prompt=[(s['doc_id'],s['arguments'],s['target']) for s in samples]
   if method=='bf16':ref=prompt
   accuracy=sum(s[metric] for s in samples)/len(samples)
   score=raw['results'][task][metric+',none']*100
   scores[model,method,task]=score
   q=json.loads((base/'quantization.json').read_text())
   expected=(196 if model=='qwen' else 112) if method in ('firon','parent') else 0
   checks=dict(full_sample_count=len(samples)==count,unique_doc_ids=len({s['doc_id'] for s in samples})==count,
      matched_actual_prompts=prompt==ref,raw_accuracy=math.isclose(accuracy*100,score,abs_tol=1e-10),
      zero_shot=raw['n-shot'][task]==0,scales_unchanged=q['scales_unchanged'],
      quantizer_coverage=len(q['quantizer_calls'])==expected and q['backbone_linears']==expected and all(c.get('prefill',0)>0 for c in q['quantizer_calls'].values()))
   row=next(r for r in appendix if r['Model']==NAMES[model] and r['Method']==METHODS[method])
   checks['appendix_score']=math.isclose(float(row[task]),score,abs_tol=1e-10)
   checks['appendix_delta']=math.isclose(float(row[task+' delta (pp)']),score-scores[model,'bf16',task],abs_tol=1e-10)
   report['checks'].append(dict(model=model,method=method,task=task,score=score,checks=checks))
 for method in registry[model]:
  vals=[scores.get((model,method,t)) for t in TASKS]
  row=next(r for r in main if r['Model']==NAMES[model] and r['Method']==METHODS[method])
  mean=sum(vals)/8 if all(v is not None for v in vals) else None
  report['averages'].append(dict(model=model,method=method,recomputed=mean,main=row['Zero-shot Avg. (8)'],pass_check=math.isclose(float(row['Zero-shot Avg. (8)']),mean,abs_tol=1e-10) if mean is not None and row['Zero-shot Avg. (8)'] else mean is None and row['Zero-shot Avg. (8)']==''))
report['pass']=not report['missing'] and all(all(r['checks'].values()) for r in report['checks']) and all(r['pass_check'] for r in report['averages'])
(RUN/'verifier/zero_shot_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
