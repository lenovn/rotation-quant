import csv,json,math,sys
from pathlib import Path
from collections import Counter
RUN=Path(__file__).resolve().parents[1]
NAMES={'qwen':'Qwen3-1.7B','llama':'Llama-3.2-1B-Instruct'}
METHODS={'bf16':'Original BF16','firon':'FIRON (QAT400, seed42)','parent':'FIRON pre-distillation (seed42)','w4a16':'Historical Phase2 GPTQ'}
models=json.loads((RUN/'models.json').read_text())
sys.path.insert(0,str(RUN.parents[1]/'worktrees/SpinQuant-multimodel'))
from experiments.phase3.architecture import tokenizer
paper=list(csv.DictReader((RUN/'main_results.csv').open()));numeric=list(csv.DictReader((RUN/'main_results_numeric.csv').open()))
extra=list(csv.DictReader((RUN/'generation_all_metrics.csv').open()))
report={'completed':[],'pending':[]}
for model in NAMES:
 tok=tokenizer(models[model]['bf16']['model_path'])
 if model=='llama':tok.pad_token_id=tok.eos_token_id
 for task,count,shots in [('gsm8k_cot',1319,8),('ifeval',541,0)]:
  refpath=RUN/model/'bf16'/task/f'samples_{task}.jsonl'
  refrows=[json.loads(s) for s in refpath.open()] if refpath.exists() else []
  def prompts(rows):return [(r['doc_id'],r['filter'],r['arguments'],r['target']) for r in rows]
  for method in models[model]:
   base=RUN/model/method/task
   if not (base/'results.json').exists():report['pending'].append(f'{model}/{method}/{task}');continue
   d=json.loads((base/'results.json').read_text());settings=json.loads((base/'settings.json').read_text());rows=[json.loads(s) for s in (base/f'samples_{task}.jsonl').open()]
   logs=[json.loads(s) for s in Path(settings['generation_log']).open()]
   decoded=[]
   for x in logs:
    ids=x['prompt_token_ids'];start=0
    while start<len(ids) and ids[start]==tok.pad_token_id:start+=1
    decoded.append(tuple(ids[start:]))
   q=json.loads((base/'quantization.json').read_text());expected=(196 if model=='qwen' else 112) if method in ('firon','parent') else 0
   filters=['strict-match','flexible-extract'] if task=='gsm8k_cot' else ['none']
   checks=dict(complete_rows=len(rows)==count*len(filters),complete_logs=len(logs)==count,matched_prompts=prompts(rows)==prompts(refrows),
    correct_shots=d['n-shot'][task]==shots and settings['num_fewshot']==shots,
    nonthinking_settings=settings['enable_thinking']==False,single_user_settings=settings['fewshot_as_multiturn']==False,
    quantizer_coverage=len(q['quantizer_calls'])==q['backbone_linears']==expected and all(c.get('prefill',0)>0 and c.get('decode',0)>0 for c in q['quantizer_calls'].values()),scales_unchanged=q['scales_unchanged'],
    generation_batch=settings['effective_batch_size']==(64 if model=='qwen' and task=='ifeval' else 16),
    greedy=all(r['arguments'][0][1]['do_sample']==False for r in rows))
   checks['selected_log_matches_prompt_set']=Counter(decoded)==Counter(tuple(tok.encode(r['arguments'][0][0],add_special_tokens=False)) for r in rows if r['filter']==filters[0])
   if model=='qwen':checks['actual_nonthinking_suffix']=all(r['arguments'][0][0].endswith('<think>\n\n</think>\n\n') for r in rows)
   if task=='gsm8k_cot':checks['actual_eight_demos']=all(r['arguments'][0][0].count('The answer is')==8 and r['arguments'][0][0].count('\nA:')==9 for r in rows)
   values={}
   for filt in filters:
    group=[r for r in rows if r['filter']==filt]
    checks['unique_'+filt]=len(group)==count and len({r['doc_id'] for r in group})==count
    for metric in group[0]['metrics']:
     items=[r[metric] for r in group];flat=[v for x in items for v in x] if isinstance(items[0],list) else items
     value=sum(flat)/len(flat);key=metric+','+filt;values[key]=100*value
     checks['aggregate_'+key]=math.isclose(value,d['results'][task][key],abs_tol=1e-12)
     cell=next(x for x in extra if x['Model']==NAMES[model] and x['Method']==METHODS[method] and x['Task']==task and x['Metric']==key)
     checks['extra_'+key]=math.isclose(float(cell['Score']),100*value,abs_tol=1e-10)
     reference=json.loads((RUN/model/'bf16'/task/'results.json').read_text())['results'][task][key]
     checks['delta_'+key]=math.isclose(float(cell['Delta_vs_BF16_pp']),100*(value-reference),abs_tol=1e-10)
   p=next(x for x in paper if x['Model']==NAMES[model] and x['Method']==METHODS[method]);n=next(x for x in numeric if x['Model']==NAMES[model] and x['Method']==METHODS[method])
   if task=='gsm8k_cot':
    strict=values['exact_match,strict-match'];flex=values['exact_match,flexible-extract']
    checks['paper_both_filters']=p['GSM8K']==f'{strict:.2f} / {flex:.2f}'
    checks['numeric_primary']=math.isclose(float(n['GSM8K']),strict,abs_tol=1e-10)
   else:checks['paper_prompt_strict']=math.isclose(float(p['IFEval']),values['prompt_level_strict_acc,none'],abs_tol=1e-10)
   exhausted=sum(not any(t in (x['eos_token_ids'] if isinstance(x['eos_token_ids'],list) else [x['eos_token_ids']]) for t in x['generated_token_ids']) and not any(s and s in x['raw_text'] for s in x['stop']) and len(x['generated_token_ids'])>=x['max_new_tokens'] for x in logs)
   report['completed'].append(dict(model=model,method=method,task=task,checks=checks,values=values,budget_exhausted=exhausted,generation_log=settings['generation_log']))
report['completed_pass']=all(all(r['checks'].values()) for r in report['completed']);report['all_complete']=not report['pending']
(RUN/'verifier/generation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(completed=len(report['completed']),pending=report['pending'],completed_pass=report['completed_pass'],failed=[(x['model'],x['method'],x['task'],[k for k,v in x['checks'].items() if not v]) for x in report['completed'] if not all(x['checks'].values())]),indent=2))
