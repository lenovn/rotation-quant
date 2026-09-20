import json,math,csv
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];BASE=RUN/'qwen/w4a16'
registry=json.loads((RUN/'models.json').read_text())['qwen']['w4a16'];package=registry['package']
tasks={'boolq':('acc',3270),'piqa':('acc_norm',1838),'social_iqa':('acc',1954),'hellaswag':('acc_norm',10042),'winogrande':('acc',1267),'arc_easy':('acc_norm',2376),'arc_challenge':('acc_norm',1172),'openbookqa':('acc_norm',500),'mmlu':('acc',14042)}
expected={f'model.layers.{i}.{s}' for i in range(28) for s in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj')}
report={'ppl':[],'tasks':[],'pending':[]}
reuse=json.loads((RUN/'ppl_reuse.json').read_text())
for kind,dataset in [('test','Salesforce/wikitext'),('c4','allenai/c4')]:
 d=json.loads((BASE/f'ppl-{kind}/result.json').read_text());refrow=next(x for x in reuse if x['model_key']=='qwen' and x['method']=='bf16' and x['dataset']==dataset);ref=json.loads(Path(refrow['evidence_path']).read_text())
 checks=dict(same_input_metadata=d['input_metadata']==ref['input_metadata'],same_token_counts=all(d[k]==ref[k] for k in ['token_count','predicted_tokens','unscored_tail_tokens']),new_gptq_package=d['package']==package,
  original_model=d['model_path']==registry['model_path'],no_training_or_calibration=d['training']==False and d['calibration']==False,
  weighted_nll=math.isclose(sum(s['nll']*s['predicted_tokens'] for s in d['segments'])/d['predicted_tokens'],d['nll'],abs_tol=1e-12),ppl_from_nll=math.isclose(math.exp(d['nll']),d['ppl'],abs_tol=1e-12))
 report['ppl'].append(dict(dataset=dataset,ppl=d['ppl'],checks=checks,reference_token_path=ref['input_token_path']))
for task,(metric,n) in tasks.items():
 base=BASE/task
 if not (base/'results.json').exists():report['pending'].append(task);continue
 d=json.loads((base/'results.json').read_text());cfg=json.loads((base/'settings.json').read_text());q=json.loads((base/'quantization.json').read_text());loading=json.loads((base/'w4a16_loading.json').read_text());files=list(base.glob('samples_*.jsonl'))
 checks=dict(package=cfg['selected']['package']==q['weight_source']==loading['package']==package,shots=cfg['num_fewshot']==(5 if task=='mmlu' else 0),no_chat=cfg['generation_chat_template']==False,
  no_a8=q['backbone_linears']==0 and q['quantizer_calls']=={},w4_forward_coverage=set(q['fixed_w4_forward_calls'])==expected and all(v>0 for v in q['fixed_w4_forward_calls'].values()),no_scale_change=q['scales_unchanged'])
 total=0;correct=0
 for f in files:
  rows=[json.loads(s) for s in f.open()];ref=[json.loads(s) for s in (RUN/'qwen/bf16'/task/f.name).open()];subject=f.stem[len('samples_'):]
  identity=lambda x:[(s['doc_id'],s['arguments'],s['target']) for s in x]
  checks['prompt_match_'+subject]=identity(rows)==identity(ref)
  checks['count_'+subject]=len(rows)==d['n-samples'][subject]['original']==d['n-samples'][subject]['effective'] and len({x['doc_id'] for x in rows})==len(rows)
  hits=sum(x[metric] for x in rows);total+=len(rows);correct+=hits
  checks['score_'+subject]=math.isclose(hits/len(rows),d['results'][subject][metric+',none'],abs_tol=1e-12)
 checks['total_count']=total==n;checks['file_count']=len(files)==(57 if task=='mmlu' else 1)
 checks['aggregate']=math.isclose(correct/total,d['results'][task][metric+',none'],abs_tol=1e-12)
 report['tasks'].append(dict(task=task,score=100*correct/total,checks=checks))
report['pass_completed']=all(all(x['checks'].values()) for x in report['ppl']+report['tasks']);report['all_complete']=not report['pending']
(RUN/'verifier/qwen_gptq_results_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(ppl=len(report['ppl']),tasks=len(report['tasks']),pending=report['pending'],passed=report['pass_completed'],failures=[(x.get('task',x.get('dataset')),[k for k,v in x['checks'].items() if not v]) for x in report['ppl']+report['tasks'] if not all(x['checks'].values())]),indent=2))
