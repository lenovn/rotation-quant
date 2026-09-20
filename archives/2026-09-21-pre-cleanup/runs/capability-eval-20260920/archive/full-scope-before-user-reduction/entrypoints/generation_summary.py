"""Inspect successful raw generations, excluding batch padding and EOS/stop endings."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/capability-eval-20260920'
rows=[]
for settings in sorted(RUN.glob('*/*/*/settings.json')):
 j=json.loads(settings.read_text()); path=j.get('generation_log')
 if not path or not (settings.parent/'results.json').exists() or not Path(path).exists():continue
 samples=[json.loads(x) for x in Path(path).read_text().splitlines()]
 truncated=0;ended_eos=0;ended_stop=0
 for x in samples:
  eos=x.get('eos_token_ids') or [];eos=[eos] if isinstance(eos,int) else eos
  has_eos=any(t in eos for t in x['generated_token_ids'])
  has_stop=any(s and s in x['raw_text'] for s in x['stop'])
  ended_eos+=has_eos;ended_stop+=has_stop
  truncated+=not has_eos and not has_stop and len(x['generated_token_ids'])>=x['max_new_tokens']
 rows.append(dict(model=settings.parts[-4],method=settings.parts[-3],task=settings.parts[-2],
  samples=len(samples),budget_exhausted_no_eos_or_stop=truncated,
  budget_exhausted_percent=100*truncated/len(samples),eos_present=ended_eos,stop_present=ended_stop,
  raw_source=path))
if rows:
 with (RUN/'generation_lengths.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
