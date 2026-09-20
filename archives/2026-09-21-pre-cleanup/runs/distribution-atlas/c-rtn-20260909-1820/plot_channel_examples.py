"""Actual within-channel histograms, with explicit selection and normalized axes."""
from pathlib import Path
import json
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/dongpeiyan/projects/rotation-quant')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'worktrees/SpinQuant-distribution-experiment'))
from experiments.distribution_atlas import collect as atlas

def main():
    stats=torch.load(OUT/'bf16/weight_channel_summary.pt',map_location='cpu',weights_only=True)['sources']
    selection=[]
    for family in sorted({s['projection_family'] for s in stats}):
        candidates=[s for s in stats if s['projection_family']==family]
        s=max(candidates,key=lambda s:float(s['tail_score_t'].max()))
        row=int(s['tail_score_t'].argmax())
        selection.append(dict(name=s['source_name'],row=row,T=float(s['tail_score_t'][row]),selection='maximum T in family'))
    s=next(s for s in stats if s['source_name']=='model.layers.1.mlp.down_proj')
    row=int((s['tail_score_t']-s['tail_score_t'].median()).abs().argmin())
    selection.append(dict(name=s['source_name'],row=row,T=float(s['tail_score_t'][row]),selection='median T in layer1 down'))
    model=atlas._load_rotated_model(ROOT/'cache/models/llama-3.2-1b-instruct',ROOT/'runs/phase2/learned-sw-c-20260909.ByFYAM/C/rotation/R.bin')
    sources=dict(atlas._weight_sources(model))
    fig,axes=plt.subplots(4,2,figsize=(15,16))
    saved=[]
    for ax,selection_row in zip(axes.flat,selection):
        x=sources[selection_row['name']].weight[selection_row['row']].detach().float().cpu()
        z=(x-x.mean())/x.std(unbiased=False)
        ax.hist(z.numpy(),bins=80,log=True,color='#4477aa')
        ax.set_title(f"{selection_row['name']} | channel {selection_row['row']}\n{selection_row['selection']}; T={selection_row['T']:.3f}",fontsize=10)
        ax.set_xlabel('(weight - row mean) / row std')
        ax.set_ylabel('Count (log scale)')
        saved.append(dict(**selection_row,weights=x))
    fig.suptitle('C-R BF16: within-output-channel distributions',fontsize=17)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(OUT/'channel_examples.png',dpi=160)
    torch.save(saved,OUT/'channel_examples.pt')
    (OUT/'channel_examples.json').write_text(json.dumps(selection,indent=2))
    print('CHANNEL EXAMPLES COMPLETE')

if __name__=='__main__':
    main()
