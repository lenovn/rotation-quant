"""C-R BF16 versus saved C RTN, with per-row INT4/SP2 reconstruction screening."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/home/dongpeiyan/projects/rotation-quant')
OUT = Path(__file__).resolve().parent
C = ROOT / 'runs/phase2/learned-sw-c-20260909.ByFYAM/C'
sys.path.insert(0, str(ROOT / 'worktrees/SpinQuant-distribution-experiment'))
from experiments.distribution_atlas import collect as atlas
from experiments.distribution_atlas import collect_weight_channels as channels

# Eq. (8), sign + 2 + 1 bits. Repeated sums are a single numerical value.
LEVELS = sorted({a+b for a in [0., .5, .25, .125] for b in [0., .5]})

def project(w, alpha, kind):
    if kind == 'int4':
        return ((w / (alpha / 7)).round().clamp(-8, 7) * (alpha / 7)).bfloat16().float()
    levels = torch.tensor(LEVELS, device=w.device)
    indices = torch.bucketize((w.abs() / alpha).contiguous(), (levels[1:]+levels[:-1])/2)
    return (w.sign() * levels[indices] * alpha).bfloat16().float()

def fit(w, kind):
    maximum = w.abs().amax(1, keepdim=True).clamp_min(1e-30)
    ratios = torch.logspace(-3, 1, 33, base=2, device=w.device)
    best = torch.full((w.shape[0],), float('inf'), device=w.device)
    alpha = maximum[:, 0].clone()
    best_index = torch.zeros(w.shape[0], dtype=torch.long, device=w.device)
    for i, ratio in enumerate(ratios):
        a = maximum * ratio
        error = (project(w, a, kind)-w).square().mean(1)
        take = error < best
        best = torch.minimum(best, error)
        alpha = torch.where(take, a[:, 0], alpha)
        best_index = torch.where(take, i, best_index)
    low = ratios[(best_index-1).clamp_min(0)].log()
    high = ratios[(best_index+1).clamp_max(32)].log()
    for fraction in torch.linspace(0, 1, 17, device=w.device):
        a = maximum * (low+(high-low)*fraction).exp()[:, None]
        error = (project(w, a, kind)-w).square().mean(1)
        take = error < best
        best = torch.minimum(best, error)
        alpha = torch.where(take, a[:, 0], alpha)
    return best.cpu(), alpha.cpu(), ((best_index == 0) | (best_index == 32)).cpu()

def save_channels(items, directory):
    directory.mkdir(exist_ok=True)
    torch.save(dict(granularity='weight_output_channel', thresholds_defined=False,
                    total_sources=len(items), total_channels=sum(x['num_channels'] for x in items),
                    sources=items), directory/'weight_channel_summary.pt')
    (directory/'weight_channel_score_summary.json').write_text(json.dumps(channels._score_summary_payload(items), indent=2))

def main():
    model = atlas._load_rotated_model(ROOT/'cache/models/llama-3.2-1b-instruct', C/'rotation/R.bin')
    checkpoint = torch.load(C/'rtn/w4_rtn_model.pt', map_location='cpu', weights_only=False)['model']
    bf_stats, rtn_stats, results, rows = [], [], [], []
    with torch.inference_mode():
        for name, module in atlas._weight_sources(model):
            w = module.weight
            q_cpu = checkpoint[name+'.module.weight']
            assert torch.equal(q_cpu, checkpoint[name+'.weight'])
            q = q_cpu.to(w.device)
            bf_stats.append(channels._source_summary(name, w))
            rtn_stats.append(channels._source_summary(name, q))
            detail = dict(source_name=name, elements_per_channel=w.shape[1])
            for key in ['energy', 'kurtosis', 'tail_energy_top1pct', 'rtn_mse', 'int4_mse', 'sp2_mse',
                        'int4_alpha', 'sp2_alpha', 'int4_edge', 'sp2_edge']:
                detail[key] = []
            for start in range(0, w.shape[0], 256):
                x = w[start:start+256].float()
                centered = x-x.mean(1, keepdim=True)
                energy = x.square().mean(1)
                detail['energy'].append(energy.cpu())
                detail['kurtosis'].append((centered.pow(4).mean(1)/centered.square().mean(1).square().clamp_min(1e-30)).cpu())
                k = max(1, round(x.shape[1]*.01))
                detail['tail_energy_top1pct'].append((x.square().topk(k, dim=1).values.sum(1)/x.square().sum(1).clamp_min(1e-30)).cpu())
                detail['rtn_mse'].append((q[start:start+256].float()-x).square().mean(1).cpu())
                for kind in ['int4', 'sp2']:
                    mse, alpha, edge = fit(x, kind)
                    for suffix, value in [('mse', mse), ('alpha', alpha), ('edge', edge)]:
                        detail[kind+'_'+suffix].append(value)
            for key, value in list(detail.items()):
                if isinstance(value, list):
                    detail[key] = torch.cat(value)
            results.append(detail)
            s = bf_stats[-1]
            row = dict(name=name, family=name.rsplit('.',1)[-1], channels=w.shape[0], elements=w.numel(),
                       tail_t_median=s['tail_score_t'].median().item(), tail_t_max=s['tail_score_t'].max().item(),
                       channels_t_gt3=int((s['tail_score_t']>3).sum()), channels_t_gt5=int((s['tail_score_t']>5).sum()),
                       kurtosis_median=detail['kurtosis'].median().item(), kurtosis_max=detail['kurtosis'].max().item(),
                       top1_energy_median=detail['tail_energy_top1pct'].median().item(),
                       sp2_win_fraction=(detail['sp2_mse']<detail['int4_mse']).float().mean().item())
            for kind in ['rtn', 'int4', 'sp2']:
                row[kind+'_nmse']=(detail[kind+'_mse'].sum()/detail['energy'].sum()).item()
                row[kind+'_sse']=detail[kind+'_mse'].sum().item()*w.shape[1]
            row['energy']=detail['energy'].sum().item()*w.shape[1]
            for kind in ['int4','sp2']:
                row[kind+'_edge_channels']=int(detail[kind+'_edge'].sum())
            rows.append(row)
            print(json.dumps(row), flush=True)
    save_channels(bf_stats, OUT/'bf16')
    save_channels(rtn_stats, OUT/'rtn')
    torch.save(results, OUT/'mapping_channels.pt')
    (OUT/'mapping_summary.json').write_text(json.dumps(dict(
        rotation=str(C/'rotation/R.bin'), checkpoint=str(C/'rtn/w4_rtn_model.pt'),
        sp2_magnitudes=LEVELS, sp2_bits='sign/2/1; 13 unique signed values',
        protocol='Per-output-row weight MSE; FP32 projection then BF16; alpha/max 2^-3..2^1, 33 log coarse +17 local refinement; no activation weighting or training',
        rows=rows), indent=2))
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    main()
