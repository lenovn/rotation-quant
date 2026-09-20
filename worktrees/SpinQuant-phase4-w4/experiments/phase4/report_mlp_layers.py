"""Summarize the fixed-parent, full-validation MLP restoration measurements."""
import argparse
import csv
import json
from pathlib import Path


def main(output):
    project = Path(__file__).resolve().parents[4]
    root = project / 'runs/phase4'
    all_cases = {}
    sources = {}
    for family in ('mlp', 'up', 'gate', 'down'):
        path = root / f'diag-layer-{family}-20260915a/results.json'
        values = json.loads(path.read_text())
        all_cases.update(values)
        sources.update({key: str(path) for key in values})
    old_path = root / 'diag-mlp-local-interaction-20260915a/results.json'
    old = json.loads(old_path.read_text())
    all_cases['all16:down@1'] = old['all16:down1']
    sources['all16:down@1'] = str(old_path)
    activation = json.loads((root / 'diag-activation-20260915a/results.json').read_text())
    base = activation['all16:none']['evaluation']
    reference = activation['all16:all']['evaluation']
    rows = []
    for layer in range(16):
        row = dict(layer=layer, ordinal_layer=layer+1)
        for family in ('mlp', 'up', 'gate', 'down'):
            evaluation = all_cases[f'all16:{family}@{layer}']['evaluation']
            if evaluation['predicted_tokens'] != 252728 or evaluation['token_count'] != 252852:
                raise ValueError('Mismatched validation protocol')
            row[family+'_nll'] = evaluation['nll']
            row[family+'_ppl'] = evaluation['ppl']
            row[family+'_delta_nll'] = base['nll'] - evaluation['nll']
        row['joint_minus_sum_delta_nll'] = row['mlp_delta_nll'] - sum(row[f+'_delta_nll'] for f in ('up','gate','down'))
        rows.append(row)
    ranking = {family: sorted(range(16), key=lambda i: rows[i][family+'_delta_nll'], reverse=True)
               for family in ('mlp', 'up', 'gate', 'down')}
    globals_ = {}
    previous = json.loads((root / 'diag-families-20260915a/results.json').read_text())
    for family in ('up', 'gate', 'down', 'gateup', 'mlp'):
        entry = (all_cases['all16:'+family] if family in ('up','gate') else
                 old['all16:mlp'] if family == 'mlp' else previous['all16:'+family])
        globals_[family] = dict(**entry['evaluation'], delta_nll=base['nll']-entry['evaluation']['nll'])
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(baseline=base, reference=reference, rows=rows, rankings=ranking,
                   global_restorations=globals_, sources=sources,
                   interpretation='Positive delta=NLL(parent W4/allA16)-NLL(restored). Independent conditional effects, not additive attribution. Zero-indexed layer; full validation viewed, no training.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (output/'layers.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# 逐层 MLP / up / gate / down 条件恢复', '',
        '原父包 W4/all-A16 PPL=15.557034492007398。数值为 ΔNLL=原父NLL−恢复后NLL，正值越大表示恢复收益越大。每个单元独立从原父包开始；同R未量化BF16权重仅恢复指定模块。层号0..15。', '',
        '| layer | 整体MLP ΔNLL | up ΔNLL | gate ΔNLL | down ΔNLL |',
        '|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+str(row['layer'])+' | '+' | '.join(f"{row[f+'_delta_nll']:+.6f}" for f in ('mlp','up','gate','down'))+' |')
    lines += ['', '## 全家族恢复（跨16层）', '', '| 家族 | PPL | ΔNLL |', '|---|---:|---:|']
    for family, value in globals_.items(): lines.append(f"| {family} | {value['ppl']:.9f} | {value['delta_nll']:+.9f} |")
    lines += ['', '完整MLP与三个投影单独恢复效果不相加；summary.json保留joint_minus_sum_delta_nll描述非加性，不能当作独立因果贡献。接近零或近邻的排名未经随机性检验，不将微小差值当稳健结论。layer1 down复用此前完全匹配结果，其余65项为本轮新完整测量。所有结果均252852输入、252728预测targets，成熟token加权NLL协议；无新权重优化/导出。', '',
              '![Conditional NLL recovery](layers.png)']
    (output/'TABLE.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    values = np.array([[row[f+'_delta_nll'] for f in ('mlp','up','gate','down')] for row in rows])
    vmax = max(abs(values.min()),abs(values.max()))
    fig, ax = plt.subplots(figsize=(7,9))
    plot=ax.imshow(values,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_xticks(range(4),['Whole MLP','up','gate','down'])
    ax.set_yticks(range(16),range(16)); ax.set_ylabel('Layer index (zero-based)')
    ax.set_title('W4 / all-A16: conditional NLL recovery\nSame-R BF16 restoration; positive = improvement')
    for i in range(16):
        for j in range(4):
            ax.text(j,i,f'{values[i,j]:+.4f}',ha='center',va='center',fontsize=9,
                    color='white' if abs(values[i,j])>.65*vmax else 'black')
    fig.colorbar(plot,ax=ax,label='Parent NLL minus restored NLL',shrink=.7)
    fig.tight_layout(); fig.savefig(output/'layers.png',dpi=170); fig.savefig(output/'layers.svg'); plt.close(fig)
    print(json.dumps(dict(output=str(output),rankings=ranking,global_restorations=globals_),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
