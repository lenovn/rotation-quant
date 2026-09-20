"""Render completed Phase5 core results and paired seed statistics from summary.csv."""
import csv
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
MODELS = ('meta-llama/Llama-3.2-1B-Instruct', 'Qwen/Qwen3-1.7B')
METHODS = ('BF16', 'SP2-PTQ', 'Uniform-PTQ', 'SP2-QAT400', 'Uniform-QAT400', 'Initial-SP2-QAT400')
SPLITS = (('Salesforce/wikitext', 'validation'), ('Salesforce/wikitext', 'test'), ('allenai/c4', 'validation'))
SEEDS = ('42', '43', '44')


def main():
    with (ROOT / 'summary.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['completion'] == 'COMPLETED']
    output = ['# Phase5 完整方法与配对结果', '',
              '由 summary.csv 生成；空缺表示尚无已完成结果。PPL 仅为显示保留六位，原始精度与证据见 summary.csv。',
              'BF16 不按 seed 重复；B100 格式对照和中间后处理不进入本表。QAT 默认 WikiText-2 train only。', '']

    def lookup(model, method, seed, split):
        matches = [row for row in rows if row['model'] == model
                   and row['stage'].lower() == method.lower()
                   and (method == 'BF16' or row['seed'] == seed)
                   and (row['dataset'], row['split']) == split]
        return matches[0] if len(matches) == 1 else None, len(matches)

    for model in MODELS:
        output += [f'## {model}', '', '| 方法 | seed | WikiText val PPL | WikiText test PPL | 固定 C4 PPL | test/C4 包路径 |',
                   '|---|---|---:|---:|---:|---|']
        for method in METHODS:
            if method == 'Initial-SP2-QAT400' and model != MODELS[0]:
                continue
            for seed in (('',) if method == 'BF16' else SEEDS if method in ('SP2-QAT400', 'Uniform-QAT400') else ('42',)):
                selected = [lookup(model, method, seed, split) for split in SPLITS]
                cells = [f"[{float(row['ppl']):.6f}]({row['evidence_path']})" if row else
                         ('多条记录，待核对' if count else '未完成') for row, count in selected]
                test, c4 = selected[1][0], selected[2][0]
                package = '待两项完成'
                if test and c4:
                    package = '相同' if test['evaluation_package'] == c4['evaluation_package'] else '不同，待核对'
                output.append('| ' + ' | '.join([method, seed or '—', *cells, package]) + ' |')
        output.append('')

    output += ['## 相对本模型原始 BF16 的指标', '',
               'ΔNLL = 当前NLL − 同模型同数据BF16 NLL；PPL比值 = 当前PPL / BF16 PPL。不同tokenizer的绝对PPL不直接排名。', '',
               '| 模型 | 方法 | seed | 数据 | PPL | NLL | ΔNLL | PPL比值 | targets |',
               '|---|---|---|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        for method in METHODS:
            if method == 'Initial-SP2-QAT400' and model != MODELS[0]:
                continue
            for seed in (('',) if method == 'BF16' else SEEDS if method in ('SP2-QAT400', 'Uniform-QAT400') else ('42',)):
                for split in SPLITS:
                    row = lookup(model, method, seed, split)[0]
                    if row is None:
                        continue
                    original = lookup(model, 'BF16', '', split)[0]
                    delta, ratio = '待核对', '待核对'
                    if original and original['targets'] == row['targets']:
                        delta = f"{float(row['nll']) - float(original['nll']):.9f}"
                        ratio = f"{float(row['ppl']) / float(original['ppl']):.6f}"
                    label = 'C4 fixed' if split[0] == 'allenai/c4' else 'WikiText ' + split[1]
                    output.append('| ' + ' | '.join([model, method, seed or '—', label,
                        f"{float(row['ppl']):.6f}", f"{float(row['nll']):.9f}",
                        delta, ratio, row['targets']]) + ' |')
    output.append('')
    output += ['## 逐 seed 配对 NLL', '',
               '| 模型 | 数据 | seed | SP2 NLL | Uniform NLL | SP2 − Uniform |',
               '|---|---|---|---:|---:|---:|']
    for model in MODELS:
        for split in SPLITS:
            for seed in SEEDS:
                a = lookup(model, 'SP2-QAT400', seed, split)[0]
                b = lookup(model, 'Uniform-QAT400', seed, split)[0]
                if a is None and b is None:
                    continue
                delta = f"{float(a['nll']) - float(b['nll']):.9f}" if a and b and a['targets'] == b['targets'] else '未齐或待核对'
                label = 'C4 fixed' if split[0] == 'allenai/c4' else 'WikiText ' + split[1]
                output.append('| ' + ' | '.join([model, label, seed,
                    f"{float(a['nll']):.9f}" if a else '未完成',
                    f"{float(b['nll']):.9f}" if b else '未完成', delta]) + ' |')
    output += ['', '## 三 seed 统计', '',
               '仅当 seed42/43/44 全部完成且每项数据的 targets 一致时计算。标准差使用样本标准差（ddof=1）。',
               '配对 ΔNLL = SP2-QAT400 − Uniform-QAT400，负值有利于 SP2；不会使用最佳 seed 替代统计。', '',
               '| 模型 | 数据 | SP2 PPL 均值±标准差 | Uniform PPL 均值±标准差 | 配对 ΔNLL 均值±标准差 | 状态 |',
               '|---|---|---:|---:|---:|---|']
    for model in MODELS:
        for split in SPLITS:
            paired = [(lookup(model, 'SP2-QAT400', seed, split)[0],
                       lookup(model, 'Uniform-QAT400', seed, split)[0]) for seed in SEEDS]
            count = sum(a is not None and b is not None for a, b in paired)
            metrics = ['—'] * 3
            state = f'完整配对 {count}/3，未汇总'
            if count == 3:
                targets = {row['targets'] for pair in paired for row in pair}
                if len(targets) == 1:
                    values = ([float(a['ppl']) for a, _ in paired],
                              [float(b['ppl']) for _, b in paired],
                              [float(a['nll']) - float(b['nll']) for a, b in paired])
                    metrics = [f'{statistics.mean(v):.6f} ± {statistics.stdev(v):.6f}' for v in values]
                    state = '完成；逐 seed 见上表及原始记录'
                else:
                    state = 'targets 不一致，未汇总'
            label = 'C4 fixed' if split[0] == 'allenai/c4' else 'WikiText ' + split[1]
            output.append('| ' + ' | '.join([model, label, *metrics, state]) + ' |')
    output += ['', '本汇总只检查记录字段；架构正确性、数据一致性和同包验收证据仍以 verifier/、auditor/ 及原始 run 为准。', '']
    target = ROOT / 'CORE_RESULTS.md'
    target.write_text('\n'.join(output))
    print(target)


if __name__ == '__main__':
    main()
