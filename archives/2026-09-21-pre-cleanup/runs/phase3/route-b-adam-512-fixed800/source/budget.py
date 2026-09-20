"""Reuse observed training windows and compare completed training budgets."""
import json
from pathlib import Path


def reference_window_indices(reference, available_windows, probe_indices):
    rows = [json.loads(line) for line in (Path(reference) / "training.jsonl").read_text().splitlines()]
    indices = list(dict.fromkeys(index for row in rows for index in row["window_indices"]))
    if not indices or any(type(index) is not int or not 0 <= index < available_windows for index in indices):
        raise ValueError("Reference training windows are empty or outside the gradient-training pool")
    if set(indices).intersection(probe_indices):
        raise ValueError("Reference gradient windows overlap the fixed probe")
    return indices


def write_comparison(output, reference, checkpoints, schedule_steps, accumulation):
    output, reference = Path(output), Path(reference)
    old_settings = json.loads((reference / "settings.json").read_text())["arguments"]
    old_results = json.loads((reference / "results.json").read_text())
    old = next(item for item in old_results if item["step"] == old_settings["steps"])
    old_horizon = old_settings.get("schedule_steps") or old_settings["steps"]
    rows = [dict(run=str(reference), schedule_steps=old_horizon, position="old_schedule_final", **old)]
    rows.extend(dict(run=str(output), schedule_steps=schedule_steps,
                     position="new_schedule_final" if item["step"] == schedule_steps else "new_schedule_intermediate",
                     cumulative_train_tokens=item["step"] * accumulation * 2048, **item)
                for item in checkpoints)
    record = dict(comparison="Old completed schedule versus new schedule checkpoints; learning-rate trajectories differ",
                  evaluation="Fixed held-out train-split probe; initial SP2 export; full WikiText validation only",
                  rows=rows)
    temporary = output / "comparison.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "comparison.json")
    lines = ["# B 联合优化步数比较", "",
             "旧 100-step 终态与新 512-step 日程中间点分别标识；不是相同学习率轨迹。",
             "固定探针使用训练配置 down-A16；完整 validation 使用初始 down-SP2 导出包。",
             "没有后处理、QAT、test 或 C4 选择。缺失的 PPL 表示尚未完成该评测。", "",
             "| 运行位置 | step | cosine 总步数 | 固定探针 NLL | 完整 validation NLL | 完整 validation PPL |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        values = [f"{row[key]:.10f}" if key in row else "未评测"
                  for key in ("training_probe", "validation_nll", "validation_ppl")]
        lines.append(f"| {row['position']} | {row['step']} | {row['schedule_steps']} | " + " | ".join(values) + " |")
    temporary = output / "COMPARISON.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output / "COMPARISON.md")
