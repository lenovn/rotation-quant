# FIRON 四指标评测入口

当前用户约定范围：WikiText-2、C4-subset、八项zero-shot平均、MMLU。
模型为Qwen BF16/FIRON/父包/原始BF16→GPTQ对照，以及Llama BF16/FIRON/父包/历史W4A16。
Qwen W4A16由build_qwen_gptq.py从原始BF16进行一次GPTQ量化，训练集128×2048、seed42、per-output-channel、groupsize=-1；无旋转学习或蒸馏，评测中不校准。它与Llama历史GPTQ对照的配方不同，标签分别保留。不调度GSM8K或IFEval。

本次仅新增Qwen W4A16四指标评测，其余结果复用。工作目录为
`/home/dongpeiyan/projects/rotation-quant`，结果目录为
`runs/capability-eval-20260920`。协议与模型路径见该目录的
`PROTOCOL.md`、`models.json`；每任务`settings.json`记录实际运行配置。

## 只重新生成表格

```bash
runs/phase5/env/bin/python scripts/capability_eval/summarize.py
runs/phase5/env/bin/python scripts/capability_eval/report.py
```

主表为`main_results.csv/.tex`，包含四项指标；八项明细和差值位于
`appendix_zero_shot.csv/.tex`，主指标差值为`metric_deltas.csv/.tex`。
`paper_tables.tex`包含主表和附录caption，需要LaTeX的booktabs、graphicx，
从结果目录引用。`incomplete.json`记录当前尚未完成项目。

## 调度入口

以下仅为复用示例；先按项目约定确认可用GPU。完成的结果自动跳过。

```bash
runs/phase5/env/bin/python scripts/capability_eval/schedule.py \
  --jobs qwen:w4a16:0 --tasks mmlu
```

省略`--tasks`时，只调度八项zero-shot和MMLU。单任务入口为`run.py`；
当前评分batch4、MMLU 5-shot、zero-shot 0-shot，不加聊天模板。
WikiText-2/C4-subset优先复用`ppl_reuse.json`中的匹配原结果；已有Llama
W4A16的两项PPL位于`llama/w4a16/ppl-*`。

使用原`runs/phase5/env/bin/python`；harness 0.4.8新增依赖隔离在结果目录
`deps/`，数据缓存位于`cache/huggingface`，完整环境见`environment.txt`。
调度器设置数据缓存环境变量，模型包从`models.json`加载，不训练、不校准。

原完整范围的表格、报告和脚本快照保存在
`archive/full-scope-before-user-reduction/`，各任务原始结果继续保留。
旧生成代码保留供追溯，当前调度列表不包含生成任务。

若将同一W4A16的PPL与选择题分配给不同调度器，另一个调度器使用`--skip-ppl`避免重复PPL。单独PPL入口：`ppl_w4a16.py --model qwen`。

## 原始 Qwen 的 GPTQ 构建

正式包位于`qwen/gptq_w4a16_package/w4_gptq_model.pt`，已完成后直接复用。
构建入口为`build_qwen_gptq.py`；它拒绝覆盖已有完成包，固定从原始BF16
模型和WikiText-2 train开始，运行项目GPTQ核心，不加载FIRON包。
`settings.json`、`calibration.json`分别记录配方与128个训练窗口起点，
`result.json`记录196个量化Linear及高精度参数保持检查；源码位于包目录的
`source/`。构建和评测均为GPU fake-quant质量实验，不代表原生INT4内核速度。

先前FIRON/A16消融结果在`qwen/firon_a16_ablation/`，不参与当前主表。
