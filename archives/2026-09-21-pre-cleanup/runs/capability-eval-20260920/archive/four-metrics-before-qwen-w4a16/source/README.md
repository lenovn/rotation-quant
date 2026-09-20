# FIRON 四指标评测入口

当前用户约定范围：WikiText-2、C4-subset、八项zero-shot平均、MMLU。
模型为Qwen BF16/FIRON/父包，以及Llama BF16/FIRON/父包/历史W4A16。
Qwen W4A16不纳入，也不列待补；不再调度GSM8K或IFEval。

全部当前结果已完成，本次范围调整没有启动GPU任务。工作目录为
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
从结果目录引用。当前`incomplete.json`为空。

## 调度入口

以下仅为复用示例；先按项目约定确认可用GPU。完成的结果自动跳过。

```bash
runs/phase5/env/bin/python scripts/capability_eval/schedule.py \
  --jobs qwen:firon:0 --tasks mmlu
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
