# FIRON 模型能力补充评测

本轮固定评测已有模型，不训练、不校准、不选择 checkpoint。默认 seed42 来自既定主实验；本轮没有根据下游分数选 seed。模型包与本地原始模型路径见 `models.json`。Llama 与 Qwen 各评测 BF16、最终 FIRON QAT400、精确蒸馏前父包。额外 Llama 历史 Phase2-C GPTQ 权重以 A16 执行，配方不同，不当作最终 FIRON 的单变量消融。Qwen 无已定位 W4A16 包，记待补。

## 协议

- lm-evaluation-harness **0.4.8** 官方任务配置；代码保存在本目录 `deps/lm_eval/`，原样任务配置同时进入各 `results.json`。
- BoolQ validation 3270、PIQA validation 1838、SIQA validation 1954、HellaSwag validation 10042、WinoGrande XL validation 1267、ARC-Easy test 2376、ARC-Challenge test 1172、OpenBookQA test 500。全部 zero-shot、原始 completion 提示，不加聊天模板。
- 八项分数：BoolQ/SIQA/WinoGrande 使用 `acc`；PIQA/HellaSwag/ARC-Easy/ARC-Challenge/OpenBookQA 使用 `acc_norm`（官方选项长度归一化）。同时保留官方输出全部指标。八项平均为上述八个百分比分数的算术平均，缺一项不计算均值。
- MMLU：官方 `mmlu`，57 科目，各取 dev 前5条示例、test 评分，原始 completion 提示，官方按样本数加权的 accuracy。逐科目结果保留。
- GSM8K：官方 `gsm8k_cot`，8 条官方固定 CoT 示例，test 1319 条，同一次生成同时评分 `exact_match,strict-match` 与 `exact_match,flexible-extract`。主表GSM8K单元格统一并列 strict / flexible，以显式区分答案表达格式和数值答案；原定strict指标保留在 main_results_numeric.csv 和主指标差值表，两个指标及相对BF16变化均在 generation_all_metrics.csv。没有改变提示、生成或按分数择优选提取器。官方默认最多256个新 token、greedy，stop 序列 `Q:`、`</s>`、`<|im_end|>`。
- IFEval：官方 `ifeval`（v4.0 配置），541 条，zero-shot、greedy、最多1280个新 token；主指标 `prompt_level_strict_acc`，保留 loose/instruction-level 指标。
- 两个生成任务使用各模型随包提供的聊天模板，fewshot 示例拼成单条 user 消息（`fewshot_as_multiturn=False`）；Qwen `enable_thinking=False`，没有采样。上下文上限8192，GSM8K及Llama IFEval使用batch16，Qwen IFEval使用batch64（最长prompt373 tokens、含生成上限的KV估算11.30GiB），评分batch4，同家族所有方法一致。所有随机种子42。Llama padding 显式使用已有 EOS 128009；首轮生成适配失败及修复见 RECOVERY.md。
- BF16 与量化使用同一模型架构实现、tokenizer、评分/生成参数。评分无 cache；生成启用 BF16 KV cache。FIRON 的 W4 按包内整数码与尺度恢复，静态 INT8/SP2 输入量化在所有 forward 包括 decode 生效；不随任务更新尺度。embedding/head/norm 保留高精度。这是 GPU fake-quant 质量证据。
- 每任务保存原始逐样本 `samples_*.jsonl`、`results.json`、`settings.json`、`quantization.json`。quantization 记录实际 prefill/decode 调用以及完整量化器状态前后相等检查。独立验证见 `verifier/REPORT.md`。

## PPL 复用

`ppl_reuse.json` 从 Phase5 summary 逐路径对应模型包，再读取每个原始 JSON 核对精确 PPL。两家族 BF16/FIRON/父包共12项已有结果直接复用。
WikiText-2 **完整 test**：不添加 BOS/EOS，双换行连接，2048-token 不重叠窗口含尾窗，按预测 target 数加权 NLL。
C4 为已有 `allenai/c4/en/validation` **固定1024窗口子集**，不是全量 C4；每模型2096128个预测 targets。沿用模型各自既有 tokenizer 与固定 token artifact，不能按跨 tokenizer 的绝对 PPL 排名。各 token artifact 身份和原始证据路径在 `ppl_reuse.json`。
历史8窗口或validation PPL不填入主表test列。

## 执行与复用

使用 `runs/phase5/env/bin/python`，新增依赖仅装在本目录 `deps`，未更改原环境。`environment.txt` 记录软件版本；`hardware.txt` 记录RTX 4090 24GiB和驱动570.181，每项GPU/PID在launch记录中。主源码为 `worktrees/SpinQuant-multimodel`，有效未提交实现的快照与 diff 在 `source/`。

```bash
# 单任务（CUDA_VISIBLE_DEVICES 由实际资源安排）
runs/phase5/env/bin/python scripts/capability_eval/run.py \
  --model qwen --method firon --task ifeval \
  --output runs/capability-eval-20260920/qwen/firon/ifeval
# 汇总：已有完成结果不复跑，未完成项保持空白
runs/phase5/env/bin/python scripts/capability_eval/summarize.py
```

调度器 `scripts/capability_eval/schedule.py` 按模型/方法独立进程执行；每项启动记录包含 PID/GPU/命令。局部失败保留 failure.json，继续其他任务。`main_results.csv/.tex` 为主表；`appendix_zero_shot.csv/.tex` 为八项明细及相对 BF16 的百分点变化；`metric_deltas.csv/.tex` 为全部主指标变化；`appendix_details.csv` 保留逐任务来源/状态。PPL delta 为绝对差、越低越好；accuracy delta 为百分点、越高越好。

所有已约定任务都保留在表中；空白/`--` 表示未完成，原因见 `incomplete.json`。不把未完成项记为0，不以已有任务的均值代替八项平均。

官方来源：https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.8/lm_eval/tasks

原始模型版本记录：Qwen/Qwen3-1.7B 的既有 revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，原记录在 `source/qwen/phase5_revision.json`。Llama 本地既有 ModelScope 记录只有 `Revision:master,CreatedAt:1740591574`（`source/llama/modelscope_revision.txt`），不能将其冒称为不可变的上游 commit；本轮所有方法均引用 models.json 中同一本地模型目录，config/tokenizer/generation_config 快照在 source/llama。复用命令和依赖位置详见 `scripts/capability_eval/README.md`。
