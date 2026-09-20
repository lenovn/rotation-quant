# Qwen seed42 第400步冷包完整test独立复核

**独立 PASS。** 按PLAN只执行一次完整WikiText-2 test；没有重跑validation、C4或其他seed，没有重新校准、训练、选择候选、调整参数或修改应用源码。

| 指标 | 主执行 | 独立冷进程 |
|---|---:|---:|
| PPL | 14.151216160997155 | 14.151216160997155 |
| NLL | 2.649800568169599 | 2.649800568169599 |
| 输入tokens | 299078 | 299078 |
| 预测targets | 298931 | 298931 |

两次各自读取完整4358行`wikitext-test.arrow`、双换行拼接并使用Qwen tokenizer重新编码，无BOS/EOS或chat template；输入IDs逐元素一致。146个2048-token整窗加70-token尾窗，共147窗。目标计数为146×2047＋69＝298931；尾窗69个预测目标已计入，未评分尾token为0。三个评测分段（128整窗、18整窗、尾窗）的PPL/NLL也完全一致。独立重算target加权NLL符合最终值。

同一固定包：`runs/phase5/qwen3-1p7b/sp2-qat400-s42/checkpoint-0400/static_w4a8.pt`。直接读取包元数据确认step=400，parent=`sp2-down-readapt-s42/static_w4a8.pt`，原生Qwen teacher、T=1、CE权重0.1。config为qwen3/28层、显式tie_word_embeddings=false；包保留56个Q/K Norm高精度张量。

实际冷加载与证据核对：196个packed W4矩阵；168个静态INT8输入＋28个静态SP2输入。独立评测的196个量化器前后state_dict全部一致，初始scale与包中参数相符；没有down INT8 overlay、动态回退或在线Hadamard。RoPE inv_freq保留FP32，KV16、use_cache=False。量化边界和状态来自实际模型检查、`model.json`与`quantizers_before/after.pt`，不是仅靠命令声称。

使用相同已接受的评测实现：BF16 logits CE，逐token loss转FP32，分段float32 PPL取log后按targets加权。主/独立run保存的architecture/common/postprocess/quantization/external_eval/validation_acceptance源码快照相同。本次属于独立冷进程完整复现与状态核查，不是第二套指标实现的交叉验证；也不验证原生kernel或decode。

启动与资源：GPU2启动前20MiB/0%；tmux socket=`rotation-quant-phase5`，session=`phase5-qwen3-1p7b-sp2-qat400-test-independent-s42`。实际Python PID=3000641，已由ps核实；launcher使用exec，pane与Python PID一致。完成耗时45.57376677508秒，评测峰值分配显存5.543839454650879GiB。进程正常退出，GPU2恢复20MiB/0%。未使用GPU4或干预他人任务。

证据：

- 独立结果：`runs/phase5/qwen3-1p7b/sp2-qat400-test-independent-s42/result.json`。
- 独立启动：同目录外`sp2-qat400-test-independent-s42.launch.json`及`.log`。
- 主执行对照：`runs/phase5/qwen3-1p7b/sp2-qat400-test-s42/result.json`。
- 独立逐项核对：`runs/phase5/verifier/QWEN_SEED42_FINAL_TEST_20260918.json`。

PLAN规定的首个最终seed42冷包独立完整test复核已完成；没有依据此结果切换包或追加保险评测。
