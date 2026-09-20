# Llama Uniform-QAT400 seed43 最终证据审计（2026-09-18）

结论：PASS。完整 Uniform 方法固定400更新完成；同一最终包的 validation、完整 WikiText-2 test 和固定 C4 均完成，原始分段聚合与summary精度一致。本审核不扩展至其他未完成seed/方法。

| split | PPL | NLL | targets |
|---|---:|---:|---:|
| validation | 18.143936807329858 | 2.898336444698109 | 252728 |
| test | 17.575531211766275 | 2.866507662660506 | 288934 |
| C4 | 39.07188227779196 | 3.66540308496798 | 2096128 |

唯一最终包：`/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-s43/checkpoint-0400/static_w4a8.pt`。父包：`/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-down-readapt-s43/static_w4a8.pt`；参考：`/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/joint100-s43/checkpoint-0100/state.pt`。前置round/range/readapt链引用 [已通过的合并审计](LLAMA_SEEDS43_44_POSTPROCESS_20260918.md)，不把B100格式覆盖当作完整Uniform训练。

实际training.jsonl严格连续1–400，无resume，所有3200微批逐条确认8×2048输入/2047targets、按(800+(step−1)×8+i)%1180循环；有效输入6553600、预测6550400 token visits。target_ppl=None，仅400完整validation，result/progress完成且无failure。正式旧环境Transformers4.44.2；与seed42相同优化/teacher/master规则，五份关键保存源码逐字节相同。初始化记录精确引用本seed参考投影到父INT4 cell，原生未旋转BF16 teacher冻结。没有读取完整optimizer来重复证明更新步数。

CPU mmap直接检查最终包：112 W4、112真实INT8激活记录，16个down全部INT8；父包至最终112个W4矩阵、112个SW和112个SA均实际变化，74个高精度张量逐位不变。日志legacy SP2梯度组16只是down尺度组名，不代表SP2格式。

test/C4 的result与model均指向同一checkpoint-0400包，实际112 INT8、KV16、use_cache=false、RoPE FP32，无overlay，无calibration/training/candidate_selection。两次评测所有112 quantizer的每个保存状态张量before/after位级相同，test和C4初始状态也相同，尺度精确等于包中尺度；均_has_scale=true、_observing=false、sample_count=0。

test输入ID和完整metadata同时精确匹配历史BF16与完整SP2-QAT400：289076输入、288934targets、141个完整窗+308-token尾窗。C4输入ID和metadata同样精确匹配两历史参考：2097152输入、2096128targets、1024完整窗。全部segment起点/长度/窗数/targets与对应历史协议一致，0未计尾token；逐段log(PPL)及token-weighted聚合精确匹配。本次未重新tokenize。

validation为252852输入、252728targets，123完整窗+948尾窗。其artifact未另存input IDs或before/after量化快照；本次核对metadata、计数、分段、checkpoint来源及已审冷加载源码，不能把外部test/C4的逐张量冻结证据扩写为validation也有相同快照。

summary三行官方模型名/seed43/Uniform-QAT400/INT8、精确父包与评测包、PPL/NLL及tokens一致。C4的split=validation指C4源数据划分，dataset区分其与WikiText validation，未误合并。完整路径和机器结果：[LLAMA_UNIFORM_QAT400_S43_20260918.json](LLAMA_UNIFORM_QAT400_S43_20260918.json)。无GPU或PPL复跑，无应用、summary及实验产物修改。
