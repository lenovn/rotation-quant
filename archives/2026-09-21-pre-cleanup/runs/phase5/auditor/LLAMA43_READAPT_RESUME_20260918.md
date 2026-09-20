# Llama seed43 再适配恢复限定审计（2026-09-18）

结论：PASS。已保存前缀被正确恢复，正式搜索从索引4（第5层）继续；这不是最终 PTQ 包或外部指标验收。

- 旧目录：`/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-down-readapt-s43`；新目录：`/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-down-readapt-s43-r4`。
- 新 launcher：PID 3869795，GPU0，正式旧环境 `rotation-quant-p0`，`PHASE5_SEED=43`。父包 `/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-range-s43/static_w4a8.pt`，参考 `/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/joint100-s43/checkpoint-0100/state.pt` 均与旧任务一致。参数仅 output、resume_prefix 不同；六份相关保存源码逐字节一致。
- 新旧 `data.json` 完全相同：seed43、32 个训练完整窗，前24窗拟合/后8窗选择，49152/16384行，选择16376 targets；没有 validation 校准或重新选择数据。
- 旧 `prefix.pt` 包含 layers0–3 的完整连续前缀、16层 down SP2 尺度，尺度逐张量等于精确父包冷加载值。初始和当前选择 NLL 均为 **2.7589552842354874**。四层全部 selected_step=0，候选均 `[0,512,2048,8192]` 且0为最小训练 NLL，因此 weights={} 合理，恢复后这四层沿用父权重。
- `restore_prefix` 核对 parent/reference/mode/targets 及连续前缀，再恢复权重、尺度和 NLL；主循环 `targets[len(completed):]` 跳过0–3层。首个新日志确为 layer4 fit，首个坐标日志为128；所有已读新日志模块编号≥4。新 layer4 的 previous_train_nll 精确接上旧前缀，按原四候选完成并选择0；没有重新生成 initial_train_selection.json，符合源码跳过初始评测的分支。
- 审计快照中，新 prefix 已保留旧0–3并完成至 `model.layers.9.mlp.down_proj`；最新 progress 为 `sequential-round-coordinate` / `model.layers.10.mlp.down_proj` / step `7936`。这是进行中的快照，不代表全16层完成。旧四层 JSON 仍保留在原目录；新目录及最终 new_module_results 只记录恢复后新层，完整逐层审计应合并两目录，不能把旧四层缺少新副本误判为漏跑。

预算说明：每层仍有父候选0及512/2048/8192三个上限候选，恢复不扩大有效搜索空间。已提交0–3层不重跑；旧 layer4 的 coordinate 状态未写入 prefix，新任务从该层起点重做，而非从384继续。旧日志最后为 layer4 step384，故额外消耗至少包括已记录384次坐标迭代和该层捕获/拟合；最后日志后的确切计算量未知。不能把物理总计算量声称为一次无中断运行的精确预算，也不能把此阶段计为梯度训练更新。

中断原因未定：旧目录无 failure.json 或最终包；宿主进程/tmux消失由主执行确认，kernel journal 不可读。沙箱 ps 不具备宿主可见性，空结果不是退出证据。本审核采用恢复日志和保存状态，不推断 OOM/外部终止等原因，未进行 GPU、forward、tokenization 或 PPL 复跑；也未做无中断轨迹逐位重放。

源位置：`experiments/phase3/sequential_postprocess.py:142–168`（保存/恢复），`:201–225`（跳过完成层与结果记录）。机器证据：[LLAMA43_READAPT_RESUME_20260918.json](LLAMA43_READAPT_RESUME_20260918.json)。
