# 生成 padding 适配修复

首轮 Llama 四个方法的 GSM8K / IFEval 均在首个生成批次触发 CUDA embedding 索引越界，未产生可用生成批次或能力分数。各 run-1.log、launch-1.json 和 failure.json 保留。

原因：项目原有 `LlamaTokenizerFast` 加载器为保持历史无 BOS 的 PPL 协议而沿用；该类附加默认 `<unk>` token，ID=128256，超出模型 embedding 的有效范围0–128255。harness 0.4.8 的 `configure_pad_token` 优先将 unk 用作 padding。之前单条短生成验证未包含 padding，未暴露这一适配问题。

修复限于新评测入口：在交给 HFLM 前将 Llama 的 `pad_token_id` 显式设为已有 EOS 128009。不扩展 embedding，不改权重、尺度、聊天模板、实际输入文本、评分或生成预算。四种 Llama 方法统一使用同一修复。Qwen 不受此问题影响。

已完成 zero-shot / MMLU 不重跑：harness causal loglikelihood 用 `pad_and_concat` 的零填充，并仅评分实际 continuation，路径不使用 tokenizer.pad_token_id。独立验证检查该源码路径，并新增两种 prompt 长度的有 padding 生成验证。

仅恢复失败的生成任务；调度器自动跳过已有结果。正式重试保留独立 run-2.log / launch-2.json 和生成批次文件。最终表格只读取成功结果，原失败记录用于追溯，不把失败当作模型零分。

## Qwen IFEval 批调度

根据全体541条输入的token长度，仅作显存/批量安排：最长输入373，batch64含1280生成token的最坏KV估算11.30GiB。Qwen三方法统一选batch64。BF16的原batch16进程在安排时已自动开始，终止时共保留48条部分原始生成；没有完成的成绩被覆盖或择优。仅这项重新调度，首批部分文件保留在generation_batches-001.jsonl，正式成功尝试以settings.json的generation_log字段为准。该安排没有修改提示、模型、尺度、生成长度、采样或评分，也不依据成绩选择批量。

## Qwen 父包 IFEval 显存分配恢复

GPU7的batch64首批在decode期间OOM，未产生完成批次：PyTorch allocated 15.12GiB，reserved-but-unallocated 7.05GiB，另有他人既有588MiB进程。按实际错误的碎片证据启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 并重试同一任务；不缩短生成、不改batch64、不触碰其他进程。分配器只影响内存安排，模型、量化、提示和评分协议不变。原failure/run/launch完整保留。
