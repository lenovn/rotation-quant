# C4 外部评测：原始 BF16 与固定 C RTN W4A8/down-SP2

测量日期：2026-09-12。两组均为本次新测，使用同一份保存的 C4 token 输入。当前量化配置相对 BF16 增加 **7.69613783 PPL（35.233245%）**，平均 NLL 增加 **0.301830843 nat/token**。这是固定模型在外部英文语料上的整体量化损失，未作 W/A 或单层归因。

| 配置 | PPL | 平均 NLL（nat/token） |
| --- | ---: | ---: |
| 原始 BF16 W16A16 | 21.8433976245 | 3.083898707666 |
| C RTN W4 + 非 down INT8 + down SP2-A8 | 29.5395354575 | 3.385729551101 |

原始结果：[summary.json](summary.json)、[BF16 result](w16a16/result.json)、[W4A8 result](w4a8/result.json)。独立验证：[verification.md](verification.md)。

## 数据与预测目标

- 来源：[allenai/c4](https://huggingface.co/datasets/allenai/c4)，英文 `en`，官方 `validation` 的全部 8 个 JSON 分片。C4 官方没有单独的 `test` split；本轮将此前未用于本项目量化训练、校准或选型的 C4 validation 子集作为外部测试。
- 加载 364,608 篇源文档。用 Python `Random(42)` 打乱全部文档行索引，按该顺序拼接 4,480 篇文本，以两个换行分隔，得到 2,150,947 tokens。
- 实际保留前 **2,097,152 tokens**；丢弃拼接文本后部 53,795 tokens。因此不声称 4,480 篇文档都被完整评测。
- 1024 个互不重叠的 2048-token 窗口，无短尾窗。每窗首 token 不计 next-token 损失，共 **2,096,128 个预测目标**，窗口间重置上下文。
- 同一 LlamaTokenizerFast、本地 Llama-3.2-1B-Instruct tokenizer，关闭自动 BOS/EOS。先对实际拼接文本分词，再切窗口。
- 两组都读取 [data/input_tokens.pt](data/input_tokens.pt)。原文、随机行索引和 URL 保存在 [data/documents.jsonl](data/documents.jsonl)，规则与源文件列表见 [data/metadata.json](data/metadata.json)。

## 固定模型与数值口径

- 原始 BF16：未融合归一化、未旋转、未量化；按既有加载器把共享 embedding 权重复制到 lm_head。
- 当前量化配置加载原有 [C RTN W4 权重](../learned-sw-c-20260909.ByFYAM/C/rtn/w4_rtn_model.pt)、配套 [R.bin](../learned-sw-c-20260909.ByFYAM/C/rotation/R.bin)、[非 down SA/SW](../learned-sw-c-20260909.ByFYAM/C/rotation/quant_scales.pt) 和此前 train 校准的 [down-SP2 尺度](../down-codebooks-c-20260909.6YRIty/results/down_scales.json)。
- 112 个 backbone Linear 为 symmetric W4 per-output-channel，实际权重与保存 checkpoint 逐元素相等；96 个非 down 输入使用固定 INT8 per-tensor 尺度，16 个 down 输入使用固定 SP2-A8 per-tensor 尺度。所有激活量化器 buffer 在评测前后相等。
- R1/R2，关闭 R3/R4；KV16；embedding、lm_head、norm 保持高精度。整个过程没有校准、尺度搜索、优化器更新或重新量化权重。
- BF16 运算承载 fake quantization；teacher-forced next-token，batch 1、PREFILL、`use_cache=False`，关闭 TF32。
- 为控制显存，把 1024 个窗口分成 8 段，每段 128 个窗口，复用原 evaluator。每段 PPL 取对数得到 NLL，再按预测 token 数加权，最后取指数；包含原 evaluator float32 PPL 的微小指数/对数往返舍入。
- 原 evaluator/ptq 日志中的 `WikiText2 PPL`、`wiki2 ppl` 是旧的固定文案；实际数据由上述 C4 token 文件提供，结果 JSON 的 dataset、split、input_token_path 已明确记录。

## 分段结果

每段有 262,016 个预测目标。8 段的量化 NLL 均高于 BF16；下表用于检查结果分布，不将分段当作新的独立模型选择。

| 段 | BF16 PPL | W4A8/SP2 PPL | NLL 增量 |
| --- | ---: | ---: | ---: |
| 1 | 21.72407913 | 29.16539764 | 0.294561712 |
| 2 | 22.06345558 | 29.64955711 | 0.295524542 |
| 3 | 22.67020607 | 30.67150116 | 0.302282368 |
| 4 | 20.32092667 | 27.44443893 | 0.300512333 |
| 5 | 23.02649879 | 31.07238197 | 0.299673711 |
| 6 | 20.58588219 | 27.86351585 | 0.302712647 |
| 7 | 22.03007698 | 29.29116058 | 0.284877129 |
| 8 | 22.47678757 | 31.40557480 | 0.334502306 |

## 验证与复现

- 独立 CPU 验证覆盖旧完整/尾窗口径、新分块无遗漏或重复、预测 token 加权、保存 token 入口、数据准备与禁止校准路径。具体命令及结果见 [verification.md](verification.md)。
- 独立核对源分片、抽样顺序、保存原文和重新分词后的输入，以及两模型结果汇总。
- 本次入口：[44_run_c4_acceptance_c_local.sh](../../../scripts/phase2/44_run_c4_acceptance_c_local.sh)。源码和执行命令快照：[source/](source/)。
- 数据准备耗时约 299.4 秒（含下载与自动续传）；BF16 evaluator 163.10 秒，W4A8/SP2 evaluator 209.75 秒。二者是本次 GPU 3 上的评测墙钟时间，不能作为推理后端延迟对照。

## 结论范围

本次结果说明当前固定量化配置在这份 C4 英文子集上存在上述整体精度损失；没有测试 INT8/PoT 的 down 替代配置，不能据此判断 SP2 在 C4 上优于它们，也不能把全部掉点归因于 SP2。

本轮 C4 只用于评测，未根据其结果调整参数或选择格式。这里的外部评测是相对于本地量化开发流程；没有审计基础模型预训练语料，也没有执行 C4 与 WikiText 的逐段文本重叠审计，不能声称绝对无语料重叠。结果仅覆盖本子集及 prefill fake-quant 精度，不代表 C4 全集、生成任务、decode/KV 量化或手机原生后端表现。
