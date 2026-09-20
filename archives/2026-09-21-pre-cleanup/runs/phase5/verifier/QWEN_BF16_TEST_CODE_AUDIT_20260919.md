# Qwen BF16 test 与 Phase5 改进结果：独立代码审查

日期：2026-09-19。角色：独立 verifier。范围：本次重新逐行审查保存源码、实际参数、原始指标和训练日志；未以旧 PASS 替代代码核查，未运行 GPU、训练、PPL、额外测试 suite，未重审量化包张量。只新增本报告。

## 当前结论

**未发现导致 BF16 test 16.715764 与 SP2-QAT400 test 14.151216（seed42）差异的具体计分、因果注意力或开发/测试数据串用错误。** 这是有界代码审查结论，不是声称所有可能的错误已排除；主执行另外运行完全绕过项目 helpers 的官方 AutoModel 全 test 复核，本报告的代码结论不预先代替其运行结果。

**不能将下降全部解释为 QAT，更不能称为“仅降低位宽便优于原模型”。** SP2-PTQ 在 QAT 之前已经 test 14.302163，且该包已经历 train CE Joint100、embedding centering、静态校准和按 train NLL 选择的后处理。没有同预算 BF16 适应对照，没有单独隔离 centering/Joint/后处理的 Qwen test 因果消融；现有证据只能说明整体方法在该 WikiText 协议的结果，不能确证各步骤贡献。

## 1. BF16 是否原生、是否偷偷改变了模型

以 `runs/phase5/qwen3-1p7b/bf16-test-s42/source/` 保存源码为准：

- `architecture.py:10–21`：Qwen 路径通过官方 `AutoModelForCausalLM.from_pretrained` 加载本地模型，`torch_dtype=bfloat16`、`attn_implementation=sdpa`、`local_files_only=True`。默认 `training=False, untie=False`，不安装旋转线性层，不解绑 embedding/head。
- `external_eval.py:68–77`：`mode=bf16` 调用上述默认加载；不走 `load_static`。只设冻结、eval、`use_cache=False`、seqlen2048、移至 CUDA。`128–132` 检查 BF16 无量化 wrappers、RoPE inv_freq 仍 FP32。
- `common.py:20–28` 的 import 会引用融合/量化函数，但 import 不调用 `build_training_model`。实际融合只在 `common.py:194–200` 的训练建模路径调用。
- 实际 `settings.json`：mode bf16、package null、overlay null；source 明确为 Phase5 worktree、新环境 Transformers4.51.3、模型 `cache/models/qwen3-1.7b`、revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。`model.json` 的 activation_formats 为空、training/calibration 均 false。
- 本地 `config.json` 为 qwen3、28 层、Q16/KV8、head_dim128、rms_norm_eps1e-6、rope_theta1e6、rope_scaling null、tie_word_embeddings true。

因此，就实际调用链而言，该 BF16 是官方模型与原始缓存权重的 BF16 推理，不包含项目的 centering/旋转/量化。此判断不等同于本次重新对全部 safetensors 做远端身份核验；本次没有网络下载或全模型张量审计。

## 2. 移位、mask、RoPE 与全 test 覆盖

- 保存的 `architecture.py:75–89` 每窗使用 `model.model(batch,use_cache=False,return_dict=True)[0]` 后接 `model.lm_head`。`logits[:,:-1]` 对 `batch[:,1:]`，严格 token t 预测 t+1，无再次 shift 或输入本 token 当标签的错误。
- 安装环境的 `transformers/models/qwen3/modeling_qwen3.py:849–866` 官方 CausalLM forward 也是调用同一 model 再接同一 lm_head，默认 `logits_to_keep=0` 保留所有 logits。项目未绕过 decoder 的 Q/K norm、RoPE 或 causal attention。
- 同文件 `529–553`：无 cache 时 position 从0开始；每个独立窗均重新生成 position/RoPE。`627–647` 允许 SDPA 无显式 mask；`transformers/integrations/sdpa_attention.py:46–61` 此时 `is_causal=query_length>1`，因此缺省 mask 不代表双向可看未来。所有计分窗长度至少2，无 padding，也无 cache 延续。
- `external_eval.py:46–64` 只加载 `wikitext-test.arrow`，双换行连接4358行、同一 tokenizer、无BOS/EOS。`validation_acceptance.py:14–49` 分块不改变每窗内容，尾窗长度70计69 targets；按目标 token 加权，不平均各段 PPL。
- 本次 CPU 直接读取3份 `input_tokens.pt` 的 `input_ids`，BF16、SP2-PTQ、SP2-QAT400 seed42 **逐值完全相同**，均 `[1,299078]` int64。146个2048窗+70尾窗，总 targets298931。这里只读取输入ID，未加载量化权重。
- 对比三次保存源码：`validation_acceptance.py` 完全相同；`architecture.py` 唯一变化是后来为 Llama 指定 tokenizer class，Qwen仍走 AutoTokenizer；`external_eval.py` 唯一变化是打印标签。Qwen计分函数相同。

独立窗意味着每个窗的首 token 不计分；这是三者相同且公开的协议，不是 sliding-window/full-context 所有 token 评测。不能直接把不同上下文/BOS/数据连接方式的外部 PPL 当作相同口径。

## 3. 确认存在的精度口径差异

保存的 `architecture.py:85–87` 在 BF16 logits 上执行 `cross_entropy(reduction='none')`，之后 per-token loss 才转 FP32 聚合。acceptance 再由 FP32 segment PPL 取 log 并按 targets 加权。原结果明确记录此口径。

官方 `transformers/loss/loss_utils.py:41–63` 的 `ForCausalLMLoss` 先 `logits.float()`，然后 shift labels、算 CE。因此官方 `model(...,labels=...)` 的 loss 与既有主表 **并非数值上相同的 CE 口径**。这是应明确的精度限制；BF16、PTQ、QAT 的原主表均使用前述同一 BF16-CE 路径，不存在只给 BF16 或只给量化模型切换 CE dtype 的分支。

原生复核应同时报告历史兼容 BF16 CE 与 FP32 CE 诊断，且不覆写原始结果。单凭代码不能量化两种 CE 的差值，也不能预先断言其能解释/不能解释全部差距。

## 4. 实际训练与后处理数据边界

以各次保存 source、settings、data 和完整轻量 `training.jsonl` 为准：

- Joint `common.py:163–191` 分别读取 train 与 validation。train逐行无BOS/EOS分词后连接，截成1229个2048窗，丢240尾 tokens；末8窗保留不进入优化，optimizer 可用1221窗。calibration 的32索引来自 `randperm(count-8)`；probe 为1221–1224。
- Joint `run.py:245–289` 的优化输入只取 `train_windows[(step*8+microstep)%len(train_windows)]`；`token_nll` 为标准 shift FP32 CE。参数首先全部冻结，再只启用 R/SA/SW 等优化组，不是零训练原始 PTQ。实际100步×8窗=800窗，日志索引逐项独立重算与0–799相同，总1,638,400输入 tokens。
- `sp2-down-readapt-s42/source/sequential_postprocess.py:171–190` 从上述 train 的32 calibration窗取前24窗拟合、后8窗选择；`54–58` 虽调用名为 `full_validation` 的通用计分函数，传入的是这8个 **train** 窗。没有从 validation/test 构造 selection。`120–139` 与 `209–240` 按该 train NLL 接受局部候选后才进行独立 validation 报告。
- QAT `distill.py:105–109` teacher 调用默认原生加载，冻结eval，不做 norm fusion。`112–152` 在同一 train 输入下使用 FP32 CE+KL，temperature1，CE权重0.1、teacher KL权重0.9；目标按t→t+1。`prepare_student:78–101` 训练量化线性层 FP32 master及量化 scale；不只是对原BF16进行无监督格式转换。
- QAT `270,298–355` 输入只来自 `train_windows[(800+step*8+microbatch)%1221]`。本次逐项检查400步日志中的3200个索引，全部符合该公式，范围0–1220，总6,553,600输入 tokens。会循环复用 train 窗；此预算不是3200个不同文本窗。实际 `target_ppl=null`、固定400步、validation_steps=[400]，没有按test提前停或选 checkpoint。
- validation加载用于诊断/开发；本任务原本允许开发validation。外部test/C4脚本不连接优化器、校准或后处理候选选择。此次源码/日志核查未发现运行级 split 泄漏；没有进行原始数据逐文去重或上游预训练语料审计，因此不声明排除了所有语料重叠。

轻量输入ID检查首次交互命令误把保存dict当Tensor，报TypeError后立即按`input_ids`字段读取成功；没有影响产物/模型/数字，没有GPU复跑。

## 5. 为什么不能只把降低归给 QAT

| 实际 seed42 固定包/原模型 | test PPL | test NLL |
|---|---:|---:|
| BF16 原模型 | 16.715764347250076 | 2.816352247049716 |
| SP2-PTQ（Joint100+后处理） | 14.302163393485294 | 2.6604108120809626 |
| SP2-QAT400 | 14.151216160997155 | 2.649800568169599 |

来源分别为 `qwen3-1p7b/{bf16-test-s42,sp2-ptq-test-s42,sp2-qat400-test-s42}/result.json`，本次直接读取。

此外，训练入口 `common.py:196–198` 先显式解绑，再调用 inherited `utils/fuse_norm_utils.py:39–74`。其中 `42–45` 对 embedding 每行减均值；这一步对 RMSNorm 模型 **不能作为严格等价变换而忽略**。其后才是 norm 权重融合、旋转、量化及训练。BF16参考没有这一步。因此“原BF16→SP2-PTQ”的比较同时含预处理和 train CE 优化、量化、train-selected后处理；不能从这一个差值识别哪个因素造成收益。norm融合/旋转的有限精度也不能一概说逐位恒等。

student被train CE直接优化，QAT又有额外训练和teacher soft targets，因此在同领域held-out WikiText上超过原始teacher并无数学矛盾；这些只是机制上的合理解释。现有证据没有同预算BF16适应对照，**不支持“量化本身使模型更好”或已证明训练适应是全部原因**。

分布外固定C4也没有普遍超过BF16：原BF16 23.136144236064272，seed42 SP2-QAT400 24.24703443394153。该事实与“WikiText训练后收益依赖评测分布”的解释相容，但也不是单独的因果证明。

“14.14”若指汇总表，应标明它是SP2-QAT400 seeds42/43/44的test均值（14.135738576092315）；seed42单次固定包为14.151216160997155。不要把均值误记为seed42结果。

## 未覆盖与交付边界

本次没有新增实验预算，没有修改应用/环境/历史产物，没有额外完整PPL或GPU验证。主执行的 `verifier/qwen-bf16-native-test-20260919/` 原生全test结果需单独读取才可写运行结论；本报告只负责代码和已有轻量日志/ID的独立核对。因果拆解和同预算BF16训练对照仍未实施，不应为了回答合理性质疑擅自扩大实验范围。
