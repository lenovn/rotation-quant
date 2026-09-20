# 当前 W4A8 validation 阶段验收

两组均为本次新测，未复用历史 PPL。

| 配置 | PPL | 平均 NLL (nat/token) |
| --- | ---: | ---: |
| 原始 BF16 W16A16 | 13.63465773 | 2.612614913 |
| C RTN W4 per-channel + 非 down INT8 + down SP2 A8 | 17.64239998 | 2.870305094 |

总体 PPL 增加 4.00774225，相对增加 29.393787%；平均 NLL 增加 0.257690181 nat/token。这是整模型端到端损失，不作模块损失分解，也不等同于任务正确率下降 29.39 个百分点。

## 统一评测口径

- 模型：本地 Llama-3.2-1B-Instruct；两组使用同一 tokenizer，禁用自动 BOS/EOS，文本通过两个换行拼接。
- 数据：Salesforce/wikitext 的 wikitext-2-raw-v1 validation 全集，252852 tokens。
- 123 个互不重叠的 2048-token 完整窗口，加一个 948-token 尾窗；每窗首 token 不计预测损失，共 252728 个预测目标。窗口之间重置上下文。
- teacher-forced next-token PPL；batch 1，use_cache=False，PREFILL，KV16。逐层执行保留全部上游量化误差传播。
- 复用项目原 evaluator，对完整窗口和尾窗分别评测，再按预测 token 数加权平均 NLL，最后取指数。分段 NLL 由原 evaluator 的 float32 PPL 取对数，含微小浮点往返误差。

## 配置核验

- 基线直接使用原始 BF16 权重，未融合归一化、旋转或量化；共享 embedding/lm_head 按既有加载器复制保持等价。
- W4A8 加载 C 的 R1/R2、RTN checkpoint 和非 down SA；未重新量化权重或校准激活。
- 112 个 backbone Linear 权重使用 W4 symmetric per-output-channel；实际加载权重与 checkpoint 逐元素相等。
- 96 个非 down 输入使用固定 INT8 scalar scale；16 个 down 输入使用之前 train 校准的固定 SP2 scalar alpha；全部量化器 buffer 评测前后相等。
- embedding、lm_head、norm 保持高精度；这是 BF16 运算承载的 fake-quant PPL 测量，不是设备整数内核或 decode 性能测量。
- 日志模型 repr 中的“Per-Token”来自旧 wrapper 展示文字；实际运行的是 RotationStaticActQuantizer/FrozenCodebookQuantizer，尺度是固定 per-tensor，非运行时 per-token 校准。

当前 SP2 配置此前已参考部分 validation 单层诊断和 test 短样本结果选定；本次是固定配置的 validation 全集验收，不声称这是完全未参与选择的独立泛化评估。

原始数字、产物路径和配置核验见 [summary.json](summary.json)。入口为 `scripts/phase2/43_run_validation_acceptance_c_local.sh`；独立核验见 [verification.md](verification.md)。
