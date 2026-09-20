# Phase 6：down 静态 INT16 新路线探索

日期：2026-09-21。用户授权建立新 phase、完成 down INT8/INT16/浮点比较，已有同口径数据不重跑。

## 本轮问题与范围

在同一个 B100 模型上，down 输入的静态 INT16 是否已接近浮点旁路，INT8 是否不足？本轮不重新训练 R/SW/SA、不做离散后处理或 QAT，不启动 down-SA 可学习性训练消融。

- 模型：Llama-3.2-1B-Instruct，Phase3 route-b-adam-100-20260914a/checkpoint-0100/static_w4a8.pt。
- 共同部分：112 个 W4 per-output-channel Linear、96 个非 down 静态 INT8 输入、离线 R1/R2；KV BF16、无 cache、prefill fake-quant。
- 三项：16 个 down 输入均浮点旁路 / 静态 signed INT8 / 静态 signed INT16。
- 两种整数格式共用每层 alpha=历史 train 捕获 absmax。scale=alpha/127 或 alpha/32767；zero=0；整数范围 [-128,127] 或 [-32768,32767]。负端多一个码位为 signed 格式本身的非对称，不单独调范围。
- 校准：复用 B100 的 32×128 train token 范围统计；原始数据索引见 B100/data.json。统计来源 checkpoint-0100/sp2_calibration.json 的 full_absmax，不使用其中搜索选择的 SP2 alpha。与旧 uniform 校准 capture_metadata.json 精确交叉核对，不重跑校准。
- 捕获语义：全部 down 浮点旁路时并行捕获；沿用历史 column-major v_proj 捕获布局。不是整数激活级联下重新校准，不宣称自适应分布校准。
- 评测：完整 WikiText-2 validation，252852 input tokens、252728 targets，123×2048 与948尾窗；同旧 evaluator 的 BF16 logits CE、token 加权 NLL/PPL。校准只来自 train，validation 不用于选择范围。

## 复用盘点

- 三项所需完整 validation 没有已完成同口径结果，需补测。
- 已有 down-float training_probe 只有4个 train probe 窗口，不当作完整 validation。
- 旧 B100 SP2 validation PPL17.11711490993314可作旁列背景，不是 INT8/16 对照。
- 旧 B100 uniform INT8 C4 PPL1330.1567504736056使用50候选输出MSE范围搜索且为C4，不能替代本次共同absmax范围的Wiki validation。
- Phase5 uniform 后处理/QAT结果改变了权重或训练预算，不能混入本组。

## 实现与解释

入口：scripts/phase6/down_precision.py；导入旧 Phase3 实现，只读父包，不改旧源码。旧 wrapper.bits=16 表示旁路，故两种整数格式都通过相同的 Linear input pre-hook 显式做FP32舍入/截断再回BF16；记录真实调用数、越界比例与BF16后改变比例。

结果仅回答该B100、该校准预算下的精度可行性。INT16接近浮点支持采用静态INT16；不证明学习down-SA无收益，也不构成NPU整数内核、速度或decode证据。若出现INT16明显损失，先据实际越界统计解释，不自动扩大校准、搜索范围或启动训练。

实施状态：三项完整评测已完成，独立CPU窄测7项PASS及最终结果审计PASS。结果与结论见 [RESULTS.md](RESULTS.md)。原工作树的dirty内容、旧Phase产物保留；未创建Git分支、未安装环境、未提交推送。
