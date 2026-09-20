# 当前 Phase3 W4 成因诊断独立审计

2026-09-15；独立实验 auditor。**PASS：6 批、16 个完整 WikiText2 validation 结果及其条件解释。** 没有新部署候选或恢复训练成品，本次未另启 GPU 冷载复评、CPU 模型 forward、重训或旧基线重跑。

## 来源与实际运行

父包统一为 `runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`；参考状态统一为 `runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/state.pt`。复用父包正式结果 PPL 16.112577060930427 / NLL 2.779600150933802，已直接核对原 `validation.json` 的数据、协议及父包链。

执行入口是 `worktrees/SpinQuant-phase4-w4/experiments/phase4/diagnose.py`，分支 `phase4/w4-per-channel`，HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`；使用 rotation-quant-p0 Python、PyTorch 2.4.1+cu121 / CUDA 12.1 / RTX 4090。每批 `.launch.json` 保存完整 argv、环境、GPU headroom 快照、PID、tmux socket/session 与日志位置；与该批 settings/cases 及实际日志逐项一致。PID 是 launcher 记录的 tmux pane 进程，不把它当作另一次独立操作系统采样。

| 运行目录（runs/phase4） | GPU | PID | 完成 cases | tmux session |
|---|---:|---:|---:|---|
| diag-activation-20260915a | 5 | 970404 | 3 | `phase4-diag-activation-20260915a` |
| diag-down-interaction-20260915a | 5 | 1001011 | 1 | `phase4-diag-down-interaction-20260915a` |
| diag-families-20260915a | 7 | 971190 | 3 | `phase4-diag-families-20260915a` |
| diag-mlp-error-components-20260915a | 0 | 997465 | 2 | `phase4-diag-mlp-error-components-20260915a` |
| diag-mlp-format-20260915a | 5 | 986965 | 3 | `phase4-diag-mlp-format-20260915a` |
| diag-mlp-local-interaction-20260915a | 7 | 987767 | 4 | `phase4-diag-mlp-local-interaction-20260915a` |

共用 socket `rotation-quant-phase4`。每批目录保存当时 `diagnose.py`（83/104/127 行三代增量）、settings、results、complete；未把较新分支逻辑误套到较早实验。仅这些小文件和文本日志，无模型包或参考权重导出。所有 complete cases 与 launch/settings/results 一致，16 个结果逐一与原日志 RESULT 字典相等。

实际 Phase3 tracked diff 与初次继承 `inheritance.diff` 字节一致；19 个继承 dirty/untracked 文件与当前 Phase4 相应文件仍相同。重新核对 common.py、postprocess.py、eval_utils.py、modeling_llama.py、quant_utils.py、quant_linear.py 相同。未提交/推送或修改 Phase3。各批 source_record 指向 Phase4 实际导入路径；新源码保存了必要快照，不依赖仅有 Git HEAD 的错误复现方式。

## 数值与协议

- 每个 case 首先以原 packed signed INT4 × 原 `[out,1]` SW 重置全部 112 矩阵，再改指定集合；不同 case 不累积恢复。仅实际 wrapper input bits 改为 all16/down16/static，output bits=16。static 保留原 96 INT8 +16 SP2，未重估 SA/SP2；在线 Hadamard 仍为加载默认关闭。
- 参考直接由原始未量化 BF16、norm fusion、B100 保存 R1/R2 的 rotated_weight 构建。相同导出旋转顺序；未用 dequantized W4 充当参考。既有高精度边界 NaN-aware 相等记录及父包链支持本次参考匹配；本次未重复加载整包做同一张量检查。
- 同 solver 格式对照只替换 48 MLP 矩阵，从同一原始同 R 参考计算 absmax/maxcode、ties-to-even round、signed clip、BF16 回写。per-output-channel 与 group128（沿输入轴）同 RTN，W8 同 RTN；其它 attention 权重均为原父 W4。不含优化数据或额外搜索。
- 截断反事实使用原 SW：B=clamp(R,-8SW,7SW)，C=B-R，G=P-B。remove_clipping 为 BF16(P-C)，remove_grid 为 BF16(P-G)≈BF16(B)。G 包含优化后的整数码偏移和 BF16 反量化误差，不能命名为纯最近舍入误差。记录了交叉项，不假定 C/G 正交；这些恢复并非合法最终 W4 格式。
- 使用原 `full_validation` → `validation_acceptance.evaluate_full_validation` → 同一 `utils.eval_utils.evaluator`。全部 252852 输入 tokens、252728 预测 targets；123×2048 加 948 尾窗，0 未评分尾 token。123 个等长窗口按 token 等价平均 NLL，再与尾窗按 251781/947 targets 合并，最后 exp；没有平均窗口 PPL。已对全部结果独立复算合并 NLL/exp PPL 并核对日志。
- 保留成熟 BF16 logits CE 后转 FP32 的既有精度，不把结果描述为 FP32 logits CE。prefill/use_cache=False、KV16；不是整数 kernel、decode 或手机部署验证。
- 独立 verifier 的 `runs/phase4/verifier/diagnosis-review.md` 已对新增数学、reset、实际旁路和三个增量给出 PASS。本审计未重复其小张量测试。

## 全部结果

| 运行目录 | 条件 | 替换矩阵数 | NLL | PPL |
|---|---|---:|---:|---:|
| diag-activation-20260915a | `all16:none` | 0 | 2.744512915240 | 15.557034492007 |
| diag-activation-20260915a | `down16:none` | 0 | 2.747396072297 | 15.601952587634 |
| diag-activation-20260915a | `all16:all` | 112 | 2.613853954849 | 13.651562102806 |
| diag-down-interaction-20260915a | `down16:down` | 16 | 2.688916962315 | 14.715729597056 |
| diag-families-20260915a | `all16:down` | 16 | 2.686289921350 | 14.677121507297 |
| diag-families-20260915a | `all16:attention` | 64 | 2.713902355379 | 15.088039673767 |
| diag-families-20260915a | `all16:gateup` | 32 | 2.694081213725 | 14.791921893127 |
| diag-mlp-error-components-20260915a | `all16:mlp:remove_clipping` | 48 | 2.744710457704 | 15.560107970493 |
| diag-mlp-error-components-20260915a | `all16:mlp:remove_grid` | 48 | 2.638977908340 | 13.998888146295 |
| diag-mlp-format-20260915a | `all16:mlp:rtn4c` | 48 | 2.800570961709 | 16.454038715685 |
| diag-mlp-format-20260915a | `all16:mlp:rtn4g128` | 48 | 2.789579545532 | 16.274175813168 |
| diag-mlp-format-20260915a | `all16:mlp:rtn8c` | 48 | 2.640669879414 | 14.022593909189 |
| diag-mlp-local-interaction-20260915a | `all16:down1` | 1 | 2.734338045536 | 15.399546262881 |
| diag-mlp-local-interaction-20260915a | `all16:down_except1` | 15 | 2.696401072794 | 14.826276901255 |
| diag-mlp-local-interaction-20260915a | `all16:mlp` | 48 | 2.639211949921 | 14.002164851643 |
| diag-mlp-local-interaction-20260915a | `static:down` | 16 | 2.726035130155 | 15.272214477380 |

## 可支持的归因与边界

1. **当前 W4 条件损失已直接确认。** 在 all-A16 下，原 W4 PPL 15.55703449 对同 R W16 13.65156210，NLL gap=0.1306589604。不是照搬旧 C 父包或早期 8×2048 数据；原 BF16 13.6346577 与同 R 参考的细小差别不能算入当前 W4 gap。
2. **主要可恢复损失在 MLP，分布跨 gate/up/down，非旧 layer1-down 单点。** 恢复全 MLP 后 PPL 14.00216485，NLL 降 0.1053009653，覆盖本 all-A16 条件 gap 的 80.59%。这是指定联合干预的覆盖比例，非唯一可加贡献率；attention/down/gateup 的单组边际不能求和当总分解。仅 layer1-down 恢复到15.39954626，而其余15个 down 恢复到14.82627690，旧单层主导判断不适用于现父包。
3. **现有 MLP 范围截断不主导任务退化；剩余离散表示误差更有直接证据。** MLP 仅101388/805306368（0.01259%）元素越过原范围；C SSE 0.06887，相对总 SSE3371.59465为0.00204%，有非零交叉项，不能仅凭 SSE 下结论。移除 C 后PPL15.56010797，未改善；保留截断但移除 G 后13.99888815，接近恢复 MLP 的14.00216485。两者约0.0033 PPL差不能过度解释为截断有益。此结果支持已定义 C/G 干预下的范围内离散表示问题，不证明某一个求解器能在 W4约束下消除它。
4. **位宽与粒度区分已有最小匹配对照。** RTN W4 per-channel16.45403872、W4 group12816.27417581、W8 per-channel14.02259391。group128在该 RTN设置仅小幅改善，W8大幅接近W16；支持粗4bit表示影响大于这一次group128细化收益。不能宣称所有细粒度无效、per-channel已最优，或把父包15.5570与RTN各条件差异只归因粒度（父包还有学习SW/码优化）。
5. **SP2有额外条件损失及交互，不能解释全部W4损失。** 原W4下down-SP2相对down-A16增加0.03220408 NLL；同样非downINT8、恢复down W16后为0.03711817，差0.00491409。支持权重/激活交互存在；恢复 down 也会改变后续层输入，故不能单独证明整数码专门学到了 SP2 补偿，亦非独立可加误差来源。all-A16下现父W4仍可含历史SP2补偿。

这些诊断主要定位当前包的误差类型与模块；未测更细的任务敏感方向、通道结构或普适因果模型，未证明所有4bit求解器达到表示极限，也未产生新的同格式收益成品。WikiText2 validation 已被用于本轮诊断路线选择，不称为盲测；没有通过C4外部确认这些诊断趋势。无需为审计流程重跑旧冠军，最终格式仍保持既定 W4 per-output-channel/static A8。

最终文档交叉核查：已阅读 `W4_DIAGNOSIS.md`、STATUS/RESULTS 最新结论及其数字。80.59% 已限定为特定干预的条件 gap 覆盖；group128 明确不能排除更小组；SP2交互明确不单独证明码补偿；未宣称同格式新方法收益或W4不可改进。结论表述 PASS。
