# 当前实验说明

核对日期：2026-09-08。本文描述现有实现和脚本；下一步实验仍需单独确定。旧 W4A8 周期重校准方案见 [原规格](delete/root-docs-2026-09-08/SPEC.md)。

## Phase 2 两阶段路线

模型为本地 Llama-3.2-1B-Instruct。只使用 R1/R2，关闭 R3/R4，冻结原始模型权重，KV16。

1. Stage 1 使用 W16：联合学习 R1/R2 与 96 个非 down_proj 的 static per-tensor A8 scale；16 个 down_proj 输入实际旁路为 A16。初始校准后学习 SA，不按旧 Phase 1 的 0/20/40/60/80/100 方案周期重校准，也不在末尾用全量校准覆盖学到的 SA。
2. 先评测固定 R/SA 的 W16，再用相同 activation 策略收集 GPTQ 输入，把 112 个 backbone Linear（含 down_proj）的权重转换为 symmetric per-channel W4。当前启动脚本 `w_groupsize=-1`。
3. 产物为 `rotation/R.bin`、`rotation/quant_scales.pt`，以及 `gptq/w4_gptq_model.pt`。它们不是已经验证的手机 NPU 模型包。

当前脚本顺序是 `scripts/phase2/30_optimize_joint_r_sa_w16a8_local.sh` → `32_eval_w16a8_downa16_local.sh` → `31_gptq_w4a8_static_local.sh`，三者须使用同一个 RUN_NAME。默认 RUN_NAME 为 `w16a8-joint-r-sa-r12-s42`，该目录已有结果，不能直接覆盖重跑。

脚本默认 Stage 1 为 100 steps、seed 42、32×128 initial calibration、2048-token training、R learning rate 1.5、SA learning rate 1.0；GPTQ 为 128 samples，PPL 为 8×2048 WikiText-2。这些是当前脚本默认值，历史运行的实际设置应以各自日志为准。

## 术语与比较口径

- R-update 是一次 optimizer update，不是 gradient-accumulation micro-step。
- SA 是按模块保存的 activation scale 集合；per-tensor 不表示全模型共用一个 scale。
- GPTQ 是权重量化求解方法；per-channel/group32 是 scale 共享粒度，两者不是替代概念。
- W4 对 W16 的损失，与 W4 per-channel 对 W4 group32 的额外损失，是两个不同问题。粒度比较应从同一 BF16 模型和同一 R 出发，不能再次量化已有 W4 checkpoint。
- 目前 PPL 路径为 prefill、`use_cache=False`；不代表 decode、量化 KV cache 或手机 NPU 验证。

## 分布分析

代码在 `worktrees/SpinQuant-distribution-experiment/experiments/distribution_atlas/`，使用 BF16 模型和 R1/R2 观察权重与激活。默认 R 来自 `runs/phase2/w16a8-joint-r-sa-r12-s42/rotation/R.bin`。

结果在 `runs/distribution-atlas/llama32-1b-r12-bf16-wt2-s42/`：weight/activation records、对应 PNG，以及 weight channel 分析产物。分布图用于诊断，不等于某种量化改进已获得 PPL 收益。

## 既有兼容能力与约束

legacy dynamic、dual-phase static、旧 rotation-static periodic calibration 实现继续保留。dual-phase 模式的以下要求仍有效：

- prefill/decode 使用各自 frozen qparams，冻结 forward 不运行 min/max 参数发现；不允许 dynamic fallback。
- phase 来自真实 cache/position，须显式传播并覆盖 checkpoint 重算；缺失或不一致拒绝执行。StaticCache host 解析和 compile 诊断边界见 [ADR 0028](docs/adr/0028-host-resolved-phase-specialization.md)。
- 七类 backbone Linear 的 input A8 coverage 排除 lm_head/out_quantizer；static 模式保留对 int8_down_proj、act_order、pretraining_tp > 1、未实现 K/V 和不完整 save/export 路径的拒绝。
- 顶层 args/config 校验早于 NCCL、权重、数据和 GPU 工作，下游完整二次校验保留，见 [ADR 0030](docs/adr/0030-static-validation-before-runtime-side-effects.md)。
- manifest 必须来自真实冻结证据，见 [ADR 0029](docs/adr/0029-manifest-follows-frozen-evidence.md)。neutral post-RoPE K、cache write 前 V、真实 INT8/UINT8 KV cache、dual-phase 校准、模型包及部署属于后续独立工作，不能宣称已完成。

以上 dual-phase 约束不能套用为当前 down-A16/KV16 实验已经满足 strict W4A8KV8。完整历史决策见 [原 DECISIONS](delete/root-docs-2026-09-08/DECISIONS.md)；既有 ADR 保留。
