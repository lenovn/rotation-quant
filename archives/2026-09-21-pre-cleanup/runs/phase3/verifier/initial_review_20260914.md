# Phase 3 独立 verifier：现有旋转 / LSQ 源码核查

时间：2026-09-14 01:46 +08:00。角色：恢复独立 verifier，不是主执行，不建立或修改主 goal。

## 结论与实际范围

- **源码推导检查 PASS（有明确前提）**：对于当前无偏置 Llama、已解开 embedding/head 权重共享且做相同 norm fusion 的模型，现有 `QuantizeLinear.rotated_weight` 的 R1/R2 方向、head 分块与 eval 离线融合的代数方向一致；量化作用在旋转后的有效权重，而非原权重。
- **数值测试 NOT RUN / Phase 3 应用验收 PENDING**：截至检查结束，主线程尚未通知 `experiments/phase3/{quantization.py,common.py,run.py}` 可读；该目录和 `tests/test_phase3_joint.py` 尚不存在。没有编写或执行测试，没有 GPU 操作。不能把本报告当作应用代码、完整实验、PPL 或手机 NPU PASS。
- 下文风险区分为现有代码事实、在特定调用下会失效的接口行为、待新实现验证的建议；没有声称主线程尚未写出的实现已经出错。
- 当前角色仅可新增/修改 `tests/test_phase3_joint.py` 与 `runs/phase3/verifier/` 报告。主线程 GPU 调度权限变化不扩大 verifier 的 CPU-only 范围。应用源码、环境、已有实验、他人进程均未修改。

## 实查工作区证据

源码根：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。

`GIT_OPTIONAL_LOCKS=0 git -C worktrees/SpinQuant-phase3-joint rev-parse HEAD`：

```text
24918316ed594848d4de797c356b120f2a4ee0f3
```

分支：`phase3/joint-r-sw-sa-sp2`。原始 `git status --short`：

```text
 M optimize_rotation.py
 M train_utils/quant_linear.py
 M train_utils/rotation_calibration.py
 M utils/process_args.py
 M utils/quant_utils.py
?? tests/test_learned_sw_training.py
?? tests/test_w4_aware_training.py
```

以上继承修改全部保留。没有将 HEAD 当作全部执行源码。读取了项目 AGENTS、STATUS、SPEC、PLAN、LESSONS、指定 HANDOFF 和 Phase 3 STATUS；所查 worktree 与报告目录下没有更深层 AGENTS。根文档的旧整理授权不覆盖本轮新指令。

缓存 `cache/models/llama-3.2-1b-instruct/config.json:5` 等位置实查：`attention_bias=false`、`mlp_bias=false`、hidden size 2048、head dim **64**、query heads 32、KV heads 8、layers 16、`tie_word_embeddings=true`、BF16。源码中“head dim = 128”的旧注释不适用于本模型，实际 `R2.shape[0]` 推导没有写死 128。

以下源码路径均相对上述 **新 worktree**，不是 `repos/SpinQuant`。

## 1. R1/R2 的具体语义

采用 `F.linear(input, weight) = input @ weight.T`、`weight[out, in]` 约定。记 R1 为正交 hidden rotation，`Bkv = I[num_kv_heads] ⊗ R2`，`Bq = I[num_attention_heads] ⊗ R2`。

| 部件 | 有效权重 / 坐标 | 训练与 eval 证据 |
| --- | --- | --- |
| q/k/gate/up | `W @ R1` | `train_utils/quant_linear.py:17`；`eval_utils/rotation_utils.py:63`、`:83` |
| v | `Bkv.T @ (Wv @ R1)` | `train_utils/quant_linear.py:35`；`utils/hadamard_utils.py:168` |
| o | `(R1.T @ Wo) @ Bq` | `train_utils/quant_linear.py:29`；`eval_utils/rotation_utils.py:71`、`utils/hadamard_utils.py:175` |
| down | `R1.T @ Wdown` | `train_utils/modeling_llama_quant.py:347`；`eval_utils/rotation_utils.py:92`，要求 `apply_r4=false` |
| embedding | `E @ R1` | 训练在 lookup 后乘：`train_utils/modeling_llama_quant.py:1092`；eval 融到 E：`eval_utils/rotation_utils.py:55` |
| lm_head | 训练 `hidden @ R1.T` 再原 head；eval `Whead @ R1` | `train_utils/modeling_llama_quant.py:1412`；`eval_utils/rotation_utils.py:107` |

v 的 R2 是对输出 head 维左乘 R2.T；o 的 R2 是对输入 head 维右乘 R2。不能把二者都写成一个方向。GQA 中 v 输出 512、o 输入 2048，因此两边 block 数分别为 8 和 32，不能按 v 的形状构造 o 的 block matrix。每层共享同一个 R2 时，重复 KV heads 与该 head 内旋转可交换；q/k 仍在原 head 坐标，不能顺手加入 R2/R3。

`QuantizeLinear.forward:53` 先求有效旋转权重，再 `quantizer.quantize(weight)`；`rotation_calibration.py:57` 按同一模块映射初始化当前 R 的 SW。导出应从量化前有效权重用最终 learned SW 一次产生整数码，不能先量化原 W 再旋转，不能对已经量化的候选再次量化。

### 具体风险与建议

1. **阶段 A 的 W16 切换不能只改 bits。** `RotationStaticWeightQuantizer.quantize`（`utils/quant_utils.py:1082`）在 learned scale 分支直接执行 `ChannelScaleQuantize`，不检查 `bits < 16`；即便仅设置 `requires_grad_(False)`，`scale_learnable` 仍因 Parameter 类型而为真（`:1006`）。新 weight quantizer 建议显式 `enabled`/stage bypass；W16 时严格返回输入，W4 时使用已注册 SW。用实际 forward 断言，不能仅检查属性。
2. **R2-only 输入会静默被忽略。** `QuantizeLinear.rotated_weight:15` 的 R2 逻辑位于 R1 分支内部；R1=None 时返回原 W。`ActQuantWrapper.forward:790` 还会在 R1=None 时丢弃 R2/transpose 参数。当前完整 R1/R2 路径不受影响；测试 R2-only 时必须传 identity R1，或由 Phase 3 接口明确拒绝这种调用，不能以该调用“验证 R2 无影响”。
3. **偏置不是通用等价。** 训练 `quant_linear.py:62` 总使用原 bias；eval o/down 会乘 R1.T（`rotation_utils.py:78`、`:102`），而 v 的 eval helper 不处理 bias。正确 v bias 应为 `Bkv.T @ bv`，o/down bias 应为 `R1.T @ bias`。当前 config 两种 bias 均 false，所以不是当前模型阻塞；新接口不能宣传支持任意带偏置 Linear。
4. **dtype 与舍入位置必须一致。** 训练先 FP64 R1 乘法，再回写 weight dtype，然后 FP64 R2 乘法，再回写（`quant_linear.py:18`、`:22`、`:33`、`:39`、`:41`）。eval 也是两次融合回写，但 R2 helper 先 `.float().cuda()`（`hadamard_utils.py:155`），因此 FP64 权重会先丢到 FP32；当前 BF16/FP32 不因这一步额外丢精度。新导出若一次 FP64 连乘到最后才舍入，可能跨 W4 rounding tie 改整数码。建议训练/导出共用同一 effective-weight 计算顺序；独立测试另外用 block-matrix oracle，不能仅把被测 helper 输出再比较自身。
5. **head 融合只保证实数代数，不保证 BF16 bitwise。** 训练先把 `hidden @ R1.T` 舍入，再乘 head；eval 将 head 预乘 R1，舍入位置不同。embedding lookup 后旋转与离线查表可以更直接对应；head 不应要求无条件 bitwise logits。应分别测试有效 backbone weights/code 完全一致，以及小模型 FP32 logits 的合理数值误差；BF16 整模差异仍须主线程实测，不能从代数 PASS 推出零 PPL 差。
6. **tied embedding/head 是当前模型真实前提。** config 为 true；既有入口先关闭 tie、clone head，再 fusion（`optimize_rotation.py:97`）。若共享 storage 直接 fusion，head 的 norm scale 会污染 embedding；顺序旋转还可能重复变换。新加载器需要保留该 untie/clone 语义，不只是设置 config 字段。
7. **norm fusion 不全部是对原 HF 的纯等价变换。** `fuse_norm_utils.py:43` 会额外逐行减 embedding 均值；RMSNorm 一般不具有平移不变性。norm gain 融入相邻 Linear 本身可代数验证，但不能把 embedding centering 也一并称为严格函数等价。三臂应共享相同准备路径；R-only 等价 oracle 从同一个准备后 FP 模型出发；相对原 BF16 的差异另算。
8. **离线导出不能保留在线 R 路径再旋转一次。** stock eval 模型应装载融合后的 embedding/head/backbone，R3/R4 关闭。仅对训练模型调用 `.eval()` 不会移除 `modeling_llama_quant.py:1094` / `:1412` 的在线 R，也不冻结或替换 quantizer。导出重载测试应使用真实目标结构，而不只序列化几个张量。

## 2. 现有 LSQ / STE 的精确定义

权重 SW 使用 `ChannelScaleQuantize`（`utils/quant_utils.py:75`）：

- `scaled = weight.float() / SW`；每输出通道 SW shape `[out, 1]`，signed W4 code `round(scaled).clamp(-8, 7)`；forward 返回反量化值并回到 weight dtype。`torch.round` 是当前 rounding 路径，不能改成 `floor(x + 0.5)`。
- `grad_weight = upstream`，包括饱和区，是 **identity weight STE**，不是 activation LSQ 的饱和 mask。
- 每行 `grad_SW = sum(upstream * term) / sqrt(in_features * 7)`；`term = -8` 当 scaled<-8，`7` 当 scaled>7，其余为 round(scaled)-scaled。边界正好落在整数端点时 term=0，严格不等号语义应保留。

非 down SA 使用 `LSQQuantize`（`utils/quant_utils.py:94`）：

- per-tensor 一个 scale，code 为 signed `[-128,127]`。
- 输入梯度只在 scaled∈[-128,127] 透传；范围外为 0。
- `grad_SA = sum(upstream * term) / sqrt(input.numel() * 127)`，term 采用相同分段形式。不是用 W4 的 qmax，也不是逐输出通道归一化。
- `RotationStaticActQuantizer` 通过 `scale` 是否为 Parameter 选择 LSQ，不通过 `.training`；`.eval()` 并不代表 scale 已从所有学习/校准路径中冻结。

### 尺度生命周期与接口风险

1. **注册时机与 FP32**：SA/SW 应在模型 BF16 转换后建立 FP32 Parameter，且在 optimizer/DDP 注册前创建完毕。注册后保持 Parameter 对象身份，更新时 `copy_`/原地正值约束，不用新 Parameter 替换。否则 optimizer 持有旧对象、重载 scale 降为 BF16、阶段切换不参与同步等问题可能被“有 grad”假象掩盖。共同 R 也保留 FP32（既有 `RotateModule`：`optimize_rotation.py:46`）。
2. **实际更新而非仅非空梯度**：明确检查 R/SW/SA/SP2 的 finite gradient、实际参数 delta、原 FP weights 无 grad/无变化，包含正学习率的一次 optimizer update。warmup lr=0 的 update 可以合法无变化，不能将其当唯一可学习性验证。每次 update 后保持严格正值；旧 quantizer 本身不自动约束正值，约束来自 callback（`rotation_calibration.py:117`、`:187`）。
3. **learned 值不能被最终校准覆盖**：旧 SW 的 begin/load/refresh 对 learned scale 已有拒绝（`quant_utils.py:1022`、`:1049`、`:1058`），应保留。SA 的 begin/finish/load 则仍可覆盖 Parameter 内容（`:644`、`:666`、`:679`），所以“Parameter 存在”不是冻结证明。新 freeze/export 必须避免这种覆盖；冻结推理 forward 不发现 min/max。A/B 的最终 down 校准仅修改其 16 个 down 参数，不能顺带重估 96 SA/112 SW；C 保留已学 down 参数。
4. **SP2 必须真正穿过 wrapper**：`ActQuantWrapper.forward:780` 在 wrapper 层按 bits16 旁路。旧 `enable_non_downproj_scale_learning` 会设 down bits16（`rotation_calibration.py:45`）；新 down quantizer 需实际 bits8 且走静态调用路径，不能掉入 dynamic `find_params/free` 或旧 phase API。建议静态 quantizer 明确提供 `bits=8`、`rotation_static_enabled=True`、`forward(input)`，并与输出量化关闭状态一起检查。
5. **SP2 exact-forward 与 backward 分开定义**：保存并重载现有码本、正值范围参数 alpha 和 tie 规则；训练 forward 与冻结 forward 对同输入输出一致。原始硬 bucketize 不提供有用的输入梯度，不能只给 alpha 加 Parameter 就说 R/SA 在完整 SP2 前向中可学。alpha 是范围还是码本 step 要明确定义；其归一化和饱和区 surrogate 不能未经说明照搬 uniform INT8 LSQ。新文件可读后才能审核实际实现，不在本轮虚构 SP2 已通过。
6. **现有 export 会漏掉新 SP2 类型**：`rotation_calibration.py:14` 用具体类过滤，`:292` 仅导出 activation/weight 的 `.scale`。新接口至少明确保存 112 SW、96 non-down SA、16 SP2 alpha 与对应量化格式；如果使用独立 SP2 类/alpha 字段，不可直接复用旧 helper 后声称完整。建议按模块路径显式分组导出/加载，并用缺项、形状、有限正值及 reload 输出测试，而不是添加新 hash/contract 框架。
7. **多卡公平性不只有 global batch**：SA LSQ 的归一化用单次 forward 的 `input.numel()`，随每卡 microbatch/序列长度变化。相同 global batch 通过不同 microbatch/accumulation/card 数实现，并不自动给出相同 SA surrogate 梯度尺度；SP2 若照此实现也有该问题。优先保持三臂每次 forward 的形状一致并记录 accumulation/卡数；若需改变，显式说明采用什么 normalizer/学习率处理。此处是公式推导风险，没有运行分布式测量。

## 3. 给三个新文件的最小接口建议

- `quantization.py`：输入量化与权重量化分工明确；W16 bypass 独立于 scale Parameter 类型；量化 forward 无隐式校准；learned FP32 scale 对象稳定；SP2 同一码本/alpha 的训练与冻结 forward 一致，并明确 STE、正值规则及饱和边界。
- `common.py`：显式区分 source/project/artifact root；共同未优化 R 初始化与 untie/norm preparation；统一按七类模块取得有效 W 的 R1/R2 参数；初始化只从当前 R 的量化前 W；冻结/重载不重新 MinMax、GPTQ 或发现 activation range。避免 eval 硬编码 CUDA helper成为 CPU 检查的隐式 GPU 路径。
- `run.py`：A 的 W16/W4 阶段切换以 forward 为准；B/C 起点不继承已优化 Phase 2 R/SA；optimizer/DDP 前完整注册，R 与尺度不同 optimizer group；记录更新和归一化口径。train probe 不改变 observer/frozen state；最终导出从当前 FP weights/R 与 learned scales 产生，不原地污染训练模型或其他候选。

这些是接口建议，不代表主线程已经采纳或实现；无需为本轮引入新冻结 contract、hash 或额外发布门禁。

## 4. 收到主线程通知后的 CPU 测试范围

仅在 `worktrees/SpinQuant-phase3-joint/tests/test_phase3_joint.py` 增加独立测试，实际数量以收集结果为准：

1. 非 identity 正交 R、rectangular MLP、GQA 不同 head 数：七类有效权重与独立 block-matrix oracle；FP32/FP64 的适当路径；不调用硬编码 CUDA 的 eval helper 冒充 CPU 验证。
2. W4 signed 边界、ties-to-even、零行、每行 SW；SA 与 SW 的分段梯度及不同归一化；W16 真 bypass。
3. SP2 精确码本/中点/范围边界与训练-冻结 forward 一致；对输入和 alpha 的实际 surrogate 梯度与更新。
4. 小模型 R/SW/SA/SP2 联合 update、checkpoint 重算；FP32 参数对象持续、原 FP weight 不动、正值/finite；无在线 R3/R4。
5. 16 层小尺寸结构的 17 R、112 SW、96 SA、16 SP2 coverage；候选隔离、freeze/export/reload 一致，缺失参数不默默动态回退。
6. 小模型 FP 无量化旋转前后输出与离线融合输出；untie/clone 和 norm preparation 前提；不将 FP64/FP32 小模型测试外推为完整 BF16 数值验收。

预定命令（**本轮未运行**）：

```bash
cd /home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
  /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest \
  -p no:cacheprovider tests/test_phase3_joint.py
```

解释器路径只核对存在，本轮没有用其 import torch/pytest 或执行数值代码。测试日志和结果将另存本 verifier 目录；会报告实际 collected/pass/fail/skip 和未覆盖项。若导入或环境阻止收集，区分环境受阻与应用断言 FAIL。旧测试只读参考，本轮不运行、不修改。任何 PPL、完整 validation token 口径、正式 GPU 训练、pack 大模型完整产物、decode/KV8/设备内核均不在当前已完成验收范围。
