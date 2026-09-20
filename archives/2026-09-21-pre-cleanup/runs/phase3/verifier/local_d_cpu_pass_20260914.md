# Local D 新路径：限定 CPU PASS

日期：2026-09-14。独立 verifier，仅写 `tests/test_phase3_joint.py` 与本 verifier 证据目录；应用代码只读，未使用 GPU、安装、提交、推送或操作训练进程。

## 结论

**限定 PASS：新增 7 个参数化测试实例全部通过，无尚未解决的应用 FAIL。** 包括 3 个捕获/数学/搜索测试、driver 的 reject / identity / diagonal 三个分支，以及既有 launcher 测试新增 local-d 参数分支。未运行旧 suite、旧 sequential 七项或 GPU 烟测。

适用输入是 **未融合既有 D 的 PTQ 父冻结包 + 对应该父包当前 R 的原始 FP reference**，可以包含先前固定 SW 离散码/SP2 后处理。不能把此 PASS 扩展到 QAT 主权重或已有 D 的父包。`experiments/phase3/README.md:231` 已明确该限制；当前 driver 记录 provenance，但**不自动识别并拒绝不支持的父包，也不证明传入的 reference 与父包对应**。生产调用者仍须选对这两个输入，不可直接给 QAT/D 父包套原 R FP。

## 实际覆盖

测试路径均为 `worktrees/SpinQuant-phase3-joint/tests/test_phase3_joint.py`。

1. `:2124`，真实 tiny EvaluationModel、112 个实际 wrapper：实际 backbone 捕获 32 个完整 2048-token 窗；up inner Linear 接收到的 INT8 量化后输入与独立 round/clamp oracle 逐位一致；gate 为实际 Linear 输出；down 为 SiLU(gate)×up 的 pre-SP2 原始输入，不是其量化值。实际 `candidate_inputs` 对齐 BF16 linear + 独立 SP2 oracle，24/8 分割得到 49152 / 16384 行；无 first128 或行抽样。ValueError/RuntimeError 原样传播，缺捕获拒绝，成功和失败均清理 hooks；模型与输入无污染。
2. `:2185`，纯数值 oracle：改变 RMS/FP 列 absmax 排名使所选通道在 1/3 之间切换，非硬编码。检查 Phase2 同一稀疏化/去 log 均值规则的解析有界例子、常数与零值边界；0/.25/.5/.75/1 强度中 0 严格 I、D 有限正、几何均值约 1、界限 [.25,4]。FP64 同输入成对线性不变量容差 1e-12；这是 BF16 舍入前的代数检查，不是声称 BF16 变换完全函数等价。
3. 同一数值测试检查实际 BF16 FP up/D 与 down×D 后的 signed INT4 codes、up SW=parent/D、down SW=max(parent, BF16 transformed rowabsmax/7)，展开行与保留行都覆盖；I rule 确实可以改变父码/SW。父记录、FP 张量不变，D 错形状、零/NaN 拒绝。源码包含零行 absmax 的 clamp_min(1e-5)，与 Phase2 现有实现一致。
4. `:2249`，3 个 D 的搜索整合：每个 D 独立收到该 D 的 FP 变换、SW、RTN 起点和重新生成的完整量化输入；不传 previous candidate 的 initial_codes。专门令后 8 窗的最大通道不同，确认方向只由前 24 窗 RMS 决定。对每个 D 的 0/512/2048/8192 共 12 次候选均经过真实 `score_candidate` 的 apply/rollback 路径；模拟 NLL 让 MSE 最优者反而最差，仍选择 step512。检查被评分模型确为该 D/该 milestone 的权重，所保存 D 向量与选中记录绑定，结束后模型/scale/records/reference 精确恢复。
5. `:2343`，driver 三分支：真实 tiny 112-wrapper 冻结包保存/reload，选择与搜索分数 mock；检查 train 索引、32×2048、24/8 与 16376 selection targets，完整 validation 使用另一份 252852 token / 252728 target 输入。训练 NLL 不优于父包时不导出/不 full validation；非 I 获胜时 I 与 D 各导出、真正冷载各一次；I 获胜时只冷载评测一次并 `reused_identity=True`。模拟完整 validation 更差也不反向改变 train 选择。
6. 同一 driver 测试检查冷载完整 112 权重/112 activation 记录与预期一致；112 activation 中的 16 SP2 scale、96 non-down SA，gate/无关权重与 HP 保留。冷载模型状态以及小窗实际 logits 与独立应用相应候选的模型逐位一致。parent 字节不变；settings/result/包/validation 保留 parent provenance，包中保存对应 parent/reference、D layer/channel/完整向量/strength 与 selection 记录。
7. `:770`，仅 launcher 的 `[local-d]`：mock subprocess 确认新入口路由 `local_d.py`、parent/reference/layer 参数、显式 CUDA_VISIBLE_DEVICES 子环境、shell 引号、pane PID/log/launch 记录与唯一 run 保护；没有实际 tmux/GPU 查询或启动。

## 源码核查与边界

- `local_d.py:25/60/81/98/110/158` 分别对应捕获、方向、BF16/SW 变换、候选输入、搜索、导出控制。CLI 要求 strengths 从 0 起有序且在 [0,1]，因此 driver 的首候选为 identity rule。
- 对照只读 `scripts/phase2/down_d_search.py:31` 与 `scripts/phase2/sparse_d_sp2.py`：有界 dense log 方向、选单通道非负分量后几何中心化，以及 SW transport/扩张规则一致；新 layer 明确由参数指定，channel 从数据选择。
- `local_d.rounding_helper` 和 `score_candidate` 直接复用已验证 sequential 路径；前者只加载外层纯 Torch `fixed_grid_rounding.py`，不导入旧 SpinQuant repo。新测试在 helper 边界提供有限候选，未重新运行已有 8192 坐标求解器数学或旧异常回滚套件。
- NLL 控制流测试使用模拟评分；没有把这些模拟分数称为新实际 PPL。真实 BF16 evaluator 的既有 CPU 验证范围不被扩大。
- dormant legacy buffer 精确比对沿用 `equal_nan=True`；没有通过放宽数值误差来掩盖失败。
- 本次读取的应用 HEAD 为 `24918316ed594848d4de797c356b120f2a4ee0f3`，工作树有未提交源码；HEAD 不代表新增应用内容。`local_d.py` 检查前后 mtime 同为 `2026-09-14 09:16:02.180582978 +0800`，未新增 hash。

## 执行及首轮测试问题

均使用已有 rotation-quant-p0 Python；工作目录为 Phase3 worktree，设置 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=<Phase3 worktree>`，执行 `python -m pytest -p no:cacheprovider`；basetemp/XML/log 均位于 verifier 目录。

| 证据前缀（runs/phase3/verifier/） | 选择范围 | 实际结果 |
| --- | --- | --- |
| `cpu-local-d-first-20260914` | 新 local-D + launcher local-d | collection ERROR：测试插入点拆开原测试末尾两行，导致 IndentationError；未执行应用测试 |
| `cpu-local-d-second-20260914` | 同上 4 项 | 3 passed / 1 failed，10.62s；失败为测试 toy 整体 `.to(bfloat16)` 错误转换了 SP2 levels/scale，与生产 FP32 码本状态不符 |
| `cpu-local-d-third-20260914` | 仅失败搜索项 + 新 driver 3 分支 | **4 passed，14.21s**；之前通过的 3 项未重复 |

两项均为 verifier 测试构建问题，不是应用修复：恢复被拆开的原测试两行；搜索 fixture 仅把线性参数转 BF16、保留并断言 SP2 levels FP32，与生产冻结冷载一致。保留独立 oracle 的 `torch.equal` 及候选污染断言，未放宽精度后判 PASS。日志与 XML 均保留，包括失败历史。

第二轮 selector：`-k 'test_local_d_ or (test_tmux_launch_explicit_child_environment_quoting_pid_and_records and local-d)'`。

第三轮 selector：`-k 'test_local_d_search_each_diagonal_independent_inputs_rtn_and_all_nll or test_local_d_driver_matched_cold_exports_fixed_boundaries_and_provenance'`。

`GIT_OPTIONAL_LOCKS=0 git diff --check -- tests/test_phase3_joint.py` 退出码 0。运行中的警告仅既有 transformers 弃用提示。

## NOT TESTED

真实 CUDA 设备迁移/RoPE、1B 实际显存与 BF16 GPU 内核、生产全规模捕获/8192 坐标吞吐、新 D 搜索实际 PPL、正式完整 validation、实际 B100/reference 输入血缘和最终实验/手机部署验收。QAT/已有 D 父包适配不在此实现限定 PASS 内。无应用 CPU 阻塞，独立核验本轮结束，等待真实新代码变化。
