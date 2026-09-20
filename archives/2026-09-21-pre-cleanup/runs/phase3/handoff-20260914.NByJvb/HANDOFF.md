# Phase 3：三条联合优化路线、初始化与训练长度的实证比较

交接日期：2026-09-14。来源对话：`01a095f3-0dfd-7d91-95ad-bc2d92839bcf`。这是一份给新独立对话的执行交接；旧对话只做交接和分支准备，没有启动 Phase 3 训练。

## 用户最新请求与有效授权

用户原话：

> 可以，那我这里给你设置一个goal，去分析对比一下
> 1.先做R+非down a8 SA联合学习之后再加入W4进去联合学习（也就是我们现在走的这条路线）
> 2.把R+非down a8 SA+w4 SW一开始就放到一起联合学习优化
> 3.把R+非down a8 SA+w4 SW+ down SP2 A8 SA 放到一起联合学习优化
> 这三个哪个的精度更好，同时请你验证是不是如果SW和SA有一个好的初始化后再和R进入联合学习的优化训练，最终训出来的效果会更好，还有就是我在观察之前的联合优化训练时loss总是在二点几附近反复震荡，对于这样的loss变化我其实不太理解，如果是这样的变化规律的话，那训100个step和10个step还有区别吗
> 请你做实验验证以上我的一些问题，同时我希望你在验证这些问题的实验过程中是会根据实验结果来去合理规划下一步的实验，形成一个loop的工作流，而不是盲目执行，在完成以上验证实验后，将最优的一个路线再加入已证明有收益效果的离散 W4 码优化、SP2 范围调整以及局部 D ，同时看看届时的完整 validation PPL ，如果仍不能满足和W16A16基线模型的PPL掉点差距在1左右的要求，请上fine-tuning或者蒸馏来试试能不能达到我们的目标，以上的实验作为我们的phase3阶段实验，请另开分支来完成，同时我想开一个新的对话去完成，而不是在当前这个对话中去执行这个goal，请你给一个handoff后直接调用一个新对话去完成这个goal，模型和思考强度用GPT-6 Astra xhigh

这是新的执行授权：在新对话直接完成必要实现、CPU/GPU 验证和自适应实验循环，不逐阶段等待批准。允许创建 Phase 3 分支（已完成）、学习 R/SW/非 down SA/down SP2 范围，并在上述量化路线与后处理仍达不到目标时尝试 fine-tuning 或蒸馏。旧的 no-gradient/frozen-R/no-distillation 和旧 R-only 待批准提案被本次授权相应取代。前期比较仍冻结原始高精度模型权重；后续若 fine-tuning 解冻主权重，或引入教师，必须单独标明方法和参数范围，不能继续称为原来的无蒸馏 PTQ。

用户随后再次纠正本段限制：GPU 1、3、5、6、7 都可以用；允许按实际可用资源多卡训练或并行独立实验，取消单卡、12 GiB 和 memory.used/total<35% 的硬门槛，也没有 GPU 时间预算。每次启动仍实测显存/进程用于合理调度；已有进程或高 utilization 不自动排除。**不得终止他人进程。** 不再将上一版的“禁止安装/改变环境/提交推送”等附加禁令整体沿用；按用户授权的 Phase 3 实际需要、项目规则和普通权限处理，避免无必要的共享环境或他人工作改动。原 Phase 2 有效产物应保留供比较。模型仍按用户明确指定 GPT-6 Astra / xhigh。此更新优先于启动 prompt 早期版本中的资源限制。

用户再次补充：**“同时handoff中后续的实验在多卡合适的时候也可以多卡，不强制单卡。”** 本条适用于整个 Phase 3，包括后续路线比较、尺度初始化与训练长度试验、后处理、fine-tuning 和蒸馏；合理时可多卡训练或并行独立实验。比较时记录实际卡数、global batch、梯度累积及有效训练 tokens，保持各臂口径可比。

## 新对话先做什么

1. 使用这个**新独立 thread**，不要 resume/fork 旧 thread。读取此交接后建立原生 goal（create_goal；不设置 token budget，用户没有要求）。然后立即工作；若原生 goal 已由先前接手或启动端建立，读取并沿用，不重复创建。不要仅回答计划或等批准。
2. 读取项目 AGENTS.md、STATUS.md、SPEC.md、PLAN.md、LESSONS.md；根文档目前仍是 9 月 8 日快照，其“本轮只整理”不能覆盖上面的用户新授权。需要历史再读 research_log.md 和下面的指定实验记录；不重新做大范围阶段 0，不扫描大 checkpoint 求新哈希。
3. 核对新 worktree 分支、源码改动和重复运行情况，确认 Phase 3 实验还未被别人启动。旧源代码和 Phase 2 产物只读。
4. 在新分支内建立最小实验驱动与记录，用项目要求的独立 verifier 做实现与数值核对；已有授权内自动通过简短正确性检查后进入正式实验。不能用自己写自己的“独立 PASS”。
5. 新对话持续落盘进度到 `runs/phase3/STATUS.md` 和每轮目录（假设、有限候选、控制变量、命令、PID/GPU、结果/失败、分析、下一步），便于网页/协调端查看。每个完整训练/评测结束立即分析再定下一轮。目标没完成时不要因为一次实验跑完就结束。

建议原生 goal objective：完成 Phase 3 的三路线公平对比、尺度初始化和 10/100 步实证诊断，以证据驱动有限实验循环；在优胜路线验证离散 W4/SP2 范围/局部 D 后处理，必要时按用户授权 fine-tuning 或蒸馏，争取同口径完整 WikiText-2 validation PPL 不超过原 BF16 +1（14.634657725643203）；保存可重载静态 W4A8 产物、独立验证与如实完整结论。不能只达到一个分数却遗漏用户要求的对照；也不能为关闭 goal 将未达目标写成完成。

## 分支、目录与环境

- 项目根：`/home/dongpeiyan/projects/rotation-quant`。
- **新源码 worktree**：`worktrees/SpinQuant-phase3-joint`，分支 **`phase3/joint-r-sw-sa-sp2`**。
- 分支起点：`24918316ed594848d4de797c356b120f2a4ee0f3`。
- 原源码仍在 `repos/SpinQuant`，原分支 `phase2/joint-r-sa-w16-gptq-w4` 保留。
- 已将原源码 5 个 tracked 未提交改动和 2 个 untracked 测试按字节复制到新 worktree，原分支/status 未改变。证据：本目录 `branch_setup.json`、`source_before.status`、`inherited_source.patch`。不要把未提交实现遗漏为干净 HEAD。
- 新增应用源码、实验入口和测试优先放新 worktree（如 `experiments/phase3/`），运行产物放项目 `runs/phase3/`。外部 Phase 2 脚本可参考/只读复用，必须确保实际导入 **新 worktree 的 utils/train_utils**。
- Python：`/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python`；该环境 torchrun 可用。系统 `python` 不存在，`python3` 可读写小 JSON。不要安装 rg，当前用 grep/find。
- 模型缓存：`cache/models/llama-3.2-1b-instruct`，沿用已有 model/tokenizer revision 与缓存 manifest；开始实验前从现有小记录核实准确 revision，不重新下载/默默升级。
- 现有 `scripts/phase2/down_d_search.py` 会按路径推导 ROOT 并优先插入旧 `repos/SpinQuant`！复制/复用时必须明确区分 project root、source root 和 artifact root，打印实际模块 `__file__`，防止新分支却执行旧代码。
- 每次修改前向用户说明文件、目的和副作用；保留兼容路径与既有测试/安全措施。不新增无理由 hash、contract/gate。CPU 检查使用 `PYTHONDONTWRITEBYTECODE=1`、pytest `-p no:cacheprovider`。

## 公平实验设计与合理的首轮规模

以下是交接建议，不是已运行结果；新对话根据源码、正确性检查和实验数据有限调整并记录原因。

**主比较：相同原始 FP 模型、未优化的共同 R 初始值、数据序列/seed、最终格式、总 optimizer updates / 训练 tokens。** 第一条的 W16 阶段也计预算。不能让第二/三条偷偷从已经经过 W16 R+SA 优化的 C 或 R0 开始，再声称是“从一开始联合”。共同 R 可以取固定 seed 的现有未优化 R 初始化；选择理由和实际生成方式必须落盘。

| 路线 | 第一阶段 | 后续阶段 | 最终评估 |
| --- | --- | --- | --- |
| A 分阶段 | W16 下学 R + 96 非 down SA，down A16 | 加入 W4 + 112 SW，继续 R/SA/SW，down A16 | 冻结 R/SW/SA；train-only 校准 down SP2；完整静态 W4A8 |
| B 从起点 W4 联合 | W4 + R + 96 SA + 112 SW，down A16 | 同样联合 | 同 A 的 down SP2 校准程序 |
| C 从起点完整前向联合 | W4 + R + 96 SA + 112 SW + 16 down SP2 范围参数 | 同样联合 | 保留所学 SP2 参数，冻结完整静态 W4A8 |

首个有用规模可用 100 总 updates：A 先 50 W16 再 50 W4，B/C 各 100 W4；这不是假定 50/50 最优。先做能够比较的三臂，然后视收敛/排名用一个有理由的阶段比例或延长预算对照，避免把单一比例的失败解释为整个路线失败。历史 100+100 若另作复现，要将总 200 与 B/C 总 200 配对，而不能与 100 相比。

全程控制 global batch（建议沿用 8；可按所用卡数分配 microbatch 与 accumulation）、2048 上下文、seed/data_seed42、次序、梯度检查点、BF16/KV16、optimizer/学习率选择原则。多卡或可用显存导致 batch/实现调整时，保持三臂可比并说明；不要偷偷扩大某一臂的有效训练 tokens。R 用现有正交约束 SGDG；SW/SA FP32。历史 LR R=1.5、SA=1、SW=.01 是可追溯起点，不是最优；先用极少短程验证实际变化/稳定性，避免不合理 LR 让某臂失效。共同参数用可比 LR 轨迹；路线特有参数启动时机和 warmup 明确记录。

**初始化问题：** 不把从旧 C 学过的 R/SA/SW 继承称为“只改初始化”。先定义合理的当前路径 MinMax/全范围初始化；再做同一个 R、同数据、同训练 schedule 下的 train-only 尺度预优化（R 暂固定）的成对初始化对照。预优化限制为尺度，不改变原 FP 权重；记录多用的样本/计算，增加等总预算对照以区分“更好初始化”与“额外优化”。先在一个有代表性/获胜联合路线测 default vs improved；如果有收益/交互证据，再扩到其他相关臂，不先盲跑 3×所有组合。以固定 probe 和最终 validation 判定，不能以校准 MSE 自我定义“好”。

**10 vs100 steps 与 loss2.x震荡：** step 明确定义 optimizer update，不是 micro-step。至少保存 0/10/25/50/100 的固定 train-probe CE/NLL 轨迹和实际参数变化；选必要 checkpoint 做同口径完整 validation（至少明确比较 10 和100 的最终静态 W4A8）。记录 raw minibatch loss、有效 tokens、LR、有限梯度、R/SW/SA/SP2 实际更新幅度、饱和比例。换批次 loss 震荡不能证明没有学习；固定 probe 改善也不能替代泛化。100-step cosine 中的第10步与总10步 cosine 训练的学习率不同，先回答同一训练轨迹继续90步的价值；若要回答独立10步训练，另做匹配/说明 schedule 的短臂。不能把 train CE≈2.x 直接指数化等同 Wiki2 validation PPL。

学习下的 SP2 必须 exact forward + 明确 STE backward；当前硬 bucketize 没有有用输入梯度。保持现有码本，正值范围参数由校准合理初始化，记录 learned range 是否扩张/收缩及饱和率；这是用户本轮主动要求的范围学习，不能误套旧 uniform-INT8 full-range 的冻结规则，但也不能暗中再加新的 clipping 搜索造成混淆。最终 per-tensor 参数冻结，不在推理重估。

主训练先用真实 next-token CE（无教师）。候选/尺度预优化仅使用 train。完整 validation 用于开发比较，明确多轮选择可能带来选择偏差；不要拿 wiki2-test 调参。已有 C4 冻结外部评估用于迁移证据，不拿 C4 validation 校准/选择。若后续采用 C4 train 做 FT/蒸馏，单独记录域/样本变化，并保持原外部输入独立。

## 获胜路线的后处理与必要的 fallback

三路线、初始化与步数问题有实测答案后，在最优产物上逐项验证已见收益的离散 W4 整数码优化、SP2 范围调整、局部 D。**历史收益不能保证在新的 R/SW/SA 下转移**；保留未处理/0-step、同 SW、D=I 对照，train-only 筛选，完整 validation 判成效。D 配对离线融合 `D^-1 W_up` 与 `W_down D`，不增加在线变换；当前权重/当前 R 坐标系的同输入 FP MLP 才是重构参考。

不要直接写死 layer1/channel1417 为通用算法：历史只在零基 layer1（第二个 Transformer block）做了全8192通道 RMS/wcol 排名，层的选择受历史敏感性影响，只试一个 top 通道；没有证明全层/其他通道无需处理。新产物先有限、系统地收集各层/通道统计，再按 train 数据选择有根据的局部 D，保存候选排名/未选原因，不能用旧归因替代最新诊断。

若完整 validation 仍高于原 BF16 +1，继续用户已授权的 fine-tuning 或蒸馏试验。先选符合实际可调度 GPU 资源、最终可离线融合/不增加在线算子的方法（例如能融合的有限适配；教师可首先选已有原 BF16 模型，序列化教师前向/离线 train logits 解决显存）。具体是否解冻哪些参数、教师/CE/KL权重与数据必须独立记录并有对照；不要将此报告为仍旧无教师 PTQ。不因为限制解除就同时换模型、码本、数据、精度边界。不能保证目标必然可达；诚实报告失败和剩余差距，不无限重复无证据搜索。

## 精度、评测与导出规则

- 112 backbone Linear 权重 W4，per-output-channel，signed `[-8,7]`，RTN `torch.round` ties-to-even；保留学到的 SW，不用最终 MinMax/GPTQ 覆盖。
- 96 非 down 输入 INT8 static per-tensor；16 down 输入 SP2 非均匀 A8 static per-tensor。R1/R2 离线融合，R3/R4 关闭；保留 embedding/lm_head/norm/KV 的现有高精度边界。GPU fake-quant PPL 不是手机 NPU native kernel 结果。
- WikiText-2 **完整 validation**：同 tokenizer/revision，`"\n\n".join` 原始文本；252852 tokens，123 个独立2048窗+948尾窗，逐窗重置上下文，**252728 个预测目标**；总 NLL 按预测 token 加权后 exp；BF16/KV16/use_cache=False。以现有 `validation_acceptance.py` 和真实结果核实，不从旧8×2048表推断。
- 若做三路归因：同一个模型自己的整数码、SW、SA、SP2 alpha 完全共享冻结，分别全A16、非downA8/downA16、完整A8；不重新量化/校准。跨条件恢复增益为条件效应，不相加当独立误差百分比。
- 检查最少覆盖：R/量化前函数等价和 D 配对 FP32/64 等价；I/形状/正值/边界；量化 signed code、ties、零行；无原地污染/候选隔离；SP2训练/评估 forward 相同；梯度确实到且实际更新目标参数；冻结/pack-reload一致；112/96/16覆盖。先小批GPU烟测检查梯度/显存，再自动正式跑，不停在烟测等批准。
- 每个候选独立从量化前有效 FP 权重生成代码；不能重复量化前个候选。冻结/导出失败、非有限或实际 OOM，停当前失败路径、记真实证据，修复后从最近有效产物继续。不可宣称 PID 启动就是实验完成。

## 已有产物与实测证据（Phase 2，不是 Phase 3 新结果）

下面路径均相对项目根。交接端本轮重读了 C launcher、7U1l7n/cohqXT JSON 和中心 loop 日志；其他具体历史值给出定位，新对话在需要使用时读原 JSON，不能称本轮重测。

| 结果 | PPL | 定位/说明 |
| --- | ---: | --- |
| 原 BF16 W16A16 | 13.634657725643203 | `runs/phase2/validation-acceptance-c-20260909.9WVDue/w16a16/result.json`；目标+1=14.634657725643203 |
| 原 C + learned SW + SP2 | 17.64239997588965 | 同 acceptance 目录；SP2 output-MSE alpha 搜索，不是 full-range INT8 |
| 原 C W4/allA16 | 16.231066669209774 | `runs/phase2/validation-diagnostics-c-20260913.fs4I9m/{report.md,summary.json}` |
| 原 C 非downA8/downA16 | 16.292812357698388 | 同上；旧68%/31%归因不适用于新产物 |
| Phase2 当前最佳 | **16.05715342622255** | `runs/phase2/down-d-search-20260913.7U1l7n/loop_results.json` selected_d，NLL2.7761544466207675；252728预测目标 |
| 该 D 的 I+同SW规则对照 | 16.118253028142952 | 同JSON identity_rule，NLL2.779952358261725；D净收益−.0610996019PPL |
| D前 parent | 16.11105493245762 | `down-d-search-20260913.qvjyDe/loop_results.json` |
| C FP变换完整性检查 | 13.647488162677584 | `down-d-search-20260913.Ie8eEI/loop_results.json`；比原BF16+.01283，不支持多点精度被FP旋转实现破坏 |

原 C 目录：`runs/phase2/learned-sw-c-20260909.ByFYAM/C/`；`rotation/R.bin`、`rotation/quant_scales.pt`、`rtn/w4_rtn_model.pt`。初始来自 `runs/phase2/w16a8-joint-r-sa-r12-s42/rotation`，SW 来自 `runs/phase2/w4aware-ab-20260909.6jhGG9/B/rotation/initial_quant_scales.pt`。C **不是从 B 最终训练结果继续**，是共同 R0/SA0/B初始SW起点；R0/SA0 本身已有 W16 R+SA 学习历史。这一点已从 C parent `launcher.sh` 再核实。

C 实际 W4-aware 训练17 R、96非downSA、112SW，100updates，**down A16**；非downINT8。历史两卡microbatch1×accum4=global8；RLR1.5/SALR1/SWLR.01、cosine/warmup10、seed42。`C/logs/rotation.log` 约229/235行和 parent `launcher.sh` 可核对。历史 `CustomJsonDataset` 行文本各自tokenize后直接拼IDs、2048分窗/drop尾，不插入`\n\n`；固定顺序，seed42不代表自动shuffle。训练protocol与validationprotocol各自固定并说明。

最佳包：`runs/phase2/down-d-search-20260913.7U1l7n/selected_d/packed_model.pt`。对象含 `weights` **26记录**、`D`（8192）、`base_checkpoint`；每记录 `packed`/`shape`/`scale`。补齐其余权重需原 C checkpoint。up1/down1 的 SW 已不同于 C，不能用“全部SW=C”断言或旧16-only/raw-record loader。FP reference 必须同样应用该D。D为layer1/channel1417的稀疏RMS方向，t1，min.999830842/max3.999322891，log均值约0；upSW=C/D，downSW=max(C,变换后BF16行absmax/7)；其它24记录保持。该包及I对照已有独立 verifier PASS（中心loop有记录）。

SP2冻结范围来自 `7U1l7n/down_scales.json` 并继承 parent；相对原SP2，layer0×16、layer2×1.5、layer3×1.25、layer13×1.25，其余同原（核对实际JSON不要手工推算export）。原SP2记录 `runs/phase2/down-codebooks-c-20260909.6YRIty/results/down_scales.json`。

最近失败：`down-d-search-20260913.cohqXT` 固定所有整数码/R/D/SA/alpha，只扩张down SW 1..1.125，beta0/.25/.5/.75/1；trainNLL分别2.725912757/2.730857445/2.730709095/2.730460386/2.730901443，选I。0次新validation，JSON `reused_frozen_parent=true`，16.05715是复用。不能据此说任何SW学习都无用。

最新已完成三路/恢复诊断属于 D前16.11105的qvjyDe：`down-d-search-20260913.EZPTam`：allA16 15.49485525361626；downA16 15.548379865182714；downW16 15.23049418236542；non-downW16 15.067222135354244。不能说是16.05715的诊断。

已证明的后处理代码与轨迹：`scripts/phase2/{precision_loop.py,fixed_grid_rounding.py,neighbor_rounding_loop.py,sp2_alpha_loop.py,non_down_rounding.py,sparse_d_sp2.py,fixed_code_sw.py}`。当前邻码算法是固定SW下 signedINT4 ±1坐标更新，analytic quadratic proxy，0/512/2048/8192有限步候选，实际trainNLL筛选；局部MSE变好但NLL变坏多次出现，不能省略NLL。早先 uniform-INT8 full-range D路线灾难退化，不能拿它替代本次SP2主线。

C4外部：`down-d-search-20260913.EH3Kju` 测的是更早Ds4vSf（Wiki2 16.12223），C4 PPL27.062581939332095，对同token BF16 21.843397624525274/旧C29.539535457544623。输入 `runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/input_tokens.pt`，1024×2048、2096128预测目标，未用于校准/选择。不是当前16.05715的外测，也不是完整C4。

中心实验历史：`runs/phase2/down-d-search-20260913.i56v3p/{loop.md,experiment_loop.json}`。其末尾 R-only 未批准提案/goal blocked 属于旧授权状态，本次用户已给新的三路线训练授权；不要再等旧问题回答或机械运行旧两臂。

## 现有训练代码陷阱

在新 worktree 阅读 `optimize_rotation.py`、`train_utils/rotation_calibration.py`、`train_utils/quant_linear.py`、`utils/quant_utils.py`、`utils/process_args.py`：

- W4-aware 会要求 learn_activation_scales；旧 helper 开非downSA学习时强制 down bits16。第三条必须显式支持16个SP2参数，不能仅传一个名字而实际仍A16。
- learnSA=false 可能触发初始/周期/末尾校准；learnSW=false W4-aware会刷新SW。精确控制各臂训练/冻结/初始化，导出时不能覆盖learned值。
- `export_rotation_scales` 当前按具体 quantizer 类筛选；新增SP2参数容易被漏存。实际重载/冻结比较必须包括16个alpha。
- `ActQuantWrapper` 的bits16旁路在wrapper层；检查烟测实际forward，不只inspectquantizer属性。
- LSQ/STE需要实际梯度与FP32参数更新，不以有grad属性/配置LR为证据。R正交坐标、当前旋转后的权重以及量化顺序不能和export不一致。

## 完成时交付

每臂初始化/训练数据/总updates/tokens/LR/校准范围与实际训练量化边界；三路线同口径完整validation NLL/PPL、10/100对照、初始化对照、固定probe曲线与解释；获胜路线后处理的逐项净收益与最终可重载包；必要时FT/蒸馏的独立结果/参数范围；相对原BF16的DeltaNLL/PPL与距+1目标差距；独立verifier结论；失败/复用结果明确标注。新对话继续执行，不把此handoff视为实验已完成。
