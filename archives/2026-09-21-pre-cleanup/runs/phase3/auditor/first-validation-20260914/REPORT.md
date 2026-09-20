# Phase 3 首个完整 validation 点：独立实验审计

审计日期：2026-09-14，完成时间约02:54 +08:00。角色：独立实验 auditor，不是主执行、算法开发者或应用代码 verifier。仅写本 auditor 目录，不创建主 goal。

## 授权和复跑边界

- 本次集合只有已完成的 A/B/C-Adam10、C-SGD10 的正式结果审计，以及原始 BF16、当前 C-Adam10 保存冷包的独立完整评测。
- **当前任务已完成并暂停，不持续跟踪或复跑中间 checkpoint。** 用户最新指定的下一次常规审计点为三条100-step路线全部完成；此后仅在阶段最终候选、最终胜者或影响结论的异常时介入。本次不提前开展这些后续审计。
- **不重复长时间训练，不重校准，不逐候选或逐 run 增加保险复跑。** 最终胜者节点的必要复现属于后续任务，不在本次提前执行。
- GPU 复测只检验已有结果的可复现性，与算法新增收益分列。没有新训练更新、尺度优化、参数筛选、后处理或新的候选。
- 主线程后续新增 `experiments/phase3/postprocess.py`，当前处于其独立 CPU verifier 流程；不属于本次已完成实验范围，不能因当前文件存在就声称 checkpoint10 使用过它。
- 未改应用或 tests、未运行 pytest、未安装或改变环境、未提交推送、未终止他人进程；仅使用指定 Python 环境和单张 GPU1。

## 结论矩阵

| 审计项 | 状态 | 范围与证据 |
| --- | --- | --- |
| A/B/C-Adam10、C-SGD10 正式结果的命令、配置、源码、数据和日志证据链 | PASS | `records_audit.json`，四个 checkpoint10 均有完整 validation 和完成事件 |
| 三条 Adam 路线的共同初值、前十步数据及记录的学习率 | PASS | 241 个初始参数逐张相等，windows 0..79，逐步记录的全部学习率完全一致 |
| 原 BF16 独立完整 GPU 重评 | PASS | `bf16.result.json`，与历史总分及两个分段分数完全一致 |
| C-Adam10 冷包独立完整 GPU 重评 | PASS | 匹配 FP32 RoPE 后，总 NLL/PPL及两个分段与正式结果完全一致，`summary.json` |
| 当前 checkpoint10 达到原 BF16 +1 目标 | FAIL | 当前最佳 22.08612814729231，大于 14.634657725643203；不是整个 Phase3 终局失败 |
| A/B/C-SGD10 的 auditor GPU 冷包重评 | NOT TESTED | 按授权仅检查它们的既有完整结果，不增加 GPU 复跑 |
| 100-step 终态排名、独立 10-step schedule、初始化收益、最终优胜包 | NOT TESTED | 本次不等待 100 更新完成，不参与自适应搜索 |
| `postprocess.py`、D/离散码后处理、FT/蒸馏、decode/KV8/NPU | NOT TESTED | 不在本次范围；CPU verifier PASS 不能代替这些 GPU/设备验收 |

## 正式 checkpoint10 结果

以下是主执行原始结果，不是 auditor 的新算法收益。每项均为 252852 输入 tokens、252728 预测 targets。

| 正式运行 | NLL | PPL | checkpoint10 的实际训练状态 |
| --- | --- | --- | --- |
| `route-a-adam-100-20260914a` | 3.5085414738242138 | 33.39951816978356 | 仍处于 W16 R/非 down SA 学习阶段，尚未开启 W4 联合 |
| `route-b-adam-100-20260914a` | 3.1054964908848506 | 22.32029804924722 | 从起点 W4/R/非 down SA/SW 联合，训练 down A16 |
| `route-c-adam-100-20260914a` | 3.0949497258037195 | 22.08612814729231 | 从起点联合 W4/R/SA/SW/down SP2 |
| `route-c-sgd-10-20260914a` | 3.09887567386752 | 22.173007569789448 | 相同 C 路线，尺度使用 SGD 配置 |

路径均为 `runs/phase3/<运行名>/checkpoint-0010/validation.json`。`static_w4a8.pt` 均实际存在，大小 2064318358 bytes；`training_probe.json` 和 `reload_check.json` 另存，未冒充完整 validation。

当前 C-Adam 比 B-Adam 低 0.2341699019549104 PPL，比 C-SGD 低 0.08687942249713743 PPL。**只支持这个 checkpoint10 的局部排名，不支持终态优胜、最佳优化器或 A 路线无效。** A 的评测已经转换为完整 W4A8，但其前十步训练并非完整 W4 联合。

所有十步均处于 **100-update schedule 的共同轨迹**。C-SGD 的运行停止预算为 10、schedule 为 100；不能把它说成独立十步 cosine。A/B/C 的计划预算各为 100，A 在 update index 50 切换 W4；此计划不是本次已完成的实测结论。

## 命令、进程和源码

主应用源码仅为 `/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。

- 分支：`phase3/joint-r-sw-sa-sp2`。
- 完整 HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`。
- 不是干净 HEAD。审计时的 tracked dirty diff 与 handoff 的 `inherited_source.patch` 按字节完全一致；五个 tracked 文件修改时间均为 01:32:31，早于被审计运行。
- `evidence/source.head`、`source.branch`、`source.status`、`source.dirty.patch`、`inherited_source.patch` 和 `source_snapshot/` 保存本次取证；未计算新哈希。
- 四个正式 run 各自保存的 `source/{run.py,common.py,quantization.py}` 彼此逐字节相同。正式保存的 `common.py`、`quantization.py` 也与本次实际重评导入源码相同。
- 正式 `run.py` 快照时间 02:02:55；当前版本后来增加可选 `--optimizer-scale-reference`，不是这四个 run 的已执行代码。差异保存在 `evidence/formal-to-live-run.py.diff`。后续 tmux launcher 和 postprocess 文件同样不回填到旧运行。
- C-Adam 前两步父运行的差异是 resume 保存方式/检查和进度记录；C-SGD 的父运行另早于 Adam 分支及保存包校验扩展。父版本快照及逐文件 diff 已保存，没有用当前文件冒充历史执行源码。

原始 `git status --short`：

```text
 M optimize_rotation.py
 M train_utils/quant_linear.py
 M train_utils/rotation_calibration.py
 M utils/process_args.py
 M utils/quant_utils.py
?? experiments/
?? tests/test_learned_sw_training.py
?? tests/test_phase3_joint.py
?? tests/test_w4_aware_training.py
```

| 正式运行 | GPU / PID | 启动记录 |
| --- | --- | --- |
| A-Adam100 | 1 / 3979478 | `runs/phase3/route-a-adam-100-20260914a.launch.json` |
| B-Adam100 | 5 / 3979813 | `runs/phase3/route-b-adam-100-20260914a.launch.json` |
| C-Adam100 | 7 / 3979995 | `runs/phase3/route-c-adam-100-20260914a.launch.json` |
| C-SGD10 | 3 / 3990988 | `runs/phase3/route-c-sgd-10-20260914a.launch.json`，已完成 |

上述文件的 `command`/`shell_command` 是真实启动命令，不是手工推定参数；本审计已复制到 `evidence/<运行名>/`，并在 `records_audit.json` 保存完整命令。运行日志中的 `full-validation`→`checkpoint-completed` 事件均与 PID、step10、JSON NLL/PPL 一致，没有失败文件冒充成功。

共同 Python：`/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python`，PyTorch `2.4.1+cu121`、CUDA `12.1`，RTX 4090。正式命令以新 worktree 的 `experiments/phase3/run.py` 为入口，明确记录 initial、route、steps、schedule、optimizer 及 C 的 resume 路径；默认参数由每个 `settings.json` 完整补齐。

## 模型、数据和配对控制

- 模型和 tokenizer：`cache/models/llama-3.2-1b-instruct`，现有 `.mv` 为 `Revision:master,CreatedAt:1740591574`。配置、tokenizer 小元数据及已有模型文件 size/mtime 已记录；没有下载升级。`master` 是现有缓存记录，不冒充可确认的上游不可变 commit。
- 数据缓存：`cache/huggingface/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3`，分别读取 train/validation Arrow。
- 训练逐 raw 行无 BOS/EOS tokenize 后拼 IDs：2435022 tokens，1188 个 2048-token 窗，丢弃 1998 尾 tokens；末八窗不进入常规训练，固定 probe 为 1180..1183。实际可用训练窗为 1180。
- 校准只用 train：seed42，在前 1180 窗范围的固定 `randperm` 中取 32 窗、每窗前 128 tokens。完整索引保存在 `data_check.json` / 原 `data.json`。这是 4096-token 校准，不是 validation 校准。
- 四个 run 的数据元信息与共同初始化逐字段一致。独立 tokenizer 重建的完整 validation IDs 与 `common.data_windows()` 输出逐 token 相等；BF16/C 使用同一保存 token tensor。
- 共同初始 `common-init-20260914a/initial.pt`：17 个未训练随机符号 Hadamard R、96 个 full-range SA、112 个 current-R per-row SW、16 个 train-only output-MSE SP2 参数。A/B 和两条 C 父运行的 step0，共 241 个参数逐张完全相等，不继承 Phase2 学过的 R。
- 读取 C 父运行前两步和正式 run 的 3..10 步，合并后恰为 1..10，不重复训练前两步；每步 microbatch1×accum8，windows 0..79，累计 163840 输入训练 tokens、163760 预测 targets。
- 三条 Adam 路线除了 route/output，以及 C 的已声明续训路径，配置无额外差异；前十步逐步记录的全部 LR 完全一致。设备不同是独立单卡运行，未改变 global batch 或样本数。
- R 使用 SGDG，base LR1.5；尺度 Adam 每张 tensor 的 base LR 为 `0.001 * initial_scale.mean()`，eps1e-12、weight_decay0，统一 warmup10/100-step cosine，梯度全局 norm clip1。A 的 SW 未激活期间即使记录 LR，也没有实际 SW 更新。
- C-SGD 是优化器**配置**诊断：R LR1.5，SA/SP2 base LR1、SW .01；它与 Adam 同时改变了尺度 optimizer 和尺度 LR 参数化，不应推广为纯优化器算法的普遍优劣。

## 实际精度开关和冻结产物

| 路线 | 前十步实际可学习组 | 训练 down | checkpoint10 冻结评测 |
| --- | --- | --- | --- |
| A | 17R、96SA；SW/SP2 无梯度和更新 | A16 | current-R MinMax W4，再用相同 train 校准 16 down SP2 |
| B | 17R、96SA、112SW；SP2 无梯度和更新 | A16 | 保留 learned SW/SA，train-only 校准 16 down SP2 |
| C-Adam / C-SGD | 17R、96SA、112SW、16SP2 均有有限梯度 | SP2 A8 | 保留 learned SW/SA/SP2，不做末尾 SP2 重校准 |

源码模式分支、`training.jsonl` 的实际梯度/变化与表格一致。C-Adam 每步 96 个 SA 和 16 个 SP2 标量均改变；C-SGD 虽全组有梯度，每步只有部分 SA/SP2 标量发生可见更新，不能仅凭有梯度宣称所有标量有效更新。

四个静态包都是 112 backbone Linear signed W4 `[-8,7]`、per-output-channel scale `(out_features,1)`；代码由 RTN ties-to-even 生成，保存布局为 `code+8` 的 offset nibble，不是声称已适配手机原生二补码内核。

96 个非 down 输入为 INT8 static per-tensor；16 个 down 为 SP2 static per-tensor。B/C 的保存 SW 与 step10 参数完全相等，所有非 down SA 与 step10 参数相等；C 的 SP2 alpha 与 learned scale×127 相等。A/B 的 16×50 个校准候选均按记录的最小 output-MSE 选范围，保存 alpha 与所选值的 FP32 表示相同。

R1/R2 在冻结权重中离线融合，R3/R4 关闭；没有在线 Hadamard。embedding/lm_head/norm 保留 BF16 高精度边界，K/V 不量化、use_cache=False。原始主权重被冻结，优化器只接收声明的 R/scale 组。这是 prefill fake-quant GPU 结果，不是 strict-static 手机 NPU、INT8 KV cache 或速度验收。

## 完整 validation 的实际计算口径

`common.full_validation` 仅以 `importlib` 导入项目 `scripts/phase2/validation_acceptance.py` 纯模块，调用新 worktree 的 `utils/eval_utils.py::evaluator`；不运行旧 acceptance main，也不导入旧 `repos/SpinQuant` 的主程序。

1. 同 tokenizer，validation raw 文本以 `"\n\n".join` 拼接，共 252852 input tokens。
2. 123 个互不重叠的 2048-token 窗，外加 start251904 的 948-token 尾窗；每窗重新开始上下文，batch1，use_cache=False/PREFILL。
3. 主段 targets=123×2047=251781，尾段 targets=947，总计252728；没有丢掉 948-token 尾窗，unscored_tail_tokens=0。
4. 原 evaluator 对 BF16 logits 做 `CrossEntropyLoss(reduction="none")`，**先得到每-token loss，再转 FP32**；FP32 window/segment 平均后取 FP32 `exp`，返回 float32 PPL 的 Python float。
5. acceptance 对两个分段的 float32 PPL 分别 `math.log`，按预测 target 数加权，最后 `math.exp`。不是先把 logits 转 FP32 的 CE，也不是对 PPL 直接平均。
6. 训练/probe 的 `token_nll` 使用 chunked **FP32 logits CE**；这四窗固定 train probe、单窗 reload probe和 minibatch loss 都没有被当成完整 validation。

因此历史原 BF16 13.634657725643203 的原 JSON 是历史测量，主线程本轮此前只是读了它；本 auditor 的 BF16 结果才是这次新 GPU 重测。不同计算口径的 training loss 不能直接指数化替代该 validation PPL。

## 独立 GPU 重评及异常如实记录

独立 tmux socket：本目录 `tmux.socket`，session 使用 `phase3-auditor-first-validation` / `phase3-auditor-c10-corrected`。通过显式 `exec env` 和独立 `run_serial.sh` 设置 CUDA_VISIBLE_DEVICES=1，不依赖主 launcher，也不借用他人会话。

每次启动前实查 GPU/进程：GPU1 首次剩余16236 MiB，纠正 C driver 后剩余16258 MiB。BF16 peak allocated 4.3361 GiB，C 约4.65 GiB；实测余量能覆盖主训练额外约3 GiB 评测峰值，没有把100% utilization当成拒绝条件。详见 `launch.resources.txt`、各 `*.gpu-before.txt`。

真实执行命令和 PID 保存在 `commands.log`、各 `*.runtime.json` / `*.pid`、`MAIN_THREAD_NOTICE.txt`。有效 BF16 PID25317；纠正后的 C10 PID59541。两模型在同一 GPU1 串行执行，没有跨卡加载一个模型。

BF16 按原 `ptq.py` 的模型构建方式读取源模型：`LlamaForCausalLM.from_pretrained(torch_dtype=bfloat16)`，解除 tied embedding 配置后从 embedding 克隆 lm_head；不加量化 wrapper、不做随机旋转、不做 norm fusion。历史日志同为 LlamaSdpaAttention。其完整 PPL/NLL 和两个分段 PPL 与历史完全一致。

C 冷载仅从保存 `static_w4a8.pt` 的 config 创建 BF16 eval 骨架，然后装入包中的全部112 packed weights/SW、96 SA、16 SP2及 high_precision 张量；不读训练 checkpoint 来重新生成权重、不调用 frozen_model/calibration/train。112 权重逐张与解包重构完全相等，高精度存储状态逐项相等，量化缓冲在评测前后不变。

**不能隐藏的 auditor 失败和额外开销：**

- 首轮 BF16/C 启动都在进入 evaluator 前因 auditor 自身检查失败：一项错误地检查 `torch.no_grad` 装饰器源码，另一项对 disabled legacy quantizer 的 NaN 哨兵用了不支持 NaN 相等的比较。已局部纠正；原日志/exit1 保存在 `driver-failed-attempt/`。不是主应用 FAIL，也不是完成的 GPU validation。
- 随后一轮 C 实际完成了全评测，但 auditor 用 `Model(config).to(bfloat16)` 将**非持久的 RoPE inv_freq** 也转成 BF16，结果 PPL22.066430771720142。这个分数不符合原模型构建口径，**判无效并排除，不是精度收益或成功复现**，保存在 `driver-invalid-rope/`。
- CPU 已证明上述错误使 32 个频率中31个改变；原构造显式保留 FP32，而 inv_freq 不在 state_dict 内。改用 `_from_config(torch_dtype=bfloat16)` 后仍只装载同一保存冷包，17个 RoPE inv_freq 均保留 FP32。这是审计 driver 修复，不是算法修改。
- 因此本次实际完整 evaluator 调用次数为 **3：2 次匹配口径评测 +1 次无效 C driver 评测**；并非隐去失败后声称只消耗了两次。没有重复 BF16 成功计分，没有扩大模型/候选集合，没有任何训练或重校准。

最终数值对照见 `summary.json`：

| 独立 GPU 评测 | NLL | PPL | 对原结果差值 | exit / evaluator用时 |
| --- | --- | --- | --- | --- |
| 原始 BF16 | 2.612614913352714 | 13.634657725643203 | ΔNLL=0，ΔPPL=0；两个分段也完全一致 | 0 / 38.4223秒 |
| 正确冷载 C-Adam10 | 3.0949497258037195 | 22.08612814729231 | ΔNLL=0，ΔPPL=0；两个分段也完全一致 | 0 / 73.6530秒 |

两项独立复现判 PASS。测量程序中的 COMPLETED（早期 BF16 的 PASS 字段）只表示程序完成，复现正式结果的 PASS 由 `summary.json` 中独立精确数值对照给出。无效 RoPE 运行即使原测量文件写了 PASS 也仍判无效，不列入上述结果。

## 解释边界和交付

- 当前 C 相对匹配原 BF16 的差为 ΔNLL=0.4823348124510054、ΔPPL=8.451470421649107；距 BF16+1 仍高7.451470421649107 PPL。
- 已有单次同 seed checkpoint 排名不是跨 seed 显著性或终态排名；验证集是开发选择集，不冒称未参与选择的外部泛化评测。
- 真实 GPU 的 uninterrupted/resume bitwise 等价没有另做独立训练对照；这里只核对保存父状态、实际数据/step/LR/日志链，不把 CPU verifier 的 resume 测试外推为 GPU等价实验。
- 没有缺少配置或日志的 run 被列入有效结果。旧 Phase2、短 probe、复用结果、失败尝试和本次完整 GPU 重测分别列账。
- 应用 CPU verifier 的54项历史限定 PASS保持其原范围；本 auditor不重做或授予应用代码 verifier PASS，不关闭主 goal/应用里程碑。

主要证据入口：`summary.json`、`records_audit.json`、`data_check.json`、`bf16.result.json`、`c-adam10.result.json`、`c-adam10.load.json`、`rope_cpu_diagnosis.json`、两个有效运行 `.log`、`commands.log`、`evidence/`。本审计到此首完整点为止，不等待或审查之后的100-step结果。
