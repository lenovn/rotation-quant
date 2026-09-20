# FT/QAT 200-step 独立实验证据审计

日期：2026-09-14。角色：原 Phase3 独立实验 auditor，非主执行、非应用代码 verifier。本报告是 **CPU/只读证据审计**，不是最终优胜包的独立 GPU 复现。

## 1. 结论与边界

**两组 200 更新及其完整冷包评测证据链：限定 PASS。参考组原 driver 无中断成功：FAIL；参考组主执行恢复评测证据：PASS。另发现 SGD0.1“没有完整25步分数”的记载与真实产物冲突：原陈述 FAIL，主端现已纠正文档并经本 auditor 窄只读复核。保留原 FAIL 及已纠正状态，不能将本报告概括为所有陈述无条件 PASS。**

| 审计对象/陈述 | 判定 | 依据或限制 |
|---|---|---|
| 中心组 1..200 更新、导出、原 driver 完整冷评测 | PASS | 逐步日志、336 项 Adam state=200、最终冻包、result/progress/log 一致 |
| 参考组 1..200 更新及最终导出 | PASS | 逐步日志、resume200、最终包及全量码/尺度核对 |
| 参考组原 driver 无中断完成全部流程 | FAIL | ENOSPC 发生于 code_changes 写入，尚未进入 full eval |
| 主执行 PID1493471 恢复冷载完整评测 | PASS | 原日志完整命令、源快照比对、两段评测、完成 JSON 与补写归档一致 |
| 两组 200 个实际 LR 字典、1600 个窗口访问相同 | PASS | 45,000 个 LR 值逐项精确相同；窗口逐项相同 |
| 同父量化初态来源/有效量化构造及初始 probe 配对 | PASS，限证据链 | 同一 D 包；中心重取码全量一致；reference 同 cell 投影；初始 probe/首更新损失一致 |
| 两组初始 FP master/reference 全量逐位数值复建 | NOT TESTED | 未保存初始 master 张量；本轮不重建原模型、不 forward；不能声称独立全量初态 tensor 比较通过 |
| 最终全 112 W4 / 96 static INT8 / 16 static SP2 | PASS | CPU 全码比对、224 个尺度与最终 resume 对应、HP 字节不变 |
| 冻结原 FP 主权重 PTQ | FAIL（不适用的分类） | 实际优化了 112 张 FP32 master 主权重，使用 teacher KL+CE；是 FT/QAT/distillation |
| SGD0.1 无完整25步分数 | FAIL，主端已纠正文档 | 完整 validation.json、checkpoints.json 和评测后训练中断栈反证，详见第7节 |
| SGD0.001 完成100但不改善；continue800 只有部分更新 | PASS | 两者分别有完整100和部分201..234/resume225的证据，不能混入两组200排名 |
| 达到原 BF16+1 PPL 目标 | FAIL | 最好 15.578922160109332，仍高于 14.634657725643203 |
| 本 auditor 新 GPU 复现 / CPU forward / pytest / 手机验收 | NOT TESTED | 本次均未执行；不能把证据审计或历史 verifier PASS 当 GPU/手机验收 |

本轮执行 **0 GPU、0 模型 forward、0 训练更新、0 重校准、0 pytest、0 算法搜索**。仅用既有 Python 对原 JSON/log/source 只读检查、CPU mmap 读取保存 tensor 并作取码/相等性统计。成功脚本的 `torch.cuda.is_initialized()` 为 false。未创建主 goal、未新设 auditor、未修改应用/tests、未安装、未改挂载/环境、未删除、未提交推送、未操作其他进程。

写入限定本目录。已阅读适用根 AGENTS、STATUS/SPEC/PLAN/LESSONS，以及更新后的 Phase3 HANDOFF/STATUS；未发现本任务路径下附加 AGENTS。未重复 Phase0 扫描。根协调文档及主 STATUS/HANDOFF **未被本 auditor 修改**；本目录保留其只读证据副本。

## 2. 真实命令、源码与 dirty 链

源码根固定为 `/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`，完整 HEAD **24918316ed594848d4de797c356b120f2a4ee0f3**，branch `phase3/joint-r-sw-sa-sp2`。`GIT_OPTIONAL_LOCKS=0` 的原始输出见 `evidence/head.txt`、`evidence/branch.txt`、`evidence/status.txt`、`evidence/dirty.patch`。

当前 `git status --short`：

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

不能用相同 HEAD 冒充 clean source：各运行保存的 `source/tracked.diff` 与当前以及首节点审计保存的 tracked dirty patch 逐字节一致；`experiments/` 属于未跟踪代码，故另核查每次运行的实际 `source/*.py` 快照。两组200快照的 `launch.py/run.py/common.py/postprocess.py/quantization.py/tracked.diff` 全部逐字节一致，只有 `distill.py` 有声明内差异。

`evidence/center-to-reference.distill.diff` 完整保留差异：新增 reference CLI/读取父 D 变换/cell 投影、prepare_student 的可选初值、train 的初始化记录、resume 的 reference 元数据。AST 核查确认 TrainableQuantLinear、parameter_groups、teacher_model、loss、optimizer、load_resume、export/evaluate_checkpoint 定义一致。`main` 因 CLI 增项不同，不掩盖为整文件相同。当前 distill.py 等于 reference 快照，不等于旧中心/SGD快照；审计旧运行使用旧快照，不把后来新增路径当成旧运行执行过。

`utils/eval_utils.py`、`eval_utils/modeling_llama.py`、`utils/quant_utils.py`、`train_utils/quant_linear.py`、`train_utils/modeling_llama_quant.py`、`train_utils/optimizer.py` 与首审快照逐字节一致；纯模块 `scripts/phase2/validation_acceptance.py` 同样未变。这里核查源码证据，不执行其代码，也不冒充应用实现 verifier。

两组真实 launcher argv 保存于 `evidence/actual_training_commands.txt`，原 `.launch.json` 副本包含完整 `shell_command`、显式 child_environment、tmux socket/session、原 GPU 快照及 PID。共同参数：

```text
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -u
/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint/experiments/phase3/distill.py
--parent /home/dongpeiyan/projects/rotation-quant/runs/phase3/post-b100-d-20260914a/static_w4a8.pt
--steps 200 --schedule-steps 200 --weight-optimizer adam --weight-lr 0.00002
--offload-teacher-body --relative-scale-lr 0.001 --temperature 1 --ce-weight 0.1 --data-start 800
--checkpoints 10 25 50 100 200 --validation-steps 25 100 200
```

各自 `--output` 不同；仅 reference 增加 `--reference-state .../route-b-adam-100-20260914a/checkpoint-0100/state.pt`。剔除这两个声明差异后，真实命令及 settings 精确相同。默认实参为 accumulation8、warmup10、resume_every25，两组均非 resume 起跑。中心 GPU1/PID1105455，参考 GPU7/PID1254098；均 tmux socket `rotation-quant-phase3`、session `phase3-<run名>`。这是历史主执行运行记录，本 auditor 未启动这些进程。

## 3. 模型、teacher、父 D 与初态配对

**Reuse**：原 BF16/model/tokenizer/data 来源及完整协议复用 `runs/phase3/auditor/first-validation-20260914/REPORT.md:1`；B100 原路线来源复用 `runs/phase3/auditor/three-routes-100-20260914/REPORT.md:1`。本轮重新比对 `.mv/config.json/tokenizer_config.json/special_tokens_map.json` 的字节，以及 model.safetensors/tokenizer.json 的 size/mtime，均与首审证据一致；不声称重新扫描了原模型所有权重或重新 tokenizer 全数据。原 BF16 **13.634657725643203** 是旧首审已独立精确复现值，本轮没有重测，也不是旋转/norm-fused student 的“BF16 baseline”。

源模型为 `cache/models/llama-3.2-1b-instruct`，缓存 `.mv` 为 `Revision:master,CreatedAt:1740591574`。两组记录 PyTorch2.4.1+cu121/CUDA12.1/RTX4090。main seed42、4 CPU threads、TF32关闭。

Teacher 的实际构造见参考快照 `distill.py:103`：原模型 BF16/from_pretrained、sdpa、显式 untie 后把 lm_head 复制为 embed_tokens、use_cache=False、requires_grad_(False).eval()；没有 R、norm fusion 或 quant wrapper。teacher body 在各 microbatch forward 前搬到 student device、随后 CPU offload；teacher head/hidden 的放置不改变声明损失。这里只读核查实际构造路径及两组同源，不把 teacher 输出称作本 auditor 新计算。

损失见参考快照 `distill.py:115`：next-token student/teacher logits 先转 FP32，T=1，**0.1 CE + 0.9 KL(teacher||student)**，head 分128-token块、按2047 targets归一；teacher no-grad。它不是原冻结权重 PTQ，也不能拿该训练 CE/probe 代替正式 BF16-logits full-validation NLL。

父包 metadata、`post-b100-d-20260914a/result.json`、selected diagonal record 与 reference `master_initialization.json` 一致：从 B100 静态包出发，`model.layers.1.mlp.down_proj`、channel1417、factor2、alpha564.63525390625（父 B100 alpha/2）。本轮 CPU 进一步核实：相对 B100 只该 down 权重的该输入列有码改变；只该 up 权重的 SW 第1417行变为一半；只该 down 的 SP2 alpha 改为一半；其他权重码/尺度及 HP 保持。源链是 B100 → 保存的局部 D 包，不是未经证据的另一模型。

Reference 不是把原 BF16 权重不加变换直接塞回 D student。快照 `postprocess.py:42` 从原模型构建、norm fusion、读取 B100 state 的保存 R，生成 current-R 浮点权重；`rotated_weight` 使用 float64 矩阵乘后落回模型 BF16 dtype。`distill.py:60` 再按父 D 记录对 up 第1417行除2、down 第1417列乘2，然后转换 FP32 并投影入父 INT4 cell。保存的 B100 state metadata 实读为 routeB/update100/training_tokens1638400，与复用的百步报告相符。

初态证据分层：

- 两组读同一父包、SA/SW/SP2/HP 相同。中心从 BF16 反量化权重提升至 FP32，不把它误写成无舍入的理想 FP32 网格中心。本轮 CPU 检查全部 **973078528** 个父码经过此 BF16→FP32重取码路径，差异为 **0**。
- Reference 使用 `[code-0.49,code+0.49]` cell 投影，饱和端点向外不限制，保留父 INT4 码的构造；FP master 的 cell 内残差与中心不同，是声明的比较变量，不是额外训练/校准。
- 两组保存的初始4窗 probe 完整对象相等：NLL **2.954445779323578**，PPL **19.191083676540114**，8188 targets；首更新8个 microbatch 的 objective/CE/KL 记录精确相同。
- 不外推为全训练轨迹逐位相同：首更新 pre-clip gradient norm 已分别为 **19.477365493774414 / 19.478227615356445**。本轮未重建完整 FP reference、未保存/比较两组初始全量 master 快照、未 forward；这些全量独立实测项为 **NOT TESTED**。支持“同父有效量化初态、不同 master 初值”的证据链，不支持“FP masters 或所有梯度都相同”。

## 4. 200 个实际更新、LR 与数据预算

两组 `training.jsonl` 均恰为 **1..200**，无缺步/重复；最终 `resume.pt` metadata=200，112 个权重 Adam state 与224个尺度 Adam state **全部 step=200**。每步记录336张梯度覆盖：W112、SA96、SW112、SP216；每一步112张 master 的抽样64位置均出现变化。抽样只证明每张有变化，不冒称所有元素每步都变化。

实际可训练集合及最终 resume 完全一致：112张W、973078528元素；112张SW、376832元素；96张SA、96元素；16张SP2 scale、16元素；合计336张、973455472个FP32元素。无 R optimizer 组；R/D已折叠，norm/embedding/head不在可训练集合。

两组200×225组的 **45000 个日志 LR 值逐项精确相同**，且全部满足保存 optimizer 的 `initial_lr × schedule(step-1,200,10)`。W=Adam2e-5、eps1e-8、weight_decay0、betas(.9,.999)；尺度Adam eps1e-12，224个独立initial_lr。尺度 initial_lr 本来取 GPU 初值 mean；本审计使用保存的实际值，不重演 CPU mean 推导冒充 bitwise 相同。第一步W LR=2e-6，最后一步=1.3669500753099586e-9，按该实现最后一步不等于0。

数据 `data.json` 五个相关目录都与首审已核定数据对象相同：WikiText-2 raw-v1/train，逐行无BOS/EOS tokenize后连接，2435022训练tokens →1188个2048窗、丢1998-token尾。最后8窗排除出优化访问；实际训练池1180窗，probe1180..1183，calibration seed42的32×128列表仍被 data_windows 返回，但 QAT `train_windows, _, probe, validation` 丢弃 calibration，**不重新校准父包**。

每组200次更新×8个microbatch×2048 = **3276800训练输入tokens、3275200 next-token targets**。索引逐条核实为 `(800+(step-1)*8+microbatch)%1180`：第一步800..807，最后一步32..39；1600次访问、1180个不同窗，两组完全相同。这里存在规定的回绕/重复，`data_start=800` 不等于200步都使用从未见过的新数据，也不能声称与B100训练窗完全不重叠。probe与正式validation不参与梯度。

## 5. 最终冻包与完整计算口径

两组 `checkpoint-0200/static_w4a8.pt` 均由独立 CPU mmap 冷读；并读取最终 resume FP32 masters/尺度。对全部112层逐行重取码，与包中保存的 packed nibble 逐元素比较：**两组都0差异**。这不是模型 forward、训练或重校准。

CPU 结果：112个正确形状的W4、uint8双nibble packing、每输出行FP32正SW；96个非down静态INT8/per-tensor正SA；16个down静态SP2/per-tensor正alpha。SW/SA逐项等于resume，SP2 alpha逐项等于resume.scale×127。保存的74个HP state条目（含未启用量化器缓冲）与父包逐字节相等。无混入W16 backbone层；embedding/head/norm仍为规定HP边界。该全码比对只说明导出一致，不等于独立 GPU cold-forward 已测。

| 最终包 | CPU独立统计 changed codes / total | 与保存 code_changes.json |
|---|---:|---|
| center200 | 1566408 / 973078528 | 112项逐项一致 |
| reference200 | 36513943 / 973078528 | 112项逐项一致；主执行后补CPU统计，不是原driver成功写出 |

实际 full eval 调用链：`evaluate_checkpoint` 导出 → `load_static(saved static_w4a8.pt)` 新建并冷载 → `common.full_validation` 动态导入旧 acceptance 的纯函数（不执行其main）→ **新 worktree** `utils.eval_utils.evaluator`。load_static 从原BF16 pretrained壳开始再装入保存W/HP/A，不重训、不重校准、不套训练FP32 CE；RoPE等非持久buffer来源沿用已核定冷载路径。最终运行是 BF16 fake-quant/dequantized prefill，KV16、use_cache=False，无在线R；不是原生INT4/INT8 kernel、量化KV cache、decode或手机NPU结果。

完整口径经源码重读、未变源码比对和每个完成 JSON 的算术核查：

- 252852 输入tokens，123×2048 +948尾窗；独立窗每窗少1个预测，251781+947=**252728 targets**，unscored_tail_tokens=0。
- evaluator 对 **BF16 logits** 做 `CrossEntropyLoss(reduction='none')`，随后 per-token loss 转FP32、分窗mean，再FP32 segment mean与exp，输出float32 segment PPL。
- acceptance 对两个 segment PPL 做Python `math.log`，按各段预测targets加权NLL，最终 `math.exp`。两组所有已存25/100/200及SGD完成JSON的分段/尾窗/加权恒等式全部匹配；没有将PPL直接平均或丢尾窗。
- 4窗固定训练probe用FP32-logits CE，只是probe；第10/50步仅probe不能列为完整validation。两组200完整结果与旧BF16/B100/D结果在评测口径上可比，但训练预算/方法不同，不能写成等预算PTQ收益。

## 6. Reference200 ENOSPC 与主执行恢复

原参考 driver PID1254098 真实完成200更新并导出包；原日志 `distill-b100-d-adam2e5-ref-200-20260914a.log:232` 记录训练200，`:233` 记录export阶段，`:249` 明确在 `code_changes.json.tmp` 创建时ENOSPC。快照中写code_changes位于 `load_static`/full_validation **之前**。随后 failure.json.tmp 也ENOSPC，故没有failure.json不代表没失败。

原 `progress.json` 仍是export-for-validation/200；原 `checkpoints.json` 仅10/25/50/100，没有伪造原200完成账本，也无原result.json。此旧状态不是已恢复测量无效的理由，更不能覆盖成“原driver无中断成功”。

主执行另用 GPU7、tmux `phase3-ref200-recovery-eval`、PID1493471，**只冷载同一已导出包做完整eval**。原日志三条物理行：preflight`:268`、start`:269`、completed`:284`；完整 argv（含原样Python `-c`代码）、预启动GPU快照、PID、源路径/HEAD、common/postprocess/quantization与运行快照逐字节比对均在原事件中。这里只读取历史GPU快照，本 auditor 没有NVML/GPU调用。

CPU逐对象比对：三条事件与 `recovery_evaluation.json` 精确相同；preflight命令尾部与start从 `/proc/self/cmdline` 获取的实际命令相同；start/completed同PID1493471；completed的数据对象与共同数据一致；完整结果与后补 `checkpoint-0200/validation.json` 精确相同。此前原driver日志的full-validation只出现25/100，不存在原200 full-validation。

恢复测量记录用时40.91183801100124秒（含数据/冷载）、peak allocated4.650002GiB左右。该新GPU测量由**主执行**实施，不是本 auditor 复现。后补validation/recovery JSON只是证据归档，后补code_changes是CPU统计；二者都不增加GPU测量次数，也不是新算法收益。本轮独立全码比对进一步确认该完整包与保存的训练200 masters一致。

## 7. 其他 FT 分支与证据冲突

### SGD0.1：中断100成立，但“无完整25分”不成立

目录 `distill-b100-d-sgd01-100-20260914a`：原命令SGD0.1/momentum.9、scaleAdam、100-step schedule；训练日志1..25、resume25，failure=KeyboardInterrupt/PID977986。训练100未完成，失稳配置必须保留，不能列作完成100/200。

然而 `checkpoint-0025/validation.json:1` **存在完整两段252728 targets结果**：NLL **8.81122709329167**、PPL **6709.146967144104**。同目录 `checkpoints.json:1` 同样记录step25的完整validation_nll/ppl及changed_codes1754480。原log有两段 evaluator 结束；随后 traceback 在旧快照 `distill.py:254` 的 **train → distillation_loss → student backbone**，不是 evaluator。按循环顺序，这是评测返回后的下一更新训练前向中断（可推断进入第26步但未完成更新）。

因此 `early_stop.json` 所写“step25 was interrupted; do not count partial evaluation”及审计当时主 STATUS 的相同说法与产物冲突（旧主STATUS副本保留于 `evidence/phase3-STATUS.md:127`）。**FAIL的是“没有完整25步分数/中断发生在25评测”的证据陈述**；不是要求重跑，也不是把失败配置提升为有效候选。本审计只指出冲突，未修改原 early_stop/failure/progress/STATUS。主执行的停机意图来自early_stop记录，本 auditor 未重新执行或独立复原其发信号/UID核查过程。

**纠正回执，2026-09-14 08:42:41+08:00**：主线程确认实证冲突并纠正文档；本 auditor 只重读相关文本，确认当前 `runs/phase3/STATUS.md:127`、`runs/phase3/STATUS.md:158`、`runs/phase3/RESULTS.md:60`、`runs/phase3/RESULTS.md:67` 已明确完整25分及其后训练前向中断、100未完成，且保留原误判的纠正说明。旧日志/early_stop证据不删，原FAIL保留为已纠正问题，不再传播“无完整25分”。本轮未因此重复CPU tensor扫描或GPU测量。另一个清理角色的盘点授权不属于本 auditor；新 `sequential_postprocess.py` 的另行代码验证也不属于本次QAT历史源码审计，不因它后来出现而认定旧run执行过。

该25分可作为原主执行完整测量的失稳结果保留；本轮没有为SGD25另做全量冻包/独立GPU验收。无论如何不能混入“完成200”的排名，不能隐藏失败分数。

### SGD0.001：完整100，但不改善父D

`distill-b100-d-sgd0001-100-20260914a` 的实际1..100、resume100、224尺度Adam state100、result/progress completed、原日志完成事件均在。25步PPL17.07365575720706；最终100步 **NLL2.8425382978723266 / PPL17.15926563313684**、252728targets，比父D的17.003450334865143差。完成与无改善的陈述PASS；没有伪装成中断，也未列为优胜。此100-step schedule与Adam200中的checkpoint100不是纯优化器等调度对照。

### Center continue800：只有201..234，持久resume225

`distill-b100-d-adam2e5-continue800-20260914a` 的真实 `--resume` 指向center200/resume.pt，metadata/optimizer恢复链相符；源快照仍是已有reference-capable代码但本任务未传reference-state。实际目录只有34条更新 **201..234**，新增557056输入tokens/556784targets；progress停233，随后原日志写progress.json.tmp ENOSPC，连failure.json.tmp也无法创建。

独立CPU读取最后成功原子保存的resume：metadata **225**，112+224项Adam state全部225，不把234条训练日志当成可恢复234权重。无新validation/result/checkpoint结果，故不能制造400/600/800完整排名。真实命令 checkpoints为300/400/600/800、validation_steps为400/600/800。

该命令明确把schedule_steps延到800；201的W LR=1.7278788103694944e-5，所有已记录LR匹配新800 schedule。不是保持200 schedule不变的自然延长；报告只确认此事实，不参与后续调度/算法建议。

## 8. 完整结果比较与暂停

| 类别 | 完整PPL | 完整NLL | 证据状态 |
|---|---:|---:|---|
| 原BF16 | 13.634657725643203 | 2.612614913352714 | reuse 首节点独立审计；本次未重测 |
| PTQ B100 | 17.11711490993314 | 2.840078834896178 | reuse 三路线100审计 |
| PTQ 父D | 17.003450334865143 | 2.8334162843368644 | 本次父包来源/保存结果只读取证 |
| QAT center200 | 15.666180015269509 | 2.7515042497043507 | 原driver完整冷评测证据PASS |
| QAT reference200 | **15.578922160109332** | **2.7459188570552207** | 200训练/导出 + 主执行恢复冷评测证据PASS |

以上完整结果均252728targets。本次两组200内reference低 **0.08725785516017659 PPL / 0.00558539264913005 NLL**，在同200实际LR/窗口及正式协议下可比较；这是单次同seed实验比较，不是统计显著性或最终全goal优胜宣告。仍高于BF16+1目标 **0.9442644344661293 PPL**。QAT收益与PTQ类别、主执行恢复测量与auditor复现必须分别列示。

存储ENOSPC来自两个原driver traceback。`evidence/storage-df.txt` 记录本次只读df的两分支余量约88.7/108.8GiB；mergerfs minfreespace100GiB策略来源复用主STATUS记载，本轮未重新验挂载参数、未改挂载/释放空间、未启动大实验。小证据目录不复制原multi-GiB模型包、resume或训练JSONL，只保存小日志/配置/源码副本及比对结果；原路径与size/mtime记在summary，审计读期间这些已核文件元数据未变。

机器明细：`summary.json`、`findings.json`；执行记录：`COMMANDS.md`、`audit.log`、`finalize-recheck.log`；原始副本和无hash的source/dirty证据：`evidence/`。保留首次小CPU探索误用recovery目录及AST比较集合修正的记录，不冒充无误差尝试；两者均未涉及GPU/forward，详见COMMANDS与findings。

**本次结束后暂停。不自动追踪、复跑中间checkpoint，不扩候选集合，不重训/重校准，不参与自适应搜索。下一次实际GPU审计仅在主线程明确给定最终候选/优胜静态包及来源链后，在新授权范围内由本角色继续；本报告不提前授予该GPU动作。**
