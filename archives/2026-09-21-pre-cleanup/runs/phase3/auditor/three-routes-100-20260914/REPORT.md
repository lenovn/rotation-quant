# 三条100-update路线：独立只读证据链审计

日期：2026-09-14，约04:57 +08:00。沿用首节点独立实验 auditor；不是主执行、应用代码 verifier 或新主 goal。**本次完成后暂停。**

## 结论

| 项目 | 状态 | 限定范围 |
| --- | --- | --- |
| A/B/C100 正式完成、源码和数据/更新/调度证据链 | PASS | 三条各100更新，C含已记录父运行前两步，没有重复计数 |
| A：50 W16后50 W4实际开关 | PASS | 源码分支、唯一切换事件、100步梯度/变化、保存Adam计数相互印证 |
| 三个冻结包112 W4 /96 INT8 /16 SP2 | PASS | CPU只读检查保存包字段、覆盖、形状、正值和保存尺度；不是新前向/冷包GPU重评 |
| 完整尾窗口径、10→100可比性、百步终态排名 | PASS | 同100-step schedule、同数据/更新预算，排名 **B<C<A**；A步数比较包含阶段切换 |
| 本阶段原BF16+1目标 | FAIL | 最佳B100仍超过阈值2.4824571842899363 PPL；不表示整个Phase3已经终局失败或完成 |
| 新GPU复现、CPU pytest、初始化终态、后处理及最终包 | NOT TESTED | 本轮均不执行或纳入结论 |

**运行边界：0次GPU评测、0次模型forward、0次训练/校准、0次pytest、0次算法搜索。** 仅CPU读取JSON/日志/源码及保存张量；应用、tests和既有产物只读，报告及取证副本仅写本目录。没有新baseline、hash、门禁、分支、安装、提交、推送或进程干预。

## 三条正式结果与10→100

所有结果均来自各 run 的 `checkpoint-0100/validation.json`，不是用户提示直接抄录，也不是训练probe。三个 `progress.json`、原始日志末尾 `checkpoint-completed` 与 `completed`、最终 `state.pt` / `resume.pt` 一致。

| 路线 | 100 NLL | 100 PPL | 10 PPL | PPL100−PPL10 | NLL100−NLL10 |
| --- | --- | --- | --- | --- | --- |
| A | 2.96737886227514 | 19.440895490065373 | 33.39951816978356 | −13.95862267971819 | −0.541162611549074 |
| B | 2.840078834896178 | 17.11711490993314 | 22.32029804924722 | −5.203183139314081 | −0.2654176559886725 |
| C | 2.849904697477057 | 17.286134349848403 | 22.08612814729231 | −4.799993797443907 | −0.24504502832666253 |

- 本阶段B优于C：−0.1690194399152638 PPL、−0.009825862580878919 NLL；B优于A：−2.323780580132233 PPL。
- B/C训练前向的精度策略各自在1..100内不切换，同一轨迹继续90更新后的完整validation确实改善。不是独立10-step cosine与100-step cosine的比较。
- A10还在W16训练阶段，A100已经历50 W4更新；因此A的改善是**延长训练并执行分阶段路线**的联合效果，不能说成固定训练量化模式下的纯步数效应。
- 三路线最终都按相同完整静态W4A8格式评测。A/B每个完整checkpoint各自按同一train-only程序确定down SP2范围；C保留所学范围。比较的是声明的完整路线，不是强制三臂共享最终R/SA/SW/SP2。
- 公平性针对相同总更新数、输入训练tokens、样本顺序及共同LR轨迹；不是相同W4训练更新数或相同FLOPs/用时。A有50次W4更新，B/C有100次，这是路线定义而非隐藏预算。
- 10步时C略优于B，100步时B略优于C；不能沿用首节点排名。这里只确定共同seed42和已声明配置下的本阶段胜者，不推论全部阶段比例、跨seed显著性或全goal最终胜者。

## 源码快照与dirty链

源码限定 `/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。

- 分支：`phase3/joint-r-sw-sa-sp2`。
- 完整HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`。
- 新取证 `evidence/{head.txt,branch.txt,status.txt,dirty.patch}`。当前tracked dirty diff逐字节等于首节点保存diff，也等于handoff的 `inherited_source.patch`，不能只用HEAD冒充完整执行源码。
- 三条run自带 `source/{run.py,common.py,quantization.py}` 均与首节点保存的对应快照逐字节相同；这三条正式路线仍使用其启动时源码，不把后来文件变化回填为百步运行内容。
- 运行快照的common/quantization与当前源码相同；当前 `utils/eval_utils.py`、eval/training Llama、quant_utils、quant_linear、optimizer及纯acceptance模块均与首节点快照按字节相同。
- 当前run.py后来增加可选optimizer-scale-reference，但三条正式settings没有使用它；首节点已保存该差异。新 `postprocess.py`、新launcher及其后续候选不属于这三条百步轨迹，本次不作算法审查。
- C前两步来自 `smoke-c-adam-20260914a`；首节点已核定父版本与正式版本的训练语义和resume链，本次重读父training/settings并合入1..100账本，不重新开展GPU不中断/续训等价实验。

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

源快照、命令、settings、完整training.jsonl、原始日志及checkpoint10/50/100的小JSON均取证至 `evidence/<运行名>/`。大冻结包保持原位，只以CPU `map_location='cpu', weights_only=True, mmap=True`读取，没有复制、修改或新解包模型前向。

## 实际命令、完成链和100步账本

| 路线 | 正式运行名 | 原GPU/PID | 本地更新记录 | 合并更新记录 |
| --- | --- | --- | --- | --- |
| A | `route-a-adam-100-20260914a` | 1 /3979478 | 1..100 | 100 |
| B | `route-b-adam-100-20260914a` | 5 /3979813 | 1..100 | 100 |
| C | `route-c-adam-100-20260914a` | 7 /3979995 | 3..100 | 父1..2 +98 =100 |

每个 `*.launch.json` 保存真实argv与shell_command；与settings/progress的PID和源码路径一致，完整命令也列入 `summary.json`。Python复用 `/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python`。不将本次读取这些历史GPU字段说成本轮运行GPU。

本轮逐条确认：

- 每条更新序列恰为1..100，无缺号、重复或超额更新；每条microbatch1×accum8，输入2048。
- 每步训练tokens16384、预测targets16376；100步累计 **1638400输入tokens /1637600预测targets**。
- 三条精确消费相同的800个训练窗0..799，逐步window_indices相等；没有在A切换或C续训时重置数据顺序。
- 最终state/resume metadata均为100，state记录1638400训练tokens；241个保存参数在最终state与resume逐张相等。
- 三个 `progress.json` 都为completed/100；日志唯一完整validation100、checkpoint-completed100、最终completed事件与各自JSON NLL/PPL相符，无failure文件冒充完成。
- 三条settings与首节点副本完全相同，data.json与首节点已核定共同数据一致。settings明确steps100、schedule_steps100、warmup10、accumulation8。

### 学习率及保存optimizer计数

以**保存的optimizer param_groups.initial_lr**核对每个日志LR；不是用新CPU reduction重新定义训练时GPU求得的scale均值。三条路线225个LR键（R组加224个尺度tensor）在全部100更新逐步精确相等，且均精确满足原源码global warmup/cosine。

- R base LR1.5；尺度Adam base LR=`0.001*初始scale均值`，eps1e-12、weight_decay0，原配置不变。
- 1..10更新warmup；后续factor=`0.5*(1+cos(pi*((update_index-10)/90)))`，update_index为0..99。切换W4时不重启global clock。
- R LR示例：step1=0.15000000000000002，step10=1.5，step50=0.9059337681133194，step51=0.8802361332501979，step100=0.0004568797356781784。
- 保存Adam状态：A的96 SA各100步、112 SW各50步、16 SP2无state/0步；B的SA/SW各100步、SP2为0；C全部224张尺度各100步。这进一步区分“配置有LR”和“确实执行更新”。

只读取证脚本最初用CPU重新求SW均值并要求与GPU初始化的LR逐位相等，造成假不匹配；54个初始LR的最大相对差仅1.5674059805625973e-7，另需保持Python浮点表达式原运算顺序。已改用真实保存initial_lr并保持原顺序后，全部100步精确一致。`audit-initial-lr-check.log`/`summary-initial-lr-check.json`保留为**auditor算术检查已纠正记录**，不是主run FAIL；期间没有任何实验重跑或GPU调用。

## A50 W16 →50 W4 的真实证据

保存源码 `source/common.py:255`：A在update_index<50时weight bits16，之后bits4；`source/run.py:239`在index50执行current-R SW重置，再进行第51次update。

- 唯一 `A-switch-to-W4` 日志event标记step50；这是已完成50更新后的切换点，不是第50次更新已经用W4。
- 更新1..50：SW gradient_tensors均0、changed_elements均0。
- 更新51..100：SW gradient_tensors每步均112、changed_elements每步均大于0；step51恰有376832个SW元素改变，最终保存Adam各SW step=50。
- step50消费windows392..399；step51消费400..407；后半段合计50×8=400窗，不重复前半段样本。
- A的16 down SP2训练参数全100步都无梯度和变化、Adam step0，与训练down A16分支一致；最终SP2是独立冻结评测模型上的train-only校准，不是暗中训练down SP2。

B/C全部100步SW都有112张梯度及实际变化；B的SP2全程未学习，C的16个SP2全程有梯度和组内实际变化。这里由源码、日志及保存optimizer交叉核实，不把单独bits配置当作已测实际训练。

## 最终冻包覆盖与来源

三包位于各 `checkpoint-0100/static_w4a8.pt`，均2064318358 bytes；metadata route/step=对应路线/100。

- 精确核对16层×q/k/v/o/gate/up/down七类=112个weight记录和112个activation记录的**完整名称集合**，不只是计数。
- 每个weight shape符合源模型：q/o2048×2048，k/v512×2048，gate/up8192×2048，down2048×8192；packed为uint8，字节数×2等于权重元素数；SW为FP32 `(out_features,1)`、有限且正。
- offset-nibble存储按已核定源码解释为signedINT4 `[-8,7]`，RTN ties-to-even。未把这称为部署端原生二补码或整数kernel包。
- **三条最终包的全部112 SW都与各自step100 learned SW逐张相等，包括A。** A100没有再用final MinMax覆盖learned SW；其step10的current-MinMax导出是不同阶段的声明行为。
- 96个非down的format均int8，SA为FP32单标量、有限正值且与step100参数相等；16个down的format均sp2、alpha有限正值。
- A/B的SP2校准有完整16层记录，各50个候选按最小output-MSE选值；最终包alpha与所选值的FP32表示一致。校准输入来源复用已核定train-only数据路径，不来自validation。
- C没有step100末尾SP2校准文件，其全部alpha与step100 learned scale×127精确相同。
- 执行快照仍为R1/R2离线融合、R3/R4关闭、96 static INT8 +16 static SP2、KV16/use_cache=False，embedding/lm_head/norm保持高精度。没有后来postprocess路径混入这三个包。
- 主执行已存的 `reload_check.json` 三者before=after、exact=true；这是**主执行单窗reload probe证据**，不能冒称本auditor重新进行了独立百步GPU冷载或完整复现。

## 完整validation及复用来源

首节点来源证据复用：`../first-validation-20260914/{REPORT.md,summary.json,data_check.json,bf16.result.json,evidence/model_metadata/}`。本次不重新加载原模型、不重tokenize、不重测baseline。

- 模型/tokenizer仍为 `cache/models/llama-3.2-1b-instruct`，`.mv`仍是 `Revision:master,CreatedAt:1740591574`。配置、tokenizer小元数据与上轮副本按字节一致；model.safetensors/tokenizer.json只核对size/mtime未变。此为复用已核定来源证据，不声称本次重新逐权重认证或确认了新的上游不可变revision。
- train/calibration/probe划分、seed42和共同241个初始参数相等性复用首节点核验；每条当前settings/data.json重新对照一致。校准32×128、probe固定四窗，未混入额外初始化预算。
- baseline复用首节点已真实GPU精确复现的原始无旋转、无norm fusion BF16：NLL2.612614913352714 /PPL13.634657725643203。它不是本轮新增测量；上轮auditor无效RoPE尝试仍单列，不重新计入有效结果。
- 每个checkpoint100都为252852输入tokens；123×2048主段产生251781 targets，948尾窗产生947 targets，总 **252728**；起点分别0/251904，unscored_tail_tokens=0。
- 两段逐一验证NLL=log(原float32段PPL)，总NLL按251781/947预测targets加权，再exp；所有JSON算术精确一致。
- evaluator/acceptance源码与首节点完全相同：BF16 logits CE先reduction none，再将loss转FP32，先得到float32段PPL，再log加权；batch1、不重叠窗、上下文重置、prefill/use_cache=False。
- 训练/probe的FP32-logits CE、短reload probe不属于上述完整validation；10和100的精度说明、路径、格式、input/target counts全部一致。

## 阶段目标与未审范围

原BF16+1阈值复用为 **14.634657725643203**。本阶段最佳B100：

- ΔPPL相对BF16：3.4824571842899363。
- ΔNLL相对BF16：0.22746392154346395。
- 超过+1阈值：2.4824571842899363 PPL。

故“B为本阶段三路线胜者”与“全goal尚未达到”同时成立。本审计不提供算法建议或搜索决策。

初始化两组不纳入本次完成100结果，也不据它们当前进度推断初始化收益；正在进行的B100离散W4/SP2/D、组合候选及最终胜者均NOT TESTED。应用CPU verifier和设备验收仍独立，不能由本报告替代。

**审计完成后保持暂停，不自动跟踪中间候选或逐个后处理复跑。下一次实际GPU审计等待主线程明确给出最终候选路径。**

## 复核入口

- 结构化结果：`summary.json`；原始只读解析日志：`audit.log`。
- 证据副本：`evidence/`；原始大包和state/resume保留在三个run目录。
- 可复核的只读解析程序：`audit_evidence.py`，只导入torch读取CPU张量，不导入模型/实验driver或执行forward。
- 本次命令：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 GIT_OPTIONAL_LOCKS=0 /home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python runs/phase3/auditor/three-routes-100-20260914/audit_evidence.py`。
