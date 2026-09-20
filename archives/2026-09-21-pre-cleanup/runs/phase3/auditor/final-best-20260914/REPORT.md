# 最终优胜包独立 GPU 审计

日期：2026-09-14。原 Phase3 独立实验 auditor；不是主执行、应用代码 verifier 或新的算法搜索角色。

## 1. 总结：限定 PASS

**已实际独立完整复跑原 BF16 与最终保存包，两者的两段结果、最终 NLL/PPL 均精确复现原记录。最终 PPL 相对本次新测原 BF16 增加 0.9469969496849142，满足 BF16+1 阈值，余量 0.05300305031508579。**

| 本次独立 GPU 测量 | PPL | NLL | 预测 targets | 与原完整记录 |
|---|---:|---:|---:|---|
| 原 BF16，无旋转/norm fusion | 13.634657725643203 | 2.612614913352714 | 252728 | 精确相同 |
| 最终 checkpoint0400 静态包 | **14.581654675328117** | **2.6797642095325815** | 252728 | 精确相同 |

目标 **14.634657725643203**；最终相对BF16的 ΔNLL=**0.06714929617986742**。这里“精确复现”指分段及聚合结果，不宣称比较了每一个logit的逐位一致性。

| 项目 | 判定 |
|---|---|
| 最终run命令、源码/dirty、数据/teacher来源、实际1..400及终态证据 | PASS |
| 同新父2e-5对照的400步窗口/scale LR一致、W LR严格减半 | PASS |
| 最终112W4/96静态INT8/16静态SP2覆盖及保存master→码/尺度对应 | PASS |
| 原BF16独立完整GPU复现 | PASS |
| 最终冷包独立完整GPU复现及评测期间A尺度不变 | PASS |
| 本次完整validation的原BF16+1精度目标 | PASS |
| “最终组每步所有112张W的64点抽样都变化”这一更强陈述 | FAIL，非必要条件；第400步有两张抽样未变，详见第5节 |
| 2e-5组再次独立GPU复跑；初始FP master全量逐位独立比较 | NOT TESTED |
| 新pytest/应用代码verifier、手机原生kernel/KV/decode/时延、C4/test/多seed泛化 | NOT TESTED |

未发现阻断本次最终run证据及两次GPU验收的FAIL。上述抽样观察原样保留，不将更强陈述悄悄写成PASS；它也不否定已核实的400次optimizer执行。方法为 **full-backbone quantization-aware distillation / FT-QAT**，不是冻结原FP主权重PTQ。

## 2. 执行范围、设备与持久记录

本任务只写 `runs/phase3/auditor/final-best-20260914/`。读取适用AGENTS及更新后的 `runs/phase3/STATUS.md`、`runs/phase3/handoff-20260914.NByJvb/HANDOFF.md`；用户本次独立审计范围优先于交接中主执行的广义训练/goal指令。主STATUS最初尚显示392步，其后已更新为400完成；本报告始终以实际最终文件及完整日志核查，不把协调文档短暂滞后当成run未完成。

没有创建/更新goal，没有新增auditor，没有修改应用代码/tests/主STATUS/RESULTS/根协调文档，没有训练、重校准、pytest、搜索、安装、环境或挂载改动、删除、提交推送、操作Phase2或他人进程。Phase2 acceptance 纯函数只读复用，旧有效产物不动。

实际 **2次完整GPU测量，2次有效，0次无效完整测量，0次失败评测尝试**；无短烟测、中间checkpoint或2e-5对照GPU复跑。这两次是独立复现，不是两个新算法候选。首次沙箱NVML查询失败后按授权用普通 `require_escalated` 实查/启动；该资源查询不算一次评测尝试。

采用GPU1，UUID `GPU-75f32c85-75f6-07fa-f335-6e1cc84aacf9`，RTX4090。启动前及每次评测前均实际查询GPU/计算进程，GPU1当时used18MiB、free24075MiB。1/3/5/6/7均按余量允许使用；本次选择一张空闲卡串行是合理调度，不是恢复单卡、12GiB、35%或GPU时间限制。

- tmux socket：`rotation-quant-phase3`；独立session：`auditor-final-best-20260914`；wrapper PID3561101。
- BF16 PID **3561110**；final PID **3563303**。已通过 `MAIN_THREAD_NOTICE.txt` 及对话通知主线程GPU/PID。
- 两次退出码均0，session shell在串行完成后自然退出，没有kill其他任务。
- BF16 evaluator耗时18.582161845988594秒、进程内总耗时26.91453150799498秒、peak allocated4.336132526397705GiB。
- 最终包 evaluator耗时23.51341075298842秒、进程内总耗时73.34639406198403秒、peak allocated4.6500020027160645GiB。总耗时含CPU壳构造、全量冷载核对、tokenizer等，不是纯forward时延。

真实完整 `exec env ... bash run_serial.sh`、环境、tmux及设备快照见 `launch.json:1`。逐次实际 `/proc/self/cmdline`、源码路径、PID及环境见 `bf16.runtime.json:1`、`final.runtime.json:1`；完整stdout/stderr、设备、PID和退出码分别在 `bf16.*`、`final.*`。`serial.completed` 是真实完成标记，`COMMANDS.md:1` 给出实际命令。命令显式设置CUDA_VISIBLE_DEVICES，未依赖tmux server旧环境。

## 3. 最终文件、源码与真实配置

已核实真实文件名：

`runs/phase3/distill-b100-refined-ref-adam1e5-400-20260914a/checkpoint-0400/static_w4a8.pt`

不是旧父D包、checkpoint200、resume或从训练内存临时取模型。最终包/训练日志/resume/父包/B100 reference的size/mtime记录于 `records.json`，CPU审计读期间不变；不新增hash。CPU mmap读取最终resume与包，GPU审计从该静态包独立构造。

源码根：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。

HEAD：**24918316ed594848d4de797c356b120f2a4ee0f3**；branch：`phase3/joint-r-sw-sa-sp2`。

原始 `git status --short`（`GIT_OPTIONAL_LOCKS=0`）：

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

原始HEAD/branch/status/dirty patch见 `evidence/`。两个400-run的tracked.diff与当前及首审dirty证据逐字节一致，不能称clean HEAD。两组保存的全部Phase3源码快照逐字节相同，且与此次读取的当前文件一致；`distill.py` 与上次QAT reference200已审快照也逐字节一致。新sequential/local_d文件出现在复制快照中不等于400-step driver调用了其搜索入口；实际distill入口/导入路径单独核查。

`common.py`、`quantization.py`、`utils/eval_utils.py`、`eval_utils/modeling_llama.py`、`utils/quant_utils.py`、`train_utils/quant_linear.py`以及acceptance的适用旧审证据未变。GPU `*.modules.json` 实际导入均指向新worktree，没有误用 `repos/SpinQuant`。这里是源码/实验来源核查，不重开应用代码verifier测试。

原final launcher PID2761545/GPU3，与settings、progress及日志完成事件一致；原2e-5对照PID2621315。主final的 `result.json`、checkpoint400 validation、progress completed/steps400/target_reached=true及日志一致，elapsed5214.290734209004秒；没有继承旧resume，也没有原driver ENOSPC恢复评测混入本次run。

两组实际完整训练argv保存在各原 `.launch.json` 的证据副本及 `records.json`，规范化后只有 `--output`、`--weight-lr` 不同。共同关键参数为：

```text
.../rotation-quant-p0/bin/python -u .../SpinQuant-phase3-joint/experiments/phase3/distill.py
--parent .../seq-b100-sp2-refine-down-20260914a/static_w4a8.pt
--reference-state .../route-b-adam-100-20260914a/checkpoint-0100/state.pt
--steps 400 --schedule-steps 400 --weight-optimizer adam --offload-teacher-body
--relative-scale-lr 0.001 --temperature 1 --ce-weight 0.1 --data-start 800
--checkpoints 100 200 400 --validation-steps 100 200 400
```

最终组W LR=1e-5，对照=2e-5；实际默认/保存配置accumulation8、warmup10、resume_every25、resume=null、seed42、TF32关闭、torch4threads。不是将旧200后续补200而改称fresh400。

## 4. Model、teacher、reference及父包来源

**Reuse且重新核对**：原模型/tokenizer/data来源复用 `runs/phase3/auditor/first-validation-20260914/REPORT.md:1`，B100来源复用 `runs/phase3/auditor/three-routes-100-20260914/REPORT.md:1`，distill/cell构造源码核查复用 `runs/phase3/auditor/qat200-20260914/REPORT.md:1`。这不复用成“本次新测分数”：本报告两行GPU结果都由本次新进程真实测得。

模型目录 `cache/models/llama-3.2-1b-instruct`，`.mv`=`Revision:master,CreatedAt:1740591574`。小模型/tokenizer配置字节与首审副本一致，model.safetensors/tokenizer.json的size/mtime一致；未扫描整个原模型求hash或下载升级。实际GPU环境torch2.4.1+cu121/CUDA12.1，tokenizer为既有LlamaTokenizerFast配置，BOS/EOS均关闭。

冻结teacher实际从该原BF16模型构造，untie后head复制embedding，requires_grad_(False).eval()，use_cache=False，无R、norm fusion、量化wrapper。T1下按next-token优化 **0.9 KL(teacher||student)+0.1 CE**；student/teacher logits先FP32，head128-token块，按2047预测targets归一；teacher主体按microbatch卸载CPU。两组teacher/目标/放置相同。此训练/4窗probe CE不同于正式BF16-logits评测口径，不能混算。

真实父链由包metadata逐级读取，配套原settings/source/data/validation有小副本，全部沿新worktree：

| 父链阶段 | 已保存完整PPL |
|---|---:|
| B100 | 17.11711490993314 |
| seq-b100-down-round-20260914b | 16.507098542639632 |
| seq-b100-rounded-sp2-contract-20260914a | 16.19254871508746 |
| **seq-b100-sp2-refine-down-20260914a（本次直接父）** | **16.112577060930427** |

这些父分数是本轮只读取证，不是本auditor新增GPU复测。当前直接父不是旧 `post-b100-d-20260914a`。两组master_initialization均指同B100 state、同新父，**diagonal_transform=null**，与当前父链没有该旧局部D的事实一致，不能错误地再施加旧L1/ch1417 D。

B100 reference state实读为routeB/update100/training_tokens1638400；同原BF16权重经norm fusion、保存R生成current-R浮点权重，再转FP32投影入当前父INT4 cell，保留父量化初值而非原FP权重不加坐标变换直接替换。两组该初始化记录、初始4窗probe完整对象相等（NLL2.9197031259536743/PPL18.53578384969452），首更新8个microbatch损失记录也相等。本次未独立重建/保存两组初始全量FP master以作逐位比较；该更强项NOT TESTED。

校准来源没有偷换：B100原有校准来源复用旧审；新sequential父处理实际用seed42的同32个train索引，但改为**32个完整2048窗**，前24窗fit49152行、后8窗heldout/selection16384行；选择NLL是16376个train targets，不是252728-target完整validation。不能将新父校准仍称32×128。当前QAT driver只读取冻结父尺度，丢弃data_windows返回的calibration，不重新校准；auditor冷复跑也不校准。

## 5. 400次实际更新、LR与窗口配对

两组训练JSONL均恰为1..400，8个microbatch/步，累计tokens逐步正确；每步W112/SA96/SW112/SP216梯度记录完整、loss和gradient norm有限。两个最终resume中W112项Adam state及尺度224项Adam state均为400，336张保存参数均FP32，实际保存步=400。原完整评测事件为100/200/400，最终run result/progress一致。

训练参数为112张FP32 master（973078528元素）、112张SW（376832元素）、96张SA与16张SP2 scale（112元素），合计336张/973455472元素；R不在optimizer中，embedding/head/norm保持高精度冻结边界。W Adam eps1e-8、weight_decay0、betas(.9,.999)，尺度Adam eps1e-12、initial_lr=0.001×各初始GPU mean(scale)。这是真实主权重QAT，不是仅R/尺度PTQ。

每组400×8×2048=**6553600训练输入tokens、6550400预测targets**。训练原始tokenize数据2435022tokens→1188整窗、drop1998尾；最后8窗保留，实际优化池1180窗。访问序列逐项符合 `(800+(step-1)*8+micro)%1180`：第一步800..807，最后452..459，共3200次访问/1180个不同窗。存在明确回绕和重复，不称完全新数据或3200个独立不同窗；probe1180..1183与validation不参与梯度。

两组**全部400步、89600个scale LR配对精确一致**；另400个W LR值中最终组均严格等于2e-5组的一半。两边每步都精确符合保存initial_lr×400cosine/warmup10；没有用CPU mean重算初始率去假装GPU逐位配对。实际窗口3200次访问也逐项一致。因参数更新不同，不能把同scale LR写成相同梯度或相同scale轨迹；例如第二步CE已分别2.6264844238758087和2.627646416425705。

**抽样观察保留**：最终组第400步W LR=1.622214173602199e-10，在 `model.layers.1.self_attn.o_proj.module.weight` 和 `model.layers.8.self_attn.o_proj.module.weight` 各64点抽样中记录0变化；其余步/张量没有该抽样0记录。2e-5组没有该情况。故审计原始 `sampled_all_masters_change_each_step=false` 保留，不能写“每张W每步抽样全变”。这不等于全张量不变/少执行optimizer：全部336个state均400，所有112张W梯度均记录。小更新的有限精度舍入是可能解释，未单独证明；不因此增加实验或改算法。

## 6. 全量导出与独立冷包构造

CPU读取最终保存master、SW和packed码：全112张/973078528个码逐行重取码，与最终包**0差异**；相对新父变化 **47471628/973078528**，112项明细与原code_changes逐项一致。全部SW/SA与resume逐项一致，16个SP2 alpha等于resume.scale×127；HP条目与父逐字节一致。

全112个预期模块/形状、uint8每字节两nibble、signedINT4[-8,7]、ties-to-even/per-output-channel正FP32 SW成立；96非down固定INT8/per-tensor、16down固定SP2/per-tensor。不是漏层、weight16 backbone例外或把dynamic A8算成static。

GPU driver沿首审**已纠正且精确复现**的方法：BF16从原pretrained加载，不旋转、不fuse norm；candidate以保存config `_from_config(..., torch_dtype=bfloat16)` 创建壳，从最终静态包完整恢复112权重及HP/A，不从训练内存取模型、不重新加载原pretrained权重替代静态内容。冷载后112权重均与包反量化值相等，HP逐项一致，16SP2/96INT8覆盖正确。未对整个模型执行 `.to(bfloat16)`；实际非持久RoPE inv_freq **全部保持FP32**，只移动GPU设备。

实测边界见 `final.load.json:1`：input bits8、output bits16、无online_full_had/partial_had、observer关闭、requires_gradFalse、KV16/use_cacheFalse；R1/R2已融合在权重，R3/R4关闭。评测前后所有输入量化器state逐项相等，**activation_scales_unchanged=true**，没有隐式观察/重校准。

## 7. 完整token与NLL口径

两次都重新从同本地validation Arrow取原文本，用`"\n\n".join` tokenize；CPU逐token比较与首审固定token tensor完全相同，第二次还与本次BF16生成token tensor相同。`validation_input_tokens.pt`保存本轮输入，不拿不同tokenization碰巧相同长度蒙混。

- 252852 inputs，**123×2048 +948 tail**；独立窗逐窗重置上下文，预测251781+947=**252728**，unscored_tail_tokens=0。
- 调用既有acceptance纯模块函数，再调用新worktree `utils.eval_utils.evaluator`，没有执行旧Phase2 CLI main或导入旧repos/SpinQuant。
- **BF16 logits**的CE使用reduction none；per-token losses再转FP32，分窗mean，FP32 segment mean→exp得到float32 segment PPL；acceptance按log(segment PPL)、预测targets加权得总NLL，再exp。
- 没有改为训练FP32-logits CE，没有直接平均PPL，没有缩成8窗probe或丢尾窗。

本次两个段的PPL也精确相等：BF16为13.629024505615234 /15.2180757522583，final为14.576189041137695 /16.1099910736084。结果对应的总NLL/PPL、日志JSON、结果JSON、PID与退出码相互一致。

## 8. 比较与未测试边界

同新父、同400schedule/数据/scale LR下，原2e-5对照400完整 **PPL15.254327283565368 / NLL2.7248632191015845**；最终1e-5组400更好。对照的完成及LR/窗口证据本次CPU独立核查通过，但没有为其增加GPU复跑。新父PTQ与final-QAT分开列示，不将QAT收益包装成冻结权重PTQ，也不把两次auditor复现计入新算法收益。

已达到本次固定WikiText2 validation的BF16+1数值判据；验证集用于多轮开发比较，存在选择偏差。本次没有独立test/C4、多seed、跨模型泛化或统计显著性测量，不能由这一分数外推。BF16/KV16 prefill fake-quant/dequantized结果不是原生INT4/INT8 kernel、量化KV/decode、Snapdragon/QNN/mllm/NPU或时延验收。

旧首审/百步/QAT200审计及已纠正的SGD25失败配置保留为历史证据；本次未重复扫描其大产物/pytest、未重复那些GPU。既有独立代码verifier报告可由主线程另列，本auditor不以自己的脚本检查替代应用代码verifier身份。

最终机器结论：`summary.json:1`；400步/父链/源码/CPU码证据：`records.json:1`；实际GPU结果：`bf16.result.json:1`、`final.result.json:1`；命令：`COMMANDS.md:1`、`launch.json:1`；原source/dirty/配置/log小副本：`evidence/`。未复制multi-GiB权重/resume，不改主产物。

**本任务完成后暂停。未创建/更新主goal，也不代主线程宣告整个Phase3所有管理项已关闭；本报告交付明确的最终精度与独立GPU审计PASS，供主线程同步STATUS/RESULTS并处理其余完成条件。无自动训练、校准、后处理搜索或中间checkpoint复跑。**
