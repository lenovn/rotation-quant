# Phase5 执行计划

更新：2026-09-18；用户已授权实现、独立验证、隔离环境和正式实验。独立 goal 已建立，无额外 token/GPU 总时长预算。

**当前状态：本轮授权范围已完成。** 固定主线、两家族完整核心矩阵、Llama代表消融、两家族42/43/44配对、必要同包完整test/C4及独立审计均齐备；没有待启动、运行中或待验收的授权内实验。最终结果见[CORE_RESULTS.md](CORE_RESULTS.md)，完整覆盖见[verifier/HANDOFF_COVERAGE_REVIEW_20260918.md](verifier/HANDOFF_COVERAGE_REVIEW_20260918.md)。下文保留既定配方与执行衔接历史，早期等待措辞不代表当前状态。

## 固定范围

- 有效继承源：`worktrees/SpinQuant-phase3-joint`，HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，包括 tracked dirty 与 untracked experiments/tests；继承证据见 `source_inheritance/`。
- 新源码：`worktrees/SpinQuant-multimodel`，分支 `phase5/multimodel`。旧源码、环境、Phase2/3/4 产物不修改，不提交推送。
- 仅新增 `Qwen/Qwen3-1.7B`。隔离环境使用 `runs/phase5/env`，继承旧 PyTorch 2.4.1+cu121，只在隔离层安装 Transformers 4.51.3 及其必要依赖。
- 客户端 GPT-6 Astra / high 由客户端设置决定；本执行不通过 prompt 冒充修改配置。

## 实施顺序

1. 复用并独立审计 Llama test/C4 原始 JSON；不复跑历史归因。
2. 官方 Qwen3 架构上接入现有旋转、W4、静态尺度；保留 Q/K Norm、RoPE 精度、GQA。显式解绑 student embedding/head，teacher 原生。先分离架构、centering、fusion/rotation、冷加载误差。
3. 独立 verifier 检查新增风险；少量真实 GPU 输入验证，旧 Llama 只窄回归。首个 seed42 最终包独立完整 PPL 复核固定为 **WikiText-2 test**，不机械复跑 validation/C4。
4. Qwen seed42：Joint100 → 初始 SP2 → down 邻码 → 范围收缩 → down 再适配 → WikiText-only QAT400。
5. 完整 INT8/SP2 × PTQ/QAT 核心矩阵，优先复用 Llama SP2；B100 INT8 overlay 只为机制对照。INT8 获同等校准、邻码、范围和 QAT 预算，完整实现缺项不阻塞 SP2 主线。
6. 代表模型 Llama：相同 Joint100 初始 SP2 直接 QAT400，与完整后处理后 QAT400 成组比较。
7. 核心矩阵完成后，完整方法及 Uniform-QAT400 配对 seed42/43/44；各 seed 重新初始化 R 和校准索引，固定评测输入。报告逐 seed、均值/标准差、配对 ΔNLL。

## 预算与数据

- Joint100：全局8×2048、100 updates；R SGDG LR1.5，尺度 Adam relative LR0.001/eps1e-12，warmup10/cosine100；训练时 down A16；32×128 train-only 校准。
- 后处理：32×2048 train 窗，24 fit/8 selection；邻码 parent/512/2048/8192；范围1/0.875/0.75/0.5/0.25/0.125；逐模型所有 down 层，父候选可保留。
- QAT400：全局8×2048、400 updates；FP32 master Adam LR1e-5、尺度 relative LR0.001，warmup10/cosine400；T1，0.9 KL+0.1 CE；data_start800循环；同R参考投影父包INT4 cell；固定最终400，不选中间包。
- 默认优化仅 WikiText train；validation 开发；完整 test 与固定 C4 不参与校准、候选选择、调参。同一最终包验收两个数据集。
- C4 原文固定使用 `runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/documents.jsonl` 保序重新tokenize，不新增文档、不用Llama token IDs；双换行、无special tokens，沿用前缀截断上限2097152，实际目标数随tokenizer记录。
- 继承 BF16 logits CE、token loss转FP32、分段PPL取log按targets加权协议；包含可评分尾窗。

## 资源与边界

GPU4 禁用；其余卡按每次真实余量调度，保持全局batch，长任务tmux并记录PID/GPU/run/命令。检查mergerfs分支实际写入余量，不清理旧产物、不终止他人进程。

外部近邻仅登记1个候选：现有Llama上的SpinQuant learned-rotation+GPTQ，W4/group32、dynamic asymmetric A8、KV16。本地适配核心调用路径已由有限smoke独立验证可运行（`verifier/external-spinquant-smoke/EXECUTION.md`）：1次真实2048-token仅R更新、112矩阵GPTQ/group32、动态A8有限前向。正式100步/128校准窗、CLI Trainer/DDP和外部PPL未验证，不冒称官方完整复现。动态A8、o_proj分组、在线Hadamard、只学R100步与GPTQ采样均不同于主线；本地A8配置先于GPTQ的顺序也不同于官方版本。详见auditor/EXTERNAL_BASELINE_SCOPE_20260918.md。本轮落实候选/范围及有限可运行性验证，不由此自动追加完整外部训练或第二个方法，内部Uniform不得填入外部方法行。混合数据QAT未授权、不执行；下游ACC和NPU不在本轮。

## 已落实的运行选择

- Qwen QAT已使用两卡decoder顺序放置，embedding/head/final norm放逻辑cuda0；同一microbatch串行经过层，不使用DDP、global8预算不变。实际GPU索引按启动时余量选。
- Phase5蒸馏默认关闭历史target-PPL提前停止；固定400，不因结果提前结束。
- seed入口为launcher `--seed` -> `PHASE5_SEED`，实际改变R初始化、校准索引及Python/Torch随机源。训练窗保持历史顺序循环，不虚构数据shuffle；统计明确标记R/校准初始化稳定性。

## 核心对照的历史复用入口

- Llama Uniform初始尺度直接复用已独立验收的 `runs/phase3/b100-uniform-calibration-20260918c/down_int8_scales.pt`，父包仍为历史Joint100初始SP2包；只物化INT8完整包，不重复50候选校准。随后在INT8实际输入上执行同三阶段后处理及QAT400。
- Qwen主线启动后，可利用独立空余GPU并行补Llama上述完整Uniform路径和“初始SP2直接QAT400”单支消融；复用已有SP2-QAT400匹配另一侧。这不更改Qwen内部阶段顺序，不启用混合语料。
- 正式所有range命令显式传 `--alpha-factors 1 .875 .75 .5 .25 .125`，不使用保留的历史CLI扩张默认。
- QAT保留100/200 checkpoint/probe，完整validation仅400；固定400不提前停。最终test/C4同包。

## 执行衔接记录（已完成，保留参数与证据）

执行器：`worktrees/SpinQuant-multimodel/experiments/phase3/launch.py`；Qwen用`runs/phase5/env/bin/python`，Llama用旧`/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python`。参数按各run同目录外`.launch.json`直接读取；新tmux session包含模型名避免跨家族同名冲突。

1. Qwen seed42两支三阶段后处理均completed：精确SP2/Uniform包为各sp2-down-readapt-s42/static_w4a8.pt、uniform-down-readapt-s42/static_w4a8.pt，validation14.935424095235582/17.491760837320335。两份PTQ test及同包C4均completed：SP2 test14.302163393485294/C4 24.903332773203157；Uniform test16.78537625006989/C4 29.322131492633147。SP2-QAT400已完成固定400步，最终validation14.813941218960903；同包test14.151216160997155/C4 24.24703443394153均完成，首次独立完整test复核PASS（verifier/QWEN_SEED42_FINAL_TEST_20260918.md）；独立auditor的400步/validation审计PASS，test/C4联合审计PASS。Uniform-QAT400已完成固定400步及validation15.715716054138214/test15.055184987698569/C4 27.10478088408928，同包三split和完整匹配矩阵审计PASS（auditor/QWEN_UNIFORM_QAT400_20260918.md）；实际global8×2048。首个主线最终包独立test复核已完成；关键seed仍在seed42核心矩阵完成后。此前初始格式C4与Llama结果均复用，不重测。
2. **已完成：Llama Uniform三阶段后处理。** `uniform-down-round-s42` → `uniform-range-s42` → `uniform-down-readapt-s42`；精确PTQ包为最后目录的`static_w4a8.pt`，同R参考保持历史Joint100 checkpoint-0100/state.pt。预算与train-only选择审计见auditor/UNIFORM_POSTPROCESS_20260918.md。不重搜、不复跑。
3. **已完成：Llama Uniform-PTQ与Uniform-QAT400同包验收。** PTQ test38.47028685874723/C4 81.13316772229233；QAT400包在`uniform-qat400-s42/checkpoint-0400/static_w4a8.pt`，test17.51898842630044/C4 38.77759462968858。B100 overlay仍仅为格式对照。最终独立核对见auditor/UNIFORM_QAT400_20260918.md。
4. **已完成：Llama初始SP2直接QAT400代表消融。** 包为`initial-sp2-qat400-s42-r25/checkpoint-0400/static_w4a8.pt`，test14.320253706327318/C4 26.730584258553062。旧run更新1–25与r25更新26–400连续，有效更新预算6553600输入token visits；迁移前未提交的部分计算不计为有效更新，不能把这个数字当设备实际处理总量。独立审计及与历史完整方法成组比较见auditor/INITIAL_SP2_ABLATION_20260918.md。
5. Qwen Joint100完成会导出checkpoint-0100/{state.pt,static_w4a8.pt,sp2_calibration.json,validation.json}；同R参考state用于三步postprocess：round→range收缩→round（预算同上），所有28层。各阶段保存父候选或接受包，不测内部候选test/C4。
6. Qwen B100匹配INT8校准使用 `--task uniform-calibration --parent <Joint100 static> --calibration-data <Joint100 data.json> --sp2-calibration <checkpoint-0100/sp2_calibration.json> --export-package`；随后初始INT8/SP2固定C4成对测量。Qwen C4 token路径`runs/phase5/qwen3-1p7b/c4-data/input_tokens.pt`；Llama用旧`runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/input_tokens.pt`。
7. Qwen精确SP2父包进入QAT400：`--layer-devices 2`，launcher `--gpu <主卡> --extra-gpus <第二卡>`，保持global8；先查看实际余量。完整Uniform分支同预算三步后处理及QAT400。所有正式最终包统一完整test/C4；独立verifier首次完整复核集固定test一次。
8. 核心seed42矩阵完成后，42/43/44配对完整方法/Uniform-QAT400：额外seed需各自Joint100/init、校准与后处理；同seed两格式共享Joint100，评测输入固定；不扩展模型/搜索/混合数据。
9. 新的完整测量用`runs/phase5/record_result.py`写summary.csv，`--model`统一官方名`meta-llama/Llama-3.2-1B-Instruct`或`Qwen/Qwen3-1.7B`（目录/launcher仍用短名），独立auditor核对来源。只有主线/核心矩阵/必要消融/关键seed/最终验收完成才能关闭goal。

## 结果汇总入口

每次summary.csv新增完成结果后运行`python3 runs/phase5/summarize_results.py`更新`CORE_RESULTS.md`。沿用已有stage区分BF16、SP2-PTQ、Uniform-PTQ、SP2-QAT400、Uniform-QAT400及Llama代表消融Initial-SP2-QAT400（匹配忽略大小写）。最终QAT行记录WikiText-only数据配方。三seed未齐时不报告整体均值/标准差；齐后用样本标准差ddof=1，配对Delta NLL定义为SP2减Uniform，保留逐seed值和原始证据链接。

## 固定C4的实际文本覆盖（已完成）

独立报告为`auditor/C4_TEXT_COVERAGE_20260918.md`与同名JSON。两模型共享4480篇有序源文档及2097152输入token上限，但保留文本不同：Llama拼接前缀9840845字符（4382完整文档+下一篇996字符），Qwen9659562字符（4325完整文档+下一篇40167字符）；字符按Unicode code point计。两者均1024完整窗、2096128预测targets，无尾窗，截断未切Unicode字符。结果解释使用各自BF16的Delta NLL/PPL比值，不把跨tokenizer绝对PPL视为同一文本上的排名。旧Llama零宽offset限制和缓存decode定位依据在报告内，现有输入/metadata均保留。

## 外部候选可运行性补项

§5.2要求候选实际可运行，仅源码核对不能证明这一点。对已登记的唯一SpinQuant本地适配候选，由独立verifier执行有限smoke：现有Llama/旧环境，真实2048-token训练窗的一次仅R更新、单窗GPTQ W4/group32和动态A8有限值前向。证据保留在`verifier/external-spinquant-smoke`；不计入正式结果矩阵，不追加完整外部训练/PPL或第二候选。正式100步/128校准窗仍未执行；本地A8先于GPTQ、embedding/head高精度等适配差异继续披露。仅当实际执行证据成立时更新“可运行”状态。

- 外部候选有限可运行性补项已PASS：见`verifier/external-spinquant-smoke/EXECUTION.md`、`result.json`和`completion_audit.json`。1次R更新已保存并复用，setup失败后仅续跑GPTQ；112矩阵全部量化、112动态A8 wrapper实际执行、16层在线down Hadamard。所有输入为train，应用源码diff/status与开始时相同，GPU2已释放。正式性能仍未测量，smoke不进入核心结果表。


## seed43/44配对执行记录（已完成）

两模型seed42核心矩阵均已完成并审计；现在推进每模型43/44。每个模型/seed各一次`init-s<seed>`→`joint100-s<seed>`，两格式共享该Joint100，再分支执行与seed42同预算的校准、三步后处理、QAT400、最终同包完整test/C4。额外seed只新增完整方法和Uniform-QAT400最终行；BF16、代表消融、初始格式C4及PTQ外部评测不乘以三。数据/训练/选择/损失/学习率配方均不变，实际启动记录和PID在STATUS及各`.launch.json`。


- seed43/44四项初始化已完成并核对真实seed、R差异和校准索引。当前Joint100：Qwen43 GPU2/PID3089096，Qwen44 GPU5/PID3098418，Llama43 GPU0/PID3090518，Llama44 GPU3/PID3100833。各run的checkpoint-0100自然导出初始SP2与同R参考，不重复SP2校准；随后匹配INT8校准和两格式三步后处理。普通seed参数实验不重跑首次架构/冷包完整test复核。


- Llama43 Joint100已completed；SP2 down邻码GPU0/PID3697102、匹配INT8校准GPU1/PID3698944已启动。依赖顺序仍为两格式各自round→range（显式1/.875/.75/.5/.25/.125）→round→QAT400；43/44不补初始格式C4和PTQ外部主表。

- Llama43匹配INT8校准已完成，Uniform down邻码已接续GPU1/PID3709002；不再等待/重复uniform-initial-s43。

- Llama44 Joint100已completed，SP2 down邻码GPU3/PID3766558、匹配INT8校准GPU7/PID3768360已启动；完成INT8校准后接本格式round→range→round。QAT启动前重新核对实际余量。

- Llama43两格式首轮round完成，range已接续SP2 GPU0/PID3785225与Uniform GPU1/PID3790349；range完成后各自接down-readapt（round原坐标预算）。Llama44 INT8校准完成，Uniform首轮round GPU7/PID3780034已启动。

- Llama43两格式range均完成，down再适配已接续SP2 GPU0/PID3816753、Uniform GPU1/PID3819641；随后分别以该seed精确父包及joint100-s43同R参考启动固定QAT400，不新增PTQ test/C4。


- Llama43 SP2原再适配异常退出，保留旧目录；从4层完整前缀恢复为`sp2-down-readapt-s43-r4`（GPU0/PID3869795），完成后以恢复结果为SP2-QAT400精确父包；无收益则沿用原range包，不伪造缺失包。Llama44两format首轮round完成，range接续GPU3/PID3869801与GPU7/PID3869807，完成后再适配。异常与恢复证据见STATUS及auditor/LLAMA43_READAPT_RESUME_20260918.md（审计进行中）。


- Llama43 Uniform精确父包已完成；`uniform-qat400-s43` GPU1/PID3893776已启动固定400步。SP2等待`sp2-down-readapt-s43-r4`完成后接QAT400。Llama44两range完成，`sp2-down-readapt-s44` GPU3/PID3893782与`uniform-down-readapt-s44` GPU7/PID3895353已接续；完成后各自QAT400，启动前重查显存。


- Llama43 SP2精确父包已完成于`sp2-down-readapt-s43-r4/static_w4a8.pt`；`sp2-qat400-s43` GPU0/PID3925847已接续。此seed两格式均执行固定QAT400，完成后各自checkpoint-0400同包完整test/C4；不要回用异常旧run或range父包。


- Llama44两份精确父包已完成并登记；`sp2-qat400-s44` GPU3/PID3984664与`uniform-qat400-s44` GPU7/PID3984670已启动。Llama43/44四条QAT现均按固定400步运行，逐项完成后用各自checkpoint-0400统一test/C4；不能用初始化或中间probe代替最终质量结果。Qwen43/44仍待各自Joint100完成后分格式后处理。


- Qwen43 Joint100 completed，初始SP2自动校准已复用；SP2首轮round GPU2/PID4075184、匹配INT8校准 GPU2/PID4075190已启动。校准完成后Uniform round可按seed42约8GiB reserved峰值与实时余量同卡并行；两支后续仍各自round→六档range→readapt→固定QAT400，不增加B100 C4或PTQ外部评测。Qwen44等待原Joint100完成。

- Qwen43 INT8校准完成，Uniform首轮round已接续GPU2/PID4080976，与SP2 round PID4075184同卡。两支各自完成后接固定六档range及readapt；不重做已完成匹配校准。


- Qwen44 Joint100完整导出；SP2 round GPU5/PID4109819与匹配INT8校准 GPU5/PID4109825已启动。校准完成接Uniform round，同seed共享Joint100、数据与参考。Qwen43/44后续均固定range→readapt→QAT400→同包test/C4；不重复SP2自动校准或新增中间外部评测。

- Qwen44 INT8校准已完成并接Uniform首轮round GPU5/PID4116221；两seed四条round均运行。Qwen43/44 Joint/匹配初始化合并限定审计由独立auditor进行，报告目标auditor/QWEN_SEEDS43_44_JOINT_20260918.md。


- Qwen43两格式round completed28层，range已接续GPU2：SP2 PID64021、Uniform PID64027，显式六档收缩。各range完成后以本格式输出包为父接round再适配，候选parent/512/2048/8192，再以精确父包进入QAT400；未增加额外seed PTQ外部评测。Qwen44仍round。


- Qwen44两format首轮round已完成并进入range：GPU5 SP2 PID109090、Uniform PID113524。Qwen43两range完成并进入down再适配：GPU2 SP2 PID113530、Uniform PID113536。后续各自精确父包干净启动QAT400；Qwen QAT仍用已验证两卡顺序放置/global8，具体GPU待实际余量。


- Llama43 Uniform-QAT400已completed并登记最终validation；同第400步包的test/C4验收在GPU1运行，PID123828/123834，完成后录入正式Uniform-QAT400 seed43行。独立最终日志/实际覆盖/包身份/同输入统计审计合并执行，不重跑PPL。其余3条Llama QAT仍需固定400及同包验收。

- Llama43 Uniform最终同包validation/test/C4均已完成并登记；独立最终限定审计输出auditor/LLAMA_UNIFORM_QAT400_S43_20260918.md/.json（进行中）。不重复这两项外部评测，等待其他方法/seed完成后生成完整配对统计。


- Qwen44两range均completed，down再适配已接续：SP2 GPU1/PID156812，Uniform GPU5/PID160186。GPU1由已完成Llama43 Uniform test/C4释放，按实时空闲调度；Qwen43两支仍GPU2再适配。四份精确父包分别完成后再接固定QAT400，Qwen用已验证双GPU放置且global8不变。


- Llama43 SP2-QAT400已completed，固定400步包的test/C4在GPU0运行PID175632/175638；Uniform43三split已完成并独立审计PASS。两seed43格式外部结果齐备后写逐seed配对，三seed汇总仍等seed44。Llama44两条QAT继续固定400。

- Llama43 SP2最终test/C4均已完成登记，seed43两格式外部配对齐备；SP2最终独立审计输出auditor/LLAMA_SP2_QAT400_S43_20260918.md/.json（进行中），Uniform43已PASS。不再重复这四项外部评测，剩Llama44两支固定400及同包test/C4。

- Llama43两格式最终审计均PASS，三split配对齐备；剩Llama44固定400导出和同包test/C4，以及Qwen43/44再适配→双卡QAT400→同包验收。已完成结果不复跑。

- Qwen44 SP2精确父包已完成，QAT400 GPU1/0 PID242215已启动；其他Qwen再适配完成后按实际空余双卡接续，global8不变。Llama44完成第400步后先统一验收test/C4，再释放卡给后续Qwen。

- Llama44两QAT均completed，SP2同包test/C4 GPU3 PID249209/249234，Uniform同包test/C4 GPU7 PID249260/249383运行；结果齐备后审计并生成Llama三seed配对统计。Qwen44 Uniform精确父包已完成，待GPU2再适配释放后可与GPU5组成下一双卡QAT，实际启动前重核余量。

- Qwen43/44全部后处理完成，精确父包齐备；44 SP2-QAT GPU1/0 PID242215、Uniform-QAT GPU2/5 PID270807运行。43两支待可用双卡接续相同400预算，优先等Llama44 C4释放GPU3/7，不在忙碌GPU6追加任务。

- Llama三seed同包val/test/C4全部测量完成，seed44最终审计与三seed统计核验由独立auditor合并进行。Qwen43 SP2-QAT已接GPU7/3 PID282520；44两QAT保持原进程。最后43 Uniform需待合适双卡释放后以uniform-down-readapt-s43精确父包和joint100-s43参考启动同配方400步，之后四Qwen最终包各自统一test/C4并审计，才可关闭Phase5。

- Llama全部必要实验与三seed最终证据审计完成，后续仅复用，不再追加复跑。余下执行集中Qwen43/44四个QAT400最终包及其test/C4/独立证据；目前三QAT运行，43 Uniform等待双卡释放。

- handoff已完成部分覆盖复核见verifier/HANDOFF_COVERAGE_REVIEW_20260918.md。未新增实验要求；按当前Qwen余项继续，不把早期历史审计missing条目重新列为待做。四条额外seed最终验收和三seed统计齐备后，更新该完成证据链并核对handoff，才关闭goal。

- 额外seed最终验收调度沿用Qwen seed42实测：四项QAT test/C4 peak_allocated约5.544GiB、reserved5.807–5.994GiB（各progress.json）。训练完成后，可按最新实际余量把单项评测安排到现有QAT的第二卡，与剩余训练并行；不能把两项约6GiB评测同时叠到已占约13GiB的同一卡。实际启动仍逐次查余量，保持global8和同一最终包，不因调度重跑测量。最后Uniform43可在一组双卡释放后直接接续。

- 最新资源快照：Qwen44 SP2约331步时，GPU6已为18MiB/24564MiB、0%利用率（先前占用已不在当前快照中，退出原因未核查；执行器未发送终止信号）。不要把早期“GPU6忙碌不追加任务”当永久限制；最终包完成时若仍有余量，可将test/C4安排GPU6，原QAT释放的双卡直接接Uniform43。只按届时真实余量调度，不重测已完成包。

- Qwen44 SP2-QAT400已completed并登记validation；同包test/C4 GPU6 PID811501/811507运行。最后43 Uniform-QAT400已在释放的GPU1/0启动PID811495，使用既有精确父包和同seedJoint参考；不再存在未启动的Qwen额外seed训练。44 Uniform和43 SP2继续原400预算，各自完成后必要同包test/C4。三seed完整汇总与最终审计仍待这些结果齐备。

- Qwen44两格式QAT400均completed；SP2 test已登记，同包C4 GPU6 PID811507仍运行；Uniform同包test GPU2 PID827890/C4 GPU5 PID827896已启动。独立auditor合并44最终证据，完成测量直接登记不复跑。43两格式训练均已启动且保持原预算，完成后各自checkpoint-0400统一test/C4。

- Qwen44两格式全部最终测量已完成登记，独立审计仅补最后Uniform C4；不重复六项评测。Qwen完整三seed统计仍等待seed43 SP2 PID282520与Uniform PID811495完成固定400及各自test/C4，两个原进程保持运行。

- Qwen44两格式同包三split及完整独立证据全部PASS，不再复跑。Phase5剩余正式测量为Qwen43 SP2/Uniform固定400与各自完整test/C4；齐备后生成Qwen三seed均值±样本SD和逐seed配对ΔNLL，补最终独立审计及handoff完成核对。


- Qwen43 SP2固定400已完成并登记validation，同包test/C4已接续GPU6/PID884898与GPU2/PID884904；完成后登记并由独立auditor核验。最后Uniform43继续原PID811495/GPU1,0，完成后再以其checkpoint-0400统一test/C4，齐备后生成Qwen完整三seed配对统计并进行最终handoff覆盖核对。

- Qwen43 SP2最终同包validation/test/C4已全部测量并登记，独立审计仅余新增C4/summary核对；不再重复这些测量。唯一未完成训练为Uniform43原PID811495（GPU1/0），固定400结束后接其同包test/C4，再完成Qwen三seed配对统计、独立最终审计及handoff覆盖更新。

- Qwen43 SP2同包三split及独立最终证据已全部PASS（auditor/QWEN_QAT400_S43_20260918.md），不再列待审或待测。唯一剩余训练Uniform43继续原进程；其第400步完整validation、同包test/C4、相应独立新增证据、Qwen三seed配对统计及最终handoff覆盖核对完成后方可关闭goal。

- 全部必要训练已完成：Uniform43固定400及完整validation已登记，最后两项同包外部验收test GPU1/PID1221890、C4 GPU2/PID1221896运行中。完成后登记summary并重生成CORE，由独立auditor补Uniform43实际证据及Qwen三seed统计；再更新最终handoff覆盖、STATUS/PLAN并逐项核对完成条件。不新增或复跑既有实验。

- 所有必要训练及最终同包评测均已完成登记，最后Uniform43 C4为27.45911062612704；完整Qwen三seed统计已生成。剩余仅独立新增C4/三seed统计审计和handoff完成覆盖核对、最终文档收尾；无需再启动GPU任务。

- 最终完成核对：最后Uniform43同包三split、Qwen三seed18值及配对统计独立PASS，最终handoff覆盖PASS；主表57项原始指标/19组同包、summary99行全部核对。后续若开展混合QAT、ACC、原生NPU/decode或外部完整性能复现，须另列任务与协议；本轮不自动追加实验。
