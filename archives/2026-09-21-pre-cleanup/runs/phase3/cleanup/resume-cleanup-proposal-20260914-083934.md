# Phase3 resume.pt 清理提案及执行记录：仅两项，已获批删除

盘点时间：2026-09-14 08:39:34 +08:00（00:39:34 UTC）。角色仅为存储清理子代理，不承担主执行、算法实现、代码 verifier 或实验 auditor，不创建主 goal。

**最新状态：用户随后明确回复“可以删除吧”，批准本对话中已逐项解释的两个SGD恢复文件；2026-09-14 08:47:55 +08:00 已仅删除这两个文件。原审批前盘点与大小保留，执行证据见文末。此次批准不包含删除 copy 项目或其他文件。**

## 范围与方法

- 已读根 `AGENTS.md`、`STATUS.md`、`SPEC.md`、`PLAN.md`、`LESSONS.md` 及 `runs/phase3/STATUS.md`、`runs/phase3/RESULTS.md`。未发现 `runs/AGENTS.md` 或 Phase3 下更深层 AGENTS。
- 文件盘点仅针对 `runs/phase3` 下准确名为 `resume.pt` 的文件，共 22 个：12 个实验恢复文件、10 个 verifier 既有测试产物。仅下列 2 个为候选，其他 20 个全部保留。
- 淘汰理由来自实际 JSON、训练记录和原日志，不以 PID 退出、未达目标或不是全局最优单独判删。只读查看源码快照的保存/加载结构；未读取或改写 Phase2 实验文件。
- 使用 `stat`、mergerfs 只读 xattr、`statvfs`，以及 ZIP 内约 165–184 KB 的 `data.pkl` 操作码查看恢复元数据；未执行 pickle 反序列化、未载入 checkpoint 张量、未做模型重载验证或 forward。
- 唯一新增文件为本文；不改既有实验文件、应用源码、tests、根协调文档或 Phase3 STATUS/RESULTS。不删除、移动、压缩、截断既有文件，不新增 hash，不测试、不启动 CPU/GPU 实验、不安装、不提交推送、不终止进程、不修改或绕过挂载策略。

## 已批准并删除的唯一两个文件（大小为删除前）

下表目录均为准确绝对路径；每个目录只拟删表中一个 `resume.pt`，**不是删除整个目录，也不匹配其他文件**。1 GiB = 1,073,741,824 bytes。

| 编号 | 确切目录 | 唯一拟删文件 | 文件长度 bytes | GiB | 实际 basepath | 恢复更新数 |
| --- | --- | --- | ---: | ---: | --- | ---: |
| A | `/home/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-sgd01-100-20260914a` | `resume.pt` | 7,789,544,420 | 7.254578564 | `/mnt/home1` | 25 |
| B | `/home/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-sgd0001-100-20260914a` | `resume.pt` | 7,789,544,420 | 7.254578564 | `/mnt/home2` | 100 |
| 合计 | 仅以上两个目录中的指定文件 | 2 文件，均已获批删除 | **15,579,088,840** | **14.509157129** | 每分支各一个 | — |

两文件均为普通文件，UID 1012（dongpeiyan），`nlink=1`；`user.mergerfs.allpaths` 各只返回一个实际路径，未见另一个分支副本。每文件 `st_blocks × 512 = 7,789,551,616 bytes`，合计已分配块 **15,579,103,232 bytes / 14.509170532 GiB**。文件长度与已分配块略有差异；实际回收量不作保证。

### A：SGD WLR 0.1，完整25分灾难性变差，未完成100更新

本节未加前缀的证据路径均相对于上表 A 的确切目录。

- `early_stop.json:2` 明确记录 `stopped_by_main_executor_for_measured_deterioration`；`:4` 为完成 25 更新，`:11` 明确下一轮从未改父包重启、**不恢复已恶化的 optimizer 或 weights**。这是明确淘汰依据，不是从进程退出推断。
- `initial_probe.json:2` 固定四窗 train-probe NLL **2.954445779323578**；`checkpoint-0010/training_probe.json:2` 恶化至 **9.804297924041748**；`checkpoint-0025/training_probe.json:2` 为 **8.60651445388794**。这些是 train-probe，不冒充完整 validation。
- `training.jsonl:25` 最后完成更新 25，minibatch CE **8.566487848758698**、KL **6.027353823184967**、objective **6.281267046928406**；共 25 条更新记录、409600 训练输入 tokens。`resume.pt` 的元数据数值为 25。`failure.json:2` 为 `KeyboardInterrupt`，只作中断背景。
- **真实淘汰依据为已完成且灾难性变差的25更新完整评测**：`checkpoint-0025/validation.json:1` 记录252852输入 tokens、**252728 targets**、123×2048加948-token尾窗、0未计尾 tokens；`:23` 为 NLL **8.81122709329167**、PPL **6709.146967144104**。`checkpoints.json:7` 也记录同值。直接父包 `post-b100-d-20260914a/validation.json:23` 的完整 NLL/PPL 为 **2.8334162843368644 / 17.003450334865143**；25分已灾难性恶化，100更新未完成。
- 对应原日志 `/home/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-sgd01-100-20260914a.log:43` 至第二分段完成处实际显示两个评测分段；`:53` 的中断栈已处于**随后训练前向** `train → distillation_loss`，不是 evaluator。此完整25分必须计入失败配置证据，不能因随后训练被中断而排除。
- **口径纠正已确认**：用户转交 Aristotle 审计及主执行重读确认，且本代理已只读核对修正后的 `runs/phase3/STATUS.md:127`、`runs/phase3/STATUS.md:158`、`runs/phase3/RESULTS.md:60`、`runs/phase3/RESULTS.md:67`。原“无完整25分”判断错误，不再作为未决矛盾处理。原 `early_stop.json:10` 的错误描述仍作为历史记录原样保留，本代理只改本文，不改实验原始文件，也不冒称自己完成独立实验审计。

**A 全部保留内容：** `checkpoint-0010/training_probe.json`；整个完整25更新评测目录 `checkpoint-0025/`，尤其 `static_w4a8.pt`（2,064,318,486 bytes）、完整 `validation.json`、`training_probe.json`、`code_changes.json`；`source/` 下全部六个 `.py` 快照及 `tracked.diff`；`settings.json`、`data.json`、`parameter_coverage.json`、`initial_probe.json`、`checkpoints.json`、`training.jsonl`、`progress.json`、`early_stop.json`、`failure.json`。同级的 `distill-b100-d-sgd01-100-20260914a.log` 和 `distill-b100-d-sgd01-100-20260914a.launch.json` 也完整保留。A 未见 `result.json`，不伪称完整100更新包已存在。失败日志及历史错误描述都不改写。

**A 删除后损失：** 无法从该 25 更新节点恢复原 FP32 master 权重、224 张学习尺度、SGD momentum、尺度 Adam 一阶/二阶状态、Python/Torch/CUDA RNG 及步数/数据游标上下文；无法原状态继续第26次更新或原样诊断该失稳优化轨迹。保留的静态 W4A8 包可供后续另行授权的冷载评测，但不等同训练恢复包；从父包重训也不等于保住该恢复点。

### B：SGD WLR 0.001，完整100更新劣于直接父包

本节未加前缀的证据路径均相对于上表 B 的确切目录。

- `result.json:2` 为 `completed_steps=100`；`:22` 至该记录末尾给出100更新 NLL **2.8425382978723266**、PPL **17.15926563313684**。`progress.json:2` 为 `completed`、`:10` 为100步。`training.jsonl:100` 为真实第100条更新，累计1638400训练输入 tokens；`resume.pt` 元数据为100。
- `checkpoint-0100/validation.json:1` 为252852输入 tokens、252728 targets、123个2048窗加948-token尾窗、0未计尾 tokens；`:23` 为上述 NLL/PPL。同口径父证据 `/home/dongpeiyan/projects/rotation-quant/runs/phase3/post-b100-d-20260914a/validation.json:23` 为 NLL **2.8334162843368644**、PPL **17.003450334865143**。
- `settings.json:4`、`result.json:4` 和100步 validation 的 `parent` 均直接指向 `post-b100-d-20260914a/static_w4a8.pt`。相对该父包，完整 PPL **+0.1558152982716976**、NLL **+0.009122013535462159**。25更新完整 PPL **17.07365575720706** 也未胜父包，见 `checkpoint-0025/validation.json:24`。
- 对应原日志 `/home/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-sgd0001-100-20260914a.log:122` 记录100更新，`:131` 记录完成、`target_reached=false`。结合已明确拒绝该配置和完整父子结果，列为候选；不是因 Adam 全局更好而扩大淘汰范围。

**B 全部保留内容：** 整个 `checkpoint-0025/` 和 `checkpoint-0100/`，每个 `static_w4a8.pt` 为2,064,318,486 bytes，并保留各自 `validation.json`、`training_probe.json`、`code_changes.json`；`checkpoint-0010/training_probe.json`、`checkpoint-0050/training_probe.json`；`source/` 全部六个 `.py` 快照及 `tracked.diff`；`settings.json`、`data.json`、`parameter_coverage.json`、`initial_probe.json`、`checkpoints.json`、`training.jsonl`、`progress.json`、`result.json`。同级 `distill-b100-d-sgd0001-100-20260914a.log`、`distill-b100-d-sgd0001-100-20260914a.launch.json` 完整保留。直接父包、父评测及其源码/配置/日志均不在清理范围。

**B 删除后损失：** 无法从该100更新节点恢复原 FP32 master 权重/尺度、SGD momentum、尺度 Adam 一阶/二阶状态、RNG 及数据游标，无法据此延长 schedule 或原状态开展续训对照。25/100静态评测包继续保留，但不能重建优化器历史或量化 cell 内的 FP32 残差。此取舍仅在用户批准后接受。

恢复内容结构依据两个 run 各自保存的 `source/distill.py:131`（parameters/optimizers/metadata/RNG）及 `source/distill.py:144`（加载逻辑），不是新运行的恢复测试。

## 必须保留的恢复文件与删除前完整盘点

- 当前最佳 `distill-b100-d-adam2e5-ref-200-20260914a/resume.pt`：200元数据、原 `reference_state` 指向B100 `checkpoint-0100/state.pt`；`checkpoint-0200/validation.json:23` 为 NLL **2.7459188570552207**、PPL **15.578922160109332**，全部保留。
- center 续训 `distill-b100-d-adam2e5-continue800-20260914a/resume.pt`：**实际恢复元数据225**；`progress.json:10` 为233，而 `training.jsonl:34` 为234，不能把较晚日志更新数误称已保存恢复点。保留225状态，不做张量完整性或恢复运行的新验证。
- center 原200、全部三路线/SGD诊断、尺度初始化及其两种后续对照恢复状态暂保留，不以排名较差擅自列为候选。共同初始化及非 `resume.pt` 文件一概不在清理范围。
- 以下路径均相对于 `/home/dongpeiyan/projects/rotation-quant/runs/phase3/`；verifier产物只记录该文件名的元数据，不开展其测试或清理。

| Phase3 相对路径 | bytes | basepath | 本提案分类 |
| --- | ---: | --- | --- |
| `distill-b100-d-adam2e5-200-20260914a/resume.pt` | 11681919936 | `/mnt/home1` | 保留 |
| `distill-b100-d-adam2e5-continue800-20260914a/resume.pt` | 11681920000 | `/mnt/home1` | 必须保留225 |
| `distill-b100-d-adam2e5-ref-200-20260914a/resume.pt` | 11681920128 | `/mnt/home2` | 必须保留最佳200 |
| `distill-b100-d-sgd0001-100-20260914a/resume.pt` | 7789544420 | `/mnt/home2` | 候选B，后已获批删除 |
| `distill-b100-d-sgd01-100-20260914a/resume.pt` | 7789544420 | `/mnt/home1` | 候选A，后已获批删除 |
| `init-c-adam-budget100-20260914a/resume.pt` | 38926202 | `/mnt/home1` | 保留 |
| `init-c-adam-joint100-20260914a/resume.pt` | 38926202 | `/mnt/home1` | 保留 |
| `route-a-adam-100-20260914a/resume.pt` | 38912890 | `/mnt/home2` | 保留 |
| `route-b-adam-100-20260914a/resume.pt` | 38912890 | `/mnt/home1` | 保留 |
| `route-c-adam-100-20260914a/resume.pt` | 38926202 | `/mnt/home2` | 保留 |
| `route-c-sgd-10-20260914a/resume.pt` | 35680186 | `/mnt/home2` | 保留 |
| `scale-init-c-10-20260914a/resume.pt` | 21881786 | `/mnt/home1` | 保留 |
| `verifier/pytest-distill-first-20260914/test_distill_resume_optimizer_0/resume.pt` | 456372 | `/mnt/home1` | 保留 |
| `verifier/pytest-distill-first-20260914/test_distill_resume_rejects_ma0/resume.pt` | 196027 | `/mnt/home1` | 保留 |
| `verifier/pytest-distill-first-20260914/test_distill_resume_rejects_ma1/resume.pt` | 196472 | `/mnt/home1` | 保留 |
| `verifier/pytest-distill-first-20260914/test_distill_resume_rejects_ma2/resume.pt` | 196664 | `/mnt/home1` | 保留 |
| `verifier/pytest-distill-first-20260914/test_distill_resume_rejects_ma3/resume.pt` | 152120 | `/mnt/home1` | 保留 |
| `verifier/pytest-distill-second-20260914/test_distill_train_builds_teac0/resume.pt` | 202810 | `/mnt/home2` | 保留 |
| `verifier/pytest-sixth-20260914/test_actual_training_loop_clam0/resume.pt` | 326970 | `/mnt/home1` | 保留 |
| `verifier/pytest-sixth-20260914/test_save_resume_interruption_0/resume.pt` | 135994 | `/mnt/home1` | 保留 |
| `verifier/pytest-sixth-20260914/test_save_resume_parameters_op0/resume.pt` | 98234 | `/mnt/home1` | 保留 |
| `verifier/pytest-sixth-20260914/test_save_resume_parameters_op1/resume.pt` | 326970 | `/mnt/home1` | 保留 |

## mergerfs 分支空间：理论估算，不是训练容量承诺

只读 `os.getxattr('/home/.mergerfs', ...)` 实测返回：

```text
user.mergerfs.branches = /mnt/home1=RW:/mnt/home2=RW
user.mergerfs.category.create = mfs
user.mergerfs.minfreespace = 107374182400
```

即 `/home` 来自 `/mnt/home1:/mnt/home2`，每分支新建门槛 **100 GiB**，不能拿两分支 aggregate 空闲判断是否可新建。沙箱 `findmnt` 所见 `/home` 为 `fuse.mergerfs`、SOURCE `1:2`，且显示沙箱只读视图；分支及真实策略以以上运行时只读 xattr 为依据，未改挂载。

候选A的 `user.mergerfs.fullpath` 为 `/mnt/home1/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-sgd01-100-20260914a/resume.pt`；候选B为 `/mnt/home2/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-sgd0001-100-20260914a/resume.pt`。仅对这些确切底层路径只读核对 size/allocated，无底层写入。

以下 `available = f_bavail × f_frsize`，采样时刻同本文首行。理论删后值 = 当前 available + 对应候选已分配块；假设其块均可回收、无并发变化，未减去随后新建本文的少量空间。

| 分支 | 当前 available bytes / GiB | 指定候选理论释放 bytes | 理论删后 bytes / GiB | 删后相对100GiB |
| --- | ---: | ---: | ---: | ---: |
| `/mnt/home1` | 95,139,962,880 / 88.605995178 | 7,789,551,616 | 102,929,514,496 / **95.860580444** | 仍低 **4.139419556 GiB** |
| `/mnt/home2` | 116,833,107,968 / 108.809310913 | 7,789,551,616 | 124,622,659,584 / **116.063896179** | 高 **16.063896179 GiB** |

**仅删这两文件不会让两个分支都达到门槛。** 按此瞬时估算 home2 写入一个约10.879636 GiB的新Adam恢复文件后约剩105.184260 GiB，但原子替换时旧文件与临时文件可并存，另有packed评测包、其他任务和持续日志写入；因此不保证后续保存，更不保证并发训练空间。未调查所有进程的开放文件句柄、文件系统共享块或未来写入，不能承诺 unlink 后立即获得全部理论空间。本提案不建议绕过100GiB策略，不追加其他清理候选，也不启动训练。

## 原审批前交接状态

仅两候选，合计文件长度15,579,088,840 bytes（14.509157129 GiB）；完整评测包、源码、配置、日志及失败证据全部保留，Phase2不动。A采用已确认的真实淘汰依据：25更新完整 PPL6709.146967144104、NLL8.81122709329167、252728 targets，灾难性变差；随后训练前向中断、100更新未完成。旧“无完整25分”判断已被主执行更正，本提案同步纠正，候选范围不变。

原提案交接时为0文件删除、等待具体批准；后续明确批准及执行如下，不把原等待状态当作当前状态。

## 2026-09-14 08:47:55 +08:00：用户批准后执行

- 当前用户原话：“可以删除吧，同时你帮我看一下/home/dongpeiyan/projects/rotation-quant copy这个文件是不是和目前的/home/dongpeiyan/projects/rotation-quant有很多重复文件”。删除授权承接本对话唯一两个SGD `resume.pt`；copy仅获只读比较授权。
- 通过 `/home` 合并挂载路径，以两个准确绝对路径执行 `rm -v --`，未用通配符、递归删除或底层分支绕行。命令成功，两个准确路径均再次确认不存在。
- 删除文件长度合计 **15,579,088,840 bytes / 14.509157129 GiB**；删除前已分配块合计15,579,103,232 bytes。
- 删除前后 `df -B1`：home1可用95,137,955,840 → 102,927,249,408 bytes；home2可用116,829,462,528 → 124,619,014,144 bytes。瞬时净增加 **15,578,845,184 bytes / 14.508930206 GiB**，与文件块数的小差异属于同时段文件系统变化，不能承诺归因给本操作。
- 删除后分支可用约 **95.858471 / 116.060501 GiB**；home1仍低于100GiB新建门槛，未改策略，不保证后续或并发训练保存空间。
- 随后只读确认：A的完整25更新 `static_w4a8.pt`（2,064,318,486 bytes）、`validation.json`（1160 bytes）、`failure.json`（68 bytes）仍在；B的25和100更新静态包各2,064,318,486 bytes仍在。
- 最佳reference200恢复文件11,681,920,128 bytes、center225恢复文件11,681,920,000 bytes、center原200恢复文件11,681,919,936 bytes均仍在。完整评测包、JSON、源码、配置、训练记录和日志不在删除命令范围。
- **最终仅删除2文件；Phase2未删除或改写，copy未删除、移动或改写。** 无测试、模型加载、CPU/GPU实验、安装、提交推送、进程终止或挂载修改。本段仅补记清理文档。
