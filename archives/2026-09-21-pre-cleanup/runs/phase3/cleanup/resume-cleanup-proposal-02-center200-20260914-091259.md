# 第二份单文件清理提案：center200 resume.pt（未批准／暂缓）

盘点范围仅为下列center200恢复文件；元数据与空间快照时间 **2026-09-14 09:12:59 +08:00**。本轮新增删除0、移动0、实验文件改写0；只新增本文，不执行或申请删除。

## 最新主线与暂缓结论

写入本文前，主线程已明确纠正下一步：**先完成Phase3 B100按Phase2方法的离散W4 → SP2 → 局部D，再从新的优胜PTQ冻结包适配蒸馏。旧ref200续训不再是当前下一步。** 本条最新调度优先于STATUS/RESULTS中较早的旧ref200续训计划。

因此本提案仅保留已经发生的只读盘点，状态为 **未批准、暂缓，不推进删除审批或执行**。不为启动旧ref200续训而释放空间；主端后续若按新父包计划确需空间，再另行明确处理。原两个SGD恢复文件的已批准执行/确认记录保持原样，不重复删除，也不将其批准扩展到center200。

## 唯一盘点候选

| 项目 | 实测值 |
| --- | --- |
| 确切目录 | `/home/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-adam2e5-200-20260914a` |
| 唯一候选文件 | `resume.pt`；不是整个目录 |
| 完整路径 | `/home/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-adam2e5-200-20260914a/resume.pt` |
| 当前状态 | 仍存在；没有删除授权 |
| 文件长度 | **11,681,919,936 bytes / 10.879635751 GiB** |
| 已分配块 | **11,681,927,168 bytes / 10.879642487 GiB** |
| 文件类型/所有者/链接数 | 普通文件，dongpeiyan，`nlink=1` |
| mergerfs实际basepath | `/mnt/home1` |
| 底层路径（仅只读定位） | `/mnt/home1/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-d-adam2e5-200-20260914a/resume.pt` |
| 恢复元数据 | `step=200`；仅读取ZIP内data.pkl操作码，没有加载张量或运行pickle对象 |

## 数值证据及不应作出的推论

本节center相对证据路径均位于上表确切目录。

- `result.json:2` 为 `completed_steps=200`，`:29` 的最终节点完整 NLL **2.7515042497043507**、PPL **15.666180015269509**。`progress.json:2` 为completed、`:10` 为200；外置 `runs/phase3/distill-b100-d-adam2e5-200-20260914a.log` 最后一条也记录completed/200，原PID1105455目前不存在。
- `checkpoint-0200/validation.json:1` 记录252852输入tokens、252728预测targets、123个2048窗及948-token尾窗、0未计尾tokens；`:23` 为上述NLL/PPL。不是训练probe或浮点master评测。
- ref200证据 `runs/phase3/distill-b100-d-adam2e5-ref-200-20260914a/checkpoint-0200/validation.json:23` 为 NLL **2.7459188570552207**、PPL **15.578922160109332**、252728 targets。center200相对ref200 PPL **+0.08725785516017659**、NLL **+0.00558539264913005**。
- 这是两个200更新节点的已测配置比较，不是“所有中心初值续训都被精度证伪”。center200自身100→200仍从PPL15.999590911200382改善到15.666180015269509；不能仅凭其不是全局最好就判为无价值。
- `runs/phase3/RESULTS.md:81` 的seq-QAT SP2完整PPL15.61729328754997属于ref200父包的另一后处理实验，不是center225的评测。它不更新当前已测最佳，也不能拿来否定center225。

## 活跃读取与计划引用核查

### 已查到的历史引用

`runs/phase3/distill-b100-d-adam2e5-continue800-20260914a/settings.json:5` 的 `resume` 及同名 `.launch.json` 真实命令均明确引用此center200文件。该历史启动PID为1358170，目前不存在；原center200 PID1105455也不存在。**因此不能写“没有任何引用”**，只能区分历史启动和当前活跃使用。

center续训的旧progress仍为233；既有训练日志记录到234，但实际恢复元数据为225。历史设置中的800是计划/schedule长度，不是实际完成数。保存center225可保留后来轨迹上的恢复点，但不能替代原center200的精确起点。

### 活跃使用快照及可见性边界

- 沙箱内对该唯一文件运行只读 `lsof` 未返回句柄、exit=1，但工具同时报告部分挂载不可见，因此没有用这个结果单独声称“全机无人读取”。
- 经受审查的只读宿主查询，**09:12:16 +08:00** 检查本用户UID1012的43个进程，以 `/home` 文件与真实basepath文件的设备/inode身份比对打开fd，并查找映射记录；未找到可见的匹配打开句柄或映射。
- 仍有本用户PID **10013、1898575、3272743** 的fd/maps不具读取权限；未越权读取，也没有调查其他用户历史。**结论仅为当前可见范围未发现活跃读取，不能保证全机绝对不存在读者。** 本次无终止进程、无GPU查询/forward/实验。
- 宿主可见的Phase3 Python任务为PID **2122482**，运行 `sequential_postprocess.py` 的 `seq-b100-down-round-20260914b`，父包和reference-state分别为B100 `checkpoint-0100/static_w4a8.pt`、`state.pt`，没有center200参数。
- 只读检查Phase3直属run的 `settings.json` 和顶层 `.launch.json`，精确匹配此resume完整路径，仅发现上述历史center续训引用。当前STATUS/RESULTS中的早期ref200续训计划随后已被本次最新主线纠正覆盖；**最新明确计划不是启动center200或旧ref200续训**。

本记录不是永久无引用证明或自动删除条件；所有观察都是瞬时、有限范围证据。当前调度已要求暂缓，不申请额外权限来强行得出全局否定结论。

## 31项完整保留清单

除唯一候选resume外，该center200 run当前其余29个文件及外置2个文件全部保留，不移动、不压缩、不截断、不改写。以下run内路径均相对于上表确切目录。

| 保留类别 | 确切相对路径/内容 | 数量 |
| --- | --- | ---: |
| 完整25更新评测包 | `checkpoint-0025/static_w4a8.pt`、`checkpoint-0025/validation.json`、`checkpoint-0025/training_probe.json`、`checkpoint-0025/code_changes.json` | 4 |
| 完整100更新评测包 | `checkpoint-0100/static_w4a8.pt`、`checkpoint-0100/validation.json`、`checkpoint-0100/training_probe.json`、`checkpoint-0100/code_changes.json` | 4 |
| 完整200更新评测包 | `checkpoint-0200/static_w4a8.pt`、`checkpoint-0200/validation.json`、`checkpoint-0200/training_probe.json`、`checkpoint-0200/code_changes.json` | 4 |
| 其他probe | `checkpoint-0010/training_probe.json`、`checkpoint-0050/training_probe.json` | 2 |
| 源码/dirty记录 | `source/common.py`、`source/distill.py`、`source/launch.py`、`source/postprocess.py`、`source/quantization.py`、`source/run.py`、`source/tracked.diff` | 7 |
| 配置/数据/覆盖 | `settings.json`、`data.json`、`parameter_coverage.json` | 3 |
| 训练与结果证据 | `initial_probe.json`、`checkpoints.json`、`training.jsonl`、`progress.json`、`result.json` | 5 |
| 外置日志/真实启动命令 | `runs/phase3/distill-b100-d-adam2e5-200-20260914a.log`（77,757 bytes）、`runs/phase3/distill-b100-d-adam2e5-200-20260914a.launch.json`（4,117 bytes） | 2 |

三个完整静态包各 **2,064,318,486 bytes**；本次只核对文件/已有记录，不新做冷载验证。父D包、三路线/尺度初始化及原项目Phase2不在本提案清理范围。

必须保留的两个恢复文件本次均仍存在：

| 文件 | bytes | 只读恢复元数据 | 说明 |
| --- | ---: | ---: | --- |
| `runs/phase3/distill-b100-d-adam2e5-continue800-20260914a/resume.pt` | 11,681,920,000 | **225** | 有效恢复节点必须保留；并未完成800，尚无此续训分支新增完整PPL，不称被精度证伪 |
| `runs/phase3/distill-b100-d-adam2e5-ref-200-20260914a/resume.pt` | 11,681,920,128 | **200** | 已测最佳ref200状态必须保留；不等于当前计划立即续训它 |

## 如果未来另获批准删除，会失去什么

center200源码快照 `source/distill.py:140` 保存112个FP32 master权重、224个学习尺度、全部优化器state、step/数据参数及Python/Torch/CUDA RNG；`:154` 为对应恢复逻辑。本配置主权重和尺度均使用Adam。

删除此单文件将失去**从中心初值恰好200更新时刻的FP32权重/尺度、Adam一阶/二阶状态及步数、RNG和数据游标上下文精确恢复**的能力，不能再原样重启200→后续的调度对照或其他从中心200出发的实验。

center225已经包含后续更新和延长schedule后的优化历史，不能倒推回完全相同的center200优化器起点；静态W4A8 checkpoint-0200可保留冷载评测能力，但不包含可重建全部FP32残差/优化器/RNG的训练状态。这是真实的信息损失，不因保留center225而消失。

## 实际空间与非执行性估算

本次运行时只读xattr：`/home` 分支 `/mnt/home1=RW:/mnt/home2=RW`，`category.create=mfs`，`minfreespace=107374182400 bytes`（100GiB）。

| 分支 | 09:12:59实际available bytes / GiB | 仅作信息的假设删后bytes / GiB |
| --- | ---: | ---: |
| `/mnt/home1` | **138,529,439,744 / 129.015594482** | 150,211,366,912 / 139.895236969 |
| `/mnt/home2` | **157,721,702,400 / 146.889781952** | 不变：157,721,702,400 / 146.889781952 |

假设删后仅将该文件已分配块加到home1，不是实际已释放，不保证并发变化、开放文件/共享块或将来checkpoint容量。**本轮实际释放0 bytes。**

需要纠正时间背景：本子会话此前另一项用户直接授权的copy整理已在09:08:04完成，重复文件71,127,301,942 bytes被从copy移除，独有文件迁入主项目新 `WXD-imported-rotation-quant-copy-20260914/`。这不是本次center200盘点造成，也不是再次删除SGD。执行清单/前后空间记录已在该独立归档的 `execution-summary.json` 中；本次没有追加该整理或扫描copy。

因此本次实测余量已明显高于早先“单文件新resume即跌破门槛”的快照；不能照抄旧空间紧迫性作为继续清理center200的理由。后续容量应随**新PTQ父包路线**和实际checkpoint/临时文件需求由主端重新规划，本提案不启动训练、不下空间保证，也不提交删除请求。

## 交接

唯一候选仍在，center225和best ref200仍在，原两SGD确认记录保留。**第二份提案已完成只读盘点；未获删除批准，按最新主线暂缓，0新增删除。**
