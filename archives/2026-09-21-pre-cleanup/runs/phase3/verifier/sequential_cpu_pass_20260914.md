# sequential_postprocess：限定 CPU PASS

## 结论与执行计数

新增 sequential driver、launcher sequential 路由及父包来源传递，在下述范围内 **PASS**。**新增7个用例**，没有尚未解决的应用 FAIL。

- 首轮仅选择新 sequential 用例：`6 passed, 1 failed, 88 deselected`，12.90秒。
- 唯一失败为 verifier 的helper来源断言未解包 `torch.no_grad` 装饰器，取到了torch包装函数的globals，而非原coordinate函数。改用 `inspect.unwrap`，未修改应用代码。
- 同时修正driver测试数据：此前按vocab取模令不同窗口内容相同；增加合法的逐窗独立token标记，避免索引错位仍通过比较。只复跑修正的helper用例与两个driver分支：`3 passed`，12.54秒。
- 当前7个不同的新用例均已通过；不是单轮7项全量复跑。旧54/13/55及其它已测套件没有重跑。

本报告不授予GPU精度、显存、生产坐标搜索收益或Phase3完整实验PASS。应用源码只读；verifier只扩展授权测试文件，测试包/日志/报告只写本目录。未运行GPU查询/forward/训练、安装、环境修改、提交、推送或进程操作。

## 已验证范围

### 1. 完整窗口、量化后输入与hook清理

`tests/test_phase3_joint.py:2123`：`test_sequential_capture_full_quantized_windows_and_exception_cleanup`

- 实际ActQuantWrapper/inner Linear pre-hook，32个各2048-token输入窗口，独立INT8 round/clamp oracle确认捕获的是wrapper量化后的inner Linear输入，不是原始输入。
- 捕获全部65536行，分割精确为24窗49152行/8窗16384行；不是前128 token或每8行抽样。
- 预定capture异常确实阻止inner Linear后续执行；其它ValueError和RuntimeError原样传播，且hook在finally中移除。
- 从未访问目标模块时拒绝不完整capture，仍清理hook。模型state不变。
- 模型为小型真实wrapper fixture；CUDA方法仅映射到CPU边界，实际捕获与量化在CPU运行。

### 2. 候选评分后精确回滚

`tests/test_phase3_joint.py:2165`：`test_sequential_candidate_weight_alpha_exact_rollback_on_failure`

- 同时修改一个weight候选和一个SP2 alpha，实际调用score_candidate/apply_records/selection_score。
- 正常返回、注入evaluation异常及非有限NLL三种情况后，模型state均精确恢复，父packed/SW/shape记录不变；没有通过alpha反算替代原scale快照恢复。
- 非有限NLL不作为候选分数继续使用。state比较沿用零容差equal_nan=True处理既有dormant sentinel，不放宽有效数值的误差容差。

### 3. 逐层round：MSE只预筛、真实NLL决定保留

`tests/test_phase3_joint.py:2194`：`test_sequential_round_nll_prefilter_parent_reuse_and_accepted_capture`

- 实际round_module/score_candidate及候选应用/恢复；coordinate候选与NLL在本控制流测试中mock。
- 通过实际rounding_helper确认加载外层纯torch fixed_grid_rounding.py，并用其真实函数签名绑定调用参数；保留parent initial_codes、固定SW、neighbor_search和指定milestones。
- mock步骤512具有最佳局部MSE但NLL较差，被拒绝；2048的MSE非最优但NLL改善，被保留；8192的MSE未过父基准，不调用NLL。
- step0仅复用当前prefix分数，不重复评测。下一模块capture被实际调用时模型已含上一模块胜出权重；下一模块候选NLL全差则保留step0和原权重。
- 验证当前进度适配器忽略helper的step16回调、仅step128落盘，数学候选不因此改变。
- 没有重新执行生产规模8192步坐标搜索；该helper的既有数学测试不重复。

### 4. 原BF16-logit选择口径与16376 targets

`tests/test_phase3_joint.py:2262`：`test_sequential_selection_actual_bf16_evaluator_and_16376_targets`

- 实际单层BF16 tiny Llama，实际8个完整2048-token窗口。直接计算BF16 logits、BF16 CE(reduction=none)，损失转FP32再按窗口/目标平均，最后使用log(float32 PPL)作独立数值期望。
- 实际调用新selection_score、common.full_validation、纯acceptance和新worktree原evaluator；仅将evaluator的设备实参从cuda映射为cpu，模型前向/CE/evaluator未mock。
- 结果与上述BF16口径精确一致，evaluator调用确实包含16376个预测targets、bsz1、eval_nsamples=None。没有调用FP32训练probe口径。
- 这是tiny模型CPU数值校验，不是生产W4模型的train NLL或GPU完整validation结果。

### 5–6. round/SP2 driver、compact prefix与来源传递

`tests/test_phase3_joint.py:2299`：`test_sequential_driver_prefix_resume_data_cold_validation_and_provenance[round/sp2]`

使用真实16层/112-wrapper frozen tiny包，实际run、round_module或range_module、score_candidate、save/restore_prefix、save_frozen和load_static。候选coordinate、本测试中的NLL返回值及源码快照/进度记录边界被mock；新输入窗口带独立标记。具体检查：

- 忽略data_windows另返回的旧128-token calibration，按calibration_window_indices重新读取32个完整train窗口；顺序及全部2048 tokens精确匹配。data.json记录24/8索引、49152/16384行、16376预测targets。
- round逐候选NLL选择，不把更好的MSE自动当NLL收益；SP2直接对每层alpha候选做实际range_module→score_candidate→selection_score调用，不构造局部MSE组合；factor1只复用当前分数。
- 第一个目标接受后，第二个目标开始时已看到其weight或SP2改变。此处注入中断，确认只保存第一目标完成的prefix，不导出或调用validation。
- compact prefix保存正确完整targets顺序、已完成列表、初始/当前NLL；round仅含已变更模块的packed记录，SP2 weights为空；另保存16张down scale，不保存整套HP/master参数。正常保存后无临时文件残留。
- 新run从该prefix恢复，仅执行尚未完成的第二个目标，不重复第一个目标、不重新计算初始/0-step/factor1分数；返回new_module_results也只包含第二目标。
- completed_targets不构成targets前缀时拒绝恢复，目标模型与records不变。正常恢复会更新用于最终导出的内存records，这是预期行为；原父包磁盘字节不变。
- 全部112 SW和96 non-down SA保持不变；SP2路径全部weight codes保持不变；round路径未选择的weight codes保持不变；最终HP状态与父一致（dormant NaN按equal_nan处理）。
- 仅接受train前缀后导出，再实际cold-load新的完整冻结包一次。最终validation callback使用独立252852-token/252728-target输入；没有把这些输入传给捕获或候选评分。
- 故意让mock最终validation NLL差于train选中值，driver仍保留已由train决定的结果，不用validation重新选择。SP2另检查全部候选NLL变差的分支：不导出、不调用validation，状态明确NOT TESTED。
- provenance使用实际父包metadata：round父记录为PTQ来源，SP2父明确记录quantization-aware distillation及original_backbone_trained=true。settings、result、validation及最终packed包metadata全部保留同一parent_metadata；stage只称本后处理无梯度，不把QAT父包重新归类为原始冻结权重PTQ。
- prefix临时文件→replace的原子写路径已只读审查并执行正常保存；本轮未额外注入磁盘写入中断或测试任意损坏prefix格式。

### 7. 新launcher路由

只运行既有mock launcher测试新增的`[sequential]`分支：driver=sequential_postprocess.py，parent/mode参数转发正确，保留显式子环境、shell引号、PID/session/log和run名唯一保护。没有真实tmux或nvidia-smi调用，未重跑旧路由。

## NOT TESTED

- 生产B100/QAT父包的完整GPU前向、实际候选NLL/PPL、GPU显存、真实跨设备embed/norm/head迁移和异常恢复。
- 生产49152/16384行的大矩阵坐标搜索、全部层/8192坐标耗时及搜索收益、正式完整252728-target GPU validation。
- 真实数据缓存重新分词/索引重建、生产长任务中断恢复、任意损坏或手工编辑prefix的全部校验。driver数据流与prefix测试仅覆盖明确列出的情况。
- 没有将Phase2历史收益、用户报告的GPU结果或本轮tiny CPU数值当作新生产实验验收。

## 证据与复现

- 源码根：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。
- 已有HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`；不代表未提交文件内容，未新增hash。
- 实读mtime：sequential_postprocess.py `2026-09-14 08:38:05.021029382 +0800`；launch.py `2026-09-14 08:35:18.103726748 +0800`。本轮包含parent provenance和每128坐标进度的新代码。
- 首轮日志/XML：`runs/phase3/verifier/cpu-sequential-first-20260914.log`、同名`.xml`。
- 定向复测日志/XML：`runs/phase3/verifier/cpu-sequential-second-20260914.log`、同名`.xml`。
- 临时产物：本目录`pytest-sequential-first-20260914`、`pytest-sequential-second-20260914`；首轮测试失败证据保留。
- 两轮warning均为已有Transformers quantized-training API deprecation，不是应用失败。

工作目录为源码根，共同命令前缀：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider
```

首轮追加`tests/test_phase3_joint.py -k sequential -q`；第二轮只指定：

```text
tests/test_phase3_joint.py::test_sequential_round_nll_prefilter_parent_reuse_and_accepted_capture
tests/test_phase3_joint.py::test_sequential_driver_prefix_resume_data_cold_validation_and_provenance
```

均指定本目录对应`--basetemp`与`--junitxml`，用`set -o pipefail`和tee保留原始输出。

本轮定向代码核验结束。后续仅按真实变更/具体失败风险复用verifier，不重复正式GPU审计或完整旧测试套件。
