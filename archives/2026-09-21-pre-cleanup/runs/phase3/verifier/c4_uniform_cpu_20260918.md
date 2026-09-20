# C4 matched uniform INT8：独立 CPU verifier

日期：2026-09-18。最终结论：**限定 CPU PASS；首轮7项通过，F001由主执行修复后仅复测失败项1 passed/10.16s，合计8个独立测试实例全部通过，无未解决应用FAIL。**

后续正式a/b校准因历史absmax不匹配拒绝，没有成功overlay/C4；原CPU PASS范围保留。主执行新增临时V布局恢复补丁后，另获**新增窄项CPU PASS：成功/异常2实例，2 passed/8.98s，原8项未重跑**。详见文末布局复核；不将CPU PASS称为GPU根因已证实。

最新补证：仅只读主执行正式c的capture_metadata/result，确认**16/16层absmax及rows精确匹配历史字段，无容差差异**。这是主端GPU产物的独立只读核查，不是verifier独立GPU复跑；未新增测试。

本结论允许主执行继续已授权的均匀INT8校准与第三项C4；不代表真实大模型校准或PPL已经由verifier验证。

## 边界

仅写 `worktrees/SpinQuant-phase3-joint/tests/test_phase3_uniform_baseline.py` 与本报告/.log；不改应用、既有 tests/status，不创建 goal、不启动 GPU/tmux、不安装/提交/推送/删除。已启动的 PTQ 父包及 B100-SP2 C4 不重复，旧 BF16/QAT400 分数不重测。

已只读项目约束及 Phase3 STATUS/C4_ATTRIBUTION；源码唯一 `worktrees/SpinQuant-phase3-joint`，HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，继承 dirty/untracked 实现，未重置。无更深 AGENTS。

## 验证覆盖

1. 实际 train Arrow、逐行无BOS/EOS分词拼接，严格重建 B100 data.json 的32个索引×128tokens；禁止 C4/validation/test 加载。
2. down 全部临时 bits16，pre-hook按每8行采样、最大值来自完整输入；成功和异常后恢复 bits及原 hooks。
3. 实际 RotationStaticActQuantizer 的 signed INT8/BF16舍入与独立算式逐值对照；50次真实候选 forward、33粗点＋围绕本格式最优17细点、输出MSE选择，与历史SP2预算核对。
4. 小overlay来源/尺度校验、仅down替换及父包/W4/SW/96非down不变、CPU tiny外部C4 driver与CLI/launcher兼容定向测试均通过。

只执行新文件8个定向测试，首轮失败后仅复测失败项；没有运行旧测试文件/全套或GPU保险测试。临时测试数据使用 `/tmp`，不覆盖正式产物。CPU数学/接口证据与大模型正式测量分开。

## 首轮实际结果与F001

收到主执行补丁完成通知后，已审查实际 `uniform_baseline.py` 及修改前后 external/launch diff。新增overlay读取是显式opt-in，默认96INT8＋16SP2路径保留；原模型冷载、token loaders与evaluate_tokens/CE/NLL未改，common序列化未改。

实际执行仅新文件 `tests/test_phase3_uniform_baseline.py` 的8个测试实例：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
tests/test_phase3_uniform_baseline.py -q -s
```

工作目录为 Phase3 worktree；exit1，7 passed/1 failed/5既有transformers弃用warnings，28.93s。

**F001（应用错误）**：`uniform_baseline.py:180` 的 `progress(args.output, "completed", **result)` 将 result 中的 `peak_allocated_gib` 传给 `run.py:70`，而 progress 构造 dict 已提供同名字段，触发 `TypeError: dict() got multiple values for keyword argument 'peak_allocated_gib'`。该异常发生在小overlay及result.json写出后，故不能把文件存在当成功结束。已立即通知主执行修应用；verifier没有改应用、没有放宽测试或mock掉progress。

首轮已PASS：实际 train 2435022 tokens/1188完整窗；按B100原32索引逐token匹配32×128＝4096校准tokens；采样成功/异常恢复；signed INT8/BF16真实数学与50次候选forward；16层历史SP2各50候选/512 sampled_rows预算核对；overlay下仅down变化及错误来源/尺度拒绝；tiny C4默认及overlay路径、冻结状态和CLI/launcher。

首轮失败的calibration driver测试已运行16层真实小矩阵搜索，但在完成通知处抛错，后续保存内容断言及历史rows/candidate-count/absmax不匹配拒绝测试当时未到达。此处保留首轮失败事实；现已由下述单项复测完成覆盖。

## F001关闭与最终PASS

主执行修复 `uniform_baseline.py:180`：完成progress仅传 `down_layers=len(scales)` 与 `elapsed_seconds=result["elapsed_seconds"]`，不再展开result；result.json仍保存原完整字段。实际核对修复源码为10841 bytes、mtime `2026-09-18 00:26:10.949652878 +0800`。verifier没有修改应用或测试以绕过F001。

按用户要求仅执行原失败项，环境与首轮相同：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
tests/test_phase3_uniform_baseline.py::test_uniform_calibration_driver_history_matching_and_small_overlay -q -s
```

**1 passed, 1 warning in 10.16s，exit0**；唯一warning是既有transformers弃用提示。此前7项没有重跑。

该项现已实际确认：16层每层50个真实INT8候选搜索后正常完成；输出仅含parent/calibration_metadata/16尺度的小overlay（小于64KiB），每层选中尺度与range_search记录一致；父模型完整state精确不变，父包字节不变。合成历史记录的sampled_rows、候选数或absmax分别不匹配时拒绝，且不写成功overlay/result。这里使用tiny CPU模型和合成capture/history来验证driver逻辑，不冒充真实B100历史absmax已匹配。

最终**8项限定CPU PASS，F001关闭，无需应用进一步修复**。报告与原始日志保留首轮失败和单项复测全过程。

## 修改路径

- `worktrees/SpinQuant-phase3-joint/tests/test_phase3_uniform_baseline.py`（本轮复测未再改测试）
- `runs/phase3/verifier/c4_uniform_cpu_20260918.md`
- `runs/phase3/verifier/c4_uniform_cpu_20260918.log`

## NOT TESTED

真实大B100量化校准、真实捕获与历史SP2 absmax实际匹配、三项正式C4 PPL/NLL、GPU/native/decode/手机部署均未由此verifier测试。tiny C4 evaluator分数为mock；主执行提供的新GPU结果未由本轮CPU测试复现。没有重复任何已运行的正式任务。

## 后续V布局源链与窄项核验

### 已核对事实

- `b100-uniform-calibration-20260918b/capture_metadata.json` 实际16层sampled_rows均512、历史候选均50；13层full_absmax不同。layer0 **28.25 vs28.125**，layer1 **824 vs816**，layer7 **2.90625 vs3.015625**（相对约-3.62694%）。b有failure.json，无result.json或down_int8_scales.pt。a/b是捕获匹配失败，不是有效校准。
- 主执行先新增的capture_metadata日志与a源码比较只多一个JSON写出；原校验阈值rel1e-6/abs1e-7、候选搜索没有变化。此纯日志新增仅review，未因此重测旧8项。
- 历史B100 `source/common.py` 与当前逐字节一致；`run.py`差异是后续optimizer-scale-reference支持，不涉及原freeze/calibrate/reload路径。
- `train_utils/quant_linear.py:35` 的V/R2/transpose=False分支在matmul后 `reshape(...).t()`；`common.py:323` 将量化重构结果通过 `destination.module.weight.data = ...` 安装，继承该结果布局。float/to/cpu默认preserve_format，源码中没有在此强制contiguous。
- `quantization.py:84` 的pack将codes展平成逻辑行序，包仅保存packed/shape/scale，不保存矩阵stride；`common.py:439` 的reload是 `.copy_`，写数值而不替换目标布局。`postprocess.py:26` 则先新建HF模型，再reload，不能由相同包推导与在线模型相同stride。
- `run.py:190` 到`:195` 的历史reload_check是在**同一个在线frozen对象**上evaluate→reload→evaluate。其before/after NLL精确相等不能排除新建冷载对象的布局差异。

### 新补丁审查与独立CPU证据

主执行修改 `capture_down_inputs`：try内只对16个V权重保存原 `.data` 引用并转换 `.t().contiguous().t()`，finally恢复原引用；down bits/hooks恢复保留，模型数值/尺度、搜索与外部评测路径不改。result追加capture_layout描述。verifier未改应用。

新增唯一测试函数 `test_uniform_capture_v_layout_restores_original_storage`，参数化success/exception，仅执行这2个实例。使用16-layer tiny CPU fixture，backbone为检查布局并触发真实down wrapper hooks的stub，不做大模型forward。

- 成功用例另执行真实 `QuantizeLinear.rotated_weight`、int4_codes/pack/unpack及重构子链：tiny V形状 `[4,8]`，在线结果stride **[1,4]**，行主序目标copy_后stride **[8,1]**，值逐位相同。该证据确认布局机制，不证明GPU舍入后果。
- capture期间全部16个V权重列主序、数值/dtype/device精确不变；非V参数storage与stride不变，非down bits不变；Parameter身份及wrapper.weight别名保持。
- 正常返回和注入forward异常两种路径均恢复原data_ptr、storage base pointer、storage_offset、stride、Parameter身份/别名、完整state、bits及原有hooks。没有只比较值而漏掉storage恢复。

实际命令（相同worktree与离线CPU环境）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
tests/test_phase3_uniform_baseline.py::test_uniform_capture_v_layout_restores_original_storage -q -s
```

**2 passed, 2 warnings in 8.98s，exit0**；warnings为既有transformers弃用。原8项未重跑，没有新增GPU/完整模型重建/应用改动。

### 结论与尚未证实项

**源链机制与临时布局补丁限定CPU PASS。** 在当前完整模型配置中V为512×2048，按此链预期在线stride(1,512)、冷载stride(2048,1)，但本verifier没有实际构造两份GPU大模型读取其stride或比较GEMM输出。布局改变是否足以解释a/b真实absmax偏差，截至该CPU核验时仍是根因假设；后续正式c补证见下一节。

最小修正范围应继续限于calibration capture期间，恢复完成后再搜索/外部评测；不改common序列化、不改父包、不放宽历史匹配阈值、不重跑已完成C4。c若通过原rows/absmax检查，支持恢复了该历史捕获统计；absmax一致本身仍不等价于全部中间输入逐位一致。

## 正式c主端GPU产物：仅只读补证

按用户通知，仅读取 `runs/phase3/b100-uniform-calibration-20260918c/capture_metadata.json` 和 `result.json`，未启动测试/GPU、未加载模型或overlay张量。

- capture_metadata实际16层：**absmax精确相等16/16、rows精确相等16/16，每层512 rows，历史候选均50**。无差异层，不是仅在rel/abs容差内通过。layer0为28.125、layer1为816、layer7为3.015625，均恢复历史值。
- result记录：父包为B100 checkpoint-0100；WikiText train、原32索引×128＝4096校准tokens，禁BOS/EOS；`down_layers=16`、`candidates_per_layer=50`、`historical_capture_matched=true`、`parent_scales_unchanged=true`、`c4_used=false`、`training=false`；耗时**23.186752076027915s**，peak allocated **2.9195966720581055GiB**。
- 上述精确相等由本verifier直接比较JSON数值确认；候选条数及原quantizer不变在本次补证中依据result记录，**未重复检查range_search/overlay/quantizer快照/96SA张量**，这些实际产物配对由主执行负责。

结论：正式c证据支持临时V列主序恢复解决了a/b的历史捕获统计不匹配，未放宽阈值。**证据属于主端真实GPU运行，不是本verifier独立GPU复跑，也不证明全部中间激活逐位相等。** 第三项C4的完成状态与PPL未在本次读取范围内，不能从已启动推断完成。
