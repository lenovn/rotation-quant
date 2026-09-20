# D / SP2 配对范围与双项排序：限定 CPU PASS

## 结论与实际执行

本次新增 `pair_range_trials`、D/I 配对候选选择及双项排序，在下述小型 CPU 数值/控制流范围内 **PASS**，没有发现阻塞该已测实现的具体问题。不代表 D 在生产模型上有净精度收益，也不代替正式 GPU 审计。

- 第一轮只运行新 pair 数值 oracle 和配对搜索的两个分支：**3 passed，6.55 秒**。
- 用户补充双项排序断言后，只扩展并重跑配对搜索两个分支：**2 passed，6.46 秒**。
- 共 3 个逻辑用例，后 2 个因新排序断言复跑；不是 5 个不同用例。两轮均使用当前 13 倍率；旧 9 点版本没有运行或另计验证。
- 未重跑已有量化/冷载/postprocess 11 项或 55 项套件。应用代码只读，仅修改授权测试文件并在 verifier 目录写测试日志和报告。未启动 GPU、真实模型搜索、训练、probe/validation 或 tmux，未安装/修改环境/提交/推送。

## 数值 oracle

`tests/test_phase3_joint.py:1178`：`test_postprocess_pair_range_bf16_oracle_and_no_mutation`

- 使用小型真实量化模块的 up/down packed INT4 记录，分别测试父记录和非 I 融合记录。
- 独立显式构造 `BF16(codes.float() * SW)` 权重，执行 BF16 up linear、gate 相乘、独立最近 SP2 格点 oracle、BF16 down linear；不调用被测 `sp2_project` 或 `pair_range_trials` 构造期望误差。
- 独立计算 FP32 每行输出 MSE 和前 384 / 后 128 行平均值，与返回的全部字典严格相等。
- 指数恰为 -8..4，13 个 alpha 为 parent × 2^exponent，即 1/256..16；指数 0 精确包含父 alpha。I 与非 I 均测相同完整 grid。
- inputs、gate、target、levels、reference、父记录和非 I 记录以及模型状态不变；SW/packed 使用精确相等比较。

## 配对选择与记录绑定

`tests/test_phase3_joint.py:1221`：`test_postprocess_diagonal_paired_range_control_and_binding[False/True]`

使用一个 4 中间通道的小层，实际执行排序、`fused_d_records`、候选生成和诊断写入，仅 mock pair 返回的局部误差及 GPU progress。生成器不调用 probe/validation，测试将这两个入口设为失败。真实 pair 的数值由上一项独立验证。

- 检查 I 调用使用父记录对象；非 I 覆盖 0.25、0.5、2、4、16、64、256 因子及代码中的三组通道数量。小 fixture 下 16 通道请求自然截为现有 4 通道，不冒充完整生产维度覆盖。
- 共 1 个 I + 21 个非 I pair 调用，各 13 个范围，总 286 条诊断；它们使用同一 inputs、gate、FP target、parent alpha 与 levels。
- 控制误差设为：原父范围 heldout=10，最佳 I 范围为 parent/2、heldout=2。其它非 I 虽能到 3，仍不得仅因优于原父范围而宣称 D 有收益。
- **无 D 额外收益分支**：最佳非 I 与最佳 I 恰好同为 2；只输出 range-only，selected 保持最佳 I，严格相等不生成 D 候选。
- **有 D 额外收益分支**：单通道 D=16 的 alpha=parent/8 得到 heldout=1；分别输出 range-only（无权重改动、alpha=parent/2）和 local-D（对应 up/down 记录、alpha=parent/8）。用对象身份断言确认输出 up/down 正是该次获胜 pair 的记录，而不是最后遍历或另一 D 的记录；诊断 factor/channels/exponent 一致。
- mock fit 最优点故意与 heldout 最优点不同，确认按 heldout 而非 fit 选择。父 records/model/samples/reference 全部保持不变。

这里验证的是两个独立候选的生成和绑定，不额外重复此前已通过的 `run` train-probe/validation 隔离测试；当前只读源码仍逐候选 train-probe 选择，随后才执行 validation。

## 双项排序的具体回归样本

在同一轻量配对 fixture 中，将一列 Wq 设置为 0 并同步 packed codes，FP reference 保留 0.03125；令该通道输入位于可精确 BF16 表示的 SP2 投影输出，其他通道为较小输入。

独立 SP2 oracle 计算 Qx 后，精确核对诊断的每通道数组、合计 score 与排序：

1. `activation_contributions = mean((Qx-x)^2) * sum(Wfp^2)`。
2. `weight_contributions = mean(Qx^2) * sum((Wq-Wfp)^2)`。
3. `channel_contributions` 为两项和，score 为其总和，channels 按其降序。

该 W4 舍零通道的 activation 项为 0、weight 项为正；单看 activation 不会排其为首，两项相加后确实排首。两个选择分支均验证这一具体遗漏风险。此和仍是忽略 cross-term 的候选排序 heuristic，不是精确误差分解或因果归因；README 已明确此边界。

## 证据与复现

- 源码根：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。
- 已有 Git HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`，不代表未提交应用文件内容，未新增 hash。
- 实读 `experiments/phase3/postprocess.py` mtime：`2026-09-14 04:06:54.273822935 +0800`；pair 为第 190 行起，双项排序为第 211 行起。
- 第一轮：`runs/phase3/verifier/cpu-pair-range-20260914.log` 与同名 `.xml`。
- 排序补充：`runs/phase3/verifier/cpu-pair-ranking-20260914.log` 与同名 `.xml`。
- pytest 临时产物均限定在 `runs/phase3/verifier/pytest-pair-range-20260914`、`runs/phase3/verifier/pytest-pair-ranking-20260914`。

工作目录为源码根，两轮共同命令前缀：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider
```

第一轮指定两个 node：

```text
tests/test_phase3_joint.py::test_postprocess_pair_range_bf16_oracle_and_no_mutation
tests/test_phase3_joint.py::test_postprocess_diagonal_paired_range_control_and_binding
```

排序补充只指定第二个 node。均加 `-q`、上述 verifier 子目录的 `--basetemp`、对应 `--junitxml`，以 `set -o pipefail` 和 `tee` 保存原始输出。

本轮定向核验结束，等待下一处实际代码变化；不重复正式 GPU 实验评测。
