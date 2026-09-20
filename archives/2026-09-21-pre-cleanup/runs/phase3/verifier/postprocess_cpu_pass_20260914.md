# Phase3 postprocess / launcher：限定 CPU PASS

## 结论与执行范围

- 新增后处理关键路径及 launcher driver 路由：**限定 CPU PASS**。
- 首轮定向执行：`10 passed, 1 failed, 55 deselected`，7.25 秒。唯一失败为候选 probe 抛错后的模型恢复。
- 主执行者修复 `postprocess.run` 的 `try/finally` 后，只复跑该异常项：**`1 passed`，9.17 秒**。此前通过的 10 项未重复执行；这是两次执行合计覆盖 11 个定向用例，不是修复后一次性重跑 11 项或全量测试。
- 应用只读；verifier 仅扩展 `tests/test_phase3_joint.py` 并写本目录测试产物/报告。没有 GPU 查询/运行、真实 tmux 启动、训练、安装、环境修改、提交或推送。

## 已通过的具体范围

| 项目 | 数值测试或定向检查及边界 |
| --- | --- |
| launcher 路由，3 项 | subprocess mock 检查默认及显式 `train` 走 `run.py`，`postprocess` 走 `postprocess.py`；既有 tmux server 环境不同情况下，子命令仍显式带 CUDA_VISIBLE_DEVICES；shell 特殊字符/空格参数及日志重定向可解析；pane PID、session/attach、日志路径和唯一 run 记录一致。未启动真实 tmux/server/GPU。 |
| `load_static`，1 项 | 使用真实 16 层小型 Llama 与实际冻结、保存、reload 路径冷载，验证 112 W4 / 96 static INT8 / 16 SP2 全覆盖、eval 类与参数冻结、状态/代码/SW/输出一致、父包字节不变；禁止校准入口及动态 find_params，加载和推理均无重标定。模型来源替换为本地 tiny fixture，不下载生产模型。 |
| `capture_inputs`，1 项 | 三个真实 ActQuantWrapper 及独立 INT8/SP2 数值 oracle，验证 Linear 模块内实际 A 量化后输入、down wrapper 前原始输入、gate wrapper 输出；32×128 行按每八行采样，512 行及 384/128 分割边界精确对齐，hook 清理且模型状态不变。 |
| fixed-grid rounding，1 项 | 实际调用外层 `scripts/phase2/fixed_grid_rounding.py` 纯 torch helper；检查加载路径和 sys.path 不变，未导入旧 repo 应用；单个小模块产生有效候选，step0 父代码进入选择集，按 heldout 选择，代码在 [-8,7]，父代码/SW/FP reference/model 不变。没有执行 112 模块搜索。 |
| `fused_d_records`，1 项 | D=I 的 packed codes/SW 精确不变且父记录不变；非 I 保持 up codes、up SW / D、down SW 不变，仅重算选中 down 列；显式舍入 grid oracle 与 BF16 离线 up/down 矩阵及 pair 运算精确一致。形状错误、零/负值、NaN/Inf D 被拒绝。不是与未量化 FP 模型等价的声明。 |
| SP2 range，1 项 | 单个小 down 的 33 个范围包括精确父 alpha，按 heldout 选取，候选正且有限，无模型修改。未执行全部 16 层生产搜索。 |
| 候选隔离与选择，3 项 | 合成候选修改真实小模型权重及 SP2 scale，执行真实 `run` 候选应用/恢复/选择流程。正常候选之间权重与原 scale 精确恢复；无改进保留父状态且不调用 validation；有改进先完成候选生成及 train probe 选择再调用 validation，即使 mock validation 很差也不改变选择；修复后 probe 异常传播且模型全 state（含 SP2 scale）精确恢复，不调用 validation。数据、probe/validation 数值、GPU progress 与来源快照写入在该 orchestration 测试中被 mock，因此这不是实际完整 validation 的数值验收。 |

## 唯一发现及修复复测

首轮 `test_postprocess_candidate_isolation_and_train_only_selection[probe-error]` 在候选已修改权重及 alpha 后注入 `RuntimeError("injected probe failure")`，零容差 state 比较发现 `model.layers.0.mlp.up_proj.weight` 未恢复。原恢复语句位于 evaluate 后，异常会跳过恢复；当前 CLI 会随异常退出，不能由此宣称已有成功候选或磁盘父包遭到污染。

主执行者将 apply_records 与 evaluate 放入 `try`，并在 `finally` 从父记录恢复受影响权重、从事先克隆的原 scale 精确恢复 SP2。复测保留异常传播断言、完整模型 state 零容差比较（既有 dormant NaN buffer 使用 equal_nan）及 validation 不被调用断言，结果 PASS。verifier 未修改应用代码。

## 可追溯证据

- 源码根：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。
- 已有 Git HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`；这是已有 revision，不代表未提交文件内容。没有新增 hash。
- 复测时 `experiments/phase3/postprocess.py` mtime：`2026-09-14 02:55:47.085406117 +0800`；异常恢复实现位于第 280–286 行。
- `experiments/phase3/launch.py` mtime：`2026-09-14 02:43:55.949597796 +0800`。
- 测试：`worktrees/SpinQuant-phase3-joint/tests/test_phase3_joint.py:768`（launcher），`:887`（postprocess 起始），`:1089`（候选隔离/异常恢复）。
- 首轮原始输出：`runs/phase3/verifier/cpu-postprocess-first-20260914.log`、同名 `.xml`。
- 修复单项输出：`runs/phase3/verifier/cpu-postprocess-exception-20260914.log`、同名 `.xml`。
- 首轮仅有一个已有 Transformers quantized-training deprecation warning；异常单项无 warning。

复测命令（工作目录为源码根）：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
'tests/test_phase3_joint.py::test_postprocess_candidate_isolation_and_train_only_selection[probe-error]' -q \
--basetemp=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/pytest-postprocess-exception-20260914 \
--junitxml=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-postprocess-exception-20260914.xml \
2>&1 | tee /home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-postprocess-exception-20260914.log
```

## 未验收范围

未执行生产模型完整后处理搜索、真实 tmux 启动、GPU 训练/后处理、候选真实 train-probe 或完整 validation/PPL 对照、最终优胜包或原生 INT4/SP2 设备内核验证。D 的完整层/通道组合搜索只读审查，数值测试集中在上述小张量离线融合原语。本报告不替代正式 GPU 实验审计或完整实验验收；本轮定向代码核验结束，等待下一处实际代码变化。
