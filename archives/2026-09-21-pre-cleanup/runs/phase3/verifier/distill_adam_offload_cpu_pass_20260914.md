# distill Adam / teacher-body offload：限定 CPU PASS

## 结论与范围

本轮只验证新增 weight Adam、teacher-body offload 和 resume 优化器类型兼容性：**3 passed，11.58 秒**，无失败。原 13 项、55 项以及其它已通过用例均未重跑。

下述 CPU 实现范围明确 **PASS**；未发现当前新增路径在该范围内的应用阻塞项。不代表真实 CUDA offload 能节约多少显存、生产模型训练稳定或 Adam 有泛化收益，也不重新审计此前 SGD 实验或停止进程行为。

verifier 应用只读，仅修改授权的 `tests/test_phase3_joint.py`，测试包/日志和本报告均在 `runs/phase3/verifier/`。未操作 GPU/训练进程、未启动新 smoke、未安装/改变环境/提交/推送。

## 1. 原生 Adam 与旧 SGD 缺省

`tests/test_phase3_joint.py:1764`：`test_distill_adam_weight_coverage_state_update_and_sgd_default`

- 使用完整 112-wrapper 的真实 16 层 tiny student fixture，实际构建优化器。
- 缺少 `args.weight_optimizer` 时仍构造 SGD，与显式 `sgd` 的 optimizer state_dict/param_groups 一致；保留 momentum0.9、foreach=False。
- Adam 新分支：主权重单组恰好覆盖 112 个不同 FP32 W Parameter，绝对 LR/initial_lr=2e-5，eps1e-8、weight_decay=0、foreach=False。
- scale Adam 仍为224组；两个 optimizer 合计336个不同 Parameter，精确等于全部可学习参数，没有 HP 或重复参数。
- 赋予确定性合成梯度并真实执行一步，112 个 W 都变化；逐张验证 FP32 exp_avg/exp_avg_sq 的形状及一阶/二阶矩数值、step=1，并与独立第一步 Adam 更新式 `W - lr*g/(abs(g)+eps)` 比较。
- 两份 Adam 状态分别覆盖112/224张量；W/SA/SW/SP2全部目标张量在本合成梯度检查中实际变化，HP 参数不变。这不是生产 loss 梯度下每张张量必然更新的声明。

## 2. teacher-body offload 的真实小模型数值一致性

`tests/test_phase3_joint.py:1817`：`test_distill_teacher_body_offload_loss_grad_state_equivalence`

- student 使用实际冻结包冷载后 prepare 的16层 tiny模型；teacher 是真实 eval Llama，权重 BF16、RoPE buffer FP32、head/embed解绑定且教师冻结。没有 mock backbone、linear、quantizer 或 loss。
- 分别执行 offload=False/True，chunk3，实际 non-reentrant checkpoint forward/backward。
- CPU 跟踪实际 body.to(ids.device) / body.cpu 调用：False 路径无迁移；True 路径顺序严格为 body.to → teacher no-grad forward → body.cpu → student forward。没有将整体 teacher/head 一起 offload；head 输入仍在 ids.device 且无梯度。
- 两条路径的 objective/CE/KL **零容差一致**，**全部336个目标参数**均须有有限梯度，且336张梯度逐张零容差一致；各组仍要求存在非零梯度，没有放松成仅检查参数 requires_grad 或少数梯度。
- teacher 无梯度、HP 无梯度，student/teacher state不变；另显式比较非持久 teacher buffers，包含 FP32 RoPE 的 dtype和值，均不变。既有 dormant NaN state继续用零容差 equal_nan=True，其余梯度仍严格要求 finite。
- 迁移调用实际发生于 CPU→CPU，只证明上述调用边界/顺序与 CPU 数值等价；没有执行或模拟出真实 CUDA 内存释放量、PCIe传输、CUDA异步时序或GPU梯度等价结果。

## 3. resume 类型检查与兼容性

`tests/test_phase3_joint.py:1903`：`test_distill_resume_optimizer_method_mismatch_and_legacy_default`

- 构建真实 optimizer 并先执行一步使状态非空，然后保存；metadata记录实际请求的 adam/sgd，缺属性保存为sgd。
- 检查 Adam包→SGD、SGD包→Adam、旧版无类型字段SGD包→Adam，均按预期抛出 `ValueError("Distillation resume changes weight optimizer")`。
- 目标参数故意设为与保存包不同，保证能发现误复制；每次拒绝后完整模型、两个 optimizer state_dict、Python/Torch RNG均不变，CUDA RNG setter也未调用（CUDA RNG以CPU tensor mock）。
- 正向检查新增 Adam包可恢复到Adam，参数与一阶/二阶矩等 optimizer状态精确一致；旧包缺类型字段、调用参数也缺weight_optimizer时仍可恢复为SGD。step与模拟CUDA RNG恢复正常。
- 仅验证这次新增的方法匹配逻辑和对应正常兼容路径；不重复此前完整 resume/原子保存/冷评测测试。

## 源码与实验边界

- 当前源码 `distill.py:95` 新增 offload 参数；`distill.py:124` 分支选择 optimizer；`distill.py:147` 保存类型、`:157` 在参数加载前拒绝不一致；train 调用将 `getattr(args, "offload_teacher_body", False)` 传给 loss。
- `chunk_objective` 的 KL/CE 数学和 checkpoint 冷加载流程未在这次改动中重写；本轮不为未变代码重复原13项验证。
- 本轮 Adam 一步数值测试采用LR2e-5，但没有执行200-step schedule实验。主执行的新LR/schedule/offload配置属于优化配置搜索，不应称纯optimizer-only公平对照；CPU PASS本身不支持任何实验效果归因。

## 证据与命令

- 源码根：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。
- 已有HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`，不代表未提交文件内容；未新增hash。
- 运行前后实读 `experiments/phase3/distill.py` mtime：`2026-09-14 05:28:38.112398571 +0800`。
- 原始输出：`runs/phase3/verifier/cpu-distill-adam-offload-20260914.log`、同名 `.xml`。
- 临时产物：`runs/phase3/verifier/pytest-distill-adam-offload-20260914/`。
- 4个warning来自已有 Transformers quantized-training 和 torch CPU autocast deprecation，无新的失败。

工作目录为源码根：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
tests/test_phase3_joint.py::test_distill_adam_weight_coverage_state_update_and_sgd_default \
tests/test_phase3_joint.py::test_distill_teacher_body_offload_loss_grad_state_equivalence \
tests/test_phase3_joint.py::test_distill_resume_optimizer_method_mismatch_and_legacy_default -q \
--basetemp=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/pytest-distill-adam-offload-20260914 \
--junitxml=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-distill-adam-offload-20260914.xml \
2>&1 | tee /home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-distill-adam-offload-20260914.log
```

本轮定向代码核验结束，等待下一处实际变更；不替代正式 GPU 实验或最终优胜包验收。
