# Quantization-aware distillation：限定 CPU 实现 PASS

## 结论

`experiments/phase3/distill.py` 与新增 launcher distill 路由，在本报告列出的数值、真实小模型反传、导出冷载和恢复控制流范围内 **PASS**。未发现这些已测范围内尚未解决的应用阻塞项。它是解冻 backbone 主权重的量化感知蒸馏分支，不是原始主权重冻结 PTQ。

- 首轮只选择新增 `distill` 用例：**8 passed / 2 failed / 69 deselected，10.10 秒**。
- 两个失败均为 verifier fixture/断言问题：测试直接冻结了初始化后仍 down-A16 的模型，导致 SP2 scale 不在计算图；另一个以 `torch.equal` 比较既有 dormant NaN buffer。测试已改为先切完整 C 量化模式、保存并真实冷载完整冻结父包；NaN buffer 使用零容差 `equal_nan=True`。应用代码未因此修改。
- 第二轮只运行修正后的两个用例以及新增 teacher/RNG 三项：**5 passed，9.60 秒**。
- 合计 **13 个不同的新增定向用例已通过**，其中两个在修正测试后复跑；不是单轮全量 13 项或已有整套测试重跑。既有量化/训练/postprocess/PPL 实验不重复核验。

## 具体已测范围

### 1. 父格点、参数身份与真实 checkpoint 反传

`tests/test_phase3_joint.py:1367` 附近的 `test_distill_parent_grid_aliases_real_checkpoint_backward`：

- 使用实际 16 层 tiny Llama、实际 112 个 wrapper、冻结包保存与 `load_static(..., device="cpu")`，再调用真实 `prepare_student`。
- 验证第一步前所有 W4 量化权重转 BF16 后与父冻结权重逐张精确相等；两个输入窗口的完整 BF16 logits 精确相等。所有 backbone 输入 quantizer 为 bits8。
- 可学习参数恰为 W112 / SA96 / SW112 / SP216，共 336 个不同 Parameter；全部 W master 与 SW 为 FP32；wrapper.weight 指向 replacement.module.weight，旧 112 个主权重对象不残留于模型参数中。
- 优化器恰好覆盖这 336 个对象且无重复：主权重 SGD momentum0.9、foreach=False；224 个 scale 各自 Adam group、eps1e-12。head/embed/norm 等 HP 参数不进入优化器，始终冻结。
- 实际开启 non-reentrant block checkpoint，真实小模型蒸馏 forward/backward；head 以 chunk3 执行三段 non-reentrant checkpoint。全部 336 个参数得到有限梯度，各组均存在非零梯度和一步实际更新。未把“有梯度”外推成每张张量所有元素均更新或全部整数码已改变。
- 教师无梯度，教师状态不变；HP 参数无梯度、无更新；没有重新插入 R1。

### 2. KL/CE 与 chunk 数值 oracle

- `test_distill_chunk_objective_direction_temperature_and_sums`：独立 float64 logsumexp/gather/概率求和 oracle 检查 KL(teacher || student)、CE、0.9/0.1 权重；T=1 和 T=2 检查 temperature²。返回为 FP32 sum，不错误地按 chunk 均值混合；反向 KL 与正确方向在测试数据上明确不同。teacher logits 即便 requires_grad 也被 detach。
- `test_distill_loss_chunk_tail_shift_and_token_normalization`：BF16 embedding/head 小模型，batch2、每窗8 tokens，共14个 next-token targets；检查目标右移、末 token 排除和总 token 归一化。chunk1、3、128 对同一独立完整 logits oracle 在 FP32 舍入容差内一致，尾 chunk 正确；实际 checkpoint backward 能到 student，teacher/head 不获得梯度。这不是声称不同 chunk 的 BF16 梯度逐位一致。

### 3. export_student 与冷载

- `test_distill_export_coldload_exact_forward_codes_and_hp`：人为改变一个 FP32 master 元素使其跨码，并改变一个 SP2 scale；实际导出 112 个 packed W4 记录，独立 round/clamp oracle 检查代码范围 [-8,7]、全部变化统计及至少一个非零变化。
- HP 保存状态与父一致，父 records 与训练模型状态不被 export 修改。
- 实际 `load_static`/既有 reload 冷载导出包，112 个 W4 记录完整、加载后全模型冻结；两个窗口的 student 与冷载冻结模型 logits 精确相等。此测试没有重新校准。

### 4. teacher 的真实本地 from_pretrained

- `test_distill_teacher_real_local_from_pretrained_preserves_weights_and_fp32_rope` 不 mock from_pretrained：在 verifier 临时目录保存一个真实 tiny tied-embedding Llama checkpoint，再从该目录调用实际 teacher_model。
- 故意使用非单位 norm 权重，检查 teacher 所有保存权重与原始 source 一致、无 norm fusion、无 R1/量化 wrapper、冻结/eval/use_cache=False。
- head/embed 值相等但存储参数已解绑定；全部 inv_freq 与构建时的原始 FP32 RoPE 精确相等且 dtype=float32，不经过 BF16 RoPE 舍入。
- 这是本地小型 source checkpoint 测试，不是生产 BF16 teacher 权重或完整 PPL 重评。

### 5. resume 的参数、优化器与 RNG

- `test_distill_resume_optimizer_rng_next_step_and_atomic_save`：真实保存/加载 336 个 FP32 W/scale 参数、SGD momentum 与 Adam 状态、step、Python/Torch RNG；CUDA RNG API 以 CPU tensor mock。改变目标模型后加载，Parameter 对象身份不变；用随机合成梯度验证下一 optimizer step 后模型与两份优化器状态精确一致。
- 注入临时文件写入中断，既有 resume.pt 字节不变；随后成功保存原子替换，临时文件消失，step 正确。
- `test_distill_resume_rejects_malformed_without_mutation[coverage/weight_shape/scale_nonpositive/optimizer_count]`：参数缺项、W shape 错误、非正 scale、少一份优化器 state 均在加载前拒绝，模型不变。不外推为任意损坏的 optimizer 内部 tensor 都经过独立验证。
- 初查发现 `zip(optimizers, saved)` 未校验长度的真实静默少恢复风险，已立即反馈。主执行在首轮运行完成前加上 optimizer list 长度检查，本轮该缺项用例直接 PASS；没有伪造“旧代码已运行失败”的记录。

### 6. 冷载评测与恢复顺序的新改动

- `test_distill_cold_validation_rng_and_optimizer_identity_cpu_mock`：实际 export/reload，仅将设备迁移、CUDA RNG/empty_cache 与 full_validation 的 GPU 部分 mock。确认 export → student/teacher offload → 冷载新包 → validation → 返回原模型对象的顺序。主动令冷构建消耗 Torch/模拟 CUDA RNG，实际 `fork_rng(devices=[current_device])` 恢复二者；保留 student/teacher 状态及原 optimizer Parameter 身份。
- 该设备边界测试不包含真实 CPU↔CUDA 复制，不能保证生产显存释放量、迁移耗时或真实 CUDA Parameter/optimizer 行为；正式 GPU 继续训练需由主执行实测。
- `test_distill_train_builds_teacher_before_restoring_rng`：在 train 的恢复分支中，让 mock teacher 构建及迁移先消耗 Python/Torch/模拟 CUDA RNG，再执行实际 load_resume；断言全部 RNG 恢复到保存值。使用已到终点的 resume 跳过训练循环，只检查新顺序，不伪装 CPU/GPU 训练烟测。

### 7. launcher

- 只运行既有 mock 测试新增的 `[distill]` 分支，验证 driver=distill.py、parent/steps 转发，并保留显式子环境、shell 引号、PID/session/log 记录和 run 名唯一保护检查。没有重新测试旧 train/postprocess 路由，没有真实启动 tmux 或调用 nvidia-smi。

## 数据与源码只读检查

- 当前 `distill.train` 使用 `(data_start + step * accumulation + microbatch) % train_windows.shape[0]`；既有 `data_windows` 返回 `windows[:-8]`。本轮只读核对该索引逻辑，未重新加载生产数据计数或新增数据实验。
- 用户随后通知 `run.make_optimizers` reference 分支修正 CPU/GPU mean 末位差异，已独立只读确认 `run.py:49` 为 `reference[parameter_name].to(parameter.device)` 后求 mean；缺省分支仍为 parameter.detach()。未重跑旧 CPU reference 用例或执行新 GPU 测试。原 CPU PASS 仍只证明其 CPU 范围，不能证明历史不同设备归约的实际 LR 逐位相同；不改写/覆盖旧实验结果。
- postprocess 的外层纯 helper 源码快照复制不改已测数学，按用户要求不新增测试。

## 证据

- 源码根：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。
- 已有 HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`；它不代表未提交文件内容，未新增 hash。
- 实读 mtime：distill.py `2026-09-14 05:04:38.785244895 +0800`；launch.py `2026-09-14 05:01:38.640122083 +0800`；run.py 设备对齐修正 `2026-09-14 05:08:06.038435420 +0800`。
- 首轮输出：`runs/phase3/verifier/cpu-distill-first-20260914.log` 与同名 `.xml`。
- 第二轮输出：`runs/phase3/verifier/cpu-distill-second-20260914.log` 与同名 `.xml`。
- 临时包、日志等只在本目录对应 `pytest-distill-first-20260914` / `pytest-distill-second-20260914` 子目录；首轮遗留测试失败证据保留。
- 输出中的 warning 为已有 Transformers quantized-training API 与 torch CPU autocast deprecation，不是精度/运行失败。

共同命令前缀，工作目录为源码根：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider
```

首轮追加 `tests/test_phase3_joint.py -k distill -q`；第二轮指定以下五个 node：

```text
tests/test_phase3_joint.py::test_distill_parent_grid_aliases_real_checkpoint_backward
tests/test_phase3_joint.py::test_distill_export_coldload_exact_forward_codes_and_hp
tests/test_phase3_joint.py::test_distill_teacher_real_local_from_pretrained_preserves_weights_and_fp32_rope
tests/test_phase3_joint.py::test_distill_cold_validation_rng_and_optimizer_identity_cpu_mock
tests/test_phase3_joint.py::test_distill_train_builds_teacher_before_restoring_rng
```

两轮均指定上述 verifier 子目录为 `--basetemp`，对应日志名为 `--junitxml`，使用 `set -o pipefail` 与 tee 留存输出。

## 明确未验收

未进行 GPU/生产 teacher-student 运行、显存测量、真实跨设备 offload/resume、生产 checkpoint 长程恢复、完整 validation/PPL 或蒸馏收益判定。没有安装、环境改动、提交、推送或修改应用文件。本报告只关闭本次已列出的 CPU 实现核验范围，不代替最终优胜包审计或完整 Phase3 实验验收。本轮暂停，等待下一处真实代码变更。
