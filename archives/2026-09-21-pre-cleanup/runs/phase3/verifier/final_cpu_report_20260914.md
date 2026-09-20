# Phase 3 独立 verifier：CPU 实现与 Adam/resume 扩展 PASS

结论时间：2026-09-14 02:10 +08:00。

## 明确结论

**限定 CPU 实现 PASS：54 passed、0 failed、0 errors、0 skipped；pytest 报告 15.37 秒。**

- 原 47 项实现测试以及新增 Adam/resume/loop-clamp 覆盖全部通过。
- F001（训练 checkpoint 构建入口错误 SW shape）和 F002（冻结包 6 项缺校验）均由主线程修复、独立 verifier 重测关闭。当前测试范围内没有未关闭的应用断言失败。
- 本结论只批准下述已实际核验的实现范围，**不是 Phase 3 完整实验验收**，也不是三臂精度排名、Adam 优于 SGD、10/100-step 收敛、目标 PPL、真实 GPU resume 或手机 NPU PASS。

原 47 项即时 PASS 保留在 `cpu_implementation_pass_20260914.md`，没有以扩展测试尚未完成为由撤回已证实的分项结果。

## 证据与源码版本

最新日志：`cpu-sixth-20260914.log`；JUnit：`cpu-sixth-20260914.xml`。JUnit 起始时间 `2026-09-14T02:09:29.487019+08:00`，suite time `15.347` 秒；终端摘要 `15.37` 秒。

源码：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。

分支 `phase3/joint-r-sw-sa-sp2`，HEAD：

```text
24918316ed594848d4de797c356b120f2a4ee0f3
```

复测前后检查到三个应用文件修改时间一致：

| 文件 | 修改时间（+08:00） |
| --- | --- |
| experiments/phase3/quantization.py | 2026-09-14 01:48:03.860332156 |
| experiments/phase3/common.py | 2026-09-14 02:01:18.524743722 |
| experiments/phase3/run.py | 2026-09-14 02:02:55.724774109 |

实际检查结束 `git status --short`：

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

未把 HEAD 当作全部执行源码；继承修改保留。verifier 只写了 `tests/test_phase3_joint.py` 和本报告目录；三个应用文件均由主线程实现/修复，verifier 没有编辑。`git diff --check` 对 tracked diff 返回 0；新增测试的 `git diff --no-index --check /dev/null ...` 无空白错误输出、返回 1（文件相对空文件存在差异），不把该返回值伪称 0。

## 已实测通过的范围

### 量化与训练分支

- SP2 现有码点构造、全部码点/相邻中点/饱和边界、tie 取较小绝对值、FP32/BF16 exact forward；输入/scale 的声明 STE 与独立 surrogate 公式一致。梯度测试不是对离散函数作 finite-difference gradcheck。
- SP2 有限非零 scale 梯度、FP32 Parameter 身份、正学习率的实际更新；同一 scale 的训练/冻结 forward 一致，`.eval()` 不偷偷改变前向量化函数；非法 alpha 被拒绝。
- INT4 signed `[-8,7]`、ties-to-even、零行、每输出通道 SW，pack/unpack 及非法边界/奇数长度拒绝。现有 pack 是 `code+8` 的 offset-nibble 表示，测试不宣称它就是部署内核的原生二补码布局。
- SW identity weight STE 与非 down SA 饱和 mask/LSQ scale 归一化分别核验，未将二者混为一种 backward。
- A 第 0/49 步实际 W16、50 步 W4，B 起点 W4/down-A16，C 起点 W4/down-SP2；检查实际 quantizer forward 是否被 wrapper 调用，而不是只看 bits。Parameter 对象不随模式切换被替换。
- warmup/cosine 的 update 索引、同轨迹 10/100 区别、A 的当前 R SW reset 及对 SA/原权重的隔离。

### 真实小模型、冻结与重载

使用真实训练/eval Llama 类，测试内只替换 `from_pretrained` 读取：16 层、hidden8、intermediate16、query heads2、KV heads1、head dim4、vocab32，8/6-token 合成输入。没有下载或读取完整模型权重。

- 17 R、96 SA、112 SW、16 SP2 覆盖；全部目标参数梯度存在且有限、四组梯度总量非零、四组均发生实际参数更新。测试不把“每组有更新”外推为“每张 tensor/每个元素均有更新”。
- 真实 next-token FP32 CE，在关闭/开启 non-reentrant gradient checkpointing 时 loss 与梯度一致；SGDG 与新 Adam 两条 optimizer 路径均实际执行 update。原始 FP 主参数 `requires_grad=False`、无 grad、值不变；embedding/head 不共享 storage。
- FP64 独立 block-matrix oracle 覆盖 q/k/v/o/gate/up/down，包含 GQA 不同 block 数；`frozen_model` 输出为真实 eval 类，无在线 R1/R2，R3/R4 禁用，W4 code 与当前有效旋转 W+learned SW 一致。
- identity R1 下实际训练/冻结 backbone 与 NLL 一致；不要求任意 R1 的 BF16 head 融合 bitwise，也未将主线程报告的 head 舍入误差冒称 verifier 的实测。
- A/B 的 down 校准仅改变独立 frozen 模型中的 16 个 down quantizer；训练模型、非 down SA、所有权重等不受污染。校准 hook 正常移除，输出-MSE 候选选择与保存值一致。
- 有效冻结包在模型权重及尺度被故意改掉的冷对象中重载，恢复原张量与输出；坏 SW/SA、缺 head、未知格式被拒绝。训练 checkpoint 的正常加载保留 Parameter 身份，coverage、shape、finite/positive 的失败路径通过。

### 新增 Adam 与 resume

- `make_optimizers`：SGD 保留四组与 R 的 Stiefel 标志；Adam 使用单独 R-SGDG 和 224 个单 Parameter scale group，总共覆盖 241 个目标 Parameter，无重复/遗漏/身份替换，不改变模型状态。
- Adam 每组 `initial_lr = 0.001 * initial_parameter.mean()`、`eps=1e-12`、`weight_decay=0`；调度乘到每组 initial_lr，不相互覆盖不同尺度的基准学习率。
- 真实小模型 CE 已验证 Adam 下 R/SA/SW/SP2 的有限梯度、实际更新、FP32 与当前小步正值。
- 另执行真实 `run.train` 的 optimizer/clip/clamp/save 控制流，以合成梯度和故意过大的相对学习率制造 Adam scale 负值；确认 optimizer 后、clamp 前确实出现负数，循环 clamp 后全部尺度有限且 >=1e-8，原 FP 权重不变。该项在测试内将 `.cuda()` 改为 CPU 恒等、替换 progress/checkpoint 调用，并以假 CUDA RNG 数据保存；它证明 CPU 上实际 loop 的正值处理，不是 GPU 训练实测。
- `save_resume`：单包包含同次 241 参数、全部 optimizer state、route/update_step、Python/Torch RNG 与 CUDA RNG 字段；参数保存不修改模型，正常保存后 tmp 文件消失。
- 两种 optimizer 均验证保存/载入后 Parameter 身份、状态恢复，以及相同 RNG/相同固定梯度/相同下一步 schedule 下的下一次 update 与未中断对象精确一致。CUDA RNG getter 在测试内替换为哨兵张量；**不声称已经验证真实 GPU RNG 或 GPU resume 的数值一致性**。
- 注入 `torch.save` 中断并留下不完整 tmp 后，上一代 `resume.pt` 字节与 metadata 不变且可加载；再次正常保存可替换为新一步并清除 tmp。验证进程写入中断保护，不外推为断电持久性/fsync 保证。

### 评测接口

- FP32 CE 训练 probe 的 chunking、next-token shift、不同尾长 token 加权、模式恢复与尺度无污染。
- 新 worktree `utils.eval_utils.evaluator` 在 CPU 小模型上实际使用 BF16 logits CE，结果与独立 BF16 CE→FP32 loss 平均→FP32 exp 公式一致。
- `full_validation` 通过纯 `validation_acceptance.py` 委托新 worktree evaluator，不执行旧脚本 main，不把旧 `repos/SpinQuant` 插入 sys.path；合成 252852-token 输入验证 123×2048+948、252728 个预测目标及 segment 加权。该接口测试用替身 evaluator，真实 evaluator 数值由上一小模型测试覆盖；**未执行真实 WikiText-2 完整 validation**。

## 历轮结果与边界

| 轮次 | 结果 | 说明 |
| --- | --- | --- |
| first | 29 pass / 1 fail | 发现 F001 |
| second | 32 pass / 11 fail | F001 关闭；6 项 F002，5 项 verifier 对 dormant NaN 哨兵的比较错误 |
| third | 37 pass / 6 fail | 纠正 verifier 比较，剩 F002 |
| fourth | 41 pass / 6 fail | 增加 validation 接口检查；进程加载的是 F002 修复前源码 |
| fifth | 47 pass / 0 fail | 新进程加载修复版，F001/F002 全关闭 |
| sixth | **54 pass / 0 fail** | 加入 Adam、实际 loop clamp、原子 resume 检查 |

最终 38 个告警来自现有 Transformers、CPU checkpoint autocast 与 SGDG deprecated API；未修改无关代码消除告警。禁用 legacy quantizer 允许保留 NaN 哨兵；比较使用零容差 `equal_nan=True`，不放宽有效 SA/SW/SP2 的 finite/positive 要求。

尚未验收：正式三臂 GPU 完整训练、C-SGD 控制、优化器精度优劣、初始化/10/100-step 实验结论、真实大模型完整 validation 与大模型导出包、真实 GPU resume、多卡、后处理 D/离散码优化、FT/蒸馏、decode/KV8/设备原生内核。对这些不授予 PASS，也不把本 CPU PASS 解释为主 goal 完成。

本 verifier 未安装、改环境、提交、推送、创建 goal，未查询或操作 GPU/他人进程；用户列出的 GPU PID 均未触碰。

## 最终运行命令

```bash
cd /home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest \
  -p no:cacheprovider tests/test_phase3_joint.py -q \
  --basetemp=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/pytest-sixth-20260914 \
  --junitxml=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-sixth-20260914.xml
```

终端输出通过 `set -o pipefail` 与 tee 保存到同目录 `.log`，工具返回 exit_code=0。
