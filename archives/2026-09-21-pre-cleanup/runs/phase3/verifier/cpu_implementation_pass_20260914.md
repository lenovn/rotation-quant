# Phase 3 独立 CPU 实现核验：限定 PASS

后续更新：Adam/resume 扩展后的最终结果为 **54/54 PASS**，详见 `final_cpu_report_20260914.md`。下文保留第五轮 47 项即时结论。

## 即时结论

**47 passed，0 failed，0 skipped，12.82 秒。** 日志 `cpu-fifth-20260914.log`，JUnit `cpu-fifth-20260914.xml`。这是最新源码的真实 CPU pytest 结果，不是初步源码推导。

- F001：训练 checkpoint 构建入口 SW `[out,1]`、非 down SA/SP2 `[1]` 形状拒绝通过；下游 coverage、finite、positive 与参数身份保持通过。
- F002：冻结包非法 SW（负值/NaN/错误 shape）、负 SA、缺 head 高精度边界、未知 activation 格式均被拒绝；有效包冷重载保持原参数、尺度和输出通过。两项已复现缺陷均由主线程修复，独立重测关闭。
- SP2 全码点/中点/饱和边界 exact forward、FP32/BF16、声明的 STE 输入/scale 梯度、实际 scale 更新、同 scale 冻结前向一致通过。
- INT4 signed bounds、ties-to-even、零行、offset-nibble pack/unpack、SW/SA LSQ 梯度差异通过。
- A/B/C 实际 W16/W4/down-SP2 wrapper 分支、0/49/50 切换、schedule 索引及当前 R 的 SW reset 通过。
- 真实 16 层小模型（hidden8、intermediate16、query heads2/KV heads1/head dim4）的 17 R、96 SA、112 SW、16 SP2 coverage、联合 backward、checkpoint 重算梯度一致、SGDG 实际更新及原 FP 权重冻结通过。
- GQA R2/block 独立 FP64 oracle、有效 W4 code、转真实 eval 类并移除在线 R1/R2、identity R1 下训练/冻结输出、禁用 R3/R4 通过；任意 R1 下 BF16 head 融合不要求 bitwise。
- frozen down 校准仅改变独立 frozen 模型的 16 down 参数；训练模型、96 SA、112 W4 及其他 frozen 状态不受污染通过。
- FP32 CE probe/chunk/tail 统计通过；旧 evaluator 在 CPU 小模型上真实 BF16 logits CE 与独立公式一致通过；新增 full_validation 对当前 worktree evaluator 的委托和 252852/252728、123 整窗+948尾窗计数通过。后者是合成 token/假 evaluator 的接口计数检查，**不是完整真实 validation 重测**。

## 可用范围与尚未覆盖

以上范围给 **CPU 实现 PASS**；不因后续补充测试未完成而把所有首次训练路径笼统标为阻塞。新 `make_optimizers` 的 Adam 分组/更新与 `save_resume` 原子恢复正追加独立测试，暂未计入这 47 项。Adam 优劣、2/10/100-step 收敛、完整真实 validation、实际大模型包/多卡/手机 NPU 均不在本 PASS 内；主线程 GPU 报告不冒称 verifier 的独立测量。launch.py、README 未列入本次数值验收。

## 源码与执行边界

- worktree：`worktrees/SpinQuant-phase3-joint`；HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，包含继承未提交修改和主线程新增源码。
- 复测开始读取的源码修改时间：quantization.py `2026-09-14 01:48:03 +08:00`，common.py `02:01:18 +08:00`，run.py `02:02:55 +08:00`。没有用 HEAD 代替全部运行源码，没有新增 hash。
- 测试仅改 `tests/test_phase3_joint.py`；报告/测试临时产物只写本 verifier 目录。未改应用源码、环境、已有 Phase 2/3 训练产物；未查询/操作 GPU 或他人进程。
- 真实模型类/量化/初始化/融合/校准/保存重载执行；测试内仅将 pretrained 模型读取替换为小配置新模型，避免读取大 checkpoint 或下载。比对禁用 legacy quantizer 的 NaN 哨兵使用零容差 `equal_nan=True`；有效尺度仍严格要求 finite/positive。
- 30 个告警来自现有 Transformers、checkpoint autocast、SGDG deprecated API，不是断言失败。

```bash
cd /home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest \
  -p no:cacheprovider tests/test_phase3_joint.py -q \
  --basetemp=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/pytest-fifth-20260914 \
  --junitxml=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-fifth-20260914.xml
```
