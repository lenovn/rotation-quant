# Phase3 C4 external driver：限定 CPU PASS

日期：2026-09-15。独立代码 verifier，不是主执行或正式 GPU 实验 auditor。

## 结论

**新增 4 项定向测试全部通过，无未解决应用 FAIL。** 只运行 `tests/test_phase3_external.py`，未运行旧 suite。应用源码、Phase2 历史文件和固定 token 包只读；未创建 goal、训练、校准、搜索、下载、安装、提交、推送或使用 GPU。

本报告支持新 external driver 的限定实现 PASS，**不代表真实 C4 PPL、泛化结论或实验验收**。两组正式 GPU 测量由主执行承担。

## 当前源码与复用边界

源码：`worktrees/SpinQuant-phase3-joint`，现有 HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，存在未提交新增实现，不把 HEAD 当作新增文件的完整版本标识。本轮读取完成后的文件状态：

- `experiments/phase3/external_eval.py`：8706 bytes，mtime `2026-09-15 02:02:18.246623168 +0800`。
- `experiments/phase3/launch.py`：4199 bytes，mtime `2026-09-15 02:01:45.122717448 +0800`。

已只读核对 `common.reload_frozen` 的完整 weights/activation/HP coverage 与 shape/scale 检查、`postprocess.load_static` 的默认 CUDA 冷载，以及外层纯函数 `scripts/phase2/validation_acceptance.py` 的分块聚合。新 driver 动态导入该纯 acceptance，不调用其历史 `main`/PTQ 准备入口；evaluator 来源实际为新 worktree 的 `utils/eval_utils.py`，不是旧 repo。

原 BF16 构造与 `runs/phase3/auditor/final-best-20260914/replicate.py` 已审路径一致：本地 pretrained、配置先解 tied、原配置 tied 时 head 复制 embedding、冻结、无旋转/norm fusion；保留 FP32 RoPE，模型 `.cuda()` 不附带全局 BF16 dtype 转换。量化分支直接使用既有 `load_static(package)` 完整恢复冻结包，不调用 reference 重建或量化搜索。

## 四项实际覆盖

测试文件：`worktrees/SpinQuant-phase3-joint/tests/test_phase3_external.py`。

### 1. 固定官方 C4 token 包与 metadata

`:27` 实际 CPU 读取用户指定的历史包：

`runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/input_tokens.pt`

- `input_ids` 实际存储 shape 为 **[1,2097152]、torch.int64**，对应 1024 个独立 2048-token 窗，不是文件中直接存成 [1024,2048]。
- metadata 为 `allenai/c4 / en / validation`；1024 windows、2097152 inputs、**2096128 targets**，8 个官方 validation shard 来源；原 tokenizer 路径、BOS/EOS 关闭、两换行拼接及 external-only 用途记录一致。
- 在内存里替换错误 split、tokenizer 路径、BOS、tensor shape/dtype、windows、target 数分别拒绝；没有改写历史包。
- 前后文件 size/mtime 不变，并重新读取确认 tokens 逐位相同。未重下载原文、重分词或复核历史所有原始 shard；官方来源依据此已审历史包 metadata，不重建数据。

### 2. 真实 acceptance 的 128 窗分块及聚合

`:61` 使用真实 `evaluate_tokens` 与真实外层 acceptance，仅 mock 底层 evaluator 返回已知 float32 PPL：

- 1024 窗精确分成 **8×128**；每段都是原 token 张量的连续对应切片，拼接后逐 token 相同，无省略/重排/WT2 data_windows。
- evaluator 参数为 eval_nsamples=None、bsz=1、capture_layer_io=False，独立窗口长度2048；目标数/尾窗计数准确。
- 聚合为预测 token 加权 `log(float32 segment PPL)` 后 exp，**不是平均 PPL**。另以 129 窗切片验证 128+1 不等权段的目标数权重，未执行额外真实模型评测。
- 模型 seqlen 保持2048；核对 evaluator 定义文件确实属于新 worktree。

### 3. 两种模型真实 tiny 冷载、配对 driver 与尺度保护

`:174` 使用真实 16-layer tiny Llama（hidden8/intermediate16、112 wrappers）的本地 pretrained fixture；非伪造 `load_static` 返回值。固定量化 fixture 用明确常量 activation scales 和保存 packed 权重，不执行训练/校准。只在 CUDA 设备边界映射为 CPU，显存计数 mock；C4 evaluator 分数 mock，无 2M token CPU 模型前向。

- BF16 分支实际本地 from_pretrained；quantized 分支实际 `postprocess.load_static` → `common.reload_frozen`。解 tied 后 head/embedding 数值相同但不共享参数存储。
- BF16 所有原权重/HP 与本地原模型逐位一致，无量化 wrapper/R；量化模型实际 112 W4 权重、96 INT8/16 SP2、输出A16及 HP 与冻结包对应模型逐位一致。两种模型都 requires_grad=False、eval、use_cache=False；参数BF16、全部 `inv_freq` FP32。
- 两个 tiny 冷载模型分别做一个实际短窗 CPU forward，与对应原/冻结 fixture 的 logits **逐位相同**；不把该短窗称 C4 实测。
- 完整 `external_eval.run` 的两种模式分别收到同一份真实 C4 token 包、同8段输入；结果保存真实协议/路径/targets以及 calibration=False、training=False、candidate_selection=False。
- 实际 quantizer before/after 快照逐位一致。人为注入 SP2 scale 变化时 driver 拒绝写成功 result，证明新状态检查不是恒真。冻结包原始字节不变。
- 新路径若调用常见训练模型构造、尺度初始化/SP2校准、norm fusion、动态 find_params 或 optimizer 创建，会由测试 trap 失败；成功运行未触发这些路径。
- dormant legacy state 只在 `quantizer.` 范围允许 `equal_nan=True`；活动尺度/完整 before-after 检查仍为精确 `torch.equal`。

### 4. launcher external 与 CLI 默认值

`:267` mock subprocess（包括 nvidia-smi/tmux），实际执行 launcher 参数解析和 tmux 命令组装：

- `--task external` 路由到 `external_eval.py`，mode/package/tokens/output argv 原样保留，包括含空格/引号/shell字符的路径。
- 子环境显式 `CUDA_VISIBLE_DEVICES=1`，exec/env、log 重定向、pane PID 与 launch JSON 一致；未查询/启动真实 GPU 或 tmux。
- 另执行 external CLI 到 mock run 边界，确认默认 `chunk_windows=128`、tokens/package resolve 后正确传递；真实模型 driver 则由上一项覆盖。

## 实际命令与结果

工作目录为 Phase3 worktree：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
tests/test_phase3_external.py -q
```

首轮 **3 passed / 1 failed，7.33s**。失败是独立聚合 oracle 把“等长分段直接平均”与“先乘目标数再除总目标数”的 float64 运算要求逐位相同：`1.4500000049562816` 对 `1.4500000049562818`，相差1 ULP，并非应用 target/accounting 错误。只将该数学 oracle 的绝对容差改为 `5e-16`（rel=0），保留 PPL=exp(实际NLL) 精确断言；token、权重、尺度、RoPE、logits 的精确断言均未放宽。

随后只执行失败项：

```text
tests/test_phase3_external.py::test_external_actual_acceptance_128_window_chunks_and_weighted_oracle
```

**1 passed，5.81s**；之前通过的三项未重复。最终为 **4 个独立测试实例 PASS**，不是第二次完整重跑4项。

完整首轮及单项复测日志保留于 `runs/phase3/verifier/c4_external_cpu_20260915.log`。仅有既有 transformers 弃用警告，无应用修复。本次项目持久写入只有新测试文件和本报告/log；pytest tiny fixture 产物在临时目录。

## NOT TESTED

实际最高精度大包本轮 C4 模型前向、GPU设备/RoPE迁移与显存、真实 BF16/量化两组 C4 NLL/PPL及其差值、完整 C4 数据集、与基础模型预训练语料的独立性、native kernel/KV/decode/手机部署。历史 WikiText 最佳包审计不是本次 C4 验收；本报告也不将 mock 分数称为泛化证据。独立代码核验本轮完成，等待真正的新代码变化。
