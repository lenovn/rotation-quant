# Phase5 外部近邻对照范围核对

日期：2026-09-18。只读检查 handoff、本地继承源码及官方 SpinQuant 文档/实现；没有启动训练、GPU、安装、源码修改或新模型下载。

## 本轮要求的解释

**本轮必须完成外部候选身份与范围登记；handoff 没有把尚未确定身份的外部完整复现明确列为必跑矩阵。** 这是对明文的范围解释，不是因为成本而删减实验。

- `HANDOFF.md:234` 的动作是“在 PLAN 确定至多 1–2 个实际可运行的近邻基线，说明原生及适配设置”，并明确身份与可运行性尚未确定。不能将仅找到源码写成已完成运行验证。
- §1 的“本轮纳入”、§5.2 表格、§7.3 交付和用户启动语明确要求固定主线、完整 Uniform/SP2×PTQ/QAT 内部对照、代表模型消融、关键 seed、必要 test/C4；均没有另列必须完成某一外部方法。
- 因此本轮把下面唯一候选写进 PLAN，标清源码就绪但未运行验证，继续完成当前核心任务。不能以这项计划登记声称论文外部强基线或官方方法复现已完成，也无需自动追加第二个方法、外部训练或搜索。

## 唯一推荐候选

**SpinQuant 的 learned-rotation + GPTQ，W4/dynamic-A8/KV16，先限现有 Llama-3.2-1B-Instruct、seed42。**

官方仓库确有小模型 W4/group32、动态 A8 的 ExecuTorch 路径，以及旋转学习与 GPTQ 两阶段入口；这使它比新引入另一个仓库更接近当前可复用资源。[官方说明](https://github.com/facebookresearch/SpinQuant#3-export-to-executorch)

状态准确写为：**源码入口已核实；目标设置尚未做运行验证/正式复现。** 旧环境和本地 Phase2 运行只能证明部分基础设施曾工作，不能证明下述动态 A8/完整在线算子组合已通过当前验证。

| 项目 | 已核实本地入口 / 最小建议 |
| --- | --- |
| 旋转学习 | `repos/SpinQuant/optimize_rotation.py`；模板 `scripts/31_optimize_rotation_executorch.sh`，W16/A8、100 updates、LR1.5/cosine、仅学习 R；原模板8进程×每卡1，应按实际健康资源改为全局8，而非直接跑8卡模板 |
| W4 导出 | `repos/SpinQuant/ptq.py` → `eval_utils/main.py::ptq_model` → `eval_utils/gptq_utils.py::gptq_fwrd`；generic `scripts/2_eval_ptq.sh` 可表达 W4/A8/KV16，增加 group32，与小模型官方量化粒度一致 |
| 动态输入 | `utils/process_args.py` 的 `legacy_dynamic` 与 `utils/static_quant_policy.py::_configure_legacy_inputs`；普通 Linear 默认 per-token，o_proj 输入另按 head_dim 分组；asymmetric A8 与本项目静态 symmetric per-tensor 不同 |
| 在线算子 | 保留该候选原有 down 在线 Hadamard 与配对离线变换，不套主线 R1/R2-only 限制后再冒称原方法；KV16 不触发低比特 K 的 post-RoPE 包装，实际开关逐项记录 |
| 数据 | 本地 rotation训练用 WikiText train；GPTQ `get_wikitext2(eval_mode=False)` 使用双换行拼接 train 的随机连续2048窗口，默认128 samples；这不同于主线按行无特殊token拼接的32校准窗，必须披露 |
| 恢复预算 | 100步 R优化后 GPTQ，无额外 QAT400；这是外部方法原生预算参照，不是等恢复预算机制对照。后者由内部 Uniform-QAT400 完成 |
| 评测 | 若以后执行，保存同一份固定导出，统一完整WikiText test及既有C4；复用目标模型BF16。不能直接采用旧ptq.py自带整窗截断指标代替本轮含尾窗协议 |
| Qwen | 本地原始ptq/optimizer仍写死 Llama 模型和tokenizer；当前Phase5架构适配并未自动使官方dynamic/GPTQ全路径适配Qwen，因此不列为“Qwen外部基线已可运行” |

## 必须披露或先解决的差异

1. **本地 checkout 不是未经修改的官方版本。** 当前本地 HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，remote upstream 为官方仓库，含此前本项目静态量化实现。最具体的差异：本地 `eval_utils/main.py` 在 GPTQ 前配置输入 A8；官方当前代码在 GPTQ 后配置 A8。这会改变 GPTQ 的实际输入/Hessian，而不仅是日志差异。直接运行本地入口最多叫“基于本地 SpinQuant 的适配对照”，不能称官方完整复现。[官方 PTQ 实现](https://raw.githubusercontent.com/facebookresearch/SpinQuant/main/eval_utils/main.py)
2. 若目标是更接近官方算法，应在独立可追踪工作区保留官方 legacy 分支的 GPTQ/激活配置顺序；已有静态主线及其安全措施不动。此次仅登记差异，**未实施此适配**。
3. 官方 `32_eval_ptq_executorch.sh --export_to_et` 还会把 embedding/head 做8bit并转为部署导出结构。本轮建议用 generic PTQ 保存可评测模型、保留 embedding/head高精度，便于统一验收；这就是明确的范围适配，不能说精确复现官方 ExecuTorch 包，也不执行手机内核评测。
4. 不得复用本项目 Joint100 的 R/state 当作官方 SpinQuant 旋转学习结果：前者是 W4-aware R/SA/SW 联合学习，与该候选 W16/A8 只学R不同。已有 Uniform 内部控制也不能替代这个外部方法。
5. 不建议新增“QuaRot官方复现”作为第二行。本地随机旋转选项存在，但它只能证明一个可配置的随机旋转路径；未经独立方法/配置核对，不能把它命名为完整 QuaRot。暂不增加第二个候选最符合最小范围。

## PLAN 可直接采用的文字

> 外部近邻候选暂定1个：SpinQuant learned-rotation+GPTQ 的 W4/group32、dynamic asymmetric A8、KV16 参考，先限现有 Llama-3.2-1B-Instruct seed42。已核实本地 optimizer/PTQ/GPTQ/legacy-dynamic 源码入口，尚未做目标设置的运行验证或正式复现。其动态A8、o_proj分组、在线Hadamard、只学R的100步预算及GPTQ训练采样与本方法不同，需单独披露；本地A8配置先于GPTQ的顺序还不同于官方实现，不得冒称原版复现。本轮完成候选登记并优先完成已授权的核心矩阵、消融、seed和最终验收；不因候选登记自动增加外部训练或第二个方法。内部Uniform结果继续单独命名，不能填入SpinQuant外部行。
