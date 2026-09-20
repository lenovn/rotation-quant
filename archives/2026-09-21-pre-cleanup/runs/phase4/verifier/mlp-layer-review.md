# MLP 逐层诊断增量 verifier

日期：2026-09-15。结论：**PASS**。独立 verifier 只审查 `experiments/phase4/diagnose.py::selected` 新增的逐层选择语义；未修改源码，未重复成熟加载/完整模型测试或增加 GPU gate。

工作树：`worktrees/SpinQuant-phase4-w4`。

- 新增合法选择为 `mlp/gateup/gate/up/down@0..15`；先解析层号并限制范围，再要求精确前缀 `model.layers.{layer}.mlp.`，最后按 projection 筛选。层号后的点保证 layer 1 不会误选 10–15，self_attn 不会进入任何逐层 MLP 组。
- `mlp@k` 精确包含该层 gate/up/down 三个矩阵，`gateup@k` 包含两个，其余各含一个。全局 `up`、`gate` 分别是 16 个对应投影。原 `down1` 与 `down@1` 等价。
- 独立轻量测试仅通过 Python AST 提取 selected，不加载模型或 torch：对标准 112 个模块名，穷举 16×5=80 种合法逐层选择，逐一比较完整名称集合；检查 up/gate 全家族各 16、down1 别名，以及越界层号/不支持投影/非法文本均抛出 ValueError。**PASS**。
- 以保存的 `runs/phase4/diag-mlp-error-components-20260915a/diagnose.py` 为此前实际执行源码对照：`configure`、`rtn_reference`、`error_components`、`run` 的 AST 完全一致。每 case 原父 q/SW 重置、激活 bits 设置、BF16 参考覆盖与成熟完整 validation 调用未受影响。

本次 65 项测量计划的算术为 16 整 MLP +16 up +16 gate +15 down +2 全家族；layer 1 down 复用既有匹配结果。此报告核查选择器和未受影响的函数，不将计划列表冒充已完成结果。逐层恢复依然是固定父包/all-A16 下的条件效应，不能将各行收益相加当作唯一因果分解。
