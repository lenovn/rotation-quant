# 独立验证报告

结论：**PASS**。验证角色独立于实现者；本角色未运行 GPU、未修改实验代码或已有产物。本文件为获授权新增的验证记录。

## 验证范围与证据

- 审查 `scripts/phase2/down_d_followup.py`、`down_d_search.py` 续轮入口及 `46_run_down_d_search_local.sh`。续轮在读取 validation 数据之前转入 train-only 分支；四模型均使用同一组 32×2048 train 窗口，来自已有 settings 的 seed 42 窗口索引。固定行抽样为每窗口 64 行，共 2048 行。
- `identity_recomputed`、`search_recomputed` 的 `source.json` 均指向首轮 `down-d-search-20260913.bpL0NE` 对应冻结 payload，`recalibrated=false`；实现直接恢复其 W、96 非 down SA、16 down alpha。
- CPU 独立读取并检查两个新候选 `packed_model.pt`：各 112 个权重、16 个 D、96 个非 down SA；非 down SA 全部与原 C 逐 tensor 相等；80 个非 up/down 权重的 packed codes 和 SW 全部与首轮重算 SW 的 identity 对照逐 tensor 相等。
- 新候选分别在所有 16 层只测试 `t=0.25` 或 `t=1.0`，每层 trial 数为 1，chosen_t 与固定强度一致。每层范围统计 65536 行、MSE 2048 行；传播检查也是 65536 行且零超范围。保存的 down alpha 与所选 trial 的 full_absmax 相等。所有 D 均在 [1/4,4]，mean(log D) 绝对值低于 1e-6。
- 独立检查全部 12 份 train 结果：split=train、windows=32、predicted_tokens=65504、frozen_unchanged=true；单项结果与 `train_results.json` 一致；从 PPL 重算 log 得到记录的 NLL。实现评分前后逐项核对实际权重与冻结 payload、量化器 buffer 与评分前快照。
- 每个 full-A8 模型均有 16 份 down 统计，每份元素数为 65536×8192=536870912，零超范围。归零比例均与原始计数独立重算一致。CPU 网格检查确认归零计数公式与真实 FrozenCodebookQuantizer INT8 输出一致。
- Python 语法/import、shell 语法检查通过。已有首轮 CPU 验证包括实际 tiny BF16 Llama 的三模型构造隔离、FP 权重不变、恢复 identity 后 codes/SW/alpha 一致，以及全部 96 个 C SA 与真实非 down quantizer 的数值比较。

## Train 结果

| 模型 | all-A16 PPL | down-A16 PPL | full-A8 PPL |
| --- | ---: | ---: | ---: |
| 冻结 I + 重算 SW | 16.1759471893 | 16.2408142090 | 1745.6102294922 |
| 冻结首轮选 D | 17.3563289642 | 17.4316825867 | 168.6699829102 |
| 全层 t=0.25 | 16.4680881500 | 16.5260009766 | 1388.3023681641 |
| 全层 t=1 | 18.9847583771 | 19.0672702789 | 182.7207794189 |

两个新增候选在本组 train full-A8 NLL/PPL 上均未超过首轮选 D。原 identity 的 layer 1 新增归零比例为 0.9999990444630384；首轮选 D 的对应比例为 0.9999989233911037。这支持 fullrange INT8 存在严重归零，但不能仅用该单层比例解释不同模型最终 PPL。

## 预算与结论边界

- 续轮 `budget.json`：本次 187.17659803299466 秒，prior 370.8911193340318 秒，累计 **558.0677173670265 秒**，低于 1800 秒。
- 续轮额外完整 validation 次数 **0**；首轮冻结完整 validation 为 **9**，累计仍为 **9**。
- 续轮 allocator 峰值 3.994140625 GiB；记录的进程显存 3748 MiB 是采样峰值，不是全时真实峰值。首轮 allocator 峰值 5.25 GiB。未观察到预算超限记录。
- 这些新增结果仅是固定 train 窗口诊断，不是新增 validation 泛化收益；首轮 full validation 的 D full-A8 PPL 161.3686601098 仍是该模型既有最终精度证据。
- CPU 全部设备相同的 tiny 检查未覆盖真实 GPU 跨设备问题；首轮初始化曾因 model-level rotary buffer 在 CPU 而失败，之后已修复并成功执行，失败耗时已纳入上述 prior。
