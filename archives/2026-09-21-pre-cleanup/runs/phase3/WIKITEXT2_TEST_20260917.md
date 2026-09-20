# Llama-3.2-1B-Instruct 固定模型 WikiText-2 test 评测

**状态：完整正式 GPU 配对测量已完成；不再按 test 分数搜索。**

| 固定模型 | 完整 test PPL | 完整 test NLL | 预测 targets |
| --- | ---: | ---: | ---: |
| 原始 BF16 | 13.162650325300651 | 2.5773832981204734 | 288934 |
| 最佳 W4/static A8（down SP2）+ QAT400，KV16 | 14.154416405154963 | 2.650026688830493 | 288934 |

差值：PPL **+0.9917660798543118（+7.5346989804020215%）**，NLL **+0.07264339071001968**。同 test 的 BF16+1 为14.162650325300651，本次最佳包低于它0.008233920145688245；这是本次固定模型/协议的实测，不是多seed稳健性结论。

## 请求与边界

2026-09-17 用户要求：邻码等策略在 validation 上选择，最终论文结果改用 WikiText-2 test 实测，不能继续以 validation 充当未用于选择的测试集。

本次固定两个既有模型，不新增候选搜索：

- 原始 BF16：`cache/models/llama-3.2-1b-instruct`，无旋转、无量化。
- 已按 validation 选定的最佳包：`runs/phase3/distill-b100-refined-ref-adam1e5-400-20260914a/checkpoint-0400/static_w4a8.pt`。112 backbone per-output-channel W4、96 non-down static INT8、16 down static SP2，KV16，R3/R4 关闭。它是 PTQ 后处理底座加 400-step 量化感知蒸馏，不是纯 PTQ。

没有训练、校准、重新选择 R/W4 码/SW/SA/SP2 范围，也不依据本次 test 分数继续调参。Phase 2 与原模型包只读保留。原 Phase 3 goal 已完成，本次是新授权的固定模型外部评测，不重建旧 goal。

## 协议

- 数据：本地缓存 `Salesforce/wikitext` / `wikitext-2-raw-v1` / 官方 `test`，读取 `wikitext-test.arrow` 全部原始行。
- tokenizer：同一本地 Llama tokenizer；无 BOS/EOS、无 truncation，全部行按原顺序以双换行拼接。
- 不抽样：2048-token 不重叠窗口，包含不少于 2 tokens 的末尾短窗；每窗首 token 没有前文，不计预测目标；若尾窗仅 1 token，明确记录不可评分尾 token。
- 推理：batch 1，BF16 fake-quant，SDPA，`use_cache=False`，PREFILL，KV16；不代表 native INT4/INT8 算子、decode/cache 或手机 NPU 性能验收。
- PPL：不改变既有 `utils/eval_utils.py` / `scripts/phase2/validation_acceptance.py` 数值口径，BF16 logits CE、单 token loss 转 FP32、各段 float32 PPL 取 log 后按预测 target 数加权，最后 exp。每段最多 128 完整窗，另评分尾窗；两个模型分段完全相同。
- 记录：完整命令、Git HEAD/dirty diff、源码副本、输入 token/metadata、模型格式、前后尺度、完整分段及 targets、PPL/NLL、tmux/PID/GPU/启动资源快照。不保存新的训练 checkpoint。

## 进度

1. 固定输入包和现有评测路径已核实；原最佳包的历史 validation PPL 为 14.581654675328117，不是本次 test 结果。
2. 仅扩展 `worktrees/SpinQuant-phase3-joint/experiments/phase3/external_eval.py` 支持 `--wikitext2-test`，保留原 C4 `--tokens` 模式。
3. 独立 verifier Dalton 已给出限定 CPU PASS：7 项首轮全部通过、21.42s，无应用修复。真实离线 test 为4358行、289076 tokens、141个2048完整窗＋308-token尾窗、288934 targets；测试分段128/13/尾窗，targets分别262016/26611/307。报告 `verifier/wikitext2_test_cpu_20260917.md` 与 `.log`；未运行旧全套或 GPU 烟测。
4. 正式 GPU 已通过既有 launcher 启动。最佳包：`wiki2-test-best-fixed-20260917a`，GPU5/PID871188；原 BF16：`wiki2-test-bf16-fixed-20260917a`，GPU6/PID871253。启动前对应显存569/633MiB（总24564MiB），记录全卡快照。tmux socket `rotation-quant-phase3`，session分别 `phase3-wiki2-test-best-fixed-20260917a` / `phase3-wiki2-test-bf16-fixed-20260917a`。完整 argv/环境在同名 `.launch.json`，输出在同名目录及 `.log`。本条记录启动，不代表评测完成。
5. 随后两项均写出完整 `result.json`、`progress.json: completed`，无 `failure.json`，两PID已退出。最佳包进程内含加载/评测50.81380989798345秒，peak allocated4.728203296661377GiB；BF16为34.221412922022864秒、4.414333820343018GiB。这些不是严格计时的decode/内核吞吐benchmark。
6. 主执行对产物做只读配对核对：输入token逐元素相等，metadata相等，所有分段及预测targets相等，两组源码副本（含tracked.diff）逐字节相等，settings的源码记录相等；结果均不训练/校准/选包，112量化器前后所有状态逐元素不变，BF16无量化器，FP32 RoPE保留。由分段NLL与targets重新计算得到完全相同的汇总PPL/NLL。机器可读汇总在 `WIKITEXT2_TEST_20260917.json`；这是产物核对，不冒充独立GPU复跑。

## 测试集独立性说明

本次 test 不参与当前包的训练、校准或新候选选择。需要区分“本轮固定模型不看 test 调参”与“整个项目从未读取过 test”：历史 `repos/SpinQuant/utils/data_utils.py::get_wikitext2(eval_mode=True)` 使用官方 test，早期评测确有此入口，不能未经历史审计宣称全项目从未接触 test。模型预训练与 WikiText 的潜在语料重叠也未审计。本次不为掩盖历史而删除/改写既有分数。

## 结果

正式结果文件：

- `runs/phase3/wiki2-test-bf16-fixed-20260917a/result.json`
- `runs/phase3/wiki2-test-best-fixed-20260917a/result.json`

每个run保留settings/model/data/progress、输入tokens、量化器前后快照、源码副本；同名目录外的launch JSON及log保留完整命令和运行过程。源码分支 `phase3/joint-r-sw-sa-sp2`，HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，有效实现包括已有dirty改动及untracked Phase3源码，不能仅用HEAD代表实验代码。

分段实际结果（总NLL按targets加权，不平均PPL）：

| start token | 窗数×长度 | targets | BF16 PPL | 最佳包 PPL |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 128×2048 | 262016 | 13.114706993103027 | 14.082803726196289 |
| 262144 | 13×2048 | 26611 | 13.580998420715332 | 14.817768096923828 |
| 288768 | 1×308 | 307 | 19.680500030517578 | 20.26766586303711 |

包含全部可评分尾窗，unscored_tail_tokens=0；每个独立窗首token不计目标，输入tokens不等于预测targets。两组均为新完整test实测，原14.5816546753属于validation，不改写为test或与本次绝对PPL作训练收益比较。

论文使用边界：固定方法的最终主表可引用本次test结果，validation继续标作开发/选择集。主结论限定为当前Llama模型固定W4/static A8/SP2+QAT路线；不根据本次test继续选邻码/SP2/LR/步数。没有新增GPU保险复跑、训练、环境安装、提交或推送，Phase2产物未改。
