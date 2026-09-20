# WikiText-2 test 接口：独立 CPU verifier

日期：2026-09-17。结论：**限定 CPU PASS；7 项首轮全部通过，21.42s，无未解决应用 FAIL。**

真实离线 WikiText-2 test：**4358 rows、289076 tokens、141 个完整 2048 窗＋308-token 尾窗，共142窗；288934 predicted targets，unscored_tail_tokens=0。**

本结论支持新增接口的代码验收；可交回主执行启动已授权固定 BF16/最佳包配对测量，不代表真实模型 PPL 已测得。

## 权限与事实来源

- 只读核对项目 AGENTS、STATUS/SPEC/PLAN/LESSONS 与 `runs/phase3/STATUS.md`；当前用户专项授权优先于历史授权。
- 唯一源码 `worktrees/SpinQuant-phase3-joint`；无更深 AGENTS。当前 HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，工作树存在既有未提交修改及 untracked `experiments/`。HEAD 不完整代表本次新增接口。
- 已读当前 external evaluator、既有四项 external 测试、common tokenizer/data 定义、实际共享 acceptance 的尾窗/target 加权逻辑。
- 主执行通知补丁完成后，对比会话内修改前源文件与实际 `external_eval.py`：新增 test loader、路由/metadata/保存 tokens、CLI 互斥；`load_c4_tokens`、`load_model`、`quantizer_snapshot`、`evaluate_tokens` 函数未改。新路径仍使用既有量化冷载、原 BF16 和既有 CE/NLL evaluator，没有插入训练或参数选择。
- 核查及测试时 `external_eval.py` 为 10865 bytes，mtime `2026-09-17 23:43:54.359779234 +0800`。完整实际 diff 已在会话审查；未新增 hash 或 gate。
- 未创建 goal；未启动 GPU/tmux；未修改应用、既有 tests 或 status；未安装、提交、推送、删除、校准、训练或选包。

## 实际通过范围

测试文件：`worktrees/SpinQuant-phase3-joint/tests/test_phase3_wikitext_test.py`。

1. 新测试 `:41`：PyArrow 只读实际 `common.DATA_PATH/wikitext-test.arrow`，以同一路径本地 LlamaTokenizerFast、禁 BOS/EOS、双换行独立构造 oracle；与新 loader 逐 token 一致。核对 dataset/subset/test、真实行数、路径与全部窗口/target metadata。限制 Dataset loader 只能读 test，禁止 `load_dataset` 与 `common.data_windows`，Arrow 前后 size/mtime 不变。metadata 使用实现已有的 `text_join=double-newline`，不要求新增另一个字段。
2. 新测试 `:79`：真实 `evaluate_tokens`/现有 acceptance，仅底层 evaluator mock。实测分块 **128×2048、13×2048、1×308**；对应 targets **262016、26611、307**。逐 token 切片/拼接完整一致，NLL 按这三个 target 数加权，非简单平均 NLL/PPL；评测后 seqlen 恢复2048。float64 oracle 使用 abs=1e-15、rel=0；tokens/尺度比较不放宽。
3. 新测试 `:114`：复用既有 tiny CPU fixture，真实 BF16/量化冷载后执行 test driver（CUDA 设备调用映射 CPU，分段 evaluator mock）。保存的 `input_tokens.pt` 与 oracle 全量一致；data/result 都是 test metadata，无训练/校准/selection；两个分支收到完全相同 tokens。112 个量化器 before/after 精确一致，完整 tiny state 与原/冻结 fixture 一致，量化 fixture 包字节未变。注入 SP2 尺度改变被拒绝且不写成功结果；未加载最佳大包做 forward。
4. 新测试 `:198`：合成2049 tokens，以真实 loader 计数和真实 run/acceptance 验证尾长1。仅完整2048窗计2047 targets，余下1 token明确记录 `unscored_tail_tokens=1`，正常完成而非强行要求0。此例是合成输入，不与真实 test 数字混淆。
5. 新测试 `:229`：实际 CLI 到 mock run 边界；必选且互斥 `--tokens`/`--wikitext2-test`，两种缺省/冲突非法组合在输出目录创建前拒绝，保留 package resolve 与默认 chunk128。
6. 既有 `test_external_paired_driver_real_coldload_rope_and_immutable_scales`：因本次修改了共享 run 的数据路由、标签及 tail 检查，针对性复跑 C4 driver。旧 args 无 `wikitext2_test` 字段仍可运行；真实旧 C4 token 包、1024窗/2096128 targets、C4/validation 标签、immutable尺度及旧拒绝错误字符串全部通过。模型是本地 tiny CPU fixture，C4 分数 mock；原测试中的短 CPU forward正常通过。
7. 既有 `test_external_launcher_routing_environment_and_driver_default`：因 CLI 从 required tokens 改为互斥必选组，针对性复跑原 launcher/CLI，原 C4 argv、路径转义、默认 chunk128 均通过；tmux/nvidia-smi 均 mock，未真实查询或启动 GPU。

测试的临时 tiny 模型和 driver 输出使用 `/tmp/pytest-of-dongpeiyan/pytest-4/`，不覆盖项目实验产物。仅运行这7项，未运行既有 external 另外2项、旧全套 suite 或 GPU 保险测试。

## 实际命令与结果

工作目录：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
tests/test_phase3_wikitext_test.py \
tests/test_phase3_external.py::test_external_paired_driver_real_coldload_rope_and_immutable_scales \
tests/test_phase3_external.py::test_external_launcher_routing_environment_and_driver_default -q -s
```

结果：**7 passed, 2 warnings in 21.42s；exit 0**。只有既有 transformers 弃用警告；tokenizer 另打印 checkpoint 类名/legacy 与长 token 流提示。应用和独立 oracle 使用同一指定 LlamaTokenizerFast，逐 token 一致；实际评测按2048及308分段，不把289076长串直接送模型。CLI 错误输出来自预期拒绝用例，不是测试失败。

对新测试执行 `git diff --no-index --check -- /dev/null tests/test_phase3_wikitext_test.py` 无空白错误。没有修应用、没有首轮失败或追加复跑。原始命令、stdout/stderr、最终 Git 状态已保存到 `runs/phase3/verifier/wikitext2_test_cpu_20260917.log`。

## 修改路径

- `worktrees/SpinQuant-phase3-joint/tests/test_phase3_wikitext_test.py`
- `runs/phase3/verifier/wikitext2_test_cpu_20260917.md`
- `runs/phase3/verifier/wikitext2_test_cpu_20260917.log`

## NOT TESTED

最佳大包与原完整 BF16 的真实 WikiText-2 test PPL/NLL、两项正式配对 GPU 测量、GPU显存/设备行为、native kernel、decode/KV cache、手机部署均 **NOT TESTED**。本轮不审计基础模型预训练语料或项目全部实验历史中的 test 接触独立性；不读取 WikiText train/validation 来拟合参数。日志中的 mock PPL/NLL 只验证数学聚合，不是质量或泛化成绩。正式分数与历史独立性说明由主执行负责。
