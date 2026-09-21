# Phase6 迁移交接

## 当前能否恢复：不能，以下前置文件仍需补齐

截至 2026-09-21 本次远端核查：master 为 `4d2fd04`；14 个续训/初始化文件已在 Git 仓库，Release 仍为 draft，压缩分片 000/001 为 `starter`（未完成）。附件列表出现文件名或预期大小不等于上传成功。以下是当前缺项，不是要求重新训练。

### 继续五项中断训练前必须补齐

|缺项|新服务器目标路径（相对项目根目录）|如何补齐|
|---|---|---|
|原始模型及 tokenizer/config|`cache/models/llama-3.2-1b-instruct/`、`cache/models/llama-3.2-3b-instruct/`、`cache/models/qwen3-1.7b/`|等待完整附件恢复，或从旧服务器复制这些完整目录；也可重新下载原始模型，但必须匹配下述来源/版本，不能换成 Base 或其他 Instruct 模型。|
|WikiText-2 原始 Arrow 缓存|`cache/huggingface/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/`|复制完整目录，包括 train/validation/test Arrow 和 dataset_info.json；代码使用这个具体缓存目录，只有在线下载记录或换一个缓存路径不够。|
|Python/CUDA 与 FHT 环境|`runs/phase5/env/`，FHT 源码位于 `repos/fast-hadamard-transform/`|依本文安装命令在新服务器重建，不能把旧 venv 当作可移植环境。需要可用 NVIDIA 驱动、nvcc、编译器、tmux。|
|代码快照与断点放回实际运行目录|`scripts/`、`worktrees/SpinQuant-multimodel/`、`runs/phase6/<运行名>/`|这部分已上传：解开 `phase6-code-results.tar.gz`，再 `cp -a phase6/state/. .`；直接留在 `phase6/state/` 不会被 launcher 找到。|
|旧启动标记和新机 GPU 映射|`runs/phase6/*-resume-T.launch.json`、`scripts/phase6/schedule.py`|归档对应旧启动标记和日志，保留结果和断点；按新机改训练 JOBS 及评测候选 GPU。不要在前置文件不齐时启动 scheduler。|

原始模型来源：Qwen3-1.7B 为 `Qwen/Qwen3-1.7B`，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。Llama3.2-1B/3B 都是 Instruct，来自 `LLM-Research/Llama-3.2-1B-Instruct` / `LLM-Research/Llama-3.2-3B-Instruct` 的 ModelScope 分发；现有记录未给出可保证重新下载完全一致的不可变版本，优先复制已有目录或等待附件。不能把 mirror 的 master 当作已锁定的 HF commit。

仅继续训练时，不必等所有历史 `checkpoint-*/model.pt` 都齐全，但必须有对应原始模型、WikiText 缓存、已上传的 initial/resume 文件以及 results.json。pilot 还必须有 resume_parameters.pt。先恢复主线 481→512 等中断段；不要为缺少历史导出而重跑已经完成的训练。

### 完成评测及复用历史结果还需补齐

|缺项|必须恢复的位置|用途|
|---|---|---|
|Qwen0.6B 原始模型|`cache/models/qwen3-0.6b/`|已训练完成的模型评测；来源 `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`。不重训512步。|
|各 checkpoint 的模型导出与参数|`runs/phase6/<运行名>/checkpoint-*/model.pt`、`parameters.pt`|最终选定 T 对应模型、no-opt 和已完成 Qwen0.6B 的评测。已有小文件包中的 validation.json 不是模型权重。|
|固定 C4 文档及 token|`runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/`、`runs/phase5/qwen3-1p7b/c4-data/`、`runs/phase6/data/`|保持旧固定子集和分词结果，不能随意换新抽样后比较。|
|能力评测依赖及 NLTK 数据|`runs/capability-eval-20260920/deps/`、`nltk_data/`|既有 lm-evaluation-harness 0.4.8 及配套依赖；标准训练环境的 requirements 不等于所有评测依赖。|
|八项评测数据缓存|`cache/huggingface/` 的其余数据缓存|BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC-E、ARC-C、OBQA；可以重新下载匹配版本，但完整附件已计划携带已有缓存。|
|已有评测原始记录和 SP2 对照|`runs/capability-eval-20260920/`、已有 evidence_path 指向的 JSON、`runs/phase6/evaluation/`|多数小结果已在代码包，完整附件补齐全部原始产物；保留相同项目绝对路径以复用已有证据。|

以上缺项均已纳入正在压缩上传的完整迁移包。最省事的方式是等待 Release 全部分片完成并发布，再运行 restore_files.py；若要提前在新机恢复，按上述表从旧服务器复制缺失目录，或者按明确版本获取可重新下载的部分。旧服务器不启动实验。

## 当前边界

旧服务器检修，用户明确禁止开启实验。当前仅打包和上传。新服务器只有在用户允许恢复后才启动训练或评测；安装、下载、解包均不能启动调度器。

本目录仍在上传，不能把 clone 成功当作迁移完成。第一批代码结果、14 个续训/初始化文件及本交接文档已上传；完整数据包正在准备并上传到 phase6 Release 草稿。发布前下载命令不可用；发布完成后链接为 https://github.com/lenovn/rotation-quant/releases/tag/phase6 。

## 目标与实验设置

从原始预训练模型初始化，联合学习 R、W4 的 SW、非 down 输入 INT8 的 SA、down 输入 signed INT16 的 SA。down 的权重仍为 INT4。BF16 fake quant，不是原生 INT16 内核。没有从 B100 继续训练。

全部采用 warmup 10、cosine 终点 512，选 T 后截断，不能改成 cosine T。先完成 Llama1B pilot512，再按 scripts/phase6/schedule.py 的既定 validation 规则确定统一 T。test/C4 不用于选 T。四模型主实验、Llama1B fixed-down 与 R-only 消融、八项能力评测和 WT2 test / 固定 C4 validation 子集仍需完成。

## 新服务器操作入口

在新服务器将仓库克隆到 `/home/dongpeiyan/projects/rotation-quant`，然后执行以下命令。下载脚本仅使用 Python 标准库，不依赖 PyTorch，不启动实验。请在全新 clone 中恢复，已有同路径实验文件会被附件覆盖。

```bash
cd /home/dongpeiyan/projects
git clone https://github.com/lenovn/rotation-quant.git
cd rotation-quant
python3 phase6/restore_files.py
```

需要下载全部分片，并为分片和解包后的文件分别预留空间，建议至少约 200 GiB 可用空间，另留环境编译空间。下载中断后重新运行同一命令，可续传 `.partial` 文件；已完成的分片按长度复用。不要在下载期间自行替换文件。若 Release 尚未发布，该命令会报下载错误；这不是训练失败。

## 文件与恢复

- phase6-code-results.tar.gz：有效代码快照、已有结果、依赖版本清单。
- state/runs/phase6/：续训及初始化文件，目录结构与原项目对应。resume-files.json 列出文件。
- resume-status.json：CPU 读取断点得到的实际 step、优化器与 RNG 存在性。
- 后续完整 phase6-data.tar.gz.part-*：原始模型、数据缓存、所有 Phase6 checkpoint 导出、评测依赖及参考结果。未上传完成前不能进行完整恢复。

不要仅复制仓库旧 scripts/ 运行。首先在仓库根目录解开最新有效快照，再恢复 state：

```bash
tar -xzf phase6/phase6-code-results.tar.gz -C .
cp -a phase6/state/. .
# 等完整数据附件上传并下载后：
cat phase6-data.tar.gz.part-* | tar -xzf - -C .
```

目前应使用与旧服务器相同的项目绝对路径 `/home/dongpeiyan/projects/rotation-quant`。launch 和部分代码按目录定位，但历史 JSON/PT 中也有绝对路径。若新服务器无法使用该路径，必须先做路径迁移；不要只改一个环境变量便启动。

## 新服务器环境

旧环境不能直接复制：Python 3.9.23，PyTorch 2.4.1+cu121，transformers 4.51.3；依赖库存见 migration/phase6-20260921/requirements-observed.txt。库存中的本地 file:// 路径不是可直接安装的远端包。

在新服务器安装匹配 CUDA 的 PyTorch，并在 `runs/phase5/env` 创建 Python 环境；FHT 从 `repos/fast-hadamard-transform` 针对新 CUDA/PyTorch 编译。能力评测使用 `runs/capability-eval-20260920/deps` 的既有 harness 0.4.8。需要 tmux。以下为安装命令，仅供新服务器执行；本机未执行安装，也未验证新服务器兼容性。需要可用的 Python 3.9、编译器及 CUDA toolkit（nvcc）；驱动和 toolkit 由新服务器管理员配置。

```bash
python3.9 -m venv runs/phase5/env
runs/phase5/env/bin/python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
runs/phase5/env/bin/python -m pip install -r phase6/requirements-runtime.txt
runs/phase5/env/bin/python -m pip install packaging ninja wheel setuptools
runs/phase5/env/bin/python -m pip install --no-build-isolation ./repos/fast-hadamard-transform
```

能力评测依赖包含于完整包 deps/，与 Python 3.9/Linux 环境配套；新机如变更 Python/架构，不能照搬其中二进制扩展。

## 已有进度和恢复顺序

实际保存步数以 resume-status.json 为准；Qwen0.6 已完成 512，不重训。其他五项先恢复中断目标：

|运行名|模型目录|arm|原中断目标|
|---|---|---|---|
|llama1b-joint512-s42-r1|llama-3.2-1b-instruct|joint|512|
|qwen1p7b-joint-s42|qwen3-1.7b|joint|256|
|llama3b-joint-s42|llama-3.2-3b-instruct|joint|128|
|llama1b-fixed-down-s42|llama-3.2-1b-instruct|fixed-down|320|
|llama1b-r-only-s42|llama-3.2-1b-instruct|r-only|320|

首次在新机恢复前，将对应旧 `*-resume-T.launch.json` 和 `.log` 移入单独历史目录，保留内容；否则 launcher 会拒绝重复启动。同名旧 PID 和 tmux 信息不能视为新机活跃进程。不要删除 resume.pt、initial.pt、results.json、checkpoint-*。pilot 的 resume_parameters.pt 必须与 resume.pt 一起恢复。

获准恢复且环境就绪后，用新机实际可用 GPU 指定 `--gpu`：

```bash
runs/phase5/env/bin/python scripts/phase6/launch.py \
  --name llama1b-joint512-s42-r1 --model llama-3.2-1b-instruct \
  --arm joint --steps 512 --gpu 0 --resume
```

其他运行按表中 name/model/arm/steps 替换，不使用旧机 GPU 映射作为新机默认。维持每卡总显存低于90%，不影响他人。先恢复中断作业，再启动 scheduler；scheduler 不会自行恢复这些已中断任务。最终 T 尚未确定。

## 结果使用

已有结果和原始 JSON 一并保留。JOINT_RESULTS.md 中的 validation PPL 不能直接减去 BF16 test PPL。最终量化模型 test 和八项能力结果仍未完成，不能据此声称达到 SP2 精度。SP2 的已有结果用于协议匹配后的对照；明显异常时检查模型、数据、量化与加载过程。

## 尚未完成

1. 已确认续训文件的远端上传成功（bbeeaf2）；交接文档首版提交为 0f015f1。
2. 完整数据包正在上传 Release；下载命令已提供，发布前不可用。
3. 环境安装命令已提供；本次要求同绝对路径恢复，不承诺不同路径自动适配。
4. 在全新目录核对恢复所需文件与引用，不启动实验。
5. 更新此文档为最终交接状态；当前为进行中交接，不是可运行验收。

## 文件检查补充

必须先 clone 仓库，再解包附件；附件本身不包含 phase6/ 的安装与交接文件。嵌套源码的 .git 不随包迁移，本次均使用 --resume；新建运行的 Git 快照逻辑尚未适配，不能把本交接流程用于从零新建实验。FHT 源码已覆盖，无需额外子模块下载。scheduler 的训练 JOBS 和评测候选 GPU 列表均使用旧机编号，新机启动前必须同时调整，不能只修改 launch 命令的 --gpu。

## 压缩传输更新

本轮按用户要求改用 gzip 压缩。Release 分片统一为 `phase6-data.tar.gz.part-NNN`，restore_files.py 已对应使用 gzip 解包。之前未压缩上传已取消，不要混用 `phase6-data.tar.part-*`。完整打包及上传仍在进行，尚未完成发布。
