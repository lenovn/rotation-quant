# Phase6 迁移交接

## 当前边界

旧服务器检修，用户明确禁止开启实验。当前仅打包和上传。新服务器只有在用户允许恢复后才启动训练或评测；安装、下载、解包均不能启动调度器。

本目录仍在上传，不能把 clone 成功当作迁移完成。第一批代码结果已上传；续训文件正在上传；完整数据包正在准备，尚未提供可用 Release 下载链接。

## 目标与实验设置

从原始预训练模型初始化，联合学习 R、W4 的 SW、非 down 输入 INT8 的 SA、down 输入 signed INT16 的 SA。down 的权重仍为 INT4。BF16 fake quant，不是原生 INT16 内核。没有从 B100 继续训练。

全部采用 warmup 10、cosine 终点 512，选 T 后截断，不能改成 cosine T。先完成 Llama1B pilot512，再按 scripts/phase6/schedule.py 的既定 validation 规则确定统一 T。test/C4 不用于选 T。四模型主实验、Llama1B fixed-down 与 R-only 消融、八项能力评测和 WT2 test / 固定 C4 validation 子集仍需完成。

## 文件与恢复

- phase6-code-results.tar.gz：有效代码快照、已有结果、依赖版本清单。
- state/runs/phase6/：续训及初始化文件，目录结构与原项目对应。resume-files.json 列出文件。
- resume-status.json：CPU 读取断点得到的实际 step、优化器与 RNG 存在性。
- 后续完整 phase6-data.tar.part-*：原始模型、数据缓存、所有 Phase6 checkpoint 导出、评测依赖及参考结果。未上传完成前不能进行完整恢复。

不要仅复制仓库旧 scripts/ 运行。首先在仓库根目录解开最新有效快照，再恢复 state：

```bash
tar -xzf phase6/phase6-code-results.tar.gz -C .
cp -a phase6/state/. .
# 等完整数据附件上传并下载后：
cat phase6-data.tar.part-* | tar -xf - -C .
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

1. 确认续训文件的远端上传成功。
2. 完整数据包上传 Release，并加入下载命令。
3. 补齐环境安装说明/入口与新路径适配。
4. 在全新目录核对恢复所需文件与引用，不启动实验。
5. 更新此文档为最终交接状态；当前为进行中交接，不是可运行验收。
