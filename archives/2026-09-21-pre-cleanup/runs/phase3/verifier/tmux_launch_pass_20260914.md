# tmux launcher 定向独立核验：PASS

时间：2026-09-14 02:42 +08:00。

**1 passed，0 failed，6.67 秒，exit_code=0。** 本轮仅执行一个 launcher 定向测试，没有重跑原 54 项或后续其他测试。

## 被测源码

`worktrees/SpinQuant-phase3-joint/experiments/phase3/launch.py:11` 的 `start_tmux` 与 `main` 相关启动/记录流程。源码只读，修改时间在测试前后均为 `2026-09-14 02:39:49.357530795 +08:00`；worktree HEAD `24918316ed594848d4de797c356b120f2a4ee0f3`，另有未提交实现。

测试新增于 `tests/test_phase3_joint.py:768`：`test_tmux_launch_explicit_child_environment_quoting_pid_and_records`。

## 实际通过的断言

- tmux 使用 `-L rotation-quant-phase3` 独立 socket、detached new-session、显式 session/cwd，并通过 `-P -F '#{pane_pid}'` 取得 pane PID。
- 启动 payload 确实以 `exec env` 开头，在 child 命令内携带全部 9 项环境变量；不依赖新客户端环境自动同步到已有 tmux server。测试模拟 server 的 `CUDA_VISIBLE_DEVICES=3`/旧 HF_HOME，子命令明确指定 GPU7/新 HF_HOME，server 原环境不被改写。
- project、run name、log 和透传参数包含空格、单引号、分号、`$literal`、`$HOME`、`$(...)`。shell payload 经独立 `shlex.split` 还原后，环境值和 Python argv 逐项保持原值，stdout/stderr 重定向仍准确指向单个 log 路径；源码使用 `shlex.join`/`shlex.quote`，不是裸字符串拼接参数。
- PID 取模拟 subprocess stdout 的 pane PID，正确转为整数；log 在启动调用前独占创建。记录中的 PID/GPU、backend、socket、session、child_environment、command、output/log、selected GPU 与 snapshot 均匹配。
- attach 命令经过 shell 解析后仍准确指向相同 socket/session；stdout 打印的 JSON 与磁盘 launch JSON 完全一致。记录的 `shell_command` 字段是 Python argv 的 shell 表示，环境和日志另有独立字段，不误称该字段本身包含完整 `exec env` payload。
- 同名重复启动在调用资源查询/tmux 前拒绝，既有 launch JSON/log 保留，mock 调用次数不增加。

## 模拟边界

`subprocess.run`、`subprocess.check_output`、tmux 可用性查询均为测试替身；路径推导只在测试内指向 verifier 临时目录。**没有真实启动 tmux server/session、shell 或训练，没有调用 nvidia-smi/GPU，没有安装或改变共享环境。** pane PID 为测试值，未验证真实进程存活、tmux 运行时接受特殊 session 名、终端 attach 或 GPU 子进程行为。

本 PASS 仅覆盖启动命令构造与记录/保护逻辑，没有发现本项代码阻塞；不替代正式实验 auditor 或 GPU 评测。本项核验结束，等待下一处具体代码变化。

## 证据与命令

日志 `cpu-tmux-20260914.log`，JUnit `cpu-tmux-20260914.xml`。

```bash
cd /home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_phase3_joint.py::test_tmux_launch_explicit_child_environment_quoting_pid_and_records \
  -q --basetemp=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/pytest-tmux-20260914 \
  --junitxml=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-tmux-20260914.xml
```

命令以 `set -o pipefail` 与 tee 保存输出。写入仅限指定测试文件及 verifier 报告/临时目录；应用源码、正式实验产物、进程均未修改。
