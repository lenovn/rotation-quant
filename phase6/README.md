# phase6

服务迁移入口：[HANDOFF.md](HANDOFF.md)。

代码、已有结果和续训文件已上传。完整模型与数据正在上传 Release 草稿，尚未完成发布；当前不能宣称空机已可直接续训。

发布后，在全新 clone 的根目录执行：

```bash
python3 phase6/restore_files.py
```

脚本只下载、解包，不安装环境，不启动训练、评测或调度器。按 HANDOFF.md 安装环境，待用户允许恢复后再继续未完成实验。

目标路径为 `/home/dongpeiyan/projects/rotation-quant`。保留 cosine512、现有优化器与随机状态；Qwen0.6B 已完成512，不重训。
