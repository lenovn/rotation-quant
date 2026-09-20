# B 联合优化步数比较

旧 100-step 终态与新 512-step 日程中间点分别标识；不是相同学习率轨迹。
固定探针使用训练配置 down-A16；完整 validation 使用初始 down-SP2 导出包。
没有后处理、QAT、test 或 C4 选择。缺失的 PPL 表示尚未完成该评测。

| 运行位置 | step | cosine 总步数 | 固定探针 NLL | 完整 validation NLL | 完整 validation PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| old_schedule_final | 100 | 100 | 2.9100092053 | 2.8400788349 | 17.1171149099 |
| new_schedule_intermediate | 0 | 512 | 3.6005548835 | 未评测 | 未评测 |
| new_schedule_intermediate | 100 | 512 | 2.9094509482 | 2.8366982881 | 17.0593473996 |
| new_schedule_intermediate | 256 | 512 | 2.8445532322 | 2.7824413933 | 16.1584218942 |
| new_schedule_final | 512 | 512 | 2.8124395609 | 2.7583813087 | 15.7742885681 |
