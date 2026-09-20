# 旧 C＋SP2：layer 1 恢复 BF16

本次单次完整 validation 已完成，进程 exit 0；独立验收见 verifier_report.md。

| 模型 | PPL | NLL |
|---|---:|---:|
| 原 BF16，历史匹配参考 | 13.63465773 | 2.61261491 |
| 旧 C＋SP2，历史匹配参考 | 17.64239998 | 2.87030509 |
| 旧 C＋SP2，仅 model.layers.1 恢复 W16A16，本次实测 | 16.47254442 | 2.80169502 |

对旧 C＋SP2：ΔPPL=-1.16985556，ΔNLL=-0.06861007。相对原BF16仍高2.83788669 PPL；距离BF16＋1目标14.63465773还差1.83788669 PPL。

恢复范围是零基索引 model.layers.1，即第二个完整Transformer block：q/k/v/o、gate/up/down七个Linear恢复同C R1/R2坐标系、norm融合后的量化前BF16，并关闭七处激活输入量化。其余105个W4权重与旧C checkpoint逐项相等，保留90个非down固定INT8和15个原SP2 alpha。未使用D，未搜索范围，未重新校准或量化。其余高精度边界含lm_head两个别名及norm与原C逐项核对；评分前后权重、激活buffer和实际bits检查通过。

完整WikiText-2 raw validation：252852输入token，123个2048窗口＋948尾窗，252728预测目标；窗口独立上下文，token加权NLL；BF16/KV16/use_cache=False。原17.64240为既有同口径结果，没有重复跑baseline。

解释：layer1恢复有约1.17 PPL净收益，但仅该层混精度不足以到目标。此实验同时恢复权重和激活，无法分别归因，也不能把整个block的收益归给down_proj。若继续定位，最直接的是固定其他条件，仅恢复该层W或仅恢复该层A，再决定是否缩小高精度范围；这些后续实验本次未运行。此结果是混精度诊断，不是严格全W4A8或设备部署验证。

本次GPU计时51.559630秒；累计609.627348秒；本次allocator峰值4.376953GiB，进程显存采样峰值2942MiB。本次用户新增授权1次完整validation，先前9次保持原记录。GPU1启动前2412/24564MiB，未终止他人进程。

配置 mixed_settings.json，原始结果 mixed_result.json，命令 command.txt，日志 experiment.log，预算 budget.json。复用启动器使目录名带down-d-search，但本次走mixed入口，未执行train校准、D搜索或训练。
