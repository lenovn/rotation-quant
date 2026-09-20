# 独立验证记录

2026-09-09，独立 verifier `/root/verify_atlas` 返回 **PASS**。本文由执行者记录 verifier 已报告的检查结果。

- 审查 `analyze_weights.py`、`collect_rtn.py`、`evaluate_mappings.py`、`run_mapping_ppl.sh`；小型 CPU 独立参考检查覆盖 zero/outlier/random rows、nearest projection 与 alpha/MSE 一致性。
- SP2 Eq.(8) 的 W4 sign/2/1 码本正确，13 个不同有符号值；INT4 为 [-8,7]。相同搜索预算，FP32 投影再转 BF16。
- 112 个模块、376,832 个输出 channel 数据完整、有限；逐模块 SSE 汇总独立重算一致。
- BF16 与 RTN 各 112 个权重、64 个激活记录，4096-bin 直方图计数和元素总数一致。
- BF16 与 RTN 各 296 张 PNG 全部经 PIL verify 可读：176 张分布图为 2400×1600，120 张 channel-score 图为 2850×1500。
- `channel_examples.png` 为 2400×2560，已目视检查标注与 channel 选择。
- 实际 RTN Atlas 在 wrapper 前捕获，表示该层输入量化前、带上游量化误差传播的轨迹；重载 R/权重/SA 路径匹配。
- 参考 PPL 14.655250549316406 与源 C RTN 结果完全相同。
- INT4/SP2 候选均从同一 C-R BF16 重建全部 112 个权重，未对 RTN 或前一个候选重复量化。wrapper 与 module 的权重别名保持正确；SA、down-A16、KV16、test 8×2048 固定。
- 端到端 PPL：当前 RTN 14.6552505493、INT4-MSE 17.1246547699、SP2-MSE 20.1759014130。

限制：有限 alpha 搜索不是数学全局最优；粗搜端点指标只描述粗搜。少数行有明显尾点，不能声称完全没有离群值。结果仅判断固定 C 后直接换映射，不覆盖格式专用重训或手机原生后端收益。
