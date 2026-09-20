# 固定 C4：蒸馏归因与匹配量化对照

日期：2026-09-18。用户批准此前列明的三项新增完整C4测量；不追加QAT训练或C4驱动搜索。

**最终状态：已完成3项新增完整C4评测，复用2项既有结果，未追加训练或按C4调参。** 全部2096128个预测targets、8段同口径、KV16/PREFILL。机器可读结果及逐段差值在 `C4_ATTRIBUTION_20260918.json`。

| 固定模型 | C4 PPL | C4 NLL | 本轮测量状态 |
| --- | ---: | ---: | --- |
| 原BF16 | 21.843397624525274 | 3.0838987076656394 | 复用旧结果 |
| B100＋匹配均匀静态INT8 | **1330.1567504736056** | 7.193052071770102 | 新完整测量 |
| B100＋初始SP2 | **29.18423776720624** | 3.373628761035482 | 新完整测量 |
| 精确PTQ父包 | **27.43961641336637** | 3.311987823655252 | 新完整测量 |
| QAT400 | 26.98478050458106 | 3.295273022058687 | 复用旧结果 |

## 结论

1. **QAT400在C4上是部分修复，不是没有修复，也不是造成额外损失。** 与精确PTQ父包相比，PPL降低0.45483590878530933（1.6575884369%），NLL降低0.016714801596564577，8/8段均改善。按相对原BF16的超额NLL计算，修复约7.328189%，幅度有限，未修复至BF16。
2. **后处理也有真实C4收益。** B100初始SP2→PTQ父包，PPL降低1.7446213538398716（5.9779575802%），NLL降低0.06164093738023002，8/8段均改善；后处理＋QAT累计降低2.199457262625181PPL（7.5364560835%）。
3. **静态非均匀激活在该匹配机制对照中有明确收益。** 同B100 R/W4码/SW/96非down SA/校准数据及50候选预算，down均匀INT8的PPL为1330.15675，SP2为29.18424，8/8段NLL均明显更低。均匀尺度不是故意使用full-range，而是逐层真实输出MSE选择；全部尺度及推理格式已核对。这个受限静态per-tensor均匀对照出现严重退化，不能把它冒称原版SpinQuant、动态A8或外部SOTA，也不证明SP2胜过所有均匀量化方案。14/16层训练校准局部MSE更偏好SP2只是辅助记录，不代替C4实测。
4. **仍有显著剩余跨域量化损失。** 最终QAT400比原BF16高5.141382880055787PPL（23.5374687053%），NLL高0.21137431439304777。收益成立不等于已实现C4近BF16；不能用WikiText达标替代这个结论。没有同预算均匀QAT400对照，不把额外训练收益全部归给SP2。

以下保留实验顺序、失败诊断与验证过程；其中启动/待确认表述是当时记录，最终状态以上述完整结果为准。

## 固定对照与执行顺序

1. 新测精确PTQ父包 `seq-b100-sp2-refine-down-20260914a/static_w4a8.pt`，比较它与既有QAT400分数以判断修复、未见修复或额外损失。启动 `c4-ptq-parent-fixed-20260918a`，GPU5/PID957862，tmux socket `rotation-quant-phase3`。
2. 新测B100初始SP2包 `route-b-adam-100-20260914a/checkpoint-0100/static_w4a8.pt`，无离散后处理/QAT。启动 `c4-b100-sp2-fixed-20260918a`，GPU6/PID957929，tmux。两项使用已有独立PASS入口，未增加保险烟测。
3. 由相同B100冻结包构造匹配均匀INT8：仅替换16个down输入量化器，R、全部W4整数码/SW、96非down SA和高精度边界不变。保存小型尺度overlay，不覆盖或复制2GB父包。此新路径由独立verifier CPU窄测通过后执行。
4. 复用旧C4原BF16 PPL21.843397624525274/NLL3.0838987076656394，与旧QAT400 PPL26.98478050458106/NLL3.295273022058687，分别在 `c4-bf16-fixed-20260915a/result.json` 和 `c4-best-fixed-20260915a/result.json`；不把历史复用称为新测。

## 均匀INT8匹配方法

- 只读B100的 `data.json`，重建其WikiText train逐行无BOS/EOS token拼接，使用原32个calibration索引、每窗前128tokens；不读取C4或WikiText validation/test参与校准。
- 与原 `common.calibrate_sp2` 一样，capture时全部down旁路A16，保留非down量化与固定W4；逐层收集真实输入、每8行取1行用于局部输出MSE，完整捕获统计absmax。
- 搜索统一的范围模板：log2 ratio在[-16,1]的33个粗点，再对各自格式粗最优附近17点细化。均匀与SP2各自选择适合本格式的尺度，不能把SP2尺度强套给INT8；两者预算/数据/目标相同。
- 使用真实静态INT8算子产生量化值，BF16输出与原SP2校准一致，输出误差目标为 `mean(linear(quantized_input - input, fixed_W4_weight)^2)`，不以C4 NLL选择尺度。
- 保存16层全部50候选、选择结果、输入rows/absmax与历史SP2采样记录比较、全部固定参数边界证据。没有改变原SP2包的参数。

## C4评测协议

只读复用 `runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/input_tokens.pt`：1024×2048输入、2096128预测targets、每128窗一段，共8段。官方en validation的既有固定子集，不是全量C4或新的盲测集。相同tokenizer、BF16 logits CE、per-token loss转FP32、分段float32 PPL取log后target加权NLL，最后exp；KV16，use_cache=False，PREFILL fake-quant。

比较链：均匀INT8 → B100初始SP2 → 完整PTQ父包 → QAT400。分别报告每一段的ΔPPL、ΔNLL及8段分数方向，微小差异不夸大为稳健收益。最终QAT比未QAT均匀基线更好不等于收益全部来自SP2；若要比较相同QAT预算的均匀方案，需另行做匹配QAT400，此轮不包含。

## 状态

三项新增正式完整评测均已完成且PID退出；均匀校准a/b失败证据保留，V布局修正后正式c已完成且16层历史统计精确匹配。最后均匀C4 `c4-b100-int8-fixed-20260918a` GPU5/PID1045611，210.9679985670秒/peak7.9763698578GiB，完整2096128 targets，无failure，量化器前后状态精确不变。所有新产物写runs/phase3，已有模型/Phase2只读，无环境安装、删除、提交或推送。原已完成goal不重建；本次收束，不在C4结果上继续优化。

独立验证：8项首轮7过1失败，发现完成日志重复`peak_allocated_gib`导致TypeError；主执行仅将最终progress改传down_layers/elapsed_seconds，保留完整result.json，独立只复测失败项1 passed/10.16s，其余7项不重复。F001已关闭，最终限定CPU PASS，报告 `verifier/c4_uniform_cpu_20260918.md` / `.log`。没有GPU保险烟测；真实校准输入与旧SP2是否匹配、正式C4结果仍以新GPU日志为准。

实际校准问题：a（GPU5/PID1005558）在layer0的历史输入匹配检查失败；b（GPU5/PID1019026）保留阈值，仅补 `capture_metadata.json`，16层均512 sampled rows/历史50候选，但13层absmax不同，layer0 28.25/28.125、layer1 824/816、layer7 2.90625/3.015625（实际/历史）。未把失败称新完整C4，也未放宽检查。

源码定位候选：`QuantizeLinear.rotated_weight` 对V的R2计算以transpose结束，使原 `frozen_model` 的V权重布局为列主序；冷载 `reload_frozen` copy_到HF预建参数保持行主序。这可能改变BF16乘加末位，而非W4码本身。新修正仅在capture时用 `.t().contiguous().t()`复原V列主序，finally恢复原storage/stride；不改权重值、不改C4推理布局。窄独立CPU检查及实际GPU对历史absmax匹配仍待确认，当前不提前称根因已证实。

后续验证结果：布局修正独立窄项 **2 passed/8.98s**，此前8项不重跑，确认16个V值精确不变、列主序、正常/异常后原storage/stride/别名恢复。正式c `b100-uniform-calibration-20260918c`（GPU6/PID1040261）成功，16层sampled_rows均512、全部absmax与历史**精确相等**，并完成全部50候选/层、输出小型 `down_int8_scales.pt`。实际23.1867520760秒，peak2.9195966721GiB；原量化器状态前后不变，0 C4数据参与。由此V布局差异得到真实GPU定位支持；没有更改C4推理布局、历史包或放宽检查。独立CPU报告不是该GPU实验的独立复跑。

## 已完成的归因结果

| 固定模型 | C4 PPL | C4 NLL | 本轮是否新测 |
| --- | ---: | ---: | --- |
| 原BF16 | 21.843397624525274 | 3.0838987076656394 | 否，复用 |
| B100初始SP2 | 29.18423776720624 | 3.373628761035482 | 是 |
| 精确PTQ父包 | 27.43961641336637 | 3.311987823655252 | 是 |
| QAT400 | 26.98478050458106 | 3.295273022058687 | 否，复用 |
| 匹配均匀INT8 | 1330.1567504736056 | 7.193052071770102 | 是，后续已完成 |

- 后处理：B100→PTQ父包 ΔPPL=-1.7446213538398716（-5.9779575802%）、ΔNLL=-0.06164093738023002；8/8段NLL均下降。
- 蒸馏：精确父包→QAT400 ΔPPL=-0.45483590878530933（-1.6575884369%）、ΔNLL=-0.016714801596564577；8/8段NLL均下降。因此当前QAT确实部分修复C4量化损失，而非额外损失；仍明显不及原BF16。
- B100→QAT400累计ΔPPL=-2.199457262625181（-7.5364560835%）、ΔNLL=-0.0783557389767946。这不是SP2与均匀INT8的比较；后者见最终结果。
- 父包228.4187190849334秒，B100230.09369091899134秒，均peak allocated6.976354598999023GiB。包含加载/评测，不作为严格吞吐benchmark。
- 主端只读核查四组完全相同的token路径/metadata/2096128 targets/8×128分段/CE-NLL口径，量化器状态前后精确不变；`common.py`/`postprocess.py`/`quantization.py`/acceptance及tracked.diff与旧QAT源码副本逐字节相同。新增WikiText入口不改变C4数值路径；当前新overlay入口尚未用于这两组。
- 部分结果与逐段差值见 `C4_ATTRIBUTION_20260918.partial.json`，不得把旧BF16/QAT复用或独立CPU核验称为新增GPU复测。

## 最终产物核对与复现入口

- 五组输入token文件/metadata、2097152输入、2096128 targets、8×128窗、尾窗计数、BF16-logit CE/加权NLL完全相同；从各段重新计算的NLL/PPL与result精确一致。全部固定尺度前后状态相同，FP32 RoPE与KV16保留。
- 均匀与SP2实际加载同一个B100包；96个非down量化器状态逐元素精确相同，16个down尺度与train overlay逐元素精确相同，signed INT8的maxq=127/zero=0、positive single scale/has_scale/非observing确认，实际格式为112 INT8而非96 INT8＋16 SP2。W4冷载源码/包路径相同，校准未改写父包；其余共享核心量化、acceptance与tracked.diff和旧QAT源码副本相同。
- 均匀overlay仅6618 bytes：`b100-uniform-calibration-20260918c/down_int8_scales.pt`。同目录保留16×50候选、capture_metadata、data/settings/result、source及before/after；配对检查摘要 `C4_UNIFORM_MATCHING_20260918.json`。
- 新正式GPU结果在 `c4-ptq-parent-fixed-20260918a/result.json`、`c4-b100-sp2-fixed-20260918a/result.json`、`c4-b100-int8-fixed-20260918a/result.json`；完整argv/环境/显存快照/PID/tmux socket/session在同名 `.launch.json`，终端日志在 `.log`。
- 均匀评测复现使用 `launch.py --task external -- --mode quantized --package B100_PACKAGE --down-int8-scales OVERLAY --tokens C4_TOKEN_FILE --chunk-windows 128`，指定未使用的run name；不要覆盖历史产物。PTQ/B100-SP2不传overlay。新均匀校准入口及V布局原因见 `worktrees/SpinQuant-phase3-joint/experiments/phase3/README.md`。
- 独立verifier范围为应用代码CPU窄测（原8项＋布局2项，记录首轮F001及仅失败项复测）；0新增独立GPU保险复跑，不能称本次C4已有Aristotle复测PASS。正式分数来自主端实际GPU测量，布局c成功匹配是主端证据。
