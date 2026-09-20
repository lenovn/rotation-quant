# 当前W4：逐层MLP及up/gate/down定位结论

2026-09-15；本轮用户明确要求覆盖每个layer及三个投影。65项新完整GPU validation全部完成，复用layer1 down与此前完全匹配的W4/W16/家族参照。不训练、不重新量化、不导出模型。

## 主要结论

1. 全家族恢复的条件NLL收益为down=0.058222994、up=0.031917836、gate=0.020429190，因此当前全A16条件下总体顺序明确为down > up > gate。三类矩阵各16个，每类总权重元素数量相同；这些是独立干预的边际效应，不是可加贡献率。
2. 整体MLP：layer1 ΔNLL=0.010770297，layer15=0.010730881，两者差仅0.000039416，视为近乎并列，不声称稳健第一；layer10=0.009805507紧随其后。layer9/13/11/12也是较敏感的后层群。
3. 最突出的单矩阵是layer1 down（ΔNLL=0.010174870），该层up/gate明显较小，恢复down已接近整MLP的恢复效应；但这不意味着全模型由这一层支配。layer15 down与layer10 down也是明确重点。
4. down在16层的单投影测量中均是数值最大者，但layer3与5的up、layer11的up、layer13的gate已接近down，不能把微小差距包装为稳健主导。
5. up最敏感的两个位置为layer10、11；gate最敏感为layer13，其次layer12。layer13 gate收益超过up，说明全局down > up > gate不能逐层机械套用。
6. 当前测量支持优先研究layer1/15/10的down，以及layer10/11的up、layer13的gate；这只是选取研究对象的任务敏感性依据，不等于已有固定SW邻码/SW拟合就能取得收益，不能重复失败配置当作新路线。

## 怎样理解这些恢复

实际MLP为 down(SiLU(gate(x)) * up(x))。单独恢复up/gate会改变该层中间特征，再经其余当前W4计算；单独恢复down直接改变投影权重。整体MLP同时恢复三个矩阵，因而与三个单项之和不同。本轮完整评测定位的是这些干预后的最终任务影响，没有声称已解释非线性误差放大的具体方向。

所有单元从同一原父q/SW干净重置，保持同R、全A16、其他模块W4。原父W4/all-A16 NLL=2.7445129152399823，PPL=15.557034492007398；ΔNLL=父NLL−该条件NLL，正值代表恢复改善。表中编号0..15对应实际第1..16层。

不能把单层恢复收益求和当成多个层同时恢复的收益，也不能把本全A16排名直接当作最终静态INT8/SP2排名。若后续要选择最终格式优化对象，应只对少量候选检查激活交互。没有随机性复跑，因此不区分近乎并列项的稳健次序；不声称外部C4同样排序。WikiText2 validation已经查看并用于诊断分析，非盲测。

## 结果文件

- [16层完整表](TABLE.md)：64格（16整MLP+48投影）及全家族对照。
- [机器可读结果](summary.json)、[CSV](layers.csv)、[热力图PNG](layers.png)、[SVG](layers.svg)。
- 每格保留完整NLL/PPL及来源，layer1 down明确复用上一轮all16:down1，与新增down@1选择严格等价。

## 运行与审查

源码：Phase4 diagnose.py只新增@layer及全up/gate selector；configure、量化、参考构建、评测逻辑保持不变。report_mlp_layers.py只聚合已完成结果和作图，不运行模型。原Phase3工作树与旧产物未修改；未提交、推送、安装或清理。

父包：runs/phase3/seq-b100-sp2-refine-down-20260914a/static_w4a8.pt；参考：runs/phase3/route-b-adam-100-20260914a/checkpoint-0100/state.pt同R的原BF16未量化权重。GPU fakequant prefill、KV16/use_cache=False；112 weights保持原signed[-8,7] per-output SW，只有被选中矩阵作BF16恢复。

完整协议：252852输入、252728预测targets、123×2048+948尾窗，沿用token加权NLL→exp，不平均PPL。无CPU模型forward。

| batch | GPU | PID | 新完整case数 |
|---|---:|---:|---:|
| diag-layer-mlp-20260915a | 5 | 1080049 | 18 |
| diag-layer-down-20260915a | 7 | 1080055 | 15 |
| diag-layer-up-20260915a | 0 | 1080061 | 16 |
| diag-layer-gate-20260915a | 1 | 1080067 | 16 |

实际命令、启动显存、session/PID、环境和日志见各同名.launch.json/.log；tmux socket rotation-quant-phase4。四批自然完成，无他人进程干扰。只新增小型结果/图表，无新大模型包；遵守用户实际可用空间限制。

独立verifier PASS：../verifier/mlp-layer-review.md。独立auditor对65个原始测量、64格汇总、全家族复用、数值/图表及结论审查，报告：../auditor/MLP_LAYER_AUDIT.md。没有新部署成品，不重审旧冠军或追加冷载复评。
