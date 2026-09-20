# distill master-reference 初始化：限定 CPU PASS

## 结论

本轮新增 master cell 投影、prepare_student 可选参考和直接 D 参考变换，针对性执行 **3 passed，6.98 秒**。下述范围明确 **CPU PASS**，未发现该已测实现范围的阻塞项。

未重跑既有13项/3项或全套测试，未运行GPU、参考模型GPU重建、新smoke、训练或完整validation。应用源码只读；只扩展授权的 `tests/test_phase3_joint.py`，临时包/日志及报告写入verifier目录。旧实验产物和源快照未回填、覆盖或重新解释为新版本结果。

## 1. 全 INT4 码、cell 边界和饱和端

`tests/test_phase3_joint.py:1956`：`test_distill_cell_reference_all_codes_saturation_and_no_mutation`

- 覆盖全部16个 signed INT4 codes，每码9个参考偏移，包含 ±0.5 边界、越界 ±0.51、cell内部以及 ±1000 外侧；per-output SW为16个严格正值，跨1e-4..1e2。
- 独立逐元素 Python scalar clamp oracle 检查原FP/SW投影到 code±0.49 再乘SW；code=-8向负侧、code=7向正侧无界，明确检查远饱和值没有被拉回有限cell边缘。
- 投影后的 round/clamp codes 与父码**逐元素精确相等**；实际 TrainableQuantLinear W4STE→BF16权重和BF16 linear前向与父格点**精确相等**。
- 错误reference矩阵形状、展平及同numel不同shape均抛ValueError；父packed/SW/shape和FP reference不变。
- 这里测试的是已验证父记录下的正尺度；不声称此函数自身独立拒绝任意非法SW。生产父包的SW shape/正finite检查仍由既有load_static/reload边界负责。

## 2. 真实 tiny 模型 master 改变但量化模型不变

`tests/test_phase3_joint.py:1992`：`test_distill_master_reference_tiny_initial_and_cold_export_match_parent`

- 使用真实16层tiny Llama、实际112个wrapper及完整冻结父包。参考矩阵包含明确非零cell残差，调用真实prepare_student(master_reference=...)，不调用训练或校准。
- 全部112张FP32 master矩阵确实不同于旧缺省初始化；旧缺省路径仍精确等于 `parent BF16 weight.float()`，没有改变旧行为。
- 仍恰好336个不同可学习Parameter：112 W +96 SA +112 SW +16 SP2；wrapper.weight正确alias新module.weight，输入全bits8，无R1。
- 父SW及全部activation scale不变；head/embed/norm等HP参数对象身份、值及冻结标志不变。此次只核验新增初始化，不重跑既有反向梯度测试，也不把requires_grad断言称为新梯度实测。
- 所有112张量化后的BF16权重与父一致；两个输入窗口的初始student logits与父**逐位相等**。
- 实际export_student生成112记录，全部changed_codes=0；实际load_static冷载导出包，其records和两个窗口logits仍与父**精确相等**。
- 明确禁止初始化/推理中进入校准与动态find_params；父模型、父records、reference矩阵不变，export不改变student状态。

## 3. 直接 D metadata 的 FP 参考变换

`tests/test_phase3_joint.py:2048`：`test_distill_reference_for_direct_d_selected_rows_columns_without_disk_mutation`

- reference_weights被替换为返回明确BF16矩阵的CPU factory，按真实helper的所有权习惯每次返回新构建的矩阵；不调用GPU参考重建。
- 实际读取小parent包metadata及相邻diagonal_candidates.json。对mode=d且candidate=local-d-指定down模块，selected channels=[0,2]、factor=2：仅对应up行除2、down列乘2。其它行列、gate和另一层矩阵全部精确不变。
- 诊断中另设一个不同层factor64作为干扰；返回变换记录只来自父metadata选定层，并完整携带该selected信息，不误用其它层或仅范围control。
- range-only候选及非d mode均不施加D，而且不读取不存在的diagonal诊断文件。
- parent包、参考checkpoint和诊断文件字节均不变，作为原始矩阵样本的fixture也不变。
- 精确边界：reference_for_parent会在其**新重建的返回矩阵**上原位施加D；本项不把它描述为对共享缓存矩阵的纯函数，也不声称从未就地修改其拥有的结果对象。已测无副作用指未选内容和磁盘输入不被污染。

## 源码与未验收范围

- 新分支只在提供reference_state时构建参考并写master_initialization.json；prepare_student缺省master_reference=None仍取原wrapper权重。
- 此初始化改变FP32 master位置，而非增加训练updates或校准；本轮不证明新的优化器轨迹或最终PPL更好。
- 只验证直接local-D父包；没有核验多级后处理继承链、生产B100参考重建、GPU数值/显存、长程resume、正式训练或完整validation。旧运行没有使用这个选项，不能追溯描述为采用本次实现。

## 证据

- 源码根：`/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint`。
- 已有HEAD：`24918316ed594848d4de797c356b120f2a4ee0f3`，不代表未提交文件内容；未新增hash。
- 运行前后实读distill.py mtime：`2026-09-14 05:52:15.937251557 +0800`。
- 原始输出：`runs/phase3/verifier/cpu-master-reference-20260914.log`、同名`.xml`。
- 临时产物：`runs/phase3/verifier/pytest-master-reference-20260914/`。
- 唯一warning为已有Transformers quantized-training API deprecation，无失败。

工作目录为源码根：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/dongpeiyan/projects/rotation-quant/worktrees/SpinQuant-phase3-joint \
/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python -m pytest -p no:cacheprovider \
tests/test_phase3_joint.py::test_distill_cell_reference_all_codes_saturation_and_no_mutation \
tests/test_phase3_joint.py::test_distill_master_reference_tiny_initial_and_cold_export_match_parent \
tests/test_phase3_joint.py::test_distill_reference_for_direct_d_selected_rows_columns_without_disk_mutation -q \
--basetemp=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/pytest-master-reference-20260914 \
--junitxml=/home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-master-reference-20260914.xml \
2>&1 | tee /home/dongpeiyan/projects/rotation-quant/runs/phase3/verifier/cpu-master-reference-20260914.log
```

本轮定向核验结束，等待下一处实际代码变更；不代替正式GPU实验或最终优胜包验收。
