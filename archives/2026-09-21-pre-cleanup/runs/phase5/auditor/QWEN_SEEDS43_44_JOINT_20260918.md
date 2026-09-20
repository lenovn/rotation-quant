# Qwen seeds43/44 Joint100 与匹配初始 INT8 审计（2026-09-18）

结论：限定 PASS。两个 seed 均完成真实100步 Joint100，匹配 INT8 与同 seed 的初始 SP2 共享相同冻结模型内容，差别限定在28层 down 激活格式及其匹配校准尺度。当前不据此声称后处理或 QAT 完成。

| seed | Joint100 validation PPL | NLL | targets |
|---|---:|---:|---:|
| 43 | 16.04792947249727 | 2.7755798364253557 | 262208 |
| 44 | 16.316429520957282 | 2.7921725462471594 | 262208 |

原始 validation 分段按各段 log(PPL)×targets 重算精确匹配聚合 NLL/PPL，summary 官方模型名 `Qwen/Qwen3-1.7B`、B100-initial-SP2 及数字一致。

初始化：直接读取保存的 initial metadata、data 和 checkpoint-0000，各 seed 所有参数张量逐项位级相等，分别为 421, 421 个。此前 seed_initialization_43_44.json 仅作线索，本次实际重新核对其身份/索引并以对应 seed 的 torch.randperm(train_windows−8) 重建32索引；未复tokenize。seed42/43/44 两两29个R张量全部不同；每个43/44最终Joint100的29个R也都不同于自身初始化。

训练：两份 training.jsonl 均恰为连续1..100，每步8微批、16384输入token /16376预测targets，实际窗索引0..799，累计1638400输入 /1637600预测 token visits。每步R/非downSA/SW分别29/168/196张量有梯度和实际更新；SP2组无梯度/更新，Joint更新后独立校准。settings实际seed、route B、global8、初始路径、completed状态一致，无failure.json。

匹配：每个 seed 的196个W4记录（包含SW和其他记录字段）逐项位级一致；168个non-down SA相同；154个高精度张量（含56个Q/K norm）位级一致，config相同。原始包实际激活格式为168 INT8+28 SP2，对照为196 INT8；R已融合进相同W/高精度内容，不在冻结包中另存独立R张量。inactive lm_head量化器占位NaN按字节比较，未误判为格式差异。INT8 parent均精确指向本seed Joint100 checkpoint-0100/static_w4a8.pt。

校准：每seed28层，同32训练窗×128=4096tokens、每层512 sampled rows；SP2、INT8与capture记录的full_absmax及historical字段精确相同。每层各50候选=33档log2[-16,1]粗搜+17档相邻区间细搜，全部候选位置、最小output MSE选中项及最终包尺度均核实；INT8记录保存的历史SP2选中项与本seed SP2记录完全相同。父包196个quantizer状态在INT8校准前后逐张量位级不变，未用C4。captured数组未保存，因此不声称捕获值逐元素相等。

环境：Qwen均使用 `runs/phase5/env/bin/python`、Transformers4.51.3；Llama正式环境为旧rotation-quant-p0/4.44.2，不能混写。四份核心run/common/architecture/quantization保存源码与已审Qwen seed42逐字节一致。

本次CPU mmap逐张量比较，无全optimizer读取、无大包无差别深拷贝，无GPU/PPL/架构suite复跑；未操作在途round进程、源码或summary。完整原始路径和精确尺度见 [机器报告](QWEN_SEEDS43_44_JOINT_20260918.json)。
