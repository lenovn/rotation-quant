# WXD-rotation-quant copy 差异文件与独有分析归档

来源：`/home/dongpeiyan/projects/rotation-quant copy`。
只读比对对象：`/home/dongpeiyan/projects/rotation-quant`，包括它已存在的 `delete/` 归档。

用户本次明确授权：仅清理copy中的重复文件，将不同/独有结果放入当前项目新文件夹；不得反向清理当前项目。

用户随后要求接收文件夹增加 `WXD-` 前缀，实际接收目录为 `/home/dongpeiyan/projects/rotation-quant/WXD-imported-rotation-quant-copy-20260914/`。这里的“当前项目/主项目”均指不带copy的 `rotation-quant`，其既有未提交改动不作修改。

执行前最新全文件比对清单：1570个copy重复文件、71,127,301,942 bytes；974个未证明重复的文件、597,843,129 bytes，后者全部先保留至 `files/`，包括独有分析及不同源码/启动脚本/元数据。对应Git和旧实验副本也包含在重复清单内，当前项目及其已有归档中的保留文件不会删除。

## 存储方式

- `files/` 按copy原相对路径保存所有未证明重复的文件，包括独有分析结果、不同的源码/启动脚本，以及不同/独有的Git元数据；内容不改写，不覆盖当前项目同名文件。
- `file-actions.json` 逐项列出来源、处理方式、保留文件对应位置及删除前大小。重复以实际 `cmp -s` 全文件逐字节比较为依据，不新增hash，不把同名/同大小直接当重复。
- `execution.jsonl` 逐项记录已完成的移动或重复删除；`execution-summary.json` 汇总实际动作及分支空间。文件不存在前，不把计划当作已完成。
- 操作顺序为先把所有非重复文件复制至新目录、逐字节核对后移除copy源文件，再移除copy中已证实重复的文件。原项目既有文件只读。

## 独有分析入口

保留原层次：`files/runs/phase2/w16a8-downa16-joint-r-sa-r12-100step-s42/`，其中包含全层down激活稳定性与token关联、layer1 down权重误差、激活范围、通道稳定性等分析。配套独有脚本保存在 `files/repos/SpinQuant/`。

## 边界

本目录是差异文件归档，**不是独立可运行的完整checkout或完整Git仓库**。已去重依赖仍保留在当前项目或其delete归档中，位置见动作清单。原脚本中的历史绝对路径不改写，若将来重新运行，需另行正确指定输入路径；本次未运行模型、测试或实验。

少量 `.git` 差异文件也按原路径保存，只用于保留元数据，不宣称可单独执行Git命令。不运行Git gc、提交、推送、分支切换或任何全局去重。

最终核实完成前，执行状态以JSON记录为准。copy根目录计划保留为空目录，不删除当前项目任何既有文件。
