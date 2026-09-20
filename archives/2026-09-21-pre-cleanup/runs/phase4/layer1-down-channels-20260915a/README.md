# Layer1 down 输出通道分布

只处理 model.layers.1.mlp.down_proj（源码layer1，即第二个block），矩阵[2048输出,8192输入]。使用原Phase3最终PTQ父包的packed INT4和已学习SW，及B100同R原BF16经norm fusion/rotated_weight的未量化参考。未重新量化/重估SW，无模型forward、校准、训练、PPL或模型导出。

[全部输出通道图](channels.png)：左上为每行|Wref|的max/P99，右上原SW，左下sqrt(mean((Wq-Wref)^2)/mean(Wref^2))，右下原整数码为0的比例。横轴均为输出通道编号0..2047；每行单独统计8192个权重，未混合整张矩阵尺度。

[单通道分布图](row_histograms.png)：参考权重除该行原SW（蓝色密度）；原整数码q按单位宽度画概率质量（橙色）。选取中位相对误差行1133、最大max/P99行1649、最大相对误差行867；不是随机样本或任务敏感性排名。三个直方图各覆盖完整8192权重，无图外省略。

整体相对权重RMSE=0.26675808；输出行relativeRMSE的min/median/P99/max=[0.1604890525341034, 0.22462084889411926, 0.5382552742958069, 0.6712092757225037]；max/P99的min/median/P99/max=[1.3016760349273682, 1.5355305671691895, 1.9103784561157227, 2.0888888835906982]。超出原[-8SW,7SW]范围的元素1184/16777216。零码比例中位数0.21752930。

误差最大的row867并非尾部最大的row1649。当前q经过输入感知后处理，weight RMSE包含码调整的结果；不能把高RMSE或零码直接解释为PPL主导因素。图用于形成下一步假设，未新增因果恢复结论。

GPU5/PID1224951，tmux socket rotation-quant-phase4，session phase4-layer1-down-channels-20260915a。真实命令/资源/日志在上级同名.launch.json/.log。只对目标矩阵调用rotated_weight；没有重扫112矩阵或重新运行Atlas。源码snapshot与设置见本目录，channels.json、row_histograms.json支持复绘，PNG和SVG均保留。独立verifier PASS：../verifier/layer1-channel-review.md；主执行已目视核对两图。无环境安装、旧文件删除或模型包写入。
