# Phase5 完整方法与配对结果

由 summary.csv 生成；空缺表示尚无已完成结果。PPL 仅为显示保留六位，原始精度与证据见 summary.csv。
BF16 不按 seed 重复；B100 格式对照和中间后处理不进入本表。QAT 默认 WikiText-2 train only。

## meta-llama/Llama-3.2-1B-Instruct

| 方法 | seed | WikiText val PPL | WikiText test PPL | 固定 C4 PPL | test/C4 包路径 |
|---|---|---:|---:|---:|---|
| BF16 | — | [13.634658](/home/dongpeiyan/projects/rotation-quant/runs/phase3/auditor/final-best-20260914/bf16.result.json) | [13.162650](/home/dongpeiyan/projects/rotation-quant/runs/phase3/wiki2-test-bf16-fixed-20260917a/result.json) | [21.843398](/home/dongpeiyan/projects/rotation-quant/runs/phase3/c4-bf16-fixed-20260915a/result.json) | 相同 |
| SP2-PTQ | 42 | [16.112577](/home/dongpeiyan/projects/rotation-quant/runs/phase3/seq-b100-sp2-refine-down-20260914a/validation.json) | [15.560179](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-ptq-test-s42/result.json) | [27.439616](/home/dongpeiyan/projects/rotation-quant/runs/phase3/c4-ptq-parent-fixed-20260918a/result.json) | 相同 |
| Uniform-PTQ | 42 | [41.547071](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-down-readapt-s42/validation.json) | [38.470287](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-ptq-test-s42/result.json) | [81.133168](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-ptq-c4-s42/result.json) | 相同 |
| SP2-QAT400 | 42 | [14.581655](/home/dongpeiyan/projects/rotation-quant/runs/phase3/distill-b100-refined-ref-adam1e5-400-20260914a/checkpoint-0400/validation.json) | [14.154416](/home/dongpeiyan/projects/rotation-quant/runs/phase3/wiki2-test-best-fixed-20260917a/result.json) | [26.984781](/home/dongpeiyan/projects/rotation-quant/runs/phase3/c4-best-fixed-20260915a/result.json) | 相同 |
| SP2-QAT400 | 43 | [14.669073](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-qat400-s43/checkpoint-0400/validation.json) | [14.204722](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-qat400-test-s43/result.json) | [26.639323](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-qat400-c4-s43/result.json) | 相同 |
| SP2-QAT400 | 44 | [14.604526](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-qat400-s44/checkpoint-0400/validation.json) | [14.197525](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-qat400-test-s44/result.json) | [26.515203](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/sp2-qat400-c4-s44/result.json) | 相同 |
| Uniform-QAT400 | 42 | [18.104945](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-s42/checkpoint-0400/validation.json) | [17.518988](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-test-s42/result.json) | [38.777595](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-c4-s42/result.json) | 相同 |
| Uniform-QAT400 | 43 | [18.143937](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-s43/checkpoint-0400/validation.json) | [17.575531](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-test-s43/result.json) | [39.071882](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-c4-s43/result.json) | 相同 |
| Uniform-QAT400 | 44 | [19.286804](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-s44/checkpoint-0400/validation.json) | [18.734456](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-test-s44/result.json) | [45.246466](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/uniform-qat400-c4-s44/result.json) | 相同 |
| Initial-SP2-QAT400 | 42 | [14.756133](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/initial-sp2-qat400-s42-r25/checkpoint-0400/validation.json) | [14.320254](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/initial-sp2-qat400-test-s42/result.json) | [26.730584](/home/dongpeiyan/projects/rotation-quant/runs/phase5/llama32-1b/initial-sp2-qat400-c4-s42/result.json) | 相同 |

## Qwen/Qwen3-1.7B

| 方法 | seed | WikiText val PPL | WikiText test PPL | 固定 C4 PPL | test/C4 包路径 |
|---|---|---:|---:|---:|---|
| BF16 | — | [17.714912](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/bf16-validation-s42/result.json) | [16.715764](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/bf16-test-s42/result.json) | [23.136144](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/bf16-c4-s42/result.json) | 相同 |
| SP2-PTQ | 42 | [14.935424](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-down-readapt-s42/validation.json) | [14.302163](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-ptq-test-s42/result.json) | [24.903333](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-ptq-c4-s42/result.json) | 相同 |
| Uniform-PTQ | 42 | [17.491761](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-down-readapt-s42/validation.json) | [16.785376](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-ptq-test-s42/result.json) | [29.322131](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-ptq-c4-s42/result.json) | 相同 |
| SP2-QAT400 | 42 | [14.813941](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-s42/checkpoint-0400/validation.json) | [14.151216](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-test-s42/result.json) | [24.247034](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-c4-s42/result.json) | 相同 |
| SP2-QAT400 | 43 | [14.807327](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-s43/checkpoint-0400/validation.json) | [14.150948](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-test-s43/result.json) | [24.042025](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-c4-s43/result.json) | 相同 |
| SP2-QAT400 | 44 | [14.803868](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-s44/checkpoint-0400/validation.json) | [14.105052](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-test-s44/result.json) | [24.170475](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/sp2-qat400-c4-s44/result.json) | 相同 |
| Uniform-QAT400 | 42 | [15.715716](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-s42/checkpoint-0400/validation.json) | [15.055185](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-test-s42/result.json) | [27.104781](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-c4-s42/result.json) | 相同 |
| Uniform-QAT400 | 43 | [15.653652](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-s43/checkpoint-0400/validation.json) | [15.013446](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-test-s43/result.json) | [27.459111](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-c4-s43/result.json) | 相同 |
| Uniform-QAT400 | 44 | [15.752338](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-s44/checkpoint-0400/validation.json) | [15.140922](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-test-s44/result.json) | [27.651998](/home/dongpeiyan/projects/rotation-quant/runs/phase5/qwen3-1p7b/uniform-qat400-c4-s44/result.json) | 相同 |

## 相对本模型原始 BF16 的指标

ΔNLL = 当前NLL − 同模型同数据BF16 NLL；PPL比值 = 当前PPL / BF16 PPL。不同tokenizer的绝对PPL不直接排名。

| 模型 | 方法 | seed | 数据 | PPL | NLL | ΔNLL | PPL比值 | targets |
|---|---|---|---|---:|---:|---:|---:|---:|
| meta-llama/Llama-3.2-1B-Instruct | BF16 | — | WikiText validation | 13.634658 | 2.612614913 | 0.000000000 | 1.000000 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | BF16 | — | WikiText test | 13.162650 | 2.577383298 | 0.000000000 | 1.000000 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | BF16 | — | C4 fixed | 21.843398 | 3.083898708 | 0.000000000 | 1.000000 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-PTQ | 42 | WikiText validation | 16.112577 | 2.779600151 | 0.166985238 | 1.181737 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-PTQ | 42 | WikiText test | 15.560179 | 2.744714991 | 0.167331693 | 1.182146 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-PTQ | 42 | C4 fixed | 27.439616 | 3.311987824 | 0.228089116 | 1.256197 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-PTQ | 42 | WikiText validation | 41.547071 | 3.726827018 | 1.114212105 | 3.047166 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-PTQ | 42 | WikiText test | 38.470287 | 3.649886173 | 1.072502875 | 2.922685 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-PTQ | 42 | C4 fixed | 81.133168 | 4.396091851 | 1.312193143 | 3.714311 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 42 | WikiText validation | 14.581655 | 2.679764210 | 0.067149296 | 1.069455 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 42 | WikiText test | 14.154416 | 2.650026689 | 0.072643391 | 1.075347 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 42 | C4 fixed | 26.984781 | 3.295273022 | 0.211374314 | 1.235375 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 43 | WikiText validation | 14.669073 | 2.685741404 | 0.073126490 | 1.075867 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 43 | WikiText test | 14.204722 | 2.653574427 | 0.076191129 | 1.079169 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 43 | C4 fixed | 26.639323 | 3.282388434 | 0.198489726 | 1.219559 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 44 | WikiText validation | 14.604526 | 2.681331480 | 0.068716566 | 1.071133 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 44 | WikiText test | 14.197525 | 2.653067676 | 0.075684378 | 1.078622 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | SP2-QAT400 | 44 | C4 fixed | 26.515203 | 3.277718285 | 0.193819578 | 1.213877 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 42 | WikiText validation | 18.104945 | 2.896185092 | 0.283570179 | 1.327862 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 42 | WikiText test | 17.518988 | 2.863285346 | 0.285902047 | 1.330962 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 42 | C4 fixed | 38.777595 | 3.657842622 | 0.573943914 | 1.775255 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 43 | WikiText validation | 18.143937 | 2.898336445 | 0.285721531 | 1.330722 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 43 | WikiText test | 17.575531 | 2.866507663 | 0.289124365 | 1.335258 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 43 | C4 fixed | 39.071882 | 3.665403085 | 0.581504377 | 1.788727 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 44 | WikiText validation | 19.286804 | 2.959421119 | 0.346806205 | 1.414543 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 44 | WikiText test | 18.734456 | 2.930364380 | 0.352981082 | 1.423304 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | Uniform-QAT400 | 44 | C4 fixed | 45.246466 | 3.812124563 | 0.728225856 | 2.071402 | 2096128 |
| meta-llama/Llama-3.2-1B-Instruct | Initial-SP2-QAT400 | 42 | WikiText validation | 14.756133 | 2.691658797 | 0.079043884 | 1.082252 | 252728 |
| meta-llama/Llama-3.2-1B-Instruct | Initial-SP2-QAT400 | 42 | WikiText test | 14.320254 | 2.661674878 | 0.084291580 | 1.087946 | 288934 |
| meta-llama/Llama-3.2-1B-Instruct | Initial-SP2-QAT400 | 42 | C4 fixed | 26.730584 | 3.285808388 | 0.201909680 | 1.223737 | 2096128 |
| Qwen/Qwen3-1.7B | BF16 | — | WikiText validation | 17.714912 | 2.874406786 | 0.000000000 | 1.000000 | 262208 |
| Qwen/Qwen3-1.7B | BF16 | — | WikiText test | 16.715764 | 2.816352247 | 0.000000000 | 1.000000 | 298931 |
| Qwen/Qwen3-1.7B | BF16 | — | C4 fixed | 23.136144 | 3.141396080 | 0.000000000 | 1.000000 | 2096128 |
| Qwen/Qwen3-1.7B | SP2-PTQ | 42 | WikiText validation | 14.935424 | 2.703735847 | -0.170670939 | 0.843099 | 262208 |
| Qwen/Qwen3-1.7B | SP2-PTQ | 42 | WikiText test | 14.302163 | 2.660410812 | -0.155941435 | 0.855609 | 298931 |
| Qwen/Qwen3-1.7B | SP2-PTQ | 42 | C4 fixed | 24.903333 | 3.215001641 | 0.073605561 | 1.076382 | 2096128 |
| Qwen/Qwen3-1.7B | Uniform-PTQ | 42 | WikiText validation | 17.491761 | 2.861729961 | -0.012676825 | 0.987403 | 262208 |
| Qwen/Qwen3-1.7B | Uniform-PTQ | 42 | WikiText test | 16.785376 | 2.820508046 | 0.004155799 | 1.004164 | 298931 |
| Qwen/Qwen3-1.7B | Uniform-PTQ | 42 | C4 fixed | 29.322131 | 3.378342572 | 0.236946492 | 1.267373 | 2096128 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 42 | WikiText validation | 14.813941 | 2.695568712 | -0.178838075 | 0.836241 | 262208 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 42 | WikiText test | 14.151216 | 2.649800568 | -0.166551679 | 0.846579 | 298931 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 42 | C4 fixed | 24.247034 | 3.188294319 | 0.046898238 | 1.048015 | 2096128 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 43 | WikiText validation | 14.807327 | 2.695122117 | -0.179284669 | 0.835868 | 262208 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 43 | WikiText test | 14.150948 | 2.649781601 | -0.166570646 | 0.846563 | 298931 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 43 | C4 fixed | 24.042025 | 3.179803346 | 0.038407266 | 1.039154 | 2096128 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 44 | WikiText validation | 14.803868 | 2.694888489 | -0.179518297 | 0.835673 | 262208 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 44 | WikiText test | 14.105052 | 2.646533018 | -0.169819229 | 0.843817 | 298931 |
| Qwen/Qwen3-1.7B | SP2-QAT400 | 44 | C4 fixed | 24.170475 | 3.185131846 | 0.043735766 | 1.044706 | 2096128 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 42 | WikiText validation | 15.715716 | 2.754661234 | -0.119745552 | 0.887146 | 262208 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 42 | WikiText test | 15.055185 | 2.711722449 | -0.104629798 | 0.900658 | 298931 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 42 | C4 fixed | 27.104781 | 3.299710129 | 0.158314048 | 1.171534 | 2096128 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 43 | WikiText validation | 15.653652 | 2.750704270 | -0.123702516 | 0.883643 | 262208 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 43 | WikiText test | 15.013446 | 2.708946190 | -0.107406057 | 0.898161 | 298931 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 43 | C4 fixed | 27.459111 | 3.312698012 | 0.171301932 | 1.186849 | 2096128 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 44 | WikiText validation | 15.752338 | 2.756988820 | -0.117417966 | 0.889213 | 262208 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 44 | WikiText test | 15.140922 | 2.717401123 | -0.098951124 | 0.905787 | 298931 |
| Qwen/Qwen3-1.7B | Uniform-QAT400 | 44 | C4 fixed | 27.651998 | 3.319697977 | 0.178301897 | 1.195186 | 2096128 |

## 逐 seed 配对 NLL

| 模型 | 数据 | seed | SP2 NLL | Uniform NLL | SP2 − Uniform |
|---|---|---|---:|---:|---:|
| meta-llama/Llama-3.2-1B-Instruct | WikiText validation | 42 | 2.679764210 | 2.896185092 | -0.216420883 |
| meta-llama/Llama-3.2-1B-Instruct | WikiText validation | 43 | 2.685741404 | 2.898336445 | -0.212595041 |
| meta-llama/Llama-3.2-1B-Instruct | WikiText validation | 44 | 2.681331480 | 2.959421119 | -0.278089639 |
| meta-llama/Llama-3.2-1B-Instruct | WikiText test | 42 | 2.650026689 | 2.863285346 | -0.213258657 |
| meta-llama/Llama-3.2-1B-Instruct | WikiText test | 43 | 2.653574427 | 2.866507663 | -0.212933236 |
| meta-llama/Llama-3.2-1B-Instruct | WikiText test | 44 | 2.653067676 | 2.930364380 | -0.277296704 |
| meta-llama/Llama-3.2-1B-Instruct | C4 fixed | 42 | 3.295273022 | 3.657842622 | -0.362569600 |
| meta-llama/Llama-3.2-1B-Instruct | C4 fixed | 43 | 3.282388434 | 3.665403085 | -0.383014651 |
| meta-llama/Llama-3.2-1B-Instruct | C4 fixed | 44 | 3.277718285 | 3.812124563 | -0.534406278 |
| Qwen/Qwen3-1.7B | WikiText validation | 42 | 2.695568712 | 2.754661234 | -0.059092523 |
| Qwen/Qwen3-1.7B | WikiText validation | 43 | 2.695122117 | 2.750704270 | -0.055582153 |
| Qwen/Qwen3-1.7B | WikiText validation | 44 | 2.694888489 | 2.756988820 | -0.062100331 |
| Qwen/Qwen3-1.7B | WikiText test | 42 | 2.649800568 | 2.711722449 | -0.061921881 |
| Qwen/Qwen3-1.7B | WikiText test | 43 | 2.649781601 | 2.708946190 | -0.059164589 |
| Qwen/Qwen3-1.7B | WikiText test | 44 | 2.646533018 | 2.717401123 | -0.070868105 |
| Qwen/Qwen3-1.7B | C4 fixed | 42 | 3.188294319 | 3.299710129 | -0.111415810 |
| Qwen/Qwen3-1.7B | C4 fixed | 43 | 3.179803346 | 3.312698012 | -0.132894666 |
| Qwen/Qwen3-1.7B | C4 fixed | 44 | 3.185131846 | 3.319697977 | -0.134566130 |

## 三 seed 统计

仅当 seed42/43/44 全部完成且每项数据的 targets 一致时计算。标准差使用样本标准差（ddof=1）。
配对 ΔNLL = SP2-QAT400 − Uniform-QAT400，负值有利于 SP2；不会使用最佳 seed 替代统计。

| 模型 | 数据 | SP2 PPL 均值±标准差 | Uniform PPL 均值±标准差 | 配对 ΔNLL 均值±标准差 | 状态 |
|---|---|---:|---:|---:|---|
| meta-llama/Llama-3.2-1B-Instruct | WikiText validation | 14.618418 ± 0.045335 | 18.511895 ± 0.671374 | -0.235702 ± 0.036759 | 完成；逐 seed 见上表及原始记录 |
| meta-llama/Llama-3.2-1B-Instruct | WikiText test | 14.185554 ± 0.027205 | 17.942992 ± 0.686011 | -0.234496 ± 0.037067 | 完成；逐 seed 见上表及原始记录 |
| meta-llama/Llama-3.2-1B-Instruct | C4 fixed | 26.713102 ± 0.243327 | 41.031981 ± 3.652816 | -0.426664 ± 0.093866 | 完成；逐 seed 见上表及原始记录 |
| Qwen/Qwen3-1.7B | WikiText validation | 14.808379 ± 0.005118 | 15.707236 ± 0.049887 | -0.058925 ± 0.003262 | 完成；逐 seed 见上表及原始记录 |
| Qwen/Qwen3-1.7B | WikiText test | 14.135739 ± 0.026576 | 15.069851 ± 0.064991 | -0.063985 ± 0.006118 | 完成；逐 seed 见上表及原始记录 |
| Qwen/Qwen3-1.7B | C4 fixed | 24.153178 ± 0.103593 | 27.405296 ± 0.277549 | -0.126292 ± 0.012910 | 完成；逐 seed 见上表及原始记录 |

本汇总只检查记录字段；架构正确性、数据一致性和同包验收证据仍以 verifier/、auditor/ 及原始 run 为准。
