# POPE popular Yes-only：域内训练与 CLEVR＋AMBER 迁移（种子仅控制初始化）

## 对照协议

LLaVA-1.5-7B、Qwen3-VL-8B 分别训练 SAM 和 SVAR。训练来源有两组：① POPE popular Yes-only train（域内）；② CLEVR9K Yes-only train＋AMBER existence Yes-only train（迁移）。各数据集沿用固定的图片级 8:2 划分，两组模型评估**完全相同的 POPE popular Yes-only test**。LLaVA test 为 262 题（239 real／23 hallucination），Qwen3 test 为 273 题（249／24）。模型答 Yes 且 GT Yes 标 real (1)，模型答 Yes 且 GT No 标 hallucination (0)。

初始化种子为 43、44、45；训练随机种子始终固定为 43。模型构造前按初始化种子设置随机状态，构造后重新设置固定训练随机状态；因此同一训练来源、方法的 minibatch 顺序和 SAM dropout 序列跨三个种子相同。数据划分、特征和样本顺序也不变。SAM 保持 128 宽单隐藏层、训练集 z-score 和训练损失检查点；SVAR 保持原生 248 宽单隐藏层。两者均只在各自来源训练集选择 Real-F1 阈值。迁移方案在 POPE 上不重训、不重新标准化、不调阈值。

| 模型 | POPE 域内 train Real/Hall | 混合来源 train Real/Hall | POPE test Real/Hall |
|---|---:|---:|---:|
| LLaVA | 944/68 | 3009/3203 | 239/23 |
| Qwen3 | 1005/70 | 3494/436 | 249/24 |

## 同一 POPE test 的完整结果

三种子均值 ± 总体标准差，单位 %。Real AUPR 与 Hall AUPR 分别把对应类别作为正类。

| 模型 | 方法 | 训练来源 | AUC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Hall Recall | Accuracy | 预测 Real 比例 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA | SAM | POPE 域内 | 86.36 ± 1.13 | 98.51 ± 0.15 | 42.11 ± 2.72 | 95.72 ± 0.56 | 36.72 ± 12.39 | 27.54 ± 11.41 | 91.98 ± 1.08 | 95.93 ± 0.95 |
| LLaVA | SAM | CLEVR＋AMBER 迁移 | 70.20 ± 1.80 | 96.24 ± 0.37 | 18.06 ± 1.74 | 0.00 ± 0.00 | 16.14 ± 0.00 | 100.00 ± 0.00 | 8.78 ± 0.00 | 0.00 ± 0.00 |
| LLaVA | SVAR | POPE 域内 | 63.65 ± 0.14 | 94.38 ± 0.02 | 14.02 ± 0.02 | 95.20 ± 0.30 | 0.00 ± 0.00 | 0.00 ± 0.00 | 90.84 ± 0.54 | 99.62 ± 0.54 |
| LLaVA | SVAR | CLEVR＋AMBER 迁移 | 49.89 ± 0.52 | 91.19 ± 0.07 | 8.75 ± 0.16 | 0.00 ± 0.00 | 16.14 ± 0.00 | 100.00 ± 0.00 | 8.78 ± 0.00 | 0.00 ± 0.00 |
| Qwen3 | SAM | POPE 域内 | 90.47 ± 0.76 | 99.02 ± 0.11 | 48.22 ± 3.63 | 95.11 ± 0.24 | 40.17 ± 4.57 | 34.72 ± 5.20 | 90.96 ± 0.46 | 93.65 ± 0.62 |
| Qwen3 | SAM | CLEVR＋AMBER 迁移 | 68.06 ± 2.55 | 95.41 ± 0.28 | 17.48 ± 2.85 | 0.00 ± 0.00 | 16.16 ± 0.00 | 100.00 ± 0.00 | 8.79 ± 0.00 | 0.00 ± 0.00 |
| Qwen3 | SVAR | POPE 域内 | 70.10 ± 0.11 | 95.83 ± 0.01 | 16.62 ± 0.29 | 95.40 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 91.21 ± 0.00 | 100.00 ± 0.00 |
| Qwen3 | SVAR | CLEVR＋AMBER 迁移 | 43.00 ± 0.93 | 91.53 ± 0.17 | 7.41 ± 0.15 | 0.00 ± 0.00 | 16.16 ± 0.00 | 100.00 ± 0.00 | 8.79 ± 0.00 | 0.00 ± 0.00 |

SAM 的 AUC 从 POPE 域内训练转为混合来源迁移后，LLaVA 下降 16.16、Qwen3 下降 22.41 个百分点；SVAR 分别下降 13.76、27.10 个百分点。SAM 的 Hall AUPR 分别由 42.11→18.06、48.22→17.48。迁移方案在两模型、两方法的 POPE test 上全部预测 hallucination；所以 Hall Recall=100% 与 Real F1=0%、Accuracy≈8.8% 同时出现，不能把幻觉召回单独解读为良好性能。

AMBER existence 的 Yes-only 样本全是 hallucination，使混合来源中的数据集与标签相关。模型可能利用 CLEVR 合成图与 AMBER 自然图像之间的域差异；本实验观察到的 POPE 阈值崩塌与这一解释一致，但不能仅凭结果证明内部机制。POPE test 的 Hall 样本只有 23/24 个，Hall Recall 的波动也应结合该样本量解读。

## 产物与验证

服务器：`/root/rivermind-data/ic-ac/outputs/qa_benchmarks/pope_yesonly_domain_vs_clevr_amber_init_only_82/`。每个模型目录含 `complete_results.md`、`per_seed_results.csv`、`per_seed_results.json`、`cohort_audit.json`；两种训练来源各自保存 SAM/SVAR 三种子的检查点。日志在 `_status/`。代码为 `scripts/run_pope_init_only_domain_vs_transfer.py`，可选的初始化种子和固定训练种子分别传给 SAM/SVAR 训练器。

逐种子 CSV 核验：每模型 12 行（2 训练来源×2 方法×3 种子），初始化种子恰为 43/44/45，训练随机种子均为 43，全部测试队列的 key/标签哈希相同。每个来源检查点重新加载并重算来源 test AUC，与训练记录偏差小于 2e-5。此前种子同时改变训练随机过程的结果保留在[旧报告](QA_CLEVR_AMBER_YESONLY_MIXED_TO_POPE_20261115.md)，本报告用于回答初始化单独变化的对照。
