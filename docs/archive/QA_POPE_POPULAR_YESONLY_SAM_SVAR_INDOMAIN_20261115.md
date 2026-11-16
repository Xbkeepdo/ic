# POPE popular Yes-only：SAM 与 SVAR 域内训练

## 协议

- LLaVA-1.5-7B 与 Qwen3-VL-8B 分别在自身 POPE popular Yes-only 样本上训练。GT Yes／模型答 Yes 为 real (1)，GT No／模型答 Yes 为 hallucination (0)。
- 沿用特征提取时的图片级 8:2 划分；训练集用于拟合模型与选择 Real-F1 阈值，测试集完全隔离。SAM 为物体词前 Log-S risk 与首个答案前 F_log 拼接；SVAR 为物体词前第 5–18 层、448 维受控特征。
- 沿用 CLEVR 迁移实验相同的单隐藏层 SAM 与原生 SVAR MLP 配置，种子 43/44/45。三种子只改变初始化和训练顺序，不改变图片划分。报告均值 ± 总体标准差，单位 %。
- 训练前核对 NPZ、问题与基线记录的 key/顺序、标签、图片划分、维度及文件 SHA-256；每个检查点重新加载，在测试集重算 AUC，与训练记录的差异小于 2e-5。

## 域内测试结果

| 模型 | 方法 | 训练 Real/Hall | 测试 Real/Hall | AUC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Hall Recall | Accuracy | 预测 Real 比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA | SAM | 944/68 | 239/23 | 85.46 ± 1.21 | 98.37 ± 0.20 | 43.27 ± 4.63 | 95.13 ± 0.36 | 36.14 ± 8.94 | 30.43 ± 10.65 | 90.97 ± 0.65 | 94.40 ± 1.83 |
| LLaVA | SVAR | 944/68 | 239/23 | 63.73 ± 0.47 | 94.39 ± 0.03 | 13.94 ± 0.35 | 94.99 ± 0.30 | 0.00 ± 0.00 | 0.00 ± 0.00 | 90.46 ± 0.54 | 99.24 ± 0.54 |
| Qwen3 | SAM | 1005/70 | 249/24 | 90.96 ± 0.62 | 99.09 ± 0.09 | 48.67 ± 3.50 | 95.13 ± 0.35 | 37.34 ± 2.83 | 30.56 ± 1.96 | 90.96 ± 0.62 | 94.38 ± 0.46 |
| Qwen3 | SVAR | 1005/70 | 249/24 | 69.91 ± 0.17 | 95.91 ± 0.08 | 16.45 ± 0.38 | 95.40 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 91.21 ± 0.00 | 100.00 ± 0.00 |

SAM 的 AUC 比 SVAR 高 21.73（LLaVA）和 21.05（Qwen3）个百分点，Hall AUPR 分别高 29.33 和 32.22 个百分点。两个测试集的 hallucination 仅有 23/24 个，且来源训练集同样类别不平衡。SVAR 的 AUC 高于随机水平，但来源训练集选择的阈值在测试集几乎全部预测 real，所以 Hall Recall=0；仅看 Accuracy 会掩盖这一点。SAM 也只检出约 30% 的幻觉。

## 与 CLEVR → POPE 迁移的同一测试集对照

| 模型 | 方法 | CLEVR 训练 → POPE test AUC | POPE 训练 → POPE test AUC |
|---|---|---:|---:|
| LLaVA | SAM | 55.60 ± 2.81 | 85.46 ± 1.21 |
| LLaVA | SVAR | 36.33 ± 0.67 | 63.73 ± 0.47 |
| Qwen3 | SAM | 64.42 ± 0.61 | 90.96 ± 0.62 |
| Qwen3 | SVAR | 50.24 ± 0.50 | 69.91 ± 0.17 |

这些对照使用同一 POPE test 样本，区别在于训练数据来源；两者的阈值均只在各自来源训练集选择。

## 产物

服务器：`/root/rivermind-data/ic-ac/outputs/qa_benchmarks/pope_popular_yesonly_sam_svar_indomain_82/`。每个模型目录包含 `complete_results.md`、`per_seed_results.csv`、`per_seed_results.json`、`cohort_audit.json`，以及 SAM/SVAR 各三个种子的检查点。执行日志在 `_status/`，运行脚本是 `scripts/run_pope_popular_yesonly_sam_svar_indomain.py`。
