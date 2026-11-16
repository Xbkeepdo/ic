# CLEVR Yes-only 检测器迁移到 POPE popular 与 AMBER existence

## 设置

- 模型分别训练：LLaVA-1.5-7B、Qwen3-VL-8B；种子 43、44、45。来源为 CLEVR9K 模型回答 Yes 的样本，GT Yes=real (1)，GT No=hallucination (0)。沿用原图片级 8:2 划分，仅在 CLEVR train 训练，CLEVR test 作为原域核对。
- SAM 输入为 `concat(risk@question_object_pre_token, F_log@prompt_last_token)`。risk 使用 K=32、tau=0.03、EMD，语义 cost 与两端点惩罚系数均为 1；语义概率为 final-norm 后的词表 softmax。LLaVA 64 维，Qwen3 72 维。单隐藏层 128、ReLU、dropout 0.3，来源训练集 z-score、Adam、最小训练损失检查点、来源训练集 Real-F1 阈值。
- SVAR 输入为物体词前第 5–18 层逐头视觉注意力比，448 维。原生单隐藏层 248 的 SVAR 检测器，按现有 strict-8:2 配置训练并在来源训练集选阈值。旧 CLEVR 与新目标 SVAR 产物均含相同的逐层矩阵和层/头元数据；检查了实际位置与维度。
- 两种方法的来源模型、SAM 标准化参数及来源阈值直接用于 POPE/AMBER。目标未参与训练、标准化、阈值选择或模型选择。`all` 是目标全部 Yes-only 样本，`test` 是目标原定图片级 test 子集；目标图像与 CLEVR 互不重合。
- 表内为三种子均值 ± 总体标准差，单位 %。AUPR 的 Real/Hall 正类分别计算。阈值指标以来源训练集阈值计算。

## 来源 CLEVR test

| 模型 | 方法 | N (Real/Hall) | AUC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Hall Recall | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA | SAM | 1464 (725/739) | 66.01 ± 0.15 | 66.49 ± 0.27 | 62.76 ± 0.14 | 63.25 ± 0.54 | 58.39 ± 0.93 | 54.26 ± 1.09 | 60.97 ± 0.71 |
| LLaVA | SVAR | 1464 (725/739) | 69.44 ± 0.07 | 69.04 ± 0.31 | 68.00 ± 0.07 | 67.87 ± 0.18 | 46.57 ± 1.96 | 34.69 ± 2.11 | 59.88 ± 0.69 |
| Qwen3 | SAM | 860 (834/26) | 90.91 ± 0.03 | 99.61 ± 0.01 | 40.96 ± 0.98 | 98.53 ± 0.10 | 38.36 ± 1.89 | 29.49 ± 1.81 | 97.13 ± 0.20 |
| Qwen3 | SVAR | 860 (834/26) | 73.26 ± 0.38 | 98.78 ± 0.05 | 12.13 ± 0.89 | 98.43 ± 0.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | 96.90 ± 0.05 |

CLEVR 来源训练集：LLaVA 3,009 real／2,778 hall；Qwen3 3,494 real／95 hall。Qwen3 来源明显不平衡。

## 迁移到 POPE popular

| 模型 | 方法 | 目标范围 | N (Real/Hall) | AUC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Hall Recall | Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA | SAM | all | 1274 (1183/91) | 59.28 ± 3.00 | 94.53 ± 0.56 | 10.76 ± 1.86 | 88.77 ± 2.60 | 14.03 ± 1.79 | 22.71 ± 6.30 | 80.25 ± 4.15 |
| LLaVA | SVAR | all | 1274 (1183/91) | 42.18 ± 0.40 | 91.67 ± 0.12 | 5.82 ± 0.04 | 96.30 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 92.86 ± 0.00 |
| LLaVA | SAM | test | 262 (239/23) | 55.60 ± 2.81 | 92.68 ± 0.62 | 10.47 ± 0.62 | 85.46 ± 2.27 | 12.64 ± 4.03 | 21.74 ± 9.39 | 75.19 ± 3.25 |
| LLaVA | SVAR | test | 262 (239/23) | 36.33 ± 0.67 | 87.87 ± 0.09 | 6.76 ± 0.10 | 95.41 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 91.22 ± 0.00 |
| Qwen3 | SAM | all | 1348 (1254/94) | 67.12 ± 2.49 | 96.05 ± 0.40 | 11.92 ± 0.68 | 94.57 ± 1.13 | 9.23 ± 3.90 | 8.16 ± 4.28 | 89.79 ± 2.01 |
| Qwen3 | SVAR | all | 1348 (1254/94) | 51.48 ± 0.90 | 93.43 ± 0.19 | 7.47 ± 0.10 | 96.39 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 93.03 ± 0.00 |
| Qwen3 | SAM | test | 273 (249/24) | 64.42 ± 0.61 | 94.33 ± 0.30 | 12.75 ± 0.21 | 91.71 ± 1.80 | 7.68 ± 5.49 | 8.33 ± 5.89 | 84.86 ± 3.00 |
| Qwen3 | SVAR | test | 273 (249/24) | 50.24 ± 0.50 | 90.94 ± 0.14 | 9.89 ± 0.33 | 95.40 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 91.21 ± 0.00 |

POPE all 的 hall 占比约 7%。SVAR 来源阈值迁移后把两模型所有 POPE Yes 样本判成 real，所以 Accuracy 看起来较高，但 Hall Recall=0。SAM 的排序指标更高，但固定来源阈值的幻觉召回仍低；尤其 Qwen3 仅 8.16%。

## 迁移到 AMBER existence

AMBER 官方 existence 的 4,924 道题 GT 全为 No。Yes-only 只保留模型回答 Yes 的幻觉样本，因此这里没有 real 类；AUC 与双类 AUPR 不定义。下表的 Accuracy 与 Hall Recall 相同。

| 模型 | 方法 | 目标范围 | Hall N | Hall Recall / Accuracy | Hall F1 | 预测 Real 比例 |
|---|---|---|---:|---:|---:|---:|
| LLaVA | SAM | all | 520 | 44.87 ± 8.55 | 61.45 ± 8.38 | 55.13 ± 8.55 |
| LLaVA | SVAR | all | 520 | 0.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 |
| LLaVA | SAM | test | 95 | 45.26 ± 9.06 | 61.79 ± 8.55 | 54.74 ± 9.06 |
| LLaVA | SVAR | test | 95 | 0.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 |
| Qwen3 | SAM | all | 419 | 11.61 ± 2.47 | 20.72 ± 3.99 | 88.39 ± 2.47 |
| Qwen3 | SVAR | all | 419 | 0.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 |
| Qwen3 | SAM | test | 78 | 13.68 ± 2.18 | 24.00 ± 3.34 | 86.32 ± 2.18 |
| Qwen3 | SVAR | test | 78 | 0.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 |

## 产物与核对

服务器：`/root/rivermind-data/ic-ac/outputs/qa_benchmarks/clevr_yesonly_sam_svar_to_pope_amber_82/`。每个模型目录有 `complete_results.md`、`per_seed_results.csv`、`per_seed_results.json`、`cohort_audit.json` 和方法/种子检查点。`_status/` 保存运行日志。

训练前逐题核对 CLEVR 方法与 SVAR 的 key、Yes-only 标签、物体位置、图片级划分、特征有限值、K/tau/系数及维度；目标矩阵与上一步导出摘要核对。加载每个检查点后重新计算 CLEVR test AUC，与训练记录偏差均小于 2e-5。目标的标准化参数和阈值直接从来源检查点读取。代码见 `scripts/run_clevr_yesonly_sam_svar_transfer.py`。
