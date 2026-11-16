# POPE popular / AMBER existence：SAM 与 SVAR 特征提取

## 协议

- 模型：LLaVA-1.5-7B、Qwen3-VL-8B。
- 题目：POPE popular 3,000 题／500 图；AMBER existence 4,924 题／1,004 图。物体词跨度按题核对，图像级划分沿用准备数据中的 `image_splits.json`。
- 仅保留模型回答 Yes 的题目；GT Yes 标为 real（1），GT No 标为 hallucination（0）。
- SAM：物体词首个 subtoken 之前的隐藏状态计算逐层 `logs_risk`，首个答案 token 之前的隐藏状态计算逐层 `f_log`，按此顺序拼接。Log-S 使用 K=32、tau=0.03、EMD；语义概率取 final-norm 后词表 softmax。
- SVAR：在物体词首个 subtoken 之前的隐藏状态提取受控 SVAR，训练向量取第 5–18 层（Python 切片 `[5,19)`）。
- 本次只提取和导出特征，没有训练二分类器。

## 完整结果

| 模型 | 数据集 | Yes 样本 | real | hallucination | SAM 维度 | SVAR 维度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-7B | POPE popular | 1,274 | 1,183 | 91 | 64 | 448 |
| Qwen3-VL-8B | POPE popular | 1,348 | 1,254 | 94 | 72 | 448 |
| LLaVA-1.5-7B | AMBER existence | 520 | 0 | 520 | 64 | 448 |
| Qwen3-VL-8B | AMBER existence | 419 | 0 | 419 | 72 | 448 |

AMBER 官方 existence 4,924 题的 GT 全为 No。按本次 Yes-only 协议，保留下来的样本自然全是 hallucination；这两组特征可用于分析，但不能单独训练或计算二分类 AUC/AUPR。

## 产物和核对

服务器根目录：`/root/rivermind-data/ic-ac/outputs/qa_benchmarks`。

每个组合位于 `<模型>/POPE-POPULAR-AMBER-EXIST-SAM-SVAR-YESONLY/<数据集>/`，其中模型名为 `llava_1_5_7b` 或 `qwen3_vl_8b`，数据集名为 `pope` 或 `amber_discriminative`。主要文件：

- `sam_object_risk_answer_f_log_vs_svar_object_yesonly.npz`：按相同 `keys` 排列的 `sam`、`svar`、`labels`、`splits` 矩阵。
- `sam_svar_cross_position_summary.json`：样本数、维度、标签/划分计数和输入及 NPZ 的 SHA-256。
- `features.pkl`、`baseline/object_hallucination_yes_only/features.pkl`：逐题完整 SAM 与 SVAR 特征。
- `_sam_svar_pope_amber_status/`：执行状态及日志。Qwen3 的 AMBER 特征提取使用 GPU0、GPU1 并行；LLaVA AMBER 提取已完成，原脚本在后续导出步骤报错，已手动重新导出并验证。

导出器逐题检查问题、SAM、SVAR 的 key 完全一致，标签和图像划分一致，物体位置已成功提取，特征维度一致且全部为有限值。四组导出均通过。

本地 `py -m py_compile`、`git diff --check`、服务器 `bash -n` 通过。仓库的 Python 单元测试未在本地运行：本地环境缺少 `torch`；服务器环境缺少 `pytest`。真实模型的四组提取与导出均已运行成功。
