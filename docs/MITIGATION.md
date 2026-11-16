# 幻觉缓解实验代码

`mitigate` 分支保存 LLaVA 与 Qwen3-VL 的生成时干预、SAR/EMS 引导、评估脚本和实验记录。基线与增强版必须使用相同图片、提示词、精度、生成上限和解码批次进行比较。这里的 Qwen3 实现是方法迁移，实验记录中保留了与原论文设置的差异。

## 代码入口

| 用途 | 文件 |
|---|---|
| LLaVA SADI 基底 | `features/llava_sadi.py` |
| LLaVA SAR/EMS 后层注意力 | `features/llava_sar_ems_attention.py` |
| Qwen3 SVAR、SADI 与 SAR/EMS 后层注意力 | `features/qwen3_sar_ems_attention.py` |
| Qwen3 批量干预 | `features/qwen3_sar_ems_attention_batch.py` |
| LLaVA 生成实验 | `scripts/generation_time_evidence_drift_llava.py` |
| Qwen3 单图生成 | `scripts/generation_time_evidence_drift_qwen3.py` |
| Qwen3 批量生成 | `scripts/generation_time_evidence_drift_qwen3_batched.py` |
| COCO CHAIR、召回、F1 与配对统计 | `scripts/evaluate_generation_time_evidence_drift_llava.py` |
| Qwen3 VQA 干预与评估 | `scripts/evaluate_qwen3_vqa_mitigation.py` |

相关探索实现包括 `features/llava_air.py`、`llava_attention_transfer.py`、`llava_spin.py`、`llava_sparc.py`、`llava_vga.py`，以及视觉支持、FFN 偏离、候选选择与回放模块。QA 特征提取、训练及跨数据集迁移脚本一并保留，以满足依赖并保存当前项目代码状态。

## 当前 500 图单图实验

2026-11-17 的实验比较 `svar` 与 `svar_sar_ems_attention`。固定 Qwen3-VL-8B-Instruct、BF16、`Describe this image.`、greedy、最多 512 个新 token；每次推理一张图，两张 GPU 各处理 250 张。固定图片和原始随机抽样顺序见 [实验快照](../scripts/experiment_snapshots/qwen3_single500/README.md)。

原生生成入口接受 `--model-root`、`--coco-root`、`--output-root`、`--cohort`、`--shards` 和 `--shard`。输出根目录需有 `cohort.json`，其中 `coco500` 为图片 ID 列表。环境依赖在仓库 `requirements.txt`；模型、COCO val2014 图片与标注、NLTK 词汇资源需单独准备。服务器配置和路径保留在配置文件及快照中。

运行环境为 Linux（共用辅助模块使用 `fcntl`）；依赖固定为 PyTorch 2.8.0、Transformers 4.57.6 和 NLTK 3.8.1 等。物体词筛选需要 WordNet，使用 `NLTK_DATA` 指向已下载的 NLTK 数据目录。

提交时单图实验仍在运行，最终结果尚未生成。已有批量结果不能替代单图实验结果：同一批图片在不同批大小下出现过生成序列变化，相关分析已记录。

## 已有结果与限制

- [Qwen3 COCO 500 图：SADI、SVAR 及其后层增强版](archive/QWEN3_SADI_SAR_EMS_COCO500_20261116.md)，含批次敏感性、配对区间与单图影响分析。
- [Qwen3 SVAR 后层增强版独立分批 100 图](archive/QWEN3_SVAR_SAR_EMS_COCO100_20261117.md)。其有利点估计未在后续 500 图批量协议中得到支持。
- [Qwen3 VQA 缓解实验](archive/QWEN3_VQA_SADI_MITIGATION_20261116.md)。
- [LLaVA SAR/EMS 后层注意力](archive/LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md)。
- [当前任务和其他实验索引](CURRENT_TASK.md)。

原始生成、逐图评分、日志和模型权重仍位于服务器的 `outputs/`、`models/` 及数据目录；本分支提交源码、配置、汇总记录及小型复现实验清单。
