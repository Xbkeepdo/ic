# 幻觉缓解 autoresearch 代码快照

本分支 `mitigate-autoresearch` 基于 `mitigate` 的提交 `e14c53dfe5c5b2dda4483f0d172ca977bd1be044`，保存截至 2026-11-20 的缓解探索源码、配置和实验报告。它包含成功与失败的探索实现，不表示所有候选都优于 SADI 或 SVAR。当前自动探索已停止，没有实验随分支发布自动启动。

## 主要入口

| 用途 | 文件 |
|---|---|
| Qwen3 SADI/SVAR 批量干预 | `features/qwen3_sar_ems_attention_batch.py` |
| 完整 SADI 后的 next-layer 指导 | `features/qwen3_full_sadi_expanded_next.py` |
| 视觉证据与 FFN 偏好分离的注意力搬移 | `features/qwen3_decoupled_semantic_route.py` |
| 语义区域重读与真实 KV 提交 | `features/qwen3_live_semantic_crop.py` |
| 固定 10 图完整生成 | `scripts/run_qwen3_live_semantic_crop10.py` |
| 独立结果与提交事件核验 | `scripts/audit_qwen3_live_semantic_crop10.py` |
| FFN 支持分量检查 | `features/qwen3_evidence_aligned_ffn.py` |
| 原 K32/tau=.03/w=1 risk 与 F_log | `features/qwen3_online_risk_f.py` |
| 显式选择 sentence/legacy 版 CHAIR | `scripts/evaluate_caption_chair.py` |

其他 Qwen3、LLaVA 候选、缓存、训练、分析和核验代码一并保留。新候选通常在完整 SADI 基底上操作后续层；具体层号与预算以各脚本及实验报告为准，不能混用不同实验的设置。

## 最近完成的结果

- [完整 10 图语义区域重读](archive/QWEN3_EARLY_SEMANTIC_CROP10_20261120.md)：三项新增候选与 SADI 的逐图物体评分相同，没有增量收益；首词改写后仍重新出现 fork，未推进新的 500 图验证。
- [语义目的地、细区域与 FFN 辅助](archive/QWEN3_DECOUPLED_SEMANTIC_ROUTE_20261120.md)。
- [语义背景、相对预算与层间检查](archive/QWEN3_SEMANTIC_SPECIFICITY_20261120.md)。
- [实验与恢复失败记录](archive/WORKFLOW_CHECK_FAILURES_20261118.md)：保留失败及评分恢复过程，恢复没有改变原始描述或 CHAIR 计算公式。

报告中的开发集、独立图片、生成批次和评估版本均有区别，结果应按各自协议读取。CHAIRs/CHAIRi 越低越好，同时检查真实物体召回、F1 和幻觉提及数。

## 环境与复跑

最近实测环境为 Linux、Python 3.12、PyTorch 2.12.1+cu130、Transformers 4.57.6，Qwen3-VL-8B-Instruct 使用 BF16/eager/greedy，两张 RTX 4080 SUPER。仓库 `requirements.txt` 是较早环境的依赖记录，不能直接当作这轮完整锁定环境。

模型、COCO val2014、标注与 NLTK 数据单独准备。实验入口可先用 `--help` 查看参数；带 `--prepare` 的研究脚本还依赖服务器上已存在的冻结协议、原生对照及缓存，不是下载后即可在空目录运行的通用入口。服务器项目为 `/root/rivermind-data/ic-ac`，模型和数据分别位于父目录的 `models/` 与 `dataset/`。

历史实验的准确源 SHA 和完整产物保存在对应服务器 `outputs/` 中；Git 文本统一 LF，重跑应在新的输出目录准备协议，保留原始冻结产物。这里提供小型 cohort、派生 recipe 和独立核验结果用于定位记录，recipe 不替代原始可执行 protocol。

本分支不包含模型权重、图片、NPZ/PKL 缓存、逐 token 长路径、原始大规模 generation 或连接凭据。完整实验数据仍在服务器，各报告注明对应目录。此前 `docs/MITIGATION.md` 为 2026-11-17 的历史说明，最新进展以本页及上述报告为准。
