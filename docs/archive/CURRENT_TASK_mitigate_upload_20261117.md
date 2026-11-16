# 当前任务

发布：按用户要求创建 GitHub `Xbkeepdo/ic` 的 `mitigate` 分支，保存当前源码、配置与实验记录。独立工作区为 `../ic-mitigate-upload`，保留 `ac` 工作区和运行中的两卡实验。500 图单图启动与评估脚本、图片清单已归档在 `scripts/experiment_snapshots/qwen3_single500/`；使用说明见该分支的 `docs/MITIGATION.md`。

进行中：用户已要求单图推理复验。Qwen3 的原 SVAR 与 SVAR＋SAR/EMS 在原固定 500 张图上以 batch size 1 生成，两卡各处理 250 张。使用项目提示词 `Describe this image.`、BF16、greedy、512 token 上限，聚合原随机顺序的 100／200／300／400／500 子集。新输出根目录为服务器 `outputs/generation_time_evidence_drift_qwen3_single500/`，完整生成后自动评分。

最近完成：Qwen3 的 SVAR＋SAR/EMS 后层注意力扩至原 COCO 500 图，复用原批次和 SVAR 缓存。首批逐 token 一致、全部逐图批次匹配。CHAIRs/i 从 40.80%／9.03% 升到 41.20%／9.39%，F1 从 77.43% 降到 76.78%，幻觉提及 361→374，修复 21 张、新增 23 张。差值区间跨零，共同未触顶 493 图同方向。新臂未在 500 图上改善原 SVAR。[完整结果](QWEN3_SADI_SAR_EMS_COCO500_20261116.md)，服务器主评分为 `outputs/generation_time_evidence_drift_qwen3_coco500/coco500/metrics.first500.svar-svar_sar_ems_attention.json`。

最近完成：解释 SADI 与 SAR/EMS 后层改造，并起草含 COCO 500 图指标的英文 rebuttal。Qwen3 的 SVAR＋同后层配方完成原随机顺序前 100 图配对，CHAIRs/i 从 44.00%／10.44% 降到 40.00%／9.28%，F1 从 75.57% 升到 76.38%，幻觉提及 79→68，修复 8 张、新增 4 张。差值区间均跨零。两卡两组在相同新批次重新生成，200 条描述与逐图评分通过核对。[完整结果](QWEN3_SVAR_SAR_EMS_COCO100_20261117.md)，服务器产物在 `outputs/generation_time_evidence_drift_qwen3_svar100/`。

最近完成：Qwen3-VL 的 SADI＋SAR/EMS 原后层配方在新增 COCO 500 图上的复验。三组同图批量解码和逐图评分已核对。后层版／SADI／SVAR 的 CHAIRs 分别为 26.00%／26.60%／40.80%，CHAIRi 为 5.01%／5.58%／9.03%。按原随机顺序复用逐图评分，已核对 200、300、400、500 张嵌套子集。后层版相对 SADI 的 CHAIR 差值区间在四档均跨零，第 376 张决定了 400、500 张上的 CHAIRi 差值方向。后层版相对 SVAR 的全量召回和 F1 低 8.85／5.16 个百分点。见[完整结果](QWEN3_SADI_SAR_EMS_COCO500_20261116.md)，产物在服务器 `outputs/generation_time_evidence_drift_qwen3_coco500/`。

最近完成：Qwen3-VL 的 VQA 幻觉缓解。POPE 600、AMBER existence 987 题，原 caption 版 SAR/EMS 两组均触发 0 次，输出逐 token 等于 SADI。问题实体适配版触发 572／948 题，POPE 答案全相同、误报 23／300，AMBER 只新增 1 个错误 Yes（SADI 79／987，适配 80／987，原模型 75／987）。AMBER 四组统一用 1,048,576 像素限额重做。两卡输出和逐题评分通过复核。见[完整结果](QWEN3_VQA_SADI_MITIGATION_20261116.md)，产物在服务器 `outputs/qa_benchmarks/qwen3_vqa_sadi_mitigation/`。

最近完成：将 LLaVA 的 SADI＋SAR/EMS 后层配方原层号迁移至 Qwen3-VL-8B-Instruct。两卡已完成同一独立 60 图的配对生成和评分。相对 Qwen3 SADI，CHAIRs/i 从 23.33%／15.29% 降至 16.67%／1.58%，修复 4 张、新增 0 张幻觉图，召回/F1 降 1.81／1.47 个百分点。大部分 CHAIRi 下降来自一张图的重复幻觉，排除该图两项 CHAIR 仍下降。比同图 Qwen3 SVAR 的召回低 8.82 个百分点。见[报告](QWEN3_SADI_SAR_EMS_TRANSFER_20261116.md)，服务器产物在 `outputs/generation_time_evidence_drift_qwen3/`。

缓解实验的最新取舍：优先让 CHAIRs 和 CHAIRi 同时低于同图 SVAR，物体召回与 F1 可以小幅下降，继续并列报告绝对幻觉提及。以 SADI 作为优先干预基底，并保留同图 SADI 以检验 SAR/EMS 的单独增益。已复核 SADI 后层版的独立 60 图优势和两批合计，见[复核结果](LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md#复核此前两批对照)。

最近完成：生成时直接迁移 SAR 的投影视觉写入，评估 SPIN、SPARC、VGA 的同图开发 24 图。SVAR＋后层 SPIN 从 24 图扩至 100 图。投影选头相对匹配对照的 CHAIRs/i 从 25.00%／6.13% 降到 23.00%／3.98%，召回／F1 从 69.64%／75.16% 升到 71.35%／77.03%。同图原 SVAR 为 21.00%／5.01% 与 71.86%／78.17%，幻觉提及 34 次对投影版 35 次。候选尚未达到 SVAR 的综合条件，封存图未验证。见[对照与文件](LLAVA_ATTENTION_METHOD_TRANSFER_20261116.md)，服务器产物在 `outputs/generation_time_evidence_drift_llava/pilot/`。

最近完成：Qwen2.5-VL-7B 和 InternVL2.5-8B 的 CLEVR 方案 B 三协议对照。No-only 按模型答 No 选样，SAM 的 No-only AUROC 分别为 0.9476、0.9445，SAM-V2 为 0.9455、0.9307。三种子共 126 条数据通过完整性核对。见[报告](QA_CLEVR_SCHEME_B_QWEN2_INTERNVL_20261116.md)，服务器产物在 `/root/rivermind-data/ic/outputs/clevr_scheme_b_qwen2_internvl/`。

最近完成：LLaVA 的视觉写入与 FFN 空间偏离消融。复用开发 100、外部 100 和保留 80 图的 SADI、AIR*、SVAR 候选，不重新生成。后两批合计，覆盖保护把联合版召回／F1 从 68.06%／74.57% 提到 73.19%／77.67%，SVAR 为 72.87%／77.55%。仅 VQA 加覆盖保护已恢复大部分覆盖，联合信号相比同候选的仅 VQA 多出 6 次幻觉提及。单独加视觉写入的召回／F1 为 73.72%／78.14%，与 SVAR 同为 83 次幻觉提及。已有回放未复现超越 SVAR 的联合缓解，封存验证未运行。见[结果和产物](LLAVA_PATCH_WRITE_COVERAGE_20261116.md)。

最近完成：前一轮有图／无图词概率差与 FFN 词表增益的 SADI/AIR* 排序归因为输出代理实验。80 图 CHAIRs/i 为 13.75%/2.99%，SVAR 为 21.25%/6.18%，物体召回和 F1 低约 8.3 和 5.8 个百分点。完整试验见[更正后的报告](LLAVA_TARGETED_GROUNDING_MITIGATION_20261116.md)。服务器根目录为 `outputs/generation_time_evidence_drift_llava/`。

最近完成：为 rebuttal 验证 LLaVA-1.5-7B 的 SAR/EMS 注意力缓解。五种新方案在同一 24 张开发图筛选，VHR 局部投影写入选头在两批各 60 张独立图上均降低 CHAIR 点估计。合并 CHAIRs/i 从 39.17%/9.65% 降到 34.17%/8.29%，区间跨零。SVAR 为 20.00%/4.67%。原 SADI＋SAR/EMS 后层版在独立 60 图上与 SADI 的 CHAIRs 持平，SADI 增益内的投影选头在开发图上降低物体召回。见[完整报告](LLAVA_REBUTTAL_SAR_EMS_MITIGATION_20261116.md)。

最近完成：LLaVA-1.5-7B 生成时 SAR/EMS 候选选词和重复控制。SADI＋SAR/EMS 视觉重复控制在 100 张和固定 40 张的 CHAIRs/i 为 12.00%/3.39% 和 32.50%/8.50%，SVAR 为 21.00%/5.01% 和 30.00%/8.97%。召回分别比 SVAR 低 4.14、5.46 个百分点。硬重排先导表现下降，校准降权 100 张的 CHAIRs 与贪心持平。见[完整报告](LLAVA_GENERATION_SAR_EMS_DECODING_20261116.md)。

最近完成：LLaVA-1.5-7B 的 SAR/EMS 区域分配探索，同图筛选 24 张。早期直接用视觉写入分配概率的 CHAIRs/i 为 33.33%/13.25%，召回 80.69%；保留中位数量纲并限幅的版本为 41.67%/11.32%，召回 76.18%。SVAR 为 20.83%/4.52%、召回 70.62%。四个新配置及完整公式见[报告](LLAVA_GENERATION_SAR_EMS_WRITE_PRIOR_20261116.md)。上一版结果分片通过 SHA-256 核对。

最近完成：LLaVA-1.5-7B 在 SADI 上叠加 SAR/EMS 后层注意力重分配。100 图的 CHAIRs/i 为 11.00%/3.05%，SADI 为 13.00%/3.34%，SVAR 为 21.00%/5.01%。40 图对照的 CHAIRs/i 为 30.00%/10.85%，与 SADI 相同，SVAR 为 30.00%/8.97%。比 SVAR 的物体召回分别低 4.58 和 4.96 个百分点。见[完整报告](LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md)。服务器产物位于 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_llava/`。

最近完成：LLaVA-1.5-7B 视觉写入和 FFN 偏离的首轮缓解探索。开发集上的 SADI 与 FFN 方向调整优势未在 40 图对照上稳定复现。见[报告](LLAVA_GENERATION_EVIDENCE_DRIFT_20261116.md)。

最近完成：POPE popular Yes-only 域内训练与 CLEVR＋AMBER existence 混合训练迁移，在同一 POPE test 对照 SAM/SVAR。种子仅改变初始化的结果见[报告](QA_POPE_DOMAIN_VS_CLEVR_AMBER_INIT_ONLY_20261115.md)，逐种子产物位于服务器 `outputs/qa_benchmarks/pope_yesonly_domain_vs_clevr_amber_init_only_82/`。

最近完成：之前的 POPE 域内与 CLEVR＋AMBER 迁移对照中，种子同时改变初始化和训练随机过程；历史结果见[报告](QA_CLEVR_AMBER_YESONLY_MIXED_TO_POPE_20261115.md)。

最近完成：POPE popular Yes-only 域内训练，SAM 与 SVAR 结果见[报告](QA_POPE_POPULAR_YESONLY_SAM_SVAR_INDOMAIN_20261115.md)。

最近完成：CLEVR Yes-only 训练 SAM 与 SVAR 并迁移到 POPE popular、AMBER existence。见[完整报告](QA_CLEVR_YESONLY_SAM_SVAR_TRANSFER_20261115.md)；逐种子结果与检查点在服务器 `outputs/qa_benchmarks/clevr_yesonly_sam_svar_to_pope_amber_82/`。

最近完成：两模型 POPE popular、AMBER existence 的 SAM/SVAR Yes-only 特征提取；四组特征均对齐并导出。见[完整报告](QA_SAM_SVAR_POPE_AMBER_20261115.md)及[过程归档](CURRENT_TASK_qa_sam_svar_pope_amber_20261115.md)。

最近完成：LLaVA/Qwen3 的 COCO4000 固定 8:2 EMS 来源向量对照，见[报告](../../outputs/log_s_p_source_vector_ablation_82_fixed_split/complete_results.md)。

最近完成：LLaVA/Qwen3 的 COCO4000 固定 8:1:1 图片划分源向量对照，见[报告](../../outputs/log_s_p_source_vector_ablation_811_fixed_split/complete_results.md)。

更早记录见上述过程归档及 `docs/archive/`。
