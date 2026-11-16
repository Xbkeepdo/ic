# 当前任务

发布完成：GitHub `Xbkeepdo/ic` 的 `mitigate` 分支已上传 68 个文件，提交 `c1364c0565867aaab59e8d73bd1854209113548f`。通过 GitHub 接口提交，远端树 `2efb01205b66fc3f22743bd4e1c74f38536a3849` 与本地已检查内容相同。独立工作区为 `../ic-mitigate-upload`，保留 `ac` 工作区和运行中的两卡实验。500 图单图启动与评估脚本、图片清单已归档在 `scripts/experiment_snapshots/qwen3_single500/`；使用说明见该分支的 `docs/MITIGATION.md`。

已完成单图复验：Qwen3 的 SVAR 与 SVAR＋SAR/EMS 在原固定 500 张图上以 batch size 1 生成，两卡各 250 图，BF16、greedy、512 token 上限、`Describe this image.`。CHAIRs 42.60%→42.80%，CHAIRi 9.70%→9.49%，召回 76.87%→76.45%，F1 76.08%→75.85%，修复 21 张、新增 22 张。配对差值区间均跨零；图 201163 减少的 16 次幻觉等于全量净减少，移除它后 CHAIRi 优势反转。产物在 `outputs/generation_time_evidence_drift_qwen3_single500/`，见[最终报告](archive/QWEN3_SVAR_SAR_EMS_SINGLE500_20261117.md)。此轮未跑 SADI 两臂的 500 图单图对照，不能混称 SADI 复验。

阶段汇报完成：100／200／300／400／500 图均已向用户汇报，见[阶段记录](archive/QWEN3_SVAR_SAR_EMS_SINGLE500_PROGRESS_20261117.md)。阶段集是生成清单按 ID 升序前 N 图；最终报告另有原 seed43 抽样顺序前 N 图，两个 100 图仅重合 17 张。最终主评分与逐图缓存、阶段评分、5,000 次配对 bootstrap 经独立复核一致。生成、监视器和诊断队列均已结束。

批次诊断完成：同图旧 100 张在 B 改变后 66 张 token 序列改变，见[同图核对](archive/QWEN3_SINGLE_BATCH_COHORT_AUDIT_20261117.md)。图 485532 的 B2／B4 首次分歧在两方法均复现：第二 token 为 image／is，候选得分 37.00:36.75→36.75:36.75。视觉表示在进入语言第 0 层前已不同；B1 两控制器、重复 B1、B4 行逆序对照均一致。定位了此例的批次相关数值差异与贪心分支切换，未定位具体算子或解释全部 66 图。见[诊断报告](archive/QWEN3_BATCH_LOGIT_DIAGNOSTIC_20261117.md)，产物在 `outputs/qwen3_batch_logit_diagnostic/`。

当前解释范围：用户所说“自己复跑”指最初 Qwen3 60 图结果之后本会话内全部实验，不是外部运行；无需再次索要命令。原 SADI 60 图与后续 500 图无重合，且单图→批量，既有样本变化也有协议变化。原 60 图的 CHAIRi 改善主要由一图贡献，修复 4／新增 0 的双侧 McNemar p=0.125，不能宣称稳定显著。后续 SVAR 迁移又改变了基底。当前证据不支持模块稳定大幅改善。此次新增诊断／监视器及报告已在本地和服务器保存，尚未更新 GitHub 上此前发布的提交。

最近完成：Qwen3 的 SVAR＋SAR/EMS 后层注意力扩至原 COCO 500 图，复用原批次和 SVAR 缓存。首批逐 token 一致、全部逐图批次匹配。CHAIRs/i 从 40.80%／9.03% 升到 41.20%／9.39%，F1 从 77.43% 降到 76.78%，幻觉提及 361→374，修复 21 张、新增 23 张。差值区间跨零，共同未触顶 493 图同方向。新臂未在 500 图上改善原 SVAR。[完整结果](archive/QWEN3_SADI_SAR_EMS_COCO500_20261116.md)，服务器主评分为 `outputs/generation_time_evidence_drift_qwen3_coco500/coco500/metrics.first500.svar-svar_sar_ems_attention.json`。

最近完成：解释 SADI 与 SAR/EMS 后层改造，并起草含 COCO 500 图指标的英文 rebuttal。Qwen3 的 SVAR＋同后层配方完成原随机顺序前 100 图配对，CHAIRs/i 从 44.00%／10.44% 降到 40.00%／9.28%，F1 从 75.57% 升到 76.38%，幻觉提及 79→68，修复 8 张、新增 4 张。差值区间均跨零。两卡两组在相同新批次重新生成，200 条描述与逐图评分通过核对。[完整结果](archive/QWEN3_SVAR_SAR_EMS_COCO100_20261117.md)，服务器产物在 `outputs/generation_time_evidence_drift_qwen3_svar100/`。

最近完成：Qwen3-VL 的 SADI＋SAR/EMS 原后层配方在新增 COCO 500 图上的复验。三组同图批量解码和逐图评分已核对。后层版／SADI／SVAR 的 CHAIRs 分别为 26.00%／26.60%／40.80%，CHAIRi 为 5.01%／5.58%／9.03%。按原随机顺序复用逐图评分，已核对 200、300、400、500 张嵌套子集。后层版相对 SADI 的 CHAIR 差值区间在四档均跨零，第 376 张决定了 400、500 张上的 CHAIRi 差值方向。后层版相对 SVAR 的全量召回和 F1 低 8.85／5.16 个百分点。见[完整结果](archive/QWEN3_SADI_SAR_EMS_COCO500_20261116.md)，产物在服务器 `outputs/generation_time_evidence_drift_qwen3_coco500/`。

最近完成：Qwen3-VL 的 VQA 幻觉缓解。POPE 600、AMBER existence 987 题，原 caption 版 SAR/EMS 两组均触发 0 次，输出逐 token 等于 SADI。问题实体适配版触发 572／948 题，POPE 答案全相同、误报 23／300，AMBER 只新增 1 个错误 Yes（SADI 79／987，适配 80／987，原模型 75／987）。AMBER 四组统一用 1,048,576 像素限额重做。两卡输出和逐题评分通过复核。见[完整结果](archive/QWEN3_VQA_SADI_MITIGATION_20261116.md)，产物在服务器 `outputs/qa_benchmarks/qwen3_vqa_sadi_mitigation/`。

最近完成：将 LLaVA 的 SADI＋SAR/EMS 后层配方原层号迁移至 Qwen3-VL-8B-Instruct。两卡已完成同一独立 60 图的配对生成和评分。相对 Qwen3 SADI，CHAIRs/i 从 23.33%／15.29% 降至 16.67%／1.58%，修复 4 张、新增 0 张幻觉图，召回/F1 降 1.81／1.47 个百分点。大部分 CHAIRi 下降来自一张图的重复幻觉，排除该图两项 CHAIR 仍下降。比同图 Qwen3 SVAR 的召回低 8.82 个百分点。见[报告](archive/QWEN3_SADI_SAR_EMS_TRANSFER_20261116.md)，服务器产物在 `outputs/generation_time_evidence_drift_qwen3/`。

缓解实验的最新取舍：优先让 CHAIRs 和 CHAIRi 同时低于同图 SVAR，物体召回与 F1 可以小幅下降，继续并列报告绝对幻觉提及。以 SADI 作为优先干预基底，并保留同图 SADI 以检验 SAR/EMS 的单独增益。已复核 SADI 后层版的独立 60 图优势和两批合计，见[复核结果](archive/LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md#复核此前两批对照)。

最近完成：生成时直接迁移 SAR 的投影视觉写入，评估 SPIN、SPARC、VGA 的同图开发 24 图。SVAR＋后层 SPIN 从 24 图扩至 100 图。投影选头相对匹配对照的 CHAIRs/i 从 25.00%／6.13% 降到 23.00%／3.98%，召回／F1 从 69.64%／75.16% 升到 71.35%／77.03%。同图原 SVAR 为 21.00%／5.01% 与 71.86%／78.17%，幻觉提及 34 次对投影版 35 次。候选尚未达到 SVAR 的综合条件，封存图未验证。见[对照与文件](archive/LLAVA_ATTENTION_METHOD_TRANSFER_20261116.md)，服务器产物在 `outputs/generation_time_evidence_drift_llava/pilot/`。

最近完成：Qwen2.5-VL-7B 和 InternVL2.5-8B 的 CLEVR 方案 B 三协议对照。No-only 按模型答 No 选样，SAM 的 No-only AUROC 分别为 0.9476、0.9445，SAM-V2 为 0.9455、0.9307。三种子共 126 条数据通过完整性核对。见[报告](archive/QA_CLEVR_SCHEME_B_QWEN2_INTERNVL_20261116.md)，服务器产物在 `/root/rivermind-data/ic/outputs/clevr_scheme_b_qwen2_internvl/`。

更早已完成记录见[本次发布时的任务归档](archive/CURRENT_TASK_mitigate_upload_20261117.md)及 `docs/archive/`。

发布验证：38 个 Python 文件语法检查通过，内部依赖与 500 图清单核对通过。普通 git push、HTTP/1.1 push 与 fetch 因 GitHub 443 连接失败；已改用 GitHub 接口完成上传并读取远端文件核验。独立本地工作区保留同内容提交 `e6b3c5a`，尚未通过 git fetch 同步接口创建的提交。
