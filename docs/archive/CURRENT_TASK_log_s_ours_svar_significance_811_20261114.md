# COCO 8:1:1：SAR+EMS 与 SVAR 的显著性检验

目标：在 LLaVA-1.5-7B 与 Qwen3-VL-8B 上，比较 SAR+EMS 与原生 SVAR。三个种子 43/44/45 同时作用于图片划分和检测器随机性。

输入：`outputs/log_s_ems_i_mean_svar_seeded_811_43_45/{model}/seed{43,44,45}/{sar_ems,svar}/scores.npz`；逐种子两个方法使用同一 object token 测试集。SVAR 的测试分数按 `(image_id,response_token_idx,target_token_id)` 与 SAR+EMS 对齐；LLaVA seed45 需重排 9 个位置，见 [对齐审计](../../outputs/log_s_ours_svar_significance_811/alignment_audit.json)。

方法：对测试图片作 3000 次配对、含放回 bootstrap，同一图片的 object token 和两个方法共用抽样权重。三个种子用测试图片 ID 并集联合重采样；计算 AUROC、Real AUPR、Hall AUPR 的均值差和 95% 百分位置信区间。双侧 p 使用重中心化的 bootstrap 分布估计，两个模型的主指标 AUROC 另作 Holm 校正。固定已有模型和三个划分，区间不覆盖重训或重新划分的不确定性。

结果：LLaVA 三种子平均 AUROC 差 +1.12 个百分点，95% CI [+0.28,+1.96]，Holm p=0.021；Qwen3 差 +0.60，CI [-0.75,+1.87]，Holm p=0.381。LLaVA 的优势有证据支持，Qwen3 的差异尚不能判定显著。全部指标、逐种子结果及原始抽样数据见 [报告](../../outputs/log_s_ours_svar_significance_811/summary.md)。

实现：`scripts/bootstrap_log_s_ours_svar_seeded_811.py`。验证包括原始 AUROC/AUPR 与既有 `result.json` 一致、加权 AUROC/AUPR 与 sklearn 对照一致，以及脚本运行和语法检查。
