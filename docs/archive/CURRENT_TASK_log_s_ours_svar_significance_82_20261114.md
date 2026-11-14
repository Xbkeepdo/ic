# Qwen3 COCO 8:2：SAR+EMS 对 SVAR 配对显著性检验

目标：检验种子 43/44/45 同时控制图片划分和训练随机性时，Qwen3 上 SAR+EMS 与原生 SVAR 的测试表现差异。

输入：`outputs/coco4000_seeded_image_split_auc_aupr/` 的 3200/800 分图、已训练检查点和原始报告，以及 `outputs/log_s_ot_rank16_rq_seeded_82_43_45/qwen3_vl_8b/seed*/paired_scores.npz` 内已恢复并核对的原 SAR+EMS 分数。用原生 SVAR 检查点和原特征恢复其测试分数，以 `(image_id,response_token_idx,target_token_id)` 对齐。seed44 的 SVAR 行次序有 10 处需要重排。

方法：逐种子重算 AUROC、Real AUPR、Hall AUPR 并与原报告核对；对 3 个测试集的 1955 张不同图片做 3000 次联合配对 bootstrap，重叠图片在各种子间同步赋权。报告三种子平均差、逐种子差、95% 百分位 CI 和重中心化双侧 p。CI 条件于现有划分和模型，不覆盖重训/重划分。

结果：三种子均值 AUROC SAR+EMS 89.04%、SVAR 87.76%，差 +1.27 个百分点，95% CI [+0.32,+2.25]，未校正 p=0.012。Real AUPR 差 +0.32，CI [-0.07,+0.68]；Hall AUPR 差 +2.81，CI [+0.11,+5.60]，后者为补充指标且未作多重比较校正。详情见 [报告](../../outputs/log_s_ours_svar_significance_82/summary.md)。

实现：`scripts/bootstrap_log_s_ours_svar_seeded_82.py`。验证：重建分数的逐种子 AUROC/AUPR 与既有报告在 1e-6 内一致，脚本已运行 3000 次抽样并通过语法检查。
