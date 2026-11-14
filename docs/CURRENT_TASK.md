# 当前任务

当前任务：无。

最近完成：LLaVA/Qwen3 的 COCO4000 固定 8:2 EMS 来源向量四组对照。CUDA 重训原版三种子的测试指标与历史 test_grid 完全一致；完整结果见[报告](../outputs/log_s_p_source_vector_ablation_82_fixed_split/complete_results.md)与[归档](archive/CURRENT_TASK_log_s_p_source_vector_ablation_fixed_82_20261115.md)。

最近完成：LLaVA 和 Qwen3 的 COCO4000 固定 8:1:1 图片划分源向量对照；种子 43/44/45 沿用原训练器控制初始化与训练随机性。见[结果](../outputs/log_s_p_source_vector_ablation_811_fixed_split/complete_results.md)及[归档](archive/CURRENT_TASK_log_s_p_source_vector_ablation_fixed_811_20261115.md)。

最近完成：同批 LLaVA 种子 43 的 100 图试验 SAR 注意力＋EMS FFN 联合干预的 FFN 纯视觉贡献版；SAR 注意力改写和最多 64 轮复检保持不变。34 图未触发而复用，66 图重跑；CHAIRs 仍为 30%，幻觉提及仍为 41 次，CHAIRi 8.04%→7.95% 仅因物体提及分母增加，物体召回 73.24%→73.18%；回滚 527→575 次。见[报告](../outputs/generation_time_sar_ems_joint_loop_fused_64_vis_ffn_seed43/summary.md)及[归档](archive/CURRENT_TASK_generation_time_sar_ems_joint_loop_fused_64_vis_ffn_seed43_20261115.md)。

最近完成：同批 LLaVA 种子 43 的 100 图把共享前向＋前缀预填充联合干预的复检上限从 32 提至 64。96 图沿用已清空结果，4 图重跑且前 32 次动作一致，最终 100 图检测器清空。CHAIRs 31%→30%，CHAIRi 8.72%→8.04%，物体召回保持 73.24%，多 32 次回滚，逐图耗时合计约多 51.70 秒；仍有 30 图被 CHAIR 判为含幻觉。见[报告](../outputs/generation_time_sar_ems_joint_loop_fused_64_seed43/summary.md)及[归档](archive/CURRENT_TASK_generation_time_sar_ems_joint_loop_fused_64_seed43_20261115.md)。

最近完成：前缀预填充基础上实现检测与干预规划共用一次完整 caption 捕获前向；同批 LLaVA 种子 43 的 100 图逐图耗时 1082.98→843.68 秒，相比最初逐 token 重放的 2032.97 秒下降 58.5%。初始触发状态 100/100 一致、66 张触发图的首风险词分数一致、首次 FFN 三层集合 99/100 一致、最终 caption 与前缀版 94/100 逐 token 相同；新变体 CHAIRs 31%、CHAIRi 8.72%、检测器清空 96 图。存在浮点路径差异，非严格等价。见[报告](../outputs/generation_time_sar_ems_joint_loop_fused_seed43/summary.md)及[归档](archive/CURRENT_TASK_generation_time_sar_ems_joint_loop_fused_seed43_20261115.md)。

最近完成：同批 LLaVA 种子 43 的 100 图尝试把多轮回滚的已知前缀一次性预填充后只续写后缀。逐图时间合计 2032.97→1082.98 秒，回滚生成 1429.51→427.70 秒；最终 caption 92/100 张逐 token 相同，因浮点路径变化不是严格等价替换，CHAIRs 32%→31%、检测器清空 96→95 图。检测与规划重复前向的可合并路径及风险写入[报告](../outputs/generation_time_sar_ems_joint_loop_prefix_seed43/summary.md)和[归档](archive/CURRENT_TASK_generation_time_sar_ems_joint_loop_prefix_seed43_20261115.md)。

最近完成：按用户纠正，逐头对图像注意力 logit 先取绝对值再求均值，选中区域加 `0.7 × 均值`，FFN 加 `0.4 × 同层完整注意力输出`，只做联合干预；同批 LLaVA 种子 43 的 100 图比较首次处理和最多 32 轮复检回滚。66 图触发；首次 CHAIRs/CHAIRi 为 49%/14.62%，多轮为 32%/9.09%，多轮物体召回 73.53%（贪心 80.54%），536 次回滚且 4 图未清空。见[报告](../outputs/generation_time_sar_ems_joint_loop_absmean_seed43/summary.md)及[归档](archive/CURRENT_TASK_generation_time_sar_ems_joint_loop_absmean_seed43_20261114.md)。

最近完成：按用户要求把局部注意力和 FFN 视觉注入的逐层系数改为固定 `1.7` 与 `0.2`，在同批 LLaVA 种子 43 的 100 张图重跑。66 图触发且全部执行联合修改；联合改写 20 图，CHAIRs 48%→47%，CHAIRi 14.75%→15.74%，物体 F1 75.36%→74.68%。强度增大后改写更多，但整体幻觉没有改善；见[报告](../outputs/generation_time_sar_ems_local_fixed_seed43/summary.md)及[归档](archive/CURRENT_TASK_generation_time_sar_ems_local_fixed_seed43_20261114.md)。

最近完成：按用户纠正，只要检测器触发就同时修改 SAR 局部注意力及 EMS 排序的 3 个 FFN 层；同批 LLaVA 100 图有 66 图触发且全部执行两项修改，联合改写 14 图，CHAIRs 48%→46%，CHAIRi 14.75%→15.38%，论文式总体物体 F1 75.36%→74.96%。见[报告](../outputs/generation_time_sar_ems_local_always_seed43/summary.md)与[归档](archive/CURRENT_TASK_generation_time_sar_ems_local_always_seed43_20261114.md)。

最近完成：LLaVA 种子 43 的 100 图检测后 SAR 局部注意力、EMS 门控 FFN 视觉注入及联合初试。66 图触发，分别改写 4/7/8 图；CHAIRs 为 47%/48%/47%，CHAIRi 为 14.77%/15.47%/15.38%，均未见可信缓解，逐图还出现词表外幻觉。见[报告](../outputs/generation_time_sar_ems_local_seed43/summary.md)与[归档](archive/CURRENT_TASK_generation_time_sar_ems_local_seed43_20261114.md)。

最近完成：LLaVA 种子 43 前 100 图使用 SAR+EMS 检测器验证集 Hall-F1 阈值 0.4013161 做逐物体循环回滚；66 图触发、760 次回滚，CHAIRs 48%→25%，CHAIRi 14.75%→6.44%，物体召回 80.54%→74.42%。详见[报告](../outputs/generation_time_closed_loop_tau0p4013161_seed43/summary.md)与[归档](archive/CURRENT_TASK_generation_time_closed_loop_train_threshold_seed43_20261114.md)。

最近完成：COCO4000 严格 8:2 下的 LLaVA/Qwen3 三种子 SAR+$E_A$ 对 SAR+$E_F$；原特征逐元素核对及训练协议核对均通过。见[结果](../outputs/log_s_p_source_vector_ablation_82/complete_results.md)与[归档](archive/CURRENT_TASK_log_s_p_source_vector_ablation_82_20261114.md)。

最近完成：SAR+EMS 原阈值 0.6925053 在 LLaVA 种子 43 前 100 图上做与阈值 0.5 相同的逐物体循环；32 张触发，共 200 次回滚，CHAIRs 48%→44%，CHAIRi 14.75%→11.75%。见[报告](../outputs/generation_time_closed_loop_tau0p6925053_seed43/summary.md)与[归档](archive/CURRENT_TASK_generation_time_closed_loop_original_threshold_seed43_20261114.md)。

最近完成：LLaVA 种子 43 前 100 图按用户原意做 SAR+EMS 检测后逐物体循环回滚，阈值 0.5；CHAIRs 48%→35%，CHAIRi 14.75%→9.76%，物体召回 80.54%→77.72%。详见[报告](../outputs/generation_time_closed_loop_tau05_seed43/summary.md)与[归档](archive/CURRENT_TASK_generation_time_closed_loop_tau05_seed43_20261114.md)。

最近完成：LLaVA 与 Qwen3 的 COCO4000 EMS 来源联合实验，比较 attention 输出、FFN 输出及残差对照；五组输入的三种子结果与容量对照已完成。见 [结果](../outputs/log_s_ems_source_union_811/complete_results.md)、[分析](../outputs/log_s_ems_source_union_811/analysis.md)与[归档](archive/CURRENT_TASK_log_s_ems_source_union_811_20261114.md)。

最近完成：SVAR 论文原始 Heads Guided Attention Intervention 在 LLaVA 种子 43 前 100 图上全程生成。论文第 5–18 层、alpha 0.5 将 CHAIRs 48%→23%、CHAIRi 14.75%→4.82%，物体召回 80.54%→71.65%；作者代码实际第 5–17 层结果也已比较。详见[报告](../outputs/generation_time_svar_official_seed43/summary.md)与[归档](archive/CURRENT_TASK_generation_time_svar_official_seed43_20261114.md)。

最近完成：三篇论文方法 VCD、DeCo、ICD 的 LLaVA 种子 43 前 100 图检测门控单步迁移，分别接入 SAR+EMS 与原生 SVAR。六组完整评测；唯一 CHAIRs 改善为 SAR+EMS + DeCo 的 48%→47%，CHAIRi 未下降。详见[报告](../outputs/generation_time_paper_transfer_seed43/summary.md)与[归档](archive/CURRENT_TASK_generation_time_paper_transfer_seed43_20261114.md)。

最近完成：按用户要求停止无检测门控的两模型 200 图实验，转而完成 LLaVA 种子 43、前 100 图的检测门控强残差试验。联合干预 CHAIRs 与原始贪心同为 48%，仅注意力干预的 CHAIRi 从 14.75% 升至 16.77%。详见 [报告](../outputs/generation_time_gated_residual_seed43/summary.md)与[归档](archive/CURRENT_TASK_generation_time_gated_residual_seed43_20261114.md)。

最近完成：LLaVA 与 Qwen3 的 COCO4000 EMS 源分布向量对照；四种 $P$ 构造的全量提取、三种子训练及数值对齐均完成。详见 [结果](../outputs/log_s_p_source_vector_ablation_811/complete_results.md)与[归档](archive/CURRENT_TASK_log_s_p_source_vector_ablation_811_20261114.md)。

最近完成：种子 43 的 generation-time SAR+EMS 三方法试验，两模型各 400 张 COCO 测试图。详见 [报告](../outputs/generation_time_sar_ems_seed43_v2/summary.md)与[归档](archive/CURRENT_TASK_generation_time_sar_ems_seed43_20261114.md)。

最近完成：Qwen3 COCO 8:2 三种子 SAR+EMS 对原生 SVAR 的配对图片级显著性检验。结果见 [报告](../outputs/log_s_ours_svar_significance_82/summary.md)，过程见 [归档](archive/CURRENT_TASK_log_s_ours_svar_significance_82_20261114.md)。
