# 当前任务

目标：在 COCO 4,000 张图像、随机种子 43/44/45 同时控制图片划分与 MLP 初始化的 8:1:1 设置下，对 LLaVA-1.5-7B 和 Qwen3-VL-8B 比较原版 [SAR, EMS] 和移除视觉来源对应关系的 [I(1+mean_m(s_m)), EMS]。其中 s_m=log(1+C p_m)，原版 SAR=I+sum_m I_m s_m。其余 object token、标签、EMS K=32、tau=0.03、训练器完全一致。

已核对：outputs/norm_semantic_transform_curves_by_label/<model>/compact_features.npz 中的 log_residual_fusion 与 outputs/log_s_mechanism_ablation_811/<model>/compact_features.npz 中 SAR 逐元素完全相同，且图片 ID、标签逐行相同。之前的 raw 概率缓存与 SAR 不一致，因此使用这个 final-norm 缓存计算 mean_m(s_m)=log_support_semantic_sum/visual_tokens。视觉 token 数 LLaVA 固定 576、Qwen3 为 70–400。原参考结果和种子划分来自 outputs/log_s_ems_i_mean_svar_seeded_811_43_45/。

实验已完成。脚本 scripts/run_log_s_source_correspondence_ablation_811.py 已通过语法检查和两模型特征同源核验；两模型各三个种子的训练已分别在本地 GPU 0/1 完成。完整报告 outputs/log_s_source_correspondence_ablation_811/complete_results.md，逐种子结果 per_seed.csv。六组比较的模型配置、train/val/test token 数和 val/test 标签逐项一致，所有测试分数有限。原版相对去对应版的 AUROC / Real AUPR 差值（百分点）：LLaVA +0.72 / +0.24；Qwen3 -0.13 / -0.07。此处仅支持 LLaVA 上的增益，Qwen3 未体现；不把小幅差异解释为显著性结论。

上一任务已归档到 docs/archive/CURRENT_TASK_before_log_s_source_correspondence_ablation_20261114.md。
