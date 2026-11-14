# 当前任务

目标：在 COCO 4000 张图像、Qwen3-VL-8B 的 8:2 设置下，用种子 43/44/45 同时控制图片划分和 MLP 初始化，比较原版 [EMS,SAR] 与移除视觉来源对应关系的 [EMS,I(1+mean_m(s_m))]。两版保持同一 object token、标签、EMS K=32、tau=0.03 和检测器配置。

原版 8:2 结果与 split 在 outputs/coco4000_seeded_image_split_auc_aupr/。去对应 SAR 的特征准备复用刚完成的 8:1:1 同源审计；最终输入顺序按 8:2 原版的 [EMS,SAR] 约定排列。8:2 无独立验证集，checkpoint 依最小训练损失，F1 阈值在训练集选取；AUROC/AUPR 无阈值。

实验已完成。脚本 scripts/run_log_s_source_correspondence_ablation_82.py 通过语法检查，Qwen3 的三个种子训练在本地 GPU 0 完成。完整结果和逐种子值见 outputs/log_s_source_correspondence_ablation_82/complete_results.md、per_seed.csv。三个种子的原版与去对应版检测器配置逐项相同，特征行顺序和公式锚点经前次 8:1:1 审计；8:2 的输入列顺序严格对齐原版 [EMS,SAR]。测试集 AUROC 原版 89.04±0.25，去对应版 88.95±0.15；Real AUPR 分别 96.38±0.24、96.41±0.13，差异很小。

上一任务记录见 docs/archive/CURRENT_TASK_log_s_source_correspondence_ablation_20261114.md。
