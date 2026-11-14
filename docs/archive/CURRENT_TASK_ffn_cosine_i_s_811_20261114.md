# 当前任务

目标：在 LLaVA-1.5-7B 和 Qwen3-VL-8B 的 COCO4000 8:1:1 设置下比较逐层 [I,S,F_mean] 与 [I,S,F_max] 两种检测器。I=sum_m ||a_m||，S=sum_m P(target|final_norm(hpre_visual_m))，F_mean/F_max 分别是物体 token 的 FFN 输出与全部视觉 hpre 状态的原始余弦均值/最大值。种子 43/44/45 同时控制 3200/400/400 图片划分与 MLP 初始化；不拼接 SAR、EMS。

实验已完成。gfchair 的 scripts/extract_ffn_cosine_mean_max_endac_811.py 完成 LLaVA/Qwen3 全量提取（12686/11751 object token）。下游脚本 scripts/run_ffn_cosine_is_811.py 通过语法检查，并按 (image_id,response_token_idx,target_token_id) 与机制缓存精确对齐；12 次训练结果齐全，同种子和原 SAR+EMS 参照的训练配置、val/test 标签及各 split token 数逐项一致，测试分数有限。完整报告在 outputs/ffn_cosine_i_s_811/complete_results.md，逐种子值在 per_seed.csv。AUROC：LLaVA F_mean 91.80±0.85、F_max 91.45±0.42，Qwen3 F_mean 89.42±1.14、F_max 89.23±0.97；原 SAR+EMS 为 91.61±0.52、88.82±0.96。所有值是三个种子的测试集均值±总体标准差，单位百分数。

最近完成的 B+EMS 消融归档在 docs/archive/CURRENT_TASK_log_s_b_ems_ablation_811_20261114.md。
