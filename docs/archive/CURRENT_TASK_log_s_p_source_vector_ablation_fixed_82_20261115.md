# 固定 8:2 下 EMS 来源向量对照

- 目标：在 LLaVA-1.5-7B、Qwen3-VL-8B 上固定 COCO4000 的 3200/800 图片划分，比较 EMS 的 (P) 由 FFN 更新 (u)、FFN 前残差 (r)、FFN 后残差 (r+u)、attention 输出 (o_A) 构造时的 SAR+EMS 检测效果。
- 协议：种子 43/44/45 控制原版 MLP 初始化、训练样本顺序与 dropout；原版无验证集 8:2 训练器，CUDA 训练；其余特征、样本、K=32、tau=0.03 不变。
- 核对：原始 EMS 与 SAR 缓存逐元素完全一致；原版 (u) 三个种子的测试指标与历史 test_grid 逐项完全一致，所有新训练的配置 JSON 与对应历史原版配置相同。
- 结果：[完整报告](../../outputs/log_s_p_source_vector_ablation_82_fixed_split/complete_results.md)，[运行脚本](../../scripts/run_log_s_p_source_vector_ablation_fixed_82.py)。
