# COCO4000 严格 8:2：SAR+E_A 与 SAR+E_F

已完成：LLaVA-1.5-7B、Qwen3-VL-8B；种子 43/44/45 同时控制 3200/800 图片划分和 MLP 初始化。对比由 attention 输出构造源分布的 $E_A$ 与原 FFN 输出构造的 $E_F$，其他特征及 $K=32,\tau=0.03$ 不变。沿用原版严格 8:2 的 [EMS,SAR] 输入顺序、无验证集、最小训练损失 checkpoint、训练集 F1 阈值。

脚本：`scripts/run_log_s_p_source_vector_ablation_82.py`；[完整结果与逐种子表](../../outputs/log_s_p_source_vector_ablation_82/complete_results.md)。复用原版 $E_F$ 三种子结果，新增 $E_A$ 三种子训练。

AUROC（均值±总体标准差）：LLaVA $E_F$ 90.96±0.46%，$E_A$ 90.85±0.43%；Qwen3 $E_F$ 89.04±0.25%，$E_A$ 89.25±0.53%。Real/Hall AUPR、F1 和 accuracy 见结果表。

检查：原 8:2 的 FFN-EMS 与 SAR 与当前缓存逐元素误差均为 0；同种子训练配置完全一致，原报告和新训练的训练/测试 token 数及测试幻觉数逐项相同；脚本 `py_compile` 与 `git diff --check` 通过。
