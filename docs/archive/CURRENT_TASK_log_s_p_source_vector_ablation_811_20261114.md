# COCO4000 EMS 源分布向量对照

已完成：在 LLaVA-1.5-7B 与 Qwen3-VL-8B 上比较四种构造 $P_m=\operatorname{softmax}_m(\cos(x,v_m)/0.03)$ 的向量：原始 FFN 输出 $u$、FFN 前残差 $r$、FFN 后残差 $r+u$、attention 输出 $o_A$。每个 softmax 仅在当前图片的视觉 token 上求。SAR、log-s 门控、$T$、cost、Top16(P)∪Top16(T)、K=32、训练器及 8:1:1 图片划分均不变，种子 43/44/45 同时控制划分和 MLP 初始化。

提取脚本：`gfchair/scripts/extract_log_s_source_vectors_811.py`；训练脚本：`scripts/run_log_s_p_source_vector_ablation_811.py`。完整结果见 [报告](../../outputs/log_s_p_source_vector_ablation_811/complete_results.md)，逐种子精度见同目录 `per_seed.csv`。

LLaVA 的原始 $u$ AUROC 为 91.61%，$r$ 为 91.31%，$r+u$ 为 91.16%，$o_A$ 为 91.75%；Qwen3 对应为 88.82%、88.16%、88.18%、89.24%。两模型的原始 $u/r$ 风险与旧缓存逐层最大绝对误差均为 0，样本与标签完全对齐。
