# COCO4000 固定 8:1:1：EMS 源向量对照

已完成：LLaVA-1.5-7B、Qwen3-VL-8B；固定同一份图片级 3200/400/400 训练/验证/测试划分，种子 43/44/45 沿用原训练器同时控制 MLP 初始化、训练样本顺序和 dropout。比较构造 EMS 源分布 $P$ 的四个向量 $u,r,r+u,o_A$；SAR、$T$、cost、$K=32$、$\tau=0.03$ 等均不变。

复用已有固定划分的 $u/r$ 实验，并用相同训练器补训 $r+u/o_A$。脚本：`scripts/run_log_s_p_source_vector_ablation_fixed_811.py`；[完整结果及逐种子表](../../outputs/log_s_p_source_vector_ablation_811_fixed_split/complete_results.md)。

AUROC 均值±总体标准差：LLaVA 的 $u$ 92.07±0.14%，$o_A$ 91.55±0.04%；Qwen3 的 $u$ 90.44±0.09%，$o_A$ 90.17±0.27%。在固定测试集下，原版 $u$ 在两个模型均领先。

验证：新缓存的 $u/r$ 与原缓存逐层完全相同；全部四组在同一种子下的验证/测试标签及顺序完全一致；两模型的测试 token 数分别固定为 1280 和 1171。`py_compile` 与 `git diff --check` 通过。
