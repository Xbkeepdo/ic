# Attention 与 FFN 来源 EMS 的联合对照

已完成：在 LLaVA-1.5-7B 和 Qwen3-VL-8B 的 COCO4000 ENDAC 8:1:1 图片划分、种子 43/44/45 下比较五组检测器输入：$SAR+E_A$、$SAR+E_F$、$SAR+E_A+E_F$、$SAR+E_A+E_{pre}$，以及维度与参数量匹配的重复输入对照 $SAR+E_A+E_A$。复用已提取风险，固定 $K=32$、$\tau=0.03$、$T$、cost 和原训练配置。

结果：[完整表与逐种子数据](../../outputs/log_s_ems_source_union_811/complete_results.md)；[解释和配对差值](../../outputs/log_s_ems_source_union_811/analysis.md)。脚本：`scripts/run_log_s_ems_source_union_811.py`。

LLaVA 中 $SAR+E_A+E_F$ 的 AUROC 为 91.77%，与 $SAR+E_A$ 的 91.75% 基本相同；Qwen3 分别为 89.88% 和 89.24%。Qwen3 的 $SAR+E_A+E_{pre}$ 已达 89.77%，故不能把增益明确归为 FFN 独有信息。

检查：五组输入在每个种子上的验证/测试标签与顺序完全一致；新训练组的隐藏层、标准化、dropout、batch size、优化器、学习率、权重衰减和 checkpoint 选择与原训练器一致。脚本 `py_compile` 与 `git diff --check` 通过。
