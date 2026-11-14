# COCO FFN 高写入区域对齐实验（2026-11-14）

目标：在 LLaVA-1.5-7B 和 Qwen3-VL-8B 上，逐层选取写入量 `||a_lm||` 最大的 32 个视觉 token，求 `cos(o_ffn(object), hpre(visual_m))` 的均值，并检验 `[I,S,F_top32(I)]` 的幻觉检测效果。

实现：`scripts/run_ffn_cosine_top32_i_811.py`。读取 `gfchair/outputs/coco4000_512_linear_cost_log_s_risk_811/{model}/features.part*.pkl` 的 `source_hpre` 和 `am_norm`，由已有 `F_mean` 和温度 `0.07` 恢复原始逐视觉 token cosine；逐样本/逐层与已有 `F_max` 核对。与 `outputs/ffn_cosine_i_s_811/{model}/compact_features.npz` 按 `(image_id,response_token_idx,target_token_id)` 对齐。

实验：COCO 8:1:1，种子 43/44/45 同时控制图片划分和 MLP 初始化。训练器与原 `[I,S,F_mean]`、`[I,S,F_max]` 实验相同。增加 `[I,S]` 对照；原始 SAR+EMS 仅作参考。两模型分别得到 12,686 和 11,751 个匹配 object token；最大 cosine 恢复误差分别为 `4.83e-08` 和 `1.38e-07`。

结果：LLaVA 的 `[I,S,F_top32(I)]` AUROC 91.75±0.65、Real AUPR 97.20±0.07；对应全视觉均值 91.80±0.85、97.21±0.13。Qwen3 的 Top32 AUROC 89.48±1.27、Real AUPR 96.60±0.44；全视觉均值 89.42±1.14、96.57±0.40。差值很小，没有稳定提升证据。

完整表、逐种子结果和审计：`outputs/ffn_cosine_top32_i_811/complete_results.md`，各模型 `alignment.json`。

验证：`py_compile` 通过；两模型全量 key、标签、层数、非有限值、Top32 cosine≤最大 cosine、三种子 cohort 都核验；12 次检测器训练完成，汇总成功。
