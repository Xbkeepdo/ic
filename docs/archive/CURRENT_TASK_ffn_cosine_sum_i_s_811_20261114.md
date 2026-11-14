# COCO FFN 视觉 cosine 直接求和（2026-11-14）

目标：在 LLaVA-1.5-7B 和 Qwen3-VL-8B 上，以 `F_sum,l=Σ_m cos(o_ffn(object),hpre(visual_m))` 替换原 `F_mean,l`，比较 `[I,S,F_sum]` 和 `[I,S,F_mean]` 的幻觉检测效果。

实现：`scripts/run_ffn_cosine_sum_i_s_811.py`。复用原始全视觉 cosine 均值，按每条样本的视觉 token 数 `M` 计算 `M*F_mean`，与直接求和数学等价；LLaVA 的全量审计给出恒定 `M=576`，Qwen3 从原始 `source_hpre` 矩阵按 object token 键读取，`M=70–400`。保持 I、S、同一 ENDAC object token、三种子 43/44/45 的 8:1:1 图片划分、MLP 训练和评价不变。

结果：LLaVA 的 `[I,S,F_sum]` AUROC `91.81±0.85`、Real AUPR `97.21±0.13`，原均值版为 `91.80±0.85`、`97.21±0.13`。Qwen3 的求和版为 `89.48±1.20`、`96.61±0.40`，均值版为 `89.42±1.14`、`96.57±0.40`。差异很小。Qwen3 的视觉 token 数单独预测 Real 的全量 AUROC 为 `0.4974`，Real/Hall 平均 token 数为 `270.35/270.80`，其数量本身未显示标签区分能力，但求和仍会改变各样本的特征尺度。

完整结果：`outputs/ffn_cosine_sum_i_s_811/complete_results.md`；逐种子指标：同目录 `per_seed.csv`；来源及样本数审计：各模型 `alignment.json`。

验证：脚本 `py_compile` 通过；两模型目标键、标签、层数、视觉 token 数范围及三种子 cohort 一致性检查通过；6 次检测器训练和汇总成功。
