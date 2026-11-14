# 检测门控强残差干预：种子 43、LLaVA 前 100 张测试图

按用户要求先停止无检测门控的两模型 200 图实验；已有部分输出保留在 `outputs/generation_time_curve_residual_seed43/`，不作为本次结果。新实验仅用 LLaVA-1.5-7B、种子 43 的前 100 张测试图，先由现有 SAR+EMS 检测器触发，再从首个高风险物体词的预测位置起增强注意力残差或削弱 FFN 残差。强系数的逐层形状来自训练集曲线差，最大 $\alpha=1.7$，最小 $\beta=0.2$；测试标签仅用于评测。

100 图中 22 图触发。门控联合干预的 CHAIRs 48% 与原始贪心 48% 相同，CHAIRi 14.77% 与原始 14.75% 接近；仅注意力 CHAIRi 升到 16.77%，配对 95% CI 的增量为 [+0.75,+3.56] 个百分点。物体召回和词数接近原始值。完整方案、逐图变化、配对区间和局限见 [报告](../../outputs/generation_time_gated_residual_seed43/summary.md)，代码见 [实验脚本](../../scripts/generation_time_gated_residual_seed43.py)。

验证：所有 100 图、三方法均生成或复用原始输出；22 个触发图全部改变输出，生成前缀逐 token 一致；`python -m py_compile` 与 CHAIR 评测完成。评测使用 `NLTK_DATA=/home/apulis-dev/userdata/nltk_data`。
