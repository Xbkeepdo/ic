# 最多 64 轮复检：LLaVA 联合干预

已在同批 LLaVA 种子 43 的 100 图完成共享前向、前缀预填充、SAR 注意力＋EMS FFN 联合干预的最多 64 轮复检。96 图沿用已在 32 轮内清空的结果；4 图重跑，前 32 次动作逐项一致。4 图均在 34–47 轮清空。100 图 CHAIRs 为 30%（32 轮为 31%），CHAIRi 为 8.04%（32 轮为 8.72%），物体召回均为 73.24%；全部图通过检测器，但 30 图仍有 CHAIR 幻觉。多 32 次回滚，逐图耗时合计多约 51.70 秒。

见[实验报告](../../outputs/generation_time_sar_ems_joint_loop_fused_64_seed43/summary.md)、[指标](../../outputs/generation_time_sar_ems_joint_loop_fused_64_seed43/llava_1_5_7b/test_100/metrics.json)与[配对比较](../../outputs/generation_time_sar_ems_joint_loop_fused_64_seed43/paired_comparison.json)。
