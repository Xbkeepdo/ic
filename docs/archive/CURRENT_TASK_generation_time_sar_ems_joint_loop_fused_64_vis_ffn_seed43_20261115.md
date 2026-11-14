# SAR 注意力＋EMS FFN 联合干预：FFN 纯视觉贡献版

同批 LLaVA 种子 43、100 图上，保持 SAR 注意力增强、检测阈值、最多 64 轮复检和共享前向不变，仅把 FFN 的 `+0.4 × 完整注意力输出` 替换成图像 token 注意力贡献，并将加入向量范数设为 FFN 输出范数的 40%。34 图未触发而复用，66 图重跑，首次风险词和 FFN 选层与旧版一致。

结果：CHAIRs 仍为 30%，幻觉物体提及仍为 41 次；CHAIRi 8.04%→7.95% 来自物体提及分母 510→516；物体召回 73.24%→73.18%；回滚 527→575 次，逐图耗时合计 895.38→906.56 秒。4 图修复、4 图新增幻觉，未见明确收益。纯视觉方向与注入范数同时改变，仍需等范数方向对照才能单独判断方向作用。

见[报告](../../outputs/generation_time_sar_ems_joint_loop_fused_64_vis_ffn_seed43/summary.md)、[指标](../../outputs/generation_time_sar_ems_joint_loop_fused_64_vis_ffn_seed43/llava_1_5_7b/test_100/metrics.json)与[配对明细](../../outputs/generation_time_sar_ems_joint_loop_fused_64_vis_ffn_seed43/comparison.json)。
