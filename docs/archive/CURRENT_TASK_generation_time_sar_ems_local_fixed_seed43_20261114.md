# 固定系数 SAR 局部注意力与 EMS FFN 干预

任务：按用户要求，将上一轮触发即干预的逐层系数改为固定值，在同批 LLaVA-1.5-7B 种子 43 的 COCO 前 100 图上复测。

实现：检测阈值 0.4013161，66 图触发，首风险词之后最多 12 步。第 5–18 层的所选图像区域在注意力 logits 上统一加 `ln(1.7)`；按 EMS 偏离选出的 3 个 FFN 层均加入范数为 FFN 输出 20% 的视觉注意力向量。保留仅注意力、仅 FFN、联合三组，并与贪心和上一轮逐层系数配对比较。[脚本](../../scripts/generation_time_sar_ems_local_intervention_seed43.py)使用 `--fixed-coefficients`。

结果：100 图全部完成，66 张触发图的联合分支均实际改写注意力和 FFN。联合改写 20 图，CHAIRs 48%→47%、CHAIRi 14.75%→15.74%、论文式总体物体 F1 75.36%→74.68%；相对上一轮联合 CHAIRs 46%、CHAIRi 15.38% 更差。逐图图像核对发现图 76731 在 CHAIR 上修复却仍编造词表外物体，图 49733 和 89395 新增虚构物体。详情见[完整报告](../../outputs/generation_time_sar_ems_local_fixed_seed43/summary.md)、[指标](../../outputs/generation_time_sar_ems_local_fixed_seed43/llava_1_5_7b/test_100/metrics.json)和[配对统计](../../outputs/generation_time_sar_ems_local_fixed_seed43/paired_stats.json)。

验证：脚本通过 `py_compile`；两个 GPU 分段生成 100 个不同图片 ID；`evaluate --fixed-coefficients` 中的 66 图实际改写断言通过；图片级配对 bootstrap 10,000 次完成。研究结论限于同一批探索图和本轮参数。
