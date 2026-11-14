# 绝对值平均注意力 logit 与完整注意力输出联合干预

任务：按用户纠正，在 LLaVA-1.5-7B 种子 43 的同批 100 张 COCO 图上，把选中图像区域的注意力 logit 设为原值加 `0.7 × 逐头全部图像 token 原始 logit 绝对值的均值`，把 EMS 所选 3 层的 FFN 输出设为原值加 `0.4 × 同层完整注意力输出`。只跑联合干预，对比首个风险词处理一次与新 caption 反复检测、逐词回滚，最多 32 轮。

实现：风险阈值 0.4013161063，66 图触发。两个分支共享第一次回滚并在该词的精确 token 前缀下排除该词首 token，联合修改持续最多 12 个生成步；多轮分支每轮重新计算 SAR+EMS 风险、选择首个超阈值词，累积前缀禁选，直到清空或达到轮次上限。用户纠正前的有符号均值试跑已停止，未并入正式数据。[脚本](../../scripts/generation_time_sar_ems_joint_loop_seed43.py)

结果：100 图全部完成；仅一次处理 CHAIRs 49%、CHAIRi 14.62%、物体召回 78.17%，检测器清空 49 图；最多 32 轮处理 CHAIRs 32%、CHAIRi 9.09%、物体召回 73.53%，清空 96 图。多轮共 536 次回滚，4 图达到 32 轮仍有风险。原始贪心 CHAIRs 48%、CHAIRi 14.75%、物体召回 80.54%。多轮相对一次处理的 CHAIRs 差值为 −17 个百分点（95% 配对 bootstrap 区间 [−25,−10]），召回差值 −4.64 个百分点（区间 [−7.56,−1.98]）。详见[完整报告](../../outputs/generation_time_sar_ems_joint_loop_absmean_seed43/summary.md)、[指标](../../outputs/generation_time_sar_ems_joint_loop_absmean_seed43/llava_1_5_7b/test_100/metrics.json)及[配对统计](../../outputs/generation_time_sar_ems_joint_loop_absmean_seed43/paired_stats.json)。

验证：100 个不同图片 ID，66 图触发，536 次回滚；全部回滚同时触发注意力和 FFN hook；脚本通过 `py_compile`，两个 GPU 分段运行及评估完成。收益包含前缀 token 禁选，不能单独归因于中间层联合修改。
