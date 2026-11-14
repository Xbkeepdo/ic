# 多轮回滚前缀预填充试验与检测/规划共享前向设计

任务：按用户建议避免每轮重生成不变前缀，并解释如何合并 SAR+EMS 检测与干预规划中重复的模型前向。

实现：[脚本](../../scripts/generation_time_sar_ems_joint_loop_seed43.py)添加 `--prefix-prefill`。用图片、原始提示与当前风险词前缀一次性预填充，`max_new_tokens` 减去前缀长度，只从回滚位置续写；原阈值 0.4013161、联合注意力与 FFN 公式、风险首 token 禁选及最多 32 轮均保留。输出独立于原实验。

结果：同批 100 图、两个 GPU。逐图耗时合计 2032.97→1082.98 秒；重生成阶段 1429.51→427.70 秒。最终 caption 92/100 图逐 token 相同，95/100 图回滚轮数相同；因浮点数值路径变化，不能称为严格等价。原版与预填充版的 32 轮 CHAIRs 为 32%/31%，CHAIRi 为 9.09%/8.90%，检测器清空 96/95 图。详见[完整报告](../../outputs/generation_time_sar_ems_joint_loop_prefix_seed43/summary.md)、[指标](../../outputs/generation_time_sar_ems_joint_loop_prefix_seed43/llava_1_5_7b/test_100/metrics.json)及[逐图比较](../../outputs/generation_time_sar_ems_joint_loop_prefix_seed43/comparison.json)。

检测/规划共享前向设计：当前检测器的 `extract_token_features_batch` 一次教师强制前向已计算全部物体词的逐层 SAR/EMS，但规划器对选中词前缀再次捕获。可在检测 forward 中保留 captures，直接复用选中词 `dgst_t_result` 的 EMS 排层，并从同一 captures 的视觉隐藏状态生成或缓存类别区域分数；wrapper 当前在返回前释放 captures，需增加内部共享返回值或回调。生成的 KV 预填充仍单独需要，合并前先核对风险分数、选层、patch 集合和 caption。

验证：脚本语法编译、1/3/9 轮样例对照、100 图完整生成与评估、逐图 token 和耗时配对比较完成。没有改动已存在的 `models/llava_wrapper.py` 工作区修改。
