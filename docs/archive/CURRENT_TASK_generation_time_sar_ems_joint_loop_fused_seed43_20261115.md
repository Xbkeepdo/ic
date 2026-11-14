# 检测与干预规划共享前向

任务：回应用户“现在已经优化完了吗”：前缀预填充已完成，继续实现检测与规划的重复前向合并，并在同批 LLaVA 100 图评估速度、数值和幻觉指标。

实现：[脚本](../../scripts/generation_time_sar_ems_joint_loop_seed43.py)新增 `--fused-forward`，每轮在完整 caption 教师强制捕获前向中同时计算所有物体词风险、首风险词 EMS 选层和视觉 top16 区域；回滚生成仍用前缀预填充及 KV 缓存。模型阈值、联合干预公式、token 禁选和最多 32 轮不变。未修改已有 `models/llava_wrapper.py` 工作区改动。

结果：同批 100 图，逐图时间合计最初逐 token 重放 2032.97 秒、前缀预填充 1082.98 秒、共享前向 843.68 秒；后者平均 8.44 秒/图。共享前向相对前缀版减少 22.1%，在最终 caption 完全一致的 94 图上减少 18.5%。100 图初始触发状态一致、66 张触发图的首风险词分数一致，99 图首次 FFN 三层集合一致；最终 caption 与前缀版 94 图相同。共享前向版最多 32 轮 CHAIRs 31%、CHAIRi 8.72%、物体召回 73.24%、检测器清空 96 图。详见[报告](../../outputs/generation_time_sar_ems_joint_loop_fused_seed43/summary.md)、[指标](../../outputs/generation_time_sar_ems_joint_loop_fused_seed43/llava_1_5_7b/test_100/metrics.json)、[配对数据](../../outputs/generation_time_sar_ems_joint_loop_fused_seed43/comparison.json)。

验证：100 个不同图片 ID、66 图触发、495 次回滚；多轮样例 1/3/9 轮核对；四张图视觉区域总交集 1343/1344；两个 GPU 分段生成、`evaluate --fused-forward` 与脚本语法编译通过。完整 caption 与独立前缀前向存在少量浮点差异，不能作为逐 token 严格等价替换。
