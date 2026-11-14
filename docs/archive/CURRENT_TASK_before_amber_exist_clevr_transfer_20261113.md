# 当前任务

当前任务：COCO ↔ CLEVR SAR/EMS 两种位置组合检测器迁移（已完成）。
LLaVA-1.5-7B、Qwen3-VL-8B；CLEVR 使用 Yes-only 物体幻觉标签；COCO 使用 object-token ENDAC 标签。
CLEVR 检测器分别为 SAR答案+EMS答案 与 SAR答案+EMS物体，同样的共享 CLEVR 测试 cohort；COCO 为原 object-token SAR+EMS 检测器。
三种子 43/44/45、K=32、τ=0.03；保存的权重、来源训练 scaler 和来源验证阈值直接跨数据集应用，目标测试集没有训练或选阈值。
报告：`outputs/coco_clevr_sar_ems_position_transfer/complete_results.md`。
逐种子：同目录 `per_seed_results.csv`；检查点回放审计：`replay_audit.json`。
实现：`scripts/evaluate_coco_clevr_sar_ems_position_transfer.py`；`vicr` 环境运行完成。
两模型两个方向 AUROC 均明显低于各自目标域内检测器；Qwen3 CLEVR 测试集仅 13 个 Hall。

前一任务状态原文：`docs/archive/CURRENT_TASK_before_coco_clevr_position_transfer_20261113.md`。
