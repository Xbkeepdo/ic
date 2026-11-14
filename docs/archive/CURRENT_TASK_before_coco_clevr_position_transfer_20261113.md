# 当前任务

当前任务：CLEVR-Exist-9K 上 SAR/EMS 预测位置组合对比（已完成）。
LLaVA-1.5-7B 与 Qwen3-VL-8B，Yes-only 物体幻觉标签、固定图片级 8:1:1、种子 43/44/45。
比较 SAR+EMS 均在答案前位置与 SAR 在答案前、EMS 在问题物体前位置。
两方案使用共同有效问题，单隐层 128 MLP，沿用正式 QA 配置；训练集标准化、验证损失选模型、验证 Real-F1 选阈值。
结果：`outputs/qa_benchmarks/clevr_sar_ems_position_ablation_811/complete_results.md`。
逐种子结果：同目录 `per_seed_results.csv`；样本审计：各模型目录 `cohort_audit.json`。
运行脚本：`scripts/train_clevr_sar_ems_position_ablation.py`；`vicr` 环境中两模型 12 个拟合均完成。
测试集 AUROC 物体 EMS 较答案 EMS 分别提高 LLaVA +4.98 pp、Qwen3 +4.63 pp；Qwen3 仅 13 个 Hall 测试样本。

前一份任务状态原文保存在 `docs/archive/CURRENT_TASK_before_clevr_sar_ems_position_20261113.md`。
