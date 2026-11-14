# 当前任务

当前任务：AMBER existence ↔ CLEVR Log-S 检测器双向迁移（已完成）。
LLaVA-1.5-7B、Qwen3-VL-8B；AMBER 只取 existence 问题，CLEVR 报告全部有效问题和 GT=No 对齐诊断。
AMBER existence Yes-only 全是 Hall，无法训练二分类，因此两边统一使用 answer_correctness_all；Real=答案正确。
CLEVR 有 SAR答案+EMS答案、SAR答案+EMS物体两种构造；AMBER 只有答案位置。K=32、τ=0.03、种子 43/44/45。
各域保存的单隐层检测器、来源训练 scaler、来源验证阈值直接迁移，目标无重训或调阈值。
完整结果：`outputs/qa_benchmarks/amber_exist_clevr_position_transfer_811/complete_results.md`。
逐种子：同目录 `per_seed_results.csv`；回放审计：`replay_audit.json`；实现：`scripts/run_amber_exist_clevr_position_transfer.py`。
AMBER existence 全部 GT=No，CLEVR GT=No 对齐子集也满足 Real 当且仅当预测 No；两子集上的满分不能证明一般跨域检测能力。完整 CLEVR 的迁移 AUROC 明显下降。
前一份任务状态原文：`docs/archive/CURRENT_TASK_before_amber_exist_clevr_transfer_20261113.md`。
