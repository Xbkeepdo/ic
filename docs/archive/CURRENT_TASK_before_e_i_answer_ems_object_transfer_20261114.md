# 当前任务

当前任务：仅用 POPE Popular 子集训练，与 CLEVR 进行双向 Yes-only 迁移对比；已完成。

沿用上一轮 LLaVA-1.5-7B、Qwen3-VL-8B、固定图片级 8:2 和 8:1:1 划分、训练种子 43/44/45。只保留模型回答 Yes 的问题，GT=Yes 是 Real，GT=No 是 Hall。POPE 原始 Popular 子集有 3,000 题；CLEVR 使用之前的完整 Yes-only 队列。SVAR 使用物体位置，SAR+EMS 拼接答案位置的 SAR 与物体位置的 EMS。源域标准化、模型和阈值直接迁移，目标域不调参。CLEVR 检测器复用上一轮相同队列和检查点，POPE 检测器按 Popular 重训。

完整结果：`outputs/qa_benchmarks/pope_popular_clevr_yes_only_transfer_82_811/complete_results.md`；逐种子 `per_seed_results.csv`；样本/划分审计 `cohort_audits.json`；实现 `scripts/run_pope_popular_clevr_transfer.py`。在 `vicr` 环境完成 96 条逐种子评估记录，无缺值；原域检查点 AUROC/Real-F1 已回放核验；脚本通过 `py_compile`。报告还在相同 CLEVR 测试集上比较 POPE Popular 与 POPE 全三子集训练。

主要结论：Popular→CLEVR 下 LLaVA 的 SAR+EMS AUROC 约 49%，Qwen3 的 SAR+EMS 为 36.62%（8:2）/43.87%（8:1:1），低于物体位置 SVAR。CLEVR→Popular 的 SAR+EMS AUROC 高于 SVAR，但 Hall-F1 多数接近零。Popular 的 Yes-only 8:1:1 测试集仅有 LLaVA 13、Qwen3 10 个 Hall，需要报告类别计数。

前一任务状态：`docs/archive/CURRENT_TASK_before_pope_popular_transfer_20261113.md`。
