# 当前任务

当前任务：比较物体位置 SVAR 与 SAR+EMS 的跨数据集迁移，并按用户指定的 Yes-only 标签规则评估；已完成。

模型为 LLaVA-1.5-7B 与 Qwen3-VL-8B；训练/测试划分分别使用既有严格图片级 8:2、8:1:1，训练种子 43/44/45。包含两项独立任务：
- AMBER existence ↔ CLEVR：`answer_correctness_all`；AMBER 无可靠物体位置，双方在 AMBER 均使用 prompt-last，CLEVR 端 SVAR 用 `question_object_pre_token`，SAR+EMS 分别用答案位置的 SAR 与物体位置的 EMS。
- POPE ↔ CLEVR：只保留模型回答 Yes 的问题；GT=Yes 为 Real，GT=No 为 Hall。双方的特征位置与上述 CLEVR 端一致。AMBER existence 的 GT 全是 No，不适用于此双类标签，故按用户要求改用 POPE。

源域训练的标准化、检测器与阈值直接迁移到目标域；目标域不适配。8:1:1 用来源验证集，8:2 不设验证集，阈值由来源训练集 Real-F1 决定。原域检查点的 AUROC 与 Real-F1 均完成逐种子回放核验；两种标签协议均有完整双向迁移结果。

完整报告：`outputs/qa_benchmarks/object_position_svar_vs_sar_ems_transfer_82_811/complete_results.md`。逐种子：同目录 `per_seed_results.csv`；样本和划分审计：`cohort_audits.json`；实现：`scripts/run_object_position_transfer_comparison.py`。已用 `vicr` 环境执行完毕，脚本 `py_compile` 通过，192 条逐种子评估记录完整且无缺值。

主要结论：POPE Yes-only 原域上 SAR+EMS 优于物体位置 SVAR；POPE→CLEVR 的跨域 AUROC 多数接近随机，且 Qwen3 上 SAR+EMS 低于 SVAR；CLEVR→POPE 的 SAR+EMS AUROC 高于 SVAR，但 Hall-F1 多数很低。Qwen3 CLEVR 的 Yes-only Hall 测试样本仅 23（8:2）/13（8:1:1），须同时报告类别计数。AMBER existence 原域近乎完美，不代表跨域有效。

前一任务状态：`docs/archive/CURRENT_TASK_before_object_position_transfer_20261113.md`。
