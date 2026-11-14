# 当前任务

当前任务：SVAR 的 AMBER existence ↔ CLEVR 检测器迁移（已完成）。
LLaVA-1.5-7B、Qwen3-VL-8B；沿用 Log-S 迁移实验的有效问题与图片级 8:1:1 划分。AMBER 只用 existence；CLEVR 报告完整集和 GT=No 诊断子集。统一采用 `answer_correctness_all`，Real=答案正确。
SVAR 用原生层 5–18 的逐头特征、248 宽单隐藏层 MLP，种子 43/44/45。来源检测器与来源验证集 Real-F1 阈值直接迁移；目标不重新训练、重新标准化或选择阈值。
完整结果：`outputs/qa_benchmarks/amber_exist_clevr_svar_transfer_811/complete_results.md`；逐种子：同目录 `per_seed_results.csv`；队列审计：`cohort_audits.json`；实现：`scripts/run_amber_exist_clevr_svar_transfer.py`。
原域检查点 AUROC 与 F1 已逐种子回放核验；脚本通过 `py_compile`，实验完整跑通。完整 CLEVR 上双向迁移较弱；GT=No 子集仅供标签极性诊断。
首次运行 `python -m scripts.run_amber_exist_clevr_svar_transfer` 因紧凑 SVAR 记录缺少 baseline schema 元数据而失败；补齐必需字段后，同一命令完整跑通。
前一任务状态保存在 `docs/archive/CURRENT_TASK_before_svar_transfer_20261113.md`。
