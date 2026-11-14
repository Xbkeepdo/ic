# 当前任务

当前任务：SVAR 的 AMBER existence ↔ CLEVR 严格图片级 8:2 检测器迁移（已完成）。
LLaVA-1.5-7B、Qwen3-VL-8B；沿用此前 8:1:1 实验的有效问题，改用项目现有的外层 8:2 图片划分（seed 42）。AMBER 只用 existence，CLEVR 报告完整集和 GT=No 诊断子集；`answer_correctness_all`，Real=答案正确。
SVAR 使用原生层 5–18 的逐头特征、248 宽单隐藏层 MLP。种子 43/44/45 控制训练；严格 8:2 无验证集，训练满 50 epoch、最后一轮权重、来源训练集 Real-F1 阈值。目标不重训、不重新标准化、不调阈值。
完整结果：`outputs/qa_benchmarks/amber_exist_clevr_svar_transfer_82/complete_results.md`；逐种子：同目录 `per_seed_results.csv`；队列审计：`cohort_audits.json`；实现：`scripts/run_amber_exist_clevr_svar_transfer_82.py`。
原域检查点 AUROC 与 F1 已逐种子回放核验；脚本通过 `py_compile`，实验完整跑通。完整 CLEVR 的双向迁移仍较弱；Qwen3 在完整 CLEVR 测试集的 Hall-F1 为 0。GT=No 仅供诊断。
初次运行因 Qwen3 CLEVR 缺少通用名 `qa_probe_splits.json` 而中断；改用现有同协议 `probe_splits.prompt_last_token.json` 后完整跑通。
前一任务状态保存在 `docs/archive/CURRENT_TASK_before_svar_transfer_82_20261113.md`。
