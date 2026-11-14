# 当前任务

当前任务：LLaVA/Qwen3 8:2 下全部六种 JS 与 OT 的显著性比较（已完成）。
复用 43/44/45 的 3200/800 图片划分、OT 和已训练的 Top16 排名 JS+R_q。
训练其余 JS、核对测试分数，逐图 Bootstrap 3,000 次；见 `outputs/log_s_ot_js_all_variants_seeded_82_43_45/`。
前次 8:1:1 对比：`outputs/log_s_js_support_variants_seeded_811_43_45/`。
前次 final-only 对比：`outputs/llava_final_only_all_visual_4000/complete_results.md`。
前次 raw+final norm/SVAR 报告：
`outputs/complexity_k032_50_all_visual_vocab_vs_svar/complete_results.md`。

## 已完成的前一任务

原始概率读取分块 32/64/128/256 速度与显存对比（已完成）：
保留 `target_probabilities` 原实现和同一 50 图样本，各分块复用同一次
teacher-forced 捕获；轮换分块运行顺序，核对 norm 概率数值，测概率阶段
wall time、峰值 allocated/reserved 显存。脚本：
`scripts/benchmark_k32_vocab_chunk_sweep.py`；结果目录：
`outputs/complexity_k032_50_vocab_chunk_sweep/`。
报告：`outputs/complexity_k032_50_vocab_chunk_sweep/complete_results.md`。
两模型各 50 图完成；LLaVA 分块 128 概率读取 93.8 ms/图（32 为 243.2），
Qwen3 分块 256 为 274.6 ms/图（32 为 515.3）；均无 OOM。
较大分块引起少量概率浮点差异，尚未重跑完整 SAR+EMS 检测结果。

## 更早的已完成任务

原始语义概率读取的投影与 `logsumexp` 分段计时（已完成）：
沿用上次相同 50 张 COCO 图、原始 raw+norm 词表计算和 `vicr` 环境，
用 CUDA event 分别测完整词表线性投影、FP32 转换、`logsumexp`；
记录整段 wall/GPU 时间并在预热图与原函数逐值核对。
脚本：`scripts/benchmark_k32_semantic_probability_stages.py`；结果目录：
`outputs/complexity_k032_50_semantic_probability_profile/`。
两模型 50 图已完成、预热图与原函数逐值一致；投影分别为
200.23/464.26 ms/图，`logsumexp` 为 21.57/27.33 ms/图。
Qwen3 首轮 event 误绑 GPU 0 的无效 JSON 已隔离，重跑结果有效。

## 历史已完成任务

原始 K=32 SAR+EMS 分阶段计时（已完成）：使用同一 50 图样本、
`vicr` 环境和原始提取数学，分别测语义概率、SAR 写入、EMS 分布、
完整视觉代价及问题准备、精确 OT 求解；在相同 Top-16/Top-16 并集上测
JS 的支持集准备与散度计算。LLaVA、Qwen3 分别使用本地 GPU 0、1。
脚本：`scripts/benchmark_k32_sar_ems_js_ot_stages.py`；结果目录：
`outputs/complexity_k032_50_stage_profile/`。
完整报告：`outputs/complexity_k032_50_stage_profile/complete_results.md`。
两模型各完成 50 图；核对样本 ID 和 object token 数与前次基准一致。
独立 SAR/EMS 和 JS/OT 各计时边界见报告，原始逐图数据见模型 JSON。

## 既往已完成任务

K=32 SAR+EMS 特征提取优化对比（已完成）：保持既有 50 图基准脚本不变，
在 `scripts/benchmark_k32_feature_optimization_branch.py` 中比较原实现、
仅 final-norm 概率、并集后局部代价、仅视觉 token 写入四条累计路径。
两模型共享 50 张 COCO 图，LLaVA 用本地 GPU 0、Qwen3 用 GPU 1；
逐 object token / layer 核对 EMS risk 和 SAR 的数值，输出至
`outputs/complexity_k032_50_optimization_branch/`。
报告：`outputs/complexity_k032_50_optimization_branch/complete_results.md`。
三项合用后的端到端计时：LLaVA 0.528→0.485 秒/图，Qwen3 0.810→0.772 秒/图；
特征存在小幅浮点差异，报告列出最大/平均误差及各步增量误差。

## 之前已完成任务

TruthPrInt 幻觉检测 baseline（已完成）。方法来源：
`github.com/jinhaoduan/TruthPrInt` commit `c61af74`。四模型使用目标词前
第 16 层隐藏状态和 128/64 ReLU MLP，种子 43/44/45。训练集内平衡类别，
保持原测试集比例；域内适配关闭跨域 subspace alignment。

- [x] COCO object-token：四模型 8:1:1 和 8:2，精确 token key 对齐。
- [x] CLEVR-Exist-9K：`prompt_last_token`、`answer_correctness_all`。
- [x] AMBER：整体及 existence、attribute、relation 三类。
- [x] 全量提取、三种子训练、测试集评估与同划分 Log-S 对比。
- [x] 第 16 层、因果位置、图片划分、标签和结果口径审计。

结果：`outputs/truthprint_baseline/complete_results.md`、
`outputs/truthprint_baseline/qa_comparison.md`、
`outputs/truthprint_baseline/coco_results.md`、
`outputs/truthprint_baseline/coco_82_comparison.md`。
实现：`scripts/extract_truthprint_states.py`、`scripts/train_truthprint.py`、
`scripts/train_truthprint_82.py`、`scripts/summarize_truthprint.py`。

此前任务记录：`docs/archive/CURRENT_TASK_before_truthprint_20260925.md`。
