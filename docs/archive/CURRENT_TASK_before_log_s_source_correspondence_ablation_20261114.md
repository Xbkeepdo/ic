# 当前任务

当前目标：评估答案位置的语义概率和 $E_l=\sum_m p_{l,m}$、视觉写入量 $I_l=\sum_m\|a_{l,m}\|_2$，与物体位置 EMS 拼接后的跨数据集迁移效果。沿用上轮 LLaVA-1.5-7B / Qwen3-VL-8B、种子 43/44/45 和相同目标测试队列：8:2 CLEVR→AMBER Existence（答案正确性标签）；8:1:1 CLEVR→POPE 全子集（仅回答 Yes，GT Yes=Real、GT No=Hall）。同域结果作为参照；与原 SAR+EMS 和 SVAR 并列。AMBER 无可靠物体 token，EMS 按上轮协议作用于答案位置；报告必须说明。

现有 QA pickle 只存 EMS 与 SAR，没有 E/I。已在 `features/logs_risk_f_log.py` 将 E 求和、E 平均及 I 求和加入现有逐层计算，并在 `features/qa_extractor.py` 保存；原 EMS/SAR 算式不改。LLaVA 的单条 POPE 特征审计：EMS 与 SAR 和旧缓存逐值完全一致，新增三个向量为有限值且层数一致。新配置 `configs/model_configs_qa_logs_e_i_answer.yaml` 只提取答案位置方法特征，不重复跑 baseline。筛选输入在 `outputs/qa_benchmarks/LOGS-K32-TAU003-EI-ANSWER/prepared/`，复用旧 generation 和标签。

两个本地 GPU 提取任务已完成：`scripts/run_qa_e_i_answer_extract.sh` 分别运行 LLaVA/cuda:0 和 Qwen3/cuda:1，顺序为 CLEVR、POPE、AMBER；六份 QA 缓存齐全、日志无报错。旧远端 SSH 端口 32081、31069、32138 不可达，故未使用远端。

实验已完成：`scripts/run_e_i_answer_ems_object_transfer.py` 逐样本对齐旧 EMS，构造 `[EMS_object,E_answer,I_answer]`；AMBER 使用 `[EMS_answer,E_answer,I_answer]`。复用旧 split、MLP、来源域标准化及阈值规则。完整报告在 `outputs/qa_benchmarks/e_i_answer_ems_object_transfer_82_811/complete_results.md`，无标准差简表 `rebuttal_table_auc_real_aupr.md`，逐种子 `per_seed_results.csv`，特征及队列审计 `feature_alignment_audits.json`、`cohort_audits.json`。六组缓存 34,740 条，新增答案位置 EMS 与旧缓存最大绝对差 0；新旧队列哈希一致，所有向量有限，24 个模型/种子训练结果齐全，36 条新方法评估结果齐全。原方法 SAR+EMS 为了避免跨域 sigmoid(float32) 饱和，也用原 checkpoint 的 raw logits 重算，详见 `comparator_raw_logit_results.csv`；SVAR 排序结果不变。来源阈值经 logit 等价变换，分类决策不变。

主要结果：CLEVR→AMBER 的 EMS+E+I AUROC 为 LLaVA 84.79%、Qwen3 2.03%；CLEVR→POPE 为 LLaVA 71.30%、Qwen3 48.40%。Qwen3→AMBER 有严重逆序，答案位置 E 的分布和标签关系明显跨域改变；不能将 AMBER 同域近 100% 视作稳健迁移。POPE 测试 Hall 仅 33/28 例。脚本通过 `py_compile`；样本端到端提取、完整队列与 checkpoint 回放验证通过。原任务归档在 `docs/archive/CURRENT_TASK_before_e_i_answer_ems_object_transfer_20261114.md`。
