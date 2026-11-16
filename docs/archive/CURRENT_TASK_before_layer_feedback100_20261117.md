# 当前任务

已完成：用户要求的新 200 图、最大 batch size 16 对照。Qwen3-VL-8B-Instruct，SVAR→SVAR＋SAR/EMS 的 CHAIRs 38.50%→40.00%，CHAIRi 8.55%→8.89%，召回 74.99%→76.45%，F1 75.81%→76.50%；修复 9 张、新增 12 张幻觉图，幻觉提及 131→141。配对区间均跨零，两项 CHAIR 未改善。两卡共生成 400 条描述，2026-11-17 北京时间 04:34:45→04:52:03，约 17 分 18 秒，自动评估退出码 0。见[完整报告](archive/QWEN3_SVAR_SAR_EMS_NEW200_B16_20261117.md)。

服务器输出：`/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_new200_b16/`。`run_status.json` 已为 complete，生成、评分均结束，无待续跑的本轮任务。实际每方法 37 个相同批次，112 图在 7 个 B16 批次中，其余 88 图为尾批；无 OOM／异常。查看结果前已声明单列 B16 子集，其 CHAIRs 43.75%→44.64%、CHAIRi 8.32%→9.08%、召回 77.64%→79.91%、F1 76.52%→77.60%；完整 200 图仍为主结果。不要称全部 200 图都在 B16 下生成。

协议：BF16、greedy、512 token、`Describe this image.`；SVAR 零起 5–18 层系数 0.5；增强版读 18 层视觉写入、20 层 FFN 信号，调整 21／22 层注意力，强度 5、温度 0.03。方法参数沿用。按视觉跨度与 prompt 长度分桶，16 是上限；保留小尾批，任一臂 OOM 时对两臂共同二分。必须报告实际批次分布及 OOM 次数。

采样已冻结：COCO val2014 40,504 图，排除旧 COCO4000、POPE、全部登记 generation_time 实验 cohort，并集 5,050 图；候选 35,454 图，用 PCG64 seed43 抽取新 200 图，与各历史来源零重合。集合 SHA `8049a9f2f9adcb32f046ec1549f3fd9d38aedcd3fe70fac12d4ef0c33b50955c`。逐来源 ID／文件哈希、清单、抽样顺序、方法源码哈希均已保存。脚本和冻结清单在 `scripts/experiment_snapshots/qwen3_new200_b16/`。准备阶段修正了旧 JSON 的数字字典键和标量元数据格式；失败均在抽样落盘和 GPU 启动前。快照语法检查、准备与运行流程独立审查通过。

最终产物：`coco200/metrics.first200.svar-svar_sar_ems_attention.json`、逐图评分、`validation.new200_b16.json`。独立复核通过：200 唯一图／400 描述、两臂 37 批有序 ID／实际大小／视觉 token 数相同；全量、B16 子集、未触顶子集的指标与 5,000 次配对 bootstrap 均重算一致。共同未触顶 197 图和去掉最大单图影响后，两项 CHAIR 仍未改善。真值缓存构建一次后复用，无重复 GPU 生成。

上一轮已完成：旧 500 图单图推理 SVAR→增强版 CHAIRs 42.60%→42.80%，CHAIRi 9.70%→9.49%，召回 76.87%→76.45%，F1 76.08%→75.85%。区间均跨零，图 201163 贡献全量净减少 16 次，剔除后 CHAIRi 反转。100／200／300／400／500 节点均已汇报。见[完整结果](archive/QWEN3_SVAR_SAR_EMS_SINGLE500_20261117.md)。当前轮与上一轮均只比较 SVAR 两臂；尚未运行 SADI 两臂的 500 图单图对照。

批次诊断已完成：图 485532 的 B2／B4 第二 token 分歧两方法均复现，image/is 得分 37.00:36.75→36.75:36.75。视觉表示在语言第 0 层前已不同；B1 两控制器、重复 B1、B4 逆序对照一致。未定位具体算子，不能解释全部 66 图或 SADI 60→500 的全部差异。见[诊断](archive/QWEN3_BATCH_LOGIT_DIAGNOSTIC_20261117.md)。

解释约束：用户“自己复跑”指原 Qwen3 SADI 60 图之后本会话的全部实验。60 图与新 500 图无重合且单图→批量，早期 CHAIRi 主要由一图贡献，修复 4／新增 0 的精确 McNemar p=0.125。后续 SVAR 又更换基底，不能混称同条件复现或宣称稳定大幅收益。见[核对](archive/QWEN3_SINGLE_BATCH_COHORT_AUDIT_20261117.md)。

GitHub 此前已上传 mitigate 分支提交 `c1364c0565867aaab59e8d73bd1854209113548f`（68 文件），独立工作区 `../ic-mitigate-upload`；此后新增诊断、监视器、报告和当前快照尚未推送。保留 ac 已有改动。更早实验、特征任务及发布详情见[本轮前归档](archive/CURRENT_TASK_before_new200_b16_20261117.md)。
