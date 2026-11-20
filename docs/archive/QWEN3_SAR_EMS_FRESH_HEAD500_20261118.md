# Qwen3 逐头引导的全新 500 图验证

固定上一轮最终梯度逐头引导参数，比较原 SADI、逐头引导和 SVAR。三组在相同的新 500 图、相同有序批次上重新生成，max256、最大 B16、BF16 eager greedy，提示为 `Describe this image.`。SADI 覆盖 5–18 层，新模块在第 31 层调整视觉注意力并重算当前查询到第 35 层。视觉支持阈值 0.35、最大强度 3、FFN 偏好温度 0.03，接受真实最终概率下降的行。模型权重与方法参数来自此前 100/500 图实验。

使用 seed 20261118，从 COCO val2014 的 40504 张图中排除已登记的 6350 张，余 34154 张中抽样。排除范围覆盖此前四模型 COCO 生成、POPE、生成时缓解实验、诊断图和预检。保留 604 项历史文件记录与哈希。排序后样本 SHA256 为 `226baddfc69edb8bba8fabd7a944e8dbfcf838192dc71e684af48f82a9ae61d1`。完整排除清单与来源保存在 `selection.json`。

新增 `scripts/run_qwen3_fresh_head500.py`，复用此前新图实验的分批、三组同步 OOM 拆批和评估流程。参数协议引用此前 `signed_feedback_head-gradient-500/protocol.json` 及其 SHA。审计同时检查三组原始文件、批次、两种 CHAIR 评估及所有配对区间。

服务器目录为 `outputs/generation_time_evidence_drift_qwen3_fresh_head500/`，已完成。本机快照在 `scripts/experiment_snapshots/qwen3_fresh_head500/`。

| 新 500 图，sentence | CHAIRs ↓ | CHAIRi ↓ | 物体召回 ↑ | F1 ↑ | 幻觉／全部物体提及 |
|---|---:|---:|---:|---:|---:|
| SADI | 25.20% | 9.57% | 70.32% | 74.41% | 454／4744 |
| SADI＋逐头最终梯度 | 25.40% | 10.83% | 70.65% | 74.87% | 505／4664 |
| SVAR | 39.20% | 8.32% | 74.60% | 75.64% | 316／3798 |

相对 SADI，CHAIRs +0.20 pp，95% 配对区间 [−2.00,+2.40]；CHAIRi +1.258 pp [−0.171,+2.810]；F1 +0.460 pp [−0.381,+1.336]；召回 +0.328 pp [−0.651,+1.377]。修正幻觉图15张、新增16张，348张序列改变。此前500图中的CHAIR改善没有在这批新图上复现。

逐图去重幻觉类别156→157，重复幻觉提及298→348。净多51次幻觉提及，其中重复项净多50次。图167128的 cow 提及从1→28，尾部循环为 mother of the calf；图435533的 bowl 从6→26，尾部循环为 bowl of the toilet。两图合计净多47次。移除最差一图后CHAIRi仍从9.587%升至10.367%；移除最差三图后为9.481%→9.784%，CHAIRs为24.748%→24.950%。全量作为主结果。

真实类别1017→1020，真实重复3273→3139。平均token为186.87→188.13，达到上限的图239→237。逐头版提出11718次调整、保留6187次（52.80%），176次立即改变候选词，所有6415次批次gradient probe的最终hidden差为0。生成GPU时间累计SADI1159.82、逐头1631.61、SVAR1111.88秒，逐头相对SADI约增加40.7%，含长度变化。三组新生成、模型加载及评估总墙钟34分42秒。

| 新 500 图，legacy | CHAIRs ↓ | CHAIRi ↓ | 物体召回 ↑ | F1 ↑ |
|---|---:|---:|---:|---:|
| SADI | 23.60% | 9.51% | 68.07% | 72.93% |
| SADI＋逐头最终梯度 | 23.80% | 10.82% | 68.18% | 73.25% |
| SVAR | 35.20% | 7.67% | 72.11% | 74.51% |

1500条生成的集合、每个有序批次、源文件哈希、两口径聚合和配对区间均独立核对。604项历史排除来源的实际文件SHA核对通过。机制审计确认probe原始hidden逐值一致，保留次数与真实概率下降次数相等。产物包括 `independent_summary.json`、`object_mention_audit.json`、`mechanism_audit.json`、`denominator_audit.json`、`failure_analysis.json`、`selection_source_audit.json`，均已取回本机。

## 训练物体词范围的后续实验

在本次新 500 图仍处于 generating 状态、评分前固定了一个词范围变体。复用旧 COCO4000 训练分割的全部目标 token ID，共 496 个，其中 358 个通过现有 WordNet 物理名词筛选。两个模型标签均用于确定训练目标词范围。该词范围与当前图片中的物体真值无关。

原 SADI 的历史 100 图中，逐图去重的 WordNet 物理候选词有 2840 个，训练物体词交集为 308 个。范围外常见词包括 image、scene、background、white、overall、setting 等。它们可能以 WordNet 中的其他词义进入候选。本统计描述候选词范围，没有逐步重新运行干预。

在此前 304 图诊断里，543 个通过 WordNet 的首次实体位置中有 541 个属于训练物体词；55 个幻觉位置中覆盖 54 个。该覆盖仅指已通过 WordNet 的子集，全部首次 CHAIR 提及共 655 个。原始 CPU 检查保存在 `scripts/experiment_snapshots/qwen3_selected_layer_cached_gate/vocabulary_scope_audit.json`。

新增 `scripts/run_qwen3_object_vocabulary_pilot.py`，参数和底层逐头算法均沿用，只把干预触发集合取为 WordNet 与训练目标词的交集。使用原 100 图、max512、有序批次和 GPU 分配，复用原 SADI 和宽词范围逐头版输出。两卡空闲后自动启动，现已完成。目录为 `outputs/generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-vocabulary-pilot/`，候选名 `sadi_head_training_vocabulary_31`。冻结文件记录 `defined_during_fresh_validation=generating` 以及新验证集 SHA。执行前修正了协议中候选名与梯度范围的映射字段，并记录修正前后源码SHA；算法参数保持原定义。

| 历史100图，sentence | CHAIRs ↓ | CHAIRi ↓ | 物体召回 ↑ | F1 ↑ | 幻觉／全部物体提及 |
|---|---:|---:|---:|---:|---:|
| SADI | 30.00% | 7.94% | 70.60% | 74.95% | 89／1121 |
| 宽词范围逐头版 | 27.00% | 6.65% | 71.42% | 75.73% | 89／1338 |
| 训练物体词范围逐头版 | 30.00% | 7.68% | 71.10% | 75.21% | 88／1146 |

收窄范围后修正1张、新增1张，去重幻觉类别36→38。118次提案中保留60次，只有1次立即换词，原宽词范围版为2279次提案、33次换词。收窄变体改变27张描述。其CHAIRs差为0 pp [−3,+3]，CHAIRi差−0.260 pp [−1.245,+0.503]，F1差+0.261 pp [−0.188,+0.884]。移除幻觉减少最多的一图后CHAIRi略升。没有扩大该变体到500图。

legacy口径为SADI29.00/8.10/68.26/73.20，收窄版29.00/7.94/68.43/73.27。实际生成与评估用时7分28秒，前置等待18分47秒；生成GPU时间714.12秒，平均token267.94（对照257.17）。因此提案数大幅减少没有按同比例减少本次总生成时间。

300条生成、两种评估的聚合与修正计数、缓存对照字节和48个配对指标区间核对通过。`mechanism_audit.json` 确认最终gradient probe hidden差为0。结果与词表在 `scripts/experiment_snapshots/qwen3_object_vocabulary_pilot/`，额外bootstrap记录在同名signed_feedback快照。

## 后续方向

当前结果表明，视觉注意力内部重分配对真实物体词的最终选择影响较小，宽词范围的收益也对重复生成敏感。代码目前在目标token已出现后跳过干预，这是此前采用的筛选选择。下一步可单独检查重复时持续引导，或把同一视觉支持与FFN偏离信号用于文本注意力路径/FFN贡献的调节。原完整检测器也可作为门控候选，需先做SADI输出迁移与缓存特征数值检查。当前两卡空闲，目标继续有效。
