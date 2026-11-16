# Qwen3-VL 的 SADI 与 SAR/EMS VQA 幻觉缓解

采用 Qwen3-VL-8B-Instruct，图像级划分沿用此前 SAM/SVAR 的种子 42 外层 8:2 测试集。POPE popular 为 100 图 600 题，GT Yes/No 各 300。AMBER existence 为 201 图 987 题，GT 全部是 No。所有方法在同一批完整题目上重新生成。提示词沿用题干加 `Answer only yes or no.`，原生聊天模板、BF16、贪心、最多 8 个新 token。 [既有 VQA 数据与协议](QA_SAM_SVAR_POPE_AMBER_20261115.md) · [POPE 官方数据](https://github.com/RUCAIBox/POPE/blob/main/output/coco/coco_pope_popular.json) · [AMBER existence 官方数据](https://github.com/junyangwang0410/AMBER/blob/master/data/query/query_discriminative-existence.json)

## 干预和评分

原模型与 SADI 作为对照。SADI 及两种增强版在第 5–18 层使用同一 SADI attention logits 干预。caption 原配方沿用第 20 层 FFN 后 logit lens 的 WordNet 暂定实体词、正 FFN 增益门控及第 21–22 层重分配。VQA 适配版只在首答案查询运行，用题干中实际对齐物体词的首个 subtoken 构造第 18 层 SAR 的目标视觉分布。仅在第 20 层 FFN 对规范的 Yes−No 分差带来正增量、且更新后分差也为正时，沿用原配方的 FFN 空间分布、温度 0.03、Hellinger 偏离和系数 5，在第 21–22 层重分配同一查询对视觉位置的注意力。适配使用题干物体，GT 只在评分时读取。 [此前 caption 配方](QWEN3_SADI_SAR_EMS_TRANSFER_20261116.md)

答案采用项目的第一个独立 Yes/No 词解析。无效答案计入准确率的错误并单列。POPE 分别以全部 GT No 和全部 GT Yes 作分母，报告错误 Yes、错误 No、准确率和 Yes-F1。AMBER 只有 GT No，报告错误 Yes 和准确率。误报和准确率的配对区间以图像为单位抽样 5,000 次。

## POPE Popular 固定测试 600 题

| 方法 | 错答 Yes ↓ | 错答 No ↓ | 准确率 ↑ | Yes-F1 ↑ | 后层触发题数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 原模型 | 23／300（7.67%） | 51／300（17.00%） | 87.67% | 87.06% | — |
| SADI | 23／300（7.67%） | 49／300（16.33%） | 88.00% | 87.46% | — |
| SADI＋caption 原配方 | 23／300（7.67%） | 49／300（16.33%） | 88.00% | 87.46% | 0 |
| SADI＋VQA 物体适配 | 23／300（7.67%） | 49／300（16.33%） | 88.00% | 87.46% | 572 |

SADI 相对原模型修正 2 个错误 No，错误 Yes 不变。增强版与 SADI 的 600 题生成 token 序列全部相同，四组均无无效答案。VQA 物体适配触发的 572 题中，有 300 题的最终答案为 No，因此第 20 层的正向分差门控也覆盖大量最终选择 No 的问题。

## AMBER Existence 固定测试 987 题

| 方法 | 错答 Yes ↓ | 准确率 ↑ | 后层触发题数 |
| --- | ---: | ---: | ---: |
| Qwen3 原模型 | 75／987（7.60%） | 92.40% | — |
| SADI | 79／987（8.00%） | 92.00% | — |
| SADI＋caption 原配方 | 79／987（8.00%） | 92.00% | 0 |
| SADI＋VQA 物体适配 | 80／987（8.11%） | 91.89% | 948 |

caption 原配方在 987 题全部与 SADI 的生成 token 相同。物体适配版的唯一变化是图像 470 的题目 `Is there a sun in this image?`。GT 为 No，SADI 答 No，适配版答 Yes。物体适配对 SADI 的误报率变化为 +0.10 个百分点，按图配对的 95% 区间为 [0.00, 0.31]。四组均无无效答案。

首答案分差的额外诊断按图像 ID 和题目键顺序各抽查 12 条 SADI 错答 Yes 的题，用原生完整输入计算首答案 logits。POPE 和 AMBER 各有 9 题的规范 Yes−No 分差被物体适配改变。原分差中位数分别是 +4.38、+3.94，变化的绝对最大值分别为 0.375、0.25。结果记录于 `false_yes_margin_diagnostic.json`。两组完整测试的误报均没有下降。

## 运行与验证

物体词跨度用实际多模态 prompt token 对齐。每张图共享完整视觉前缀和 DeepStack，只对带问题的文本后缀运行各臂。复用冻结的 KV 时按独立动态缓存分支拼接后续 token，各分支保持原生 mRoPE 的文本位置偏移。两卡启动校验涵盖两数据集、四臂的 64 组缓存路径与原生生成对照，输出 token、归一化答案和触发次数完全一致。初次启动在预检查发现模型默认上限仍为 512，显式应用此前 VQA 的 8-token 配置后重新启动，初次未写入结果。

AMBER 首轮的少数超高分辨率图在 eager 视觉注意力出现显存不足。为获得完整对照，AMBER 的 987 题四臂均按同一 1,048,576 像素上限重新生成，首次的部分输出移出正式评分。POPE 测试图的补齐尺寸最大为 409,600 像素，对该上限的处理不改变图像，已完成的 600 题保留。

服务器根目录 `/root/rivermind-data/ic-ac/outputs/qa_benchmarks/qwen3_vqa_sadi_mitigation/`。冻结配置为 `protocol.json`，每题四臂配对结果为 `pope.shard*.jsonl` 和 `amber_discriminative.shard*.jsonl`，完整评分为 `summary.json`。 [运行和评分脚本](../../scripts/evaluate_qwen3_vqa_mitigation.py) · [干预实现](../../features/qwen3_sar_ems_attention.py)
