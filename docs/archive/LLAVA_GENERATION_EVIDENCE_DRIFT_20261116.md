# LLaVA 生成期视觉写入与 FFN 偏离实验

## 同图对照

LLaVA-1.5-7B FP16 使用 `Describe this image.` 和最多 512 个生成 token。开发集为 COCO4000 种子 43 的 8:1:1 test 按 ID 排序后的前 100 张，前 24 张用于筛选。预留集为紧随其后的 40 张，方法和系数在读取其 CHAIR 结果前确定。图像清单与哈希在 `cohort.json`。每组测试使用同一评估器，逐图 CHAIR 标签不进入生成过程。物体召回和 F1 为逐图平均，CHAIRi 为跨图合计的幻觉提及数除以物体提及总数。

当前服务器重新生成全部贪心及 SVAR 对照。历史输出未随仓库迁移，用户归档的生成与历史报告的 CHAIR 数字也有差异。SVAR 使用第 5–18 层（代码从 0 开始计数）、系数 0.5，每步仅修改视觉注意力 logits 的最后一行。[SVAR 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Devils_in_Middle_Layers_of_Large_Vision-Language_Models_Interpreting_Detecting_CVPR_2025_paper.html)

## 方法

- **SADI**。对第 5–18 层的视觉注意力分数先按 head 求绝对值中位数与总体标准差。标准差归一化到每块预算 0.25–0.80。各 head 的额外分数为预算乘以中位数，再乘偏离中位数的软掩码。[SADI 方法 §3](https://arxiv.org/html/2605.24957v1#S3)
- **候选词重排**。在第 12、18、22 层计算视觉 patch 对候选 (y) 的有符号写入。仅比较当前正常 logits 前 12 中的 WordNet 实体名词，最多低于首选 3 logit。支持项为写入的有符号和除以绝对值和，平均后以系数 2 加到候选 logit。加入 FFN 的版本另扣 3 倍平均的 `D*tanh(max(0, Δ)/2)`，其中 `Δ` 是该 FFN 经 logit lens 带来的局部候选增量。支持和偏离是代理量。候选集合依据通用词典，输出标点与 EOS 不参加重排。[Same Attention, Different Truths 的视觉语义核对](https://arxiv.org/html/2608.07302v1#S4)
- **SADI＋FFN 方向调整**。第 20 层 FFN 更新后先用 logit lens 取得暂定名词。第 18 层该词的正写入决定目标证据分布，第 20 层 FFN 与本层视觉 hpre 的余弦决定偏好分布。修改第 20 层当前预测位置的 FFN 向量，保留其范数。后续层在同一前向内读取该更新。[MemVR 的 FFN 视觉信息注入](https://arxiv.org/html/2410.03577v2#S4) · [FADE 的 FFN 衰减对照](https://arxiv.org/html/2606.29431v1#S3)

### FFN 调整计算

令视觉位置为 $m$，head 为 $h$，候选词 $y$ 的语言头权重为 $w_y$，最终 RMSNorm 的逐维权重为 $\gamma$。第 18 层的写入通过将语言头方向拉回 $W_O$ 直接计算。

$$
c_m(y)=\sum_h A_{hm}\langle v_{hm},[W_O^\top(\gamma\odot w_y)]_h\rangle,\qquad
Q_m(y)=\frac{[c_m(y)]_+}{\sum_j[c_j(y)]_+}.
$$

第 20 层的单位视觉 hpre 向量为 $\hat v_m$，$u=f/\lVert f\rVert_2$ 是当前 FFN 方向。同一个视觉 patch 索引连接两层。

$$
P=\mathrm{softmax}(\hat V u / 0.03),\qquad
D=1-\sum_m\sqrt{Q_mP_m},\qquad
t=(I-uu^\top)\hat V^\top(Q-P).
$$

$$
f'=\lVert f\rVert_2\operatorname{normalize}\left(
u+0.6D\tanh\!\left([\Delta_y]_+/2\right)\operatorname{normalize}(t)
\right).
$$

仅在正写入和为正且 $D$ 下降时接受。写入是带最终 RMSNorm 权重的线性 readout，偏离是平方 Hellinger 距离。选择暂定名词可在真正选词前编辑 FFN，这也意味着其在后续层可能被其他词取代。此轮试验使用低成本代理。

两个 GPU 各跑不同图片。同一批次的变体共享每张图到最后一个文本位置前的提示 KV，后续只前向当前预测位置。视觉 V、hpre 及每个候选词的投影按层缓存，CHAIR 真实物体集合在评估时按图缓存。扩展到 100 张时复用已保存的前 24 张生成结果。各方法均无回滚和额外语言模型前向。耗时是生成阶段的单图实测，另加一次共同预填充。不同并行阶段的 GPU 负载和文本长度可影响绝对时间。

## 前 24 张筛选

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及 | 物体召回 ↑ | 物体 F1 ↑ |
|---|---:|---:|---:|---:|---:|
| 贪心 | 41.67% | 10.00% | 16 | 83.61% | 82.20% |
| SVAR | 20.83% | 4.52% | 8 | 70.62% | 77.59% |
| 写入支持重排 | 50.00% | 12.50% | 20 | 81.39% | 78.43% |
| 写入＋FFN 偏离重排 | 50.00% | 12.10% | 19 | 80.14% | 77.43% |
| SVAR＋写入和 FFN 重排 | 25.00% | 4.17% | 7 | 69.65% | 76.50% |
| SADI | 12.50% | 2.08% | 4 | 78.54% | 84.84% |
| SADI＋第 20 层 FFN 衰减 50% | 12.50% | 2.87% | 5 | 76.18% | 83.02% |
| SADI＋FFN 方向调整 | 12.50% | 2.08% | 4 | 77.15% | 84.01% |
| SADI 增量 70%＋FFN 方向调整 | 50.00% | 15.15% | 30 | 82.92% | 78.40% |

前 24 张重排跑完后，代码把 FP16 分数平分时的首选规则对齐到贪心 `argmax`，并提前排除超出置信窗口的词。表中重排是筛选时的结果。扩展到 100 张的 FFN 方向调整修改层输出并用原 `argmax` 选词。

## 开发集 100 张

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及／全部 | 物体召回 ↑ | 物体 F1 ↑ | 秒／图 ↓ | 解码 ms／token ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 贪心 | 47.00% | 14.43% | 88／610 | 80.21% | 77.51% | 5.22 | 47.2 |
| SVAR | 21.00% | 5.01% | 34／679 | 71.86% | 78.17% | 5.54 | 49.4 |
| SADI | 13.00% | 3.34% | 22／659 | 67.95% | 75.60% | 6.58 | 58.0 |
| SADI＋FFN 方向调整 | 16.00% | 3.53% | 26／737 | 68.85% | 76.19% | 6.98 | 61.3 |

SADI 相对 SVAR 修复 14 张并新增 6 张幻觉。配对 bootstrap 的 CHAIRs 差值 95% 区间为 [−17, +1] 个百分点，CHAIRi 为 [−4.24, +0.59] 个百分点。SADI＋FFN 相对 SVAR 分别为 [−13, +2] 和 [−3.57, +0.48] 个百分点。两者的点估计下降，区间包含零。

SADI＋FFN 相对 SADI 改变了 38 张的生成序列，修复 1 张并新增 4 张幻觉，物体召回提高 0.90 个百分点。在 3,651 个暂定名词位置中，接受 2,650 次更新，接受步的局部 $D$ 平均降低 0.159。偏离下降是接受条件，当前对照中它与 CHAIR 改善没有一致对应。

## 预留集 40 张

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及／全部 | 物体召回 ↑ | 物体 F1 ↑ | 秒／图 ↓ | 解码 ms／token ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 30.00% | 8.97% | 27／301 | 71.25% | 75.55% | 5.37 | 52.0 |
| SADI | 30.00% | 10.85% | 42／387 | 66.29% | 72.29% | 7.67 | 57.4 |
| SADI＋FFN 方向调整 | 27.50% | 11.55% | 44／381 | 67.25% | 73.27% | 7.53 | 57.0 |

固定设置迁移到预留队列后，开发集上两个 CHAIR 指标的同时下降没有复现。SADI＋FFN 调整比 SVAR 少 1 张含幻觉的 caption，幻觉提及增加 17 个，召回低 4.00 个百分点。SADI、SADI＋FFN 相对 SVAR 的配对 CHAIRi 差值 95% 区间分别为 [−4.10, +8.62] 和 [−4.12, +9.86] 个百分点。

SADI 与 SADI＋FFN 都在 174284、184172、215303 达到 512 token，SVAR 在这三图分别为 100、78、127 token。215303 的幻觉提及分别为 18、19 和 4 个。作为敏感性检查，将这三张图从各方法共同排除后，SVAR、SADI、SADI＋FFN 的 CHAIRi 分别为 8.30%、8.00%、8.59%。上表保留全部输出。

本轮证据支持 SADI 作为值得继续比较的单次前向基线。视觉写入重排和第 20 层 FFN 调整在现有对照中没有稳定提升 CHAIR。FFN 实验验证了可低成本调整局部视觉偏好，效果仍受暂定目标的准确性和描述重复影响。

## 文件与验证

- 服务器 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_llava/` 保存图片清单、逐图 captions 和 token、指标、配对图像 bootstrap 与日志。指标从保存的生成输出重算，沿用同一个 CHAIR 标注缓存。
- 生成入口为 `scripts/generation_time_evidence_drift_llava.py`，评估入口为 `scripts/evaluate_generation_time_evidence_drift_llava.py`。核心在 `features/llava_decoding_evidence.py`、`features/llava_sadi.py`。
- 起初服务器的 NLTK 资源下载不完整。更换为校验过的 NLTK 数据并解包后，从已生成的 captions 继续评估。抽查两张图，共享 KV 解码与原 greedy、SVAR 生成入口逐 token 对齐。有符号写入原式与拉回等式及 SADI 小张量对照均通过。
