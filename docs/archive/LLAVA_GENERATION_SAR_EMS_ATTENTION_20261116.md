# LLaVA 上用 SAR 和 EMS 引导生成时注意力

## 评估协议

使用 LLaVA-1.5-7B FP16、`Describe this image.` 和最多 512 个生成 token。图像来自 COCO4000 种子 43 的 8:1:1 test，按原清单排序。先在开发集前 24 张筛选干预基底，再扩展到同一开发集 100 张。另有 40 张固定对照，已在上一轮方法探索中使用。各组用同一 CHAIR 实现和图像清单，CHAIRi 汇总物体提及数后计算，物体召回及 F1 取逐图平均。

图像和提示一次预填充，同图的不同方法共享提示 KV。扩展时跳过已经生成的 24 张，复用已有 SADI、SVAR 的生成和 CHAIR 标签缓存。新方法每个 token 做一次模型前向，视觉语义概率的分母及写入投影内积在每图首次触发时缓存。生成时间按图实测，单列提示预填充。

## 干预

第 5–18 层沿用 SADI 的 logit 重标定。新增干预在第 21、22 层调整当前预测位置对图像 patch 的注意力。本文的层编号从 0 起算。第 20 层 FFN 后的 logit lens 给出暂定词 $y$。WordNet 判断该词为实体名词且该层 FFN 对 $y$ 的 logit 增量 $\Delta_y>0$ 时触发。[SADI 方法](https://arxiv.org/html/2605.24957v1#S3)

第 18 层视觉 patch $m$ 经各头对当前预测位置写入 $a_m=\sum_h A_{hm}W_{O,h}v_{hm}$。语义概率取该视觉位置的 hpre 经模型最终 RMSNorm、语言头与全词表 softmax 后赋给 $y$ 的概率 $p_m(y)$。将原 SAR 的每个 patch 加权写入项归一化，并使用 EMS 第 20 层的 FFN 偏好分布。

$$
s_m=\|a_m\|_2\bigl(1+\log(1+|\mathcal V|p_m(y))\bigr),\qquad
Q_m=\frac{s_m}{\sum_j s_j},\qquad
P_m=\operatorname{softmax}_m\left(\frac{\cos(f_{20},h^{\mathrm{pre}}_{20,m})}{0.03}\right).
$$

在两个后续层分别使用各自的视觉 logit $z_{hm}$。从 SADI 借鉴按空间位置的跨头分歧分配预算，新增部分取当前层 $\sigma_m=\operatorname{std}_h(|z_{hm}|)$，预算同时受 SAR 和 EMS 的差异、暂定词增量 $\Delta_y$ 控制。

$$
D=1-\sum_j\sqrt{P_jQ_j},\qquad r_m=[Q_m-P_m]_+,\qquad
b_m=5D\tanh(\Delta_y/2)\frac{r_m}{\max_j r_j},\qquad
\beta_m=0.25+0.75\frac{\sigma_m}{\max_j\sigma_j}.
$$

使用 $\tilde z_{hm}=z_{hm}+\beta_mb_m$，再减去每个头的视觉 log-sum-exp 增量 $\operatorname{LSE}_m(\tilde z_{hm})-\operatorname{LSE}_m(z_{hm})$。各头在该层分给视觉位置的总注意力质量保持相同，变化发生在不同视觉位置之间。分布和预算的分母有数值保护。$D$ 是平方 Hellinger 距离，用于生成时的快速比较。原检测器的 EMS 来源分布在这里用于直接引导，原 OT risk、Top-K 及检测 MLP 均无需计算。$y$ 由当前前向内部决定，最终生成词可能发生变化。

## 前 24 张筛选

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及 | 物体召回 ↑ | 物体 F1 ↑ |
|---|---:|---:|---:|---:|---:|
| SVAR | 20.83% | 4.52% | 8 | 70.62% | 77.59% |
| SVAR＋SAR/EMS 后层注意力 | 20.83% | 5.08% | 9 | 70.62% | 77.59% |
| SADI | 12.50% | 2.08% | 4 | 78.54% | 84.84% |
| SADI＋SAR/EMS 后层注意力 | 8.33% | 1.56% | 3 | 78.54% | 85.24% |

依据前 24 张的 CHAIRs、CHAIRi 及召回，选择 SADI 基底扩展。SVAR 对照采用论文公开实现的第 5–18 层、系数 0.5。[SVAR 作者代码](https://github.com/xjtu-wangzihan/SVAR)

## 开发集 100 张

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及／全部 | 物体召回 ↑ | 物体 F1 ↑ | 解码秒／图 ↓ | 解码 ms／token ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 21.00% | 5.01% | 34／679 | 71.86% | 78.17% | 5.54 | 49.44 |
| SADI | 13.00% | 3.34% | 22／659 | 67.95% | 75.60% | 6.58 | 57.95 |
| SADI＋SAR/EMS 后层注意力 | 11.00% | 3.05% | 20／655 | 67.27% | 75.18% | 6.21 | 54.33 |

相对 SADI，新增干预改变 9 张的 token 序列，修复 2 张幻觉图，新增 0 张。按图片配对 bootstrap 5,000 次，CHAIRs 差值的 95% 区间为 [−5.00, 0.00] 个百分点。触发 2,644 个位置，暂定词与最终生成 token 一致 809 次（30.60%）。SADI 和加干预版本各有 3 张触及 512 token 上限，图像 ID 相同。SVAR 有 1 张触及上限。基线耗时取自此前同协议运行，速度差异包含运行时负载差别。

## 固定对照 40 张

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及／全部 | 物体召回 ↑ | 物体 F1 ↑ | 解码秒／图 ↓ | 解码 ms／token ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 30.00% | 8.97% | 27／301 | 71.25% | 75.55% | 5.37 | 51.97 |
| SADI | 30.00% | 10.85% | 42／387 | 66.29% | 72.29% | 7.67 | 57.36 |
| SADI＋SAR/EMS 后层注意力 | 30.00% | 10.85% | 42／387 | 66.29% | 72.29% | 8.21 | 61.79 |

新增干预相对 SADI 只改变 1 张的 token 序列，表内的质量指标相同。1,108 次触发中暂定词和最终生成 token 一致 395 次（35.65%）。两种 SADI 版本均有相同 3 张触及 512 token 上限，SVAR 为 0 张。图像 215303 在两个 SADI 版本中各产生 18 次幻觉提及，占各自全部幻觉提及的 42.86%。

开发集上加干预版本的两个 CHAIR 指标低于 SVAR，物体召回低 4.58 个百分点。40 张对照的 CHAIRs 与 SVAR 持平，CHAIRi 高 1.88 个百分点，物体召回低 4.96 个百分点。SAR/EMS 在 SADI 上的增量只在开发集观测到。开发集包含筛选用的 24 张，固定对照也已用于早先方法探索。此处报告同图配对的探索结果。

## 复核此前两批对照

复用固定 40 张与此前未用的独立 60 张，各图在两批之间无重叠。下表是对已有结果的回顾性汇总。CHAIRi 从原始提及数汇总，召回和 F1 取各图平均。独立 60 张的单批明细保存在[原报告](LLAVA_REBUTTAL_SAR_EMS_MITIGATION_20261116.md#独立-60-图验证)。

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉／全部提及 | 物体召回 ↑ | 物体 F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SVAR | 25.00% | 6.94% | 49／706 | 73.04% | 76.94% |
| SADI | 21.00% | 5.63% | 54／959 | 69.71% | 75.44% |
| SADI＋SAR/EMS 后层注意力 | 21.00% | 5.66% | 55／972 | 69.41% | 75.29% |

后层注意力相对 SVAR 修复 8 张、新增 4 张幻觉图。相对 SADI，100 张的逐图幻觉判定均相同，额外增加了 1 次幻觉提及。最有利的独立 60 张上，后层版比 SVAR 的 CHAIRs/i 低 6.67／3.21 个百分点，召回/F1 低 2.74／0.56 个百分点。相对普通 SADI，后层信号在这两批中的独立缓解增益尚未验证。

## 数值核对

在随机 FP32 张量上逐 patch 比对缓存内积恢复和直接写入投影，归一化 SAR 分布最大绝对差为 $2.98\times10^{-8}$。对有限 logits 检查 softmax 后每头总视觉注意力与各文本位置概率，总视觉质量最大绝对差为 $5.96\times10^{-8}$。

## 产物

服务器根目录为 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_llava/`。

- 前 24 张 `pilot/metrics.first24.sadi-sadi_sar_ems_attention-svar-svar_sar_ems_attention.json`
- 开发集 100 张 `pilot/metrics.sadi-sadi_sar_ems_attention-svar.json`
- 固定对照 40 张 `confirmation/metrics.first40.sadi-sadi_sar_ems_attention-svar.json`
- 对应逐图评估使用相同后缀的 `chair_per_image.*.jsonl`。原始生成保存在各集合的 `sadi_sar_ems_attention.shard*.jsonl`，历史基线仍在同目录。
