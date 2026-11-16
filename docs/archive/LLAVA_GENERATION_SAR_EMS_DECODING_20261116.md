# LLaVA 生成时的 SAR/EMS 选词与重复干预

## 评估范围

使用 LLaVA-1.5-7B FP16、提示词 `Describe this image.`、最多 512 token。原有 COCO4000 种子 43 图片级 8:1:1 test 的前 24 张做候选筛选，开发集为该批前 100 张，另用之前已跑基线的固定 40 张比较。全部使用同图、同 CHAIR 实现。入选和基线输出分片分开保存，复用图像标注与基线生成。物体召回和 F1 为逐图平均，CHAIRi 使用实际提及数汇总。

## 同一步候选词重排

借鉴 SADT 先核对词与视觉证据再干预。当前预测用一次模型前向得到末层贪心首选。仅当首选是 WordNet 候选实体首词，并且末层前 12 存在相差至多 2 logit 的替代实体，才检查最多 3 个备选。第 12、18、22 层保留当前预测位置的注意力和 FFN 输出。层编号从 0 起算。[SADT 方法](https://arxiv.org/html/2608.07302v1#S4)

令 $s_{\ell m}(y)=\log(1+|\mathcal V|p_{\ell m}(y))$，$p_{\ell m}(y)$ 来自视觉 hpre 经过模型最终 RMSNorm 和全词表语言读出。SAR 支持取 $B_\ell(y)=(F_{\log,\ell}(y)-I_\ell)/I_\ell=\sum_m\|a_{\ell m}\|s_{\ell m}(y)/\sum_m\|a_{\ell m}\|$，分母有数值保护。平均 $B$ 至少提高 0.10 后才为首选和合格备选计算精确 EMS。沿用 FFN 来源 $P$，另以头均视觉注意力乘 $s_{\ell m}(y)$ 构造目标 $Q$。三个层上的 EMS 均使用 $K=32$，$\tau=0.03$，成本和两端点惩罚系数 1。选择满足平均风险降低超过 0.02 的最高原始 logit 候选。

每张图缓存视觉写入投影内积、词表概率分母及 EMS 视觉几何成本。只在已有词表分数上选词。使用同一步理由进行比较时，FFN 和视觉写入均来自该前缀的同一次前向。

| 同一前 24 图 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及数 | 物体召回 ↑ | 物体 F1 ↑ |
|---|---:|---:|---:|---:|---:|
| 贪心 | 41.67% | 10.00% | 16 | 83.61% | 82.20% |
| SVAR | 20.83% | 4.52% | 8 | 70.62% | 77.59% |
| SADI | 12.50% | 2.08% | 4 | 78.54% | 84.84% |
| 早期 SADI＋后层 SAR/EMS | 8.33% | 1.56% | 3 | 78.54% | 85.24% |
| 贪心＋SAR/EMS 重排 | 45.83% | 12.42% | 20 | 80.62% | 78.68% |
| SADI＋SAR/EMS 重排 | 16.67% | 9.84% | 18 | 73.19% | 79.99% |

SADI 重排改变了 14 张的 token 序列。图像 13061 占 12 次幻觉提及，生成多次重复 television remote。用于预测的相对证据差适用于当前位置，换词后的连续描述仍可能转向重复。

## 视觉约束的重复控制

研究提示模型可能反复依赖少量已生成文本。这里保留早期 SADI 与已有后层 SAR/EMS 注意力干预。在该解码器上，核对将要重复的 4-token 片段，要求片段以候选实体 token 结尾。记录上一回生成同一片段时末词的第 18 层逐 patch SAR 分布 $E_{\mathrm{prev}}$，当前同理计算 $E_m \propto \|a_m\|(1+s_m)$。[OPERA 的动机](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_OPERA_Alleviating_Hallucination_in_Multi-Modal_Large_Language_Models_via_Over-Trust_CVPR_2024_paper.html) · [已有注意力方案](LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md)

$$
\operatorname{BC}(E,E_{\mathrm{prev}})=\sum_m\sqrt{E_m E_{\mathrm{prev},m}},
\qquad D=1-\sum_m\sqrt{P_m E_m}.
$$

当跨次证据重叠至少 0.90 时，从贪心候选 logit 扣除 $4(0.5+0.5D)$，然后按词表重新选词。$P$ 为原 EMS 的第 18 层 FFN 来源分布，$D$ 为该来源与 SAR 分布间平方 Hellinger 距离。用文本对照核对贡献，采用相同的 4-token 候选和最高惩罚，直接扣 4。

如果基底序列中不存在边界条件匹配的重复片段，两个变体与基底逐步相同，直接复用保存的生成。开发集复用 20 张，固定对照复用 9 张。其余重新解码并逐图验证首次 token 变化等于日志中第一个干预位置。复用图片沿用基底时长，因此聚合耗时包含实测和历史时长。

| 图像 | 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及／全部 | 物体召回 ↑ | 物体 F1 ↑ | 解码秒／图 |
|---|---|---:|---:|---:|---:|---:|---:|
| 100 张开发集 | SVAR | 21.00% | 5.01% | 34／679 | 71.86% | 78.17% | 5.54 |
| | SADI | 13.00% | 3.34% | 22／659 | 67.95% | 75.60% | 6.58 |
| | 早期 SADI＋后层 SAR/EMS | 11.00% | 3.05% | 20／655 | 67.27% | 75.18% | 6.21 |
| | SADI＋SAR/EMS＋文本重复控制 | 11.00% | 3.03% | 22／725 | 67.79% | 75.46% | 6.60 |
| | SADI＋SAR/EMS＋视觉重复控制 | 12.00% | 3.39% | 23／679 | 67.72% | 75.53% | 6.06 |
| 40 张固定对照 | SVAR | 30.00% | 8.97% | 27／301 | 71.25% | 75.55% | 5.37 |
| | SADI | 30.00% | 10.85% | 42／387 | 66.29% | 72.29% | 7.67 |
| | 早期 SADI＋后层 SAR/EMS | 30.00% | 10.85% | 42／387 | 66.29% | 72.29% | 8.21 |
| | SADI＋SAR/EMS＋文本重复控制 | 32.50% | 8.55% | 29／339 | 65.46% | 70.98% | 6.57 |
| | SADI＋SAR/EMS＋视觉重复控制 | 32.50% | 8.50% | 30／353 | 65.79% | 71.36% | 6.10 |

40 张里的图 215303 在基底上生成 512 token 和 18 次幻觉提及，文本控制后为 136 token 和 4 次，视觉控制后为 147 token 和 3 次。两个控制的聚合 CHAIR 接近，视觉方案改动的图片数更少，100 张为 36 对 80，40 张为 18 对 30。40 张视觉控制的 CHAIRi 比 SVAR 低 0.47 个百分点，绝对幻觉提及多 3 次，物体召回低 5.46 个百分点。

## 用已有特征校准词表降权

从服务器已审计的 K32/tau0.03/w1 特征读取压缩矩阵，恢复种子 43 的 3,200/400/400 图片划分。检查两组生成图像清单均在 test 内。训练、验证、测试缓存有 10,166、1,267、1,253 个物体 token。比较同三个层的 \([F_{\log},R]\) 与 \([B,R]\) 两套输入，均用训练集标准化与逻辑回归。验证集 Hall-F0.5 选择了原 \(F_{\log}\) 和 EMS 组，阈值 \(T=0.433902\)，验证 Hall precision 63.94%，recall 53.42%，AUROC 82.99%。固定阈值和系数进入生成过程。选择的三层模型与旧全层 MLP 是不同模型。

借鉴 ReWEIGH 对置信候选施加有上限的词表降权。当前 logits 前两名间隔小于 2、首选是已在训练里出现的实体词时，计算同样的 SAR 和精确 EMS。首选与已注册的第二名均用风险分数 \(r(y)\) 计算罚项 \(2[r(y)-T]_+/(1-T)\)。两个候选中取分数调整后的胜者。每步共享视觉与 FFN 缓存。[ReWEIGH §4](https://arxiv.org/html/2608.19075v1#S4)

| 前 24 张 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及 | 物体召回 ↑ | 物体 F1 ↑ | 解码秒／图 |
|---|---:|---:|---:|---:|---:|---:|
| 贪心 | 41.67% | 10.00% | 16 | 83.61% | 82.20% | 4.57 |
| 贪心＋校准降权 | 37.50% | 8.23% | 13 | 82.92% | 82.34% | 5.01 |
| 早期 SADI＋后层 SAR/EMS | 8.33% | 1.56% | 3 | 78.54% | 85.24% | 5.36 |
| 早期 SADI＋后层 SAR/EMS＋校准降权 | 8.33% | 1.56% | 3 | 78.54% | 85.24% | 5.45 |

先导阶段贪心版本共换 13 个 token，涉及 10 张。已有 SADI＋SAR/EMS 基底上的前 24 张保持相同序列。

| 开发集 100 张 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及／全部 | 物体召回 ↑ | 物体 F1 ↑ | 解码秒／图 |
|---|---:|---:|---:|---:|---:|---:|
| 贪心 | 47.00% | 14.43% | 88／610 | 80.21% | 77.51% | 5.22 |
| 贪心＋校准降权 | 47.00% | 14.24% | 86／604 | 80.45% | 77.61% | 5.22 |
| SVAR | 21.00% | 5.01% | 34／679 | 71.86% | 78.17% | 5.54 |

校准降权改变 36 张序列，修复 2 张并在另外 2 张新增幻觉。按图片配对 bootstrap 的 CHAIRi 增量 95% 区间为 [−1.13, +0.73] 个百分点。先导阶段观察到的下降在完整开发集上很小。

在这两批图像上，视觉重复控制让已有 SADI＋SAR/EMS 的 CHAIRi 都处在 SVAR 之下。固定 40 张的 CHAIRs 高 2.50 个百分点，召回低 5.46 个百分点。最明确的增益是减少长循环里的重复幻觉提及。文本对照取得相近的总体指标，SAR 筛选在相近的聚合指标下减少了修改图像数。前 24 张包含在 100 张中，固定 40 张也在之前的探索中使用过。

## 复查与产物

缓存成本与直接构造的 EMS 问题逐项核对，并与在成本矩阵中显式加两个端点惩罚的 EMD 数值一致。复查了重复干预的首次 token 改变、基底分片哈希及 100/40 张样本数。语法检查与 Git 空白检查通过。

服务器根目录为 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_llava/`。

- 前 24 张硬重排 `pilot/metrics.first24.greedy-sadi-sadi_sar_ems_attention-sadi_sar_ems_rerank-sar_ems_rerank-svar.json`。
- 前 24 张校准降权 `pilot/metrics.first24.greedy-sadi-sadi_sar_ems_attention-sadi_sar_ems_calibrated-sar_ems_calibrated-svar.json`。
- 100 张校准降权 `pilot/metrics.greedy-sadi_sar_ems_attention-sadi_sar_ems_visual_repeat-sar_ems_calibrated-svar.json`。
- 重复控制 `pilot/metrics.sadi-sadi_sar_ems_attention-sadi_sar_ems_text_repeat-sadi_sar_ems_visual_repeat-svar.json` 及 `confirmation/metrics.first40.sadi-sadi_sar_ems_attention-sadi_sar_ems_text_repeat-sadi_sar_ems_visual_repeat-svar.json`。
- 对应逐图评估保存在同后缀的 `chair_per_image.*.jsonl`。阈值和系数由 `sar_ems_decoding_gate_seed43.json` 保存。
