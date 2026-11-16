# 用视觉写入和 SAR/EMS 分配生成时注意力

实验保留上一版 SADI＋SAR/EMS 及其生成结果。新版本借鉴逐图像区域分配干预这一思路。在早期用逐 patch 的注意力写入量分配视觉预算，在中后期用 SAR 证据与 EMS 偏好的差异分配。两段都用本头的视觉分布与证据分布的距离调节空间分配。[SADI 的区域自适应干预](https://arxiv.org/html/2605.24957v1#S3) · [上一版研究记录](LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md)

## 早期视觉写入

令 \(A_{hm}\) 为当前预测位置对视觉 patch \(m\) 的完整 softmax 注意力，\(I_m\) 为跨头注意力写入范数。在第 5、10、15 层先按原模型计算 \(A\)，再生成至下一次更新前沿用的视觉证据分布 \(\rho\)。层从 0 起算。

$$
I_m=\left\|\sum_h A_{hm}W_{O,h}v_{hm}\right\|_2,\qquad
\rho_m = \frac{\sqrt{I_m}}{\sum_j\sqrt{I_j}}.
$$

平方根压低高写入位置对分布的独占。对第 5–18 层，每个头在该层的视觉总质量为 \(M_h=\sum_{m\in\mathrm{vis}}A_{hm}\)，其条件分布为 \(V_{hm}=A_{hm}/M_h\)。用总变差距离 \(d_h=\tfrac12\lVert V_h-\rho\rVert_1\) 控制原本视觉分布和写入证据的混合。给视觉位置加入以下权重，再在全部位置归一化。

$$
B_{hm}=6M_h(1-M_h)\bigl[(1-d_h)V_{hm}+d_h\rho_m\bigr].
$$

在正常的非空视觉分布下，该头对视觉与文本的注意力赔率乘以 \(1+6(1-M_h)\)。这个阶段按已写入当前预测位置的视觉信息引导注意力，适用于尚无稳定候选词的层。视觉质量为零或写入范数全零时无此增量。

## 后续 SAR/EMS

第 12、18 层末尾读取经过最终 RMSNorm 和语言头的前 8 个候选 token。只有 WordNet 实体名词且受 FFN 正向推动的候选获得权重。令 \(\ell_y\) 为 FFN 后的该候选 logit，\(\Delta_y\) 为 FFN 前后的 logit 增量。

$$
w_y=\operatorname{softmax}_{y\in\mathrm{top8}}(\ell_y)
    \mathbf 1_{\mathrm{实体名词}}(y)
    \tanh\bigl([\Delta_y]_+/2\bigr),\qquad
g=\sum_y w_y.
$$

当 \(g>0\)，使用候选的归一化权重 \(\hat w_y=w_y/g\)。\(p_m(y)\) 是视觉 patch 在该层输入 hpre 经最终 RMSNorm、语言头和全词表 softmax 后的词概率。FFN 输出为 \(f\)。

$$
Q_m\propto I_m\left(1+\sum_y\hat w_y\log(1+|\mathcal V|\,p_m(y))\right),\qquad
P_m=\operatorname{softmax}_m\!\left(\frac{\cos(f,h_m^{\mathrm{pre}})}{0.03}\right).
$$

对第 13–18 层使用第 12 层的 \(Q,P,g\)，第 19–22 层用第 18 层重算。令 \(V_h,M_h\) 为待编辑层的条件视觉分布与视觉总质量，在这两个层段各次前向给视觉位置加以下权重，再在全部位置归一化。

$$
C_{hm}=4g M_h\frac{\|V_h-Q\|_1}{2}[Q_m-P_m]_+.
$$

两个干预同时作用于第 13–18 层，先用写入证据再用 SAR/EMS。每层只编辑当前预测位置。一次语言模型前向完成一步解码。逐图、逐层缓存视觉值向量投影的 Gram 矩阵和所需语义概率分母。运行参数为 `--variants sar_ems_write_prior`，只启用后段的筛选版本为 `sar_ems_patch_mixture`。

## 保留中位数量纲的变体

`sar_ems_write_logits` 使用同一写入分布 \(\rho\)、层配置和后段 SAR/EMS。在第 5–18 层，视觉 logits \(z_{hm}\) 的跨头绝对值中位数 \(\mu_m\) 沿用 SADI 作为量纲。区域预算取 \(\rho_m\) 与均匀分布 \(1/N\) 的相对份额，逐头筛选取该头对该区域的条件视觉注意力 \(V_{hm}\) 相对 \(\rho_m\) 的不足幅度。早期 logit 的增量为

$$
\Delta z_{hm} =
\underbrace{\mu_m}_{\text{SADI 量纲}}\,
\underbrace{\left(0.25+0.9\frac{\rho_m}{\rho_m+1/N}\right)}_{\text{区域预算}}\,
\underbrace{\frac{[\rho_m-V_{hm}]_+}{\max(\rho_m,10^{-12})}}_{\text{逐头覆盖不足}} .
$$

SADI 通过跨头方差设定区域预算，并按各头的绝对 logit 偏离中位数筛选。这里两个决定均由写入分布及该头的相对注意力给出。候选词成形后，后段使用 SAR 和 EMS 的方向差异继续补给证据区域。[SADI 公式 1–6](https://arxiv.org/html/2605.24957v1#S3)

24 张上，直接定标版本把生成长度从 SVAR 的 107.5 token／图降到 34.4，物体召回降到 29.93%。限幅版 `sar_ems_write_logits_bounded` 用 \(2\tanh(\Delta z_{hm}/2)\) 替换早期增量，使单次修正上限为 2 logit，对应单个位置的未归一化注意力倍率至多 \(e^2\)。其它候选筛选、分布和层配置沿用直接定标版。

## 评估

LLaVA-1.5-7B FP16，使用与上一轮相同的 COCO4000 种子 43 Test 图像、提示词 `Describe this image.` 和 512 token 上限。前 24 张用于探索，基线生成和 CHAIR 图像标注缓存直接复用。CHAIRi 使用汇总物体提及作为分母，物体召回和 F1 取逐图平均。

首轮只启用后段 SAR/EMS，24 张上得到 CHAIRs 41.67%、CHAIRi 11.25%、召回 85.69%。2,569 个层与 token 触发事件里，权重最高的候选与最终 token 一致 263 次（10.24%）。因此第二轮增加了在候选词形成前即可计算的视觉写入干预。

早期视觉写入在概率上增加视觉质量，24 张的 CHAIRs 降至 33.33%，CHAIRi 升到 13.25%。图像 13061 有 8 次幻觉提及，计入全部 22 次。随后复用相同的写入证据和后段干预，测试用中位数量纲按区域覆盖不足调整 logit。

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉提及／全部 | 物体召回 ↑ | 物体 F1 ↑ | token／图 | 解码秒／图 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 贪心 | 41.67% | 10.00% | 16／160 | 83.61% | 82.20% | 106.8 | 4.57 |
| SVAR | 20.83% | 4.52% | 8／177 | 70.62% | 77.59% | 107.5 | 4.78 |
| SADI | 12.50% | 2.08% | 4／192 | 78.54% | 84.84% | 99.5 | 6.06 |
| 上一版 SADI＋SAR/EMS | 8.33% | 1.56% | 3／192 | 78.54% | 85.24% | 99.5 | 5.36 |
| 后段 SAR/EMS | 41.67% | 11.25% | 18／160 | 85.69% | 82.50% | 105.5 | 5.34 |
| 早期写入概率分配＋SAR/EMS | 33.33% | 13.25% | 22／166 | 80.69% | 79.38% | 110.8 | 6.14 |
| 写入筛选＋中位数量纲＋SAR/EMS | 16.67% | 26.53% | 13／49 | 29.93% | 39.20% | 34.4 | 2.08 |
| 限幅写入筛选＋中位数量纲＋SAR/EMS | 41.67% | 11.32% | 18／159 | 76.18% | 76.66% | 105.9 | 6.31 |

直接定标版有 5 张生成中无 COCO 物体提及，全部提及仅 49 次。它的 CHAIRs 较低伴随明显的描述缩短和召回下降。限幅后，描述长度和召回恢复，CHAIRs 与贪心持平。四个新配置的 CHAIRi 均高于 SVAR。这轮结果集中在同一批 24 张筛选图，最初的 SADI＋SAR/EMS 沿用此前的 100 张和 40 张结果。

## 验证

FP32 样例核对缓存 Gram 恢复的写入范数、单候选及多候选 SAR，核对概率重分配的行归一化以及零预算情形。保留中位数函数抽取前后的 SADI 样例逐元素一致。验证只有低于视觉写入证据占比的 patch 获得正 logit 增量，限幅样例的增量在 0 到 2 之间。运行语法检查及 Git 空白检查。复核上一版开发与固定对照的四个生成分片 SHA-256。

## 产物

服务器结果位于 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_llava/pilot/`。

- 完整 24 张对照 `metrics.first24.greedy-sadi-sadi_sar_ems_attention-sar_ems_patch_mixture-sar_ems_write_logits-sar_ems_write_logits_bounded-sar_ems_write_prior-svar.json`。
- 逐图评估为同后缀的 `chair_per_image.*.jsonl`。四个新配置分别保存在对应名称的两个 `.shard*.jsonl` 文件里。
- 本地实现见 [解码入口](../../scripts/generation_time_evidence_drift_llava.py)、[干预层配置](../../features/llava_decoding_evidence.py) 和 [视觉写入与注意力分配](../../features/llava_sar_ems_attention.py)。
