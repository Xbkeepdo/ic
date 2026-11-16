# LLaVA 的 SAR/EMS 幻觉缓解实验

## 实验协议

使用 LLaVA-1.5-7B 在 COCO val2014 上生成描述。同一图片的所有方法共用提示词 `Describe this image.`、图像和提示词 KV 预填充、贪心解码及 512 个新 token 的上限。报告 CHAIRs、汇总所有物体提及计算的 CHAIRi，以及逐图平均的物体召回率与 F1。

- 开发筛选使用此前开发集顺序中最前的 24 张图。沿用已有的 SVAR、SADI 和 SADI＋SAR/EMS 后层注意力输出。
- 独立验证清单从原种子 43 的 400 张 test 图中，按 ID 排序取第 201–400 张为候选池，再用 NumPy `default_rng(20261116)` 无放回抽取 60 张并排序。审计了服务器 `ic`、`ic-ac` 目录下历史生成实验中的 114 个 LLaVA JSONL 文件，这 60 张均无已保存的缓解输出。清单锁定在开发筛选结果产生之前，SHA-256 为 `16d12227b1f838c921c179a3aa86afebb78dc30df043d4912e0b9216c6af7031`。
- 开发集上优先选择相对各自基础版本 CHAIRs 降幅最大的新增方案，持平时看 CHAIRi。对候选要求物体召回率下降不超过 2 个百分点。独立图上固定报告 SVAR、SADI、此前的 SADI＋SAR/EMS 后层注意力，以及选定的新方案及其基础版本。配对区间对图片自助重采样 5,000 次。
- 输出根目录为服务器 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_llava/`。开发与独立图分别使用 `pilot/`、`rebuttal_holdout/`，`cohort.json` 包含图片顺序。

## 五种新增方案

| 基础版本 | SAR/EMS 干预 | 代码中的实验名 |
| --- | --- | --- |
| SVAR | 在 SVAR 提升注意力后逐头重分配视觉区域，层 13–18 | `svar_sar_ems_allocation` |
| SADI | 在 SADI 提升注意力后逐头重分配视觉区域，层 13–18 | `sadi_sar_ems_allocation` |
| PAI 注意力组件 | 保持每头视觉 logits 的 PAI 干预，在层 13–18 重分配视觉区域 | `pai_sar_ems_allocation` |
| PADE 最终层局部实现 | 在最终层 PAD 和模板补偿之后，重分配视觉区域 | `pade_sar_ems_allocation` |
| VHR 局部近似 | 将全上下文与屏蔽视觉 key 的注意力头输出差投影到残差空间，用投影范数选头 | `vhr_local_sar` |

层编号从 0 开始。PAI 在层 5–18 使用 $z+0.5|z|$，与 SVAR 的层区间对齐。这一臂只运行 PAI 的注意力组件。PADE 用当前生成查询相邻层的正注意力增量之和构建 PAD，除以区域最大值，乘每头视觉 logits 的 MAD 与 0.1 后加到最终层视觉 logits。对图像前模板 logits 减去该头视觉 logit 增量的平均值。VHR 局部两臂都在第一步选头，用同一局部去图公式、离群过滤和半数头配额，在第 1 与 18–31 层将所选头的完整输出乘以 2。VHR 变体检验 SAR 的投影写入思想，不使用目标词语义或 EMS。

四种区域分配法在第 12 层 FFN 完成后，从 final-norm logit lens 的 top-8 中对正向 FFN logit 增益的候选物体词赋权。令 $c$ 为权重和，$w_y$ 为候选词归一化权重。层内区域证据分布和 FFN 偏好分别为

\[
Q_m\propto\left\|\sum_h A_{hm}W_{O,h}v_{hm}\right\|_2
  \left(1+\sum_y w_y\log(1+|\mathcal V|p_m(y))\right),
\qquad
P_m=\operatorname{softmax}_m\bigl(\cos(f,h^{\rm pre}_m)/0.03\bigr).
\]

证据分布 $Q$ 来自 SAR，偏好分布 $P$ 取自 EMS。用平方 Hellinger 距离 $D=1-\sum_m\sqrt{Q_mP_m}$ 控制强度，并令 $R\propto[Q-P]_+$ 定位区域。设该头在基础注意力干预后的视觉权重向量为 $a$，总量为 $M$。新权重为 $(1-\rho)a+\rho MR$，其中 $\rho=0.6cD$。这使同一前缀、同一层的视觉总量和每项文本注意力在 FP32 混合中保持相同。完整 EMS 的最优传输风险用于原检测任务。缓解实验只使用偏好分布组件。

基础机制来源为 [SVAR](https://arxiv.org/html/2411.16724#S4)、[SADI](https://arxiv.org/html/2605.24957v1#S3)、[PAI](https://arxiv.org/html/2407.21771v1#S4)、[PADE](https://aclanthology.org/2026.findings-acl.748/)、[VHR](https://aclanthology.org/2025.acl-long.175/) 和 [VHR 作者代码](https://github.com/jinghan1he/VHR/blob/main/vhr.py)。此前 SADI 与后层注意力版本见 [SADI＋SAR/EMS 报告](LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md)。

## 24 图开发筛选

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉／全部提及 | 物体召回 ↑ | F1 ↑ | token／图 | 修复／新增幻觉图 | 改变序列 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVAR | 20.83% | 4.52% | 8／177 | 70.62% | 77.59% | 107.5 | — | — |
| SVAR＋SAR/EMS | 25.00% | 9.57% | 18／188 | 70.62% | 77.08% | 108.8 | 0／1 | 4 |
| SADI | 12.50% | 2.08% | 4／192 | 78.54% | 84.84% | 99.5 | — | — |
| SADI＋SAR/EMS 区域分配 | 12.50% | 2.06% | 4／194 | 77.71% | 84.10% | 100.5 | 0／0 | 2 |
| PAI 注意力组件 | 25.00% | 10.88% | 16／147 | 73.68% | 78.85% | 103.9 | — | — |
| PAI 注意力组件＋SAR/EMS | 25.00% | 10.88% | 16／147 | 73.68% | 78.85% | 103.5 | 0／0 | 2 |
| PADE 最终层局部实现 | 41.67% | 10.00% | 16／160 | 83.61% | 82.20% | 107.2 | — | — |
| PADE 局部版＋SAR/EMS | 41.67% | 10.00% | 16／160 | 83.61% | 82.20% | 107.2 | 0／0 | 0 |
| VHR 局部近似 | 41.67% | 11.59% | 19／164 | 74.17% | 75.18% | 106.9 | — | — |
| VHR 局部版＋投影写入选头 | 33.33% | 9.21% | 14／152 | 75.83% | 78.46% | 102.0 | 2／0 | 21 |
| SADI＋SAR/EMS 后层注意力（此前候选） | 8.33% | 1.56% | 3／192 | 78.54% | 85.24% | 99.5 | 1／0 | 2 |

新方案按预定规则选中 VHR 投影写入选头。独立图上增加选头错位对照。对每一层，将投影写入选出的固定 head mask 按编号循环平移 $\lfloor H/3\rfloor$，保留增益和选头数量。锁定配置见服务器 `rebuttal_holdout_plan.json`。

开发版 PADE 的新增臂在第 12 层积累 PAD 时读取 FP32 注意力，基础版读取模型已经转换到 FP16 的注意力。两臂在全部 24 张图上的生成 token 序列相同。已统一代码中的取值精度。

开发图没有描述到达 512 token 上限。区域分配触发后，赋权最大的第 12 层候选与最终生成 token 的一致率为 1.49%–4.73%，此前第 20 层后层注意力候选的一致率为 29.43%。前者使用多个候选的混合分布，这里的一致率只衡量权重最大的候选。每个触发层及每头搬移的平均注意力质量为 0.00147–0.00359。SVAR 区域分配的 10 次新增幻觉提及中，7 次来自同一张图。

## 独立 60 图验证

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉／全部提及 | 物体召回 ↑ | F1 ↑ | token／图 | 到达 512 token 上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVAR | 21.67% | 5.43% | 22／405 | 74.23% | 77.86% | 106.33 | 1 |
| SADI | 15.00% | 2.10% | 12／572 | 71.99% | 77.55% | 119.32 | 4 |
| SADI＋SAR/EMS 后层注意力 | 15.00% | 2.22% | 13／585 | 71.49% | 77.30% | 121.35 | 4 |
| VHR 局部近似 | 36.67% | 8.99% | 31／345 | 77.02% | 76.97% | 96.82 | 0 |
| VHR 局部版＋投影写入选头 | 35.00% | 8.22% | 30／365 | 77.48% | 77.46% | 99.50 | 0 |
| VHR 等头数错位对照 | 43.33% | 10.86% | 39／359 | 76.27% | 76.02% | 94.88 | 0 |

投影写入相对 VHR 局部基底修复 6 张并新增 5 张幻觉图，改变 51 张的生成序列。CHAIRs 差为 −1.67 个百分点（配对 95% 区间 [−11.67, 8.33]），CHAIRi 差为 −0.77 个百分点（[−3.66, 1.77]）。与错位对照相比修复 12 张并新增 7 张，CHAIRs 差为 −8.33 个百分点（[−21.67, 5.00]），CHAIRi 差为 −2.64 个百分点（[−6.48, 1.05]）。物体召回和描述长度相对 VHR 基底略增。点估计和开发集方向一致，配对区间仍宽。

后层注意力相对 SADI 改变 8 张的生成序列，60 张的逐图 CHAIRs 完全相同。CHAIRi 差为 +0.12 个百分点（[−0.12, 0.51]），额外的一次幻觉提及发生在图像 538844。后层注意力触发 1,697 次，暂定词与生成 token 一致 632 次。两个 SADI 版本都在图像 367452、404128、480225 和 574785 到达长度上限。

共同排除任一方法触顶的 5 张图后，55 张图上 SVAR、SADI 和 SADI 后层版的 CHAIRs/i 分别为 23.64%/5.84%、16.36%/3.18% 和 16.36%/3.33%。VHR 局部基底、投影写入和错位对照依次为 38.18%/9.20%、36.36%/8.43% 和 41.82%/10.42%。

完整评估记录位于 `rebuttal_holdout/metrics.first60.sadi-sadi_sar_ems_attention-svar-vhr_local-vhr_local_sar-vhr_local_sar_permuted.json`，同目录有逐图评分与生成分片。

## 后续复验

在看过第一批独立结果后，以种子 20261117 从原候选范围剩余的 140 张中无放回抽取并排序第二批 60 张。清单 SHA-256 为 `4755c222154395fe006c196e5392a8e14ec067266ac195c064edcaad4223ba36`。复验固定原 VHR 局部版、投影写入选头、错位对照和 SVAR。两批结果分别报告。120 张合并值按提及次数汇总 CHAIRi，配对 bootstrap 按批分层，各批每次重采样 60 张，合并属于顺序扩展后的描述性分析。

另外在开发图尝试将投影写入选头接入 SADI。层 5–18 用第一个生成查询的未加增益注意力，计算每个头的全上下文输出与局部遮图输出之差 $\Delta_h$。两臂采用同一离群过滤与 16 个头的配额，分别以 $\|\Delta_h\|_2^2$ 和 $\|W_{O,h}\Delta_h\|_2^2$ 排序。固定每层选头，分别对所选和其余头的 SADI logit 增量乘 1.25 和 0.75。逐头修正视觉 logits 的 log-sum-exp，使视觉总概率与普通 SADI 对齐。脚本用相同采样协议保存了选头索引。张量检查通过。

对新增 SADI 方案，若投影版在 24 图开发集相对普通 SADI 和局部 VHD 版都降低 CHAIRs，且物体召回较普通 SADI 至多下降 2 个百分点，则扩展到新的图像子集。CHAIRs 与两者持平时，若三个方法召回相差在 2 个百分点以内，且投影版相对两者的 CHAIRi 均至少降低 1 个百分点，也进入新图复验。

| 方法 | 第二批 60 图 CHAIRs/i | 第二批召回／F1 | 合并 120 图 CHAIRs/i | 合并召回／F1 | 合并幻觉／全部提及 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SVAR | 18.33%／3.95% | 75.07%／80.77% | 20.00%／4.67% | 74.65%／79.31% | 39／835 |
| VHR 局部近似 | 41.67%／10.23% | 82.58%／81.30% | 39.17%／9.65% | 79.80%／79.13% | 71／736 |
| VHR 局部版＋投影写入选头 | 33.33%／8.35% | 83.43%／83.34% | 34.17%／8.29% | 80.45%／80.40% | 63／760 |
| VHR 等头数错位对照 | 40.00%／9.52% | 80.14%／80.60% | 41.67%／10.18% | 78.20%／78.31% | 75／737 |

第二批相对 VHR 局部基底，投影写入选头修复 8 张并新增 3 张，CHAIRs 差为 −8.33 个百分点（配对 95% 区间 [−18.33, 1.67]），CHAIRi 差为 −1.88 个百分点（[−5.24, 1.33]）。合并 120 张修复 14 张并新增 8 张，CHAIRs 差为 −5.00 个百分点（[−12.50, 2.50]），CHAIRi 差为 −1.36 个百分点（[−3.54, 0.67]）。合并结果相对错位对照修复 21 张并新增 12 张，CHAIRs 差为 −7.50 个百分点（[−16.67, 1.67]），CHAIRi 差为 −1.89 个百分点（[−4.30, 0.63]）。相对基底，逐图去重的幻觉物体类别数从 65 降至 52。

第二批产物位于 `rebuttal_replication/`，合并结果为 `rebuttal_vhr_two_cohorts120.json`。两批主结果和分层配对区间另行复算一致。两批 CHAIR 的点估计支持投影写入选头作为一个可行线索，VHR 变体的绝对 CHAIR 指标仍高于 SVAR。

### SADI 增益选头开发对照

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 幻觉／全部提及 | 物体召回 ↑ | F1 ↑ | token／图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SADI | 12.50% | 2.08% | 4／192 | 78.54% | 84.84% | 99.50 |
| SADI＋局部 VHD 选头 | 8.33% | 1.66% | 3／181 | 69.93% | 79.47% | 103.54 |
| SADI＋投影写入选头 | 12.50% | 2.56% | 4／156 | 68.40% | 77.84% | 83.79 |

投影版相对普通 SADI 的 CHAIRs 相同，物体召回下降 10.14 个百分点，平均描述缩短 15.71 token。相对等预算的 VHD 版新增 1 张幻觉图。按开发筛选条件，该版本止于 24 图。聚合文件为 `pilot/metrics.first24.sadi-sadi_head_projected-sadi_head_vhd-sadi_sar_ems_attention-svar.json`。

## Rebuttal 可用表述

> We tested whether SAR's projected-write principle can guide attention interventions during generation. In a paired VHR-style ablation on LLaVA-1.5-7B, replacing the local head divergence with its output-projected magnitude reduced both CHAIR measures in two disjoint evaluation cohorts. Across their 120 images, CHAIRs decreased from 39.17% to 34.17% and CHAIRi from 9.65% to 8.29%, with object recall increasing from 79.80% to 80.45%. This provides preliminary evidence for the projected-write component. The paired 95% interval for the CHAIRs change was [−12.50, +2.50] percentage points. The joint SAR/EMS late-attention variant yielded no CHAIRs gain over SADI in the 60-image holdout.

## 运行验证

用 Windows 的 `py -3` 和服务器实验环境的 Python 对改动文件通过 `compileall`。Windows 的 `python` 应用别名没有指向可用解释器。服务器张量检查验证了后层 logit 再分配保持视觉 log-sum-exp，以及 FP32 概率区域再分配保持每头视觉总量和各项文本概率。生成运行采用双卡图片分片，CHAIR 评估核对每组恰有完整的同图 ID。
