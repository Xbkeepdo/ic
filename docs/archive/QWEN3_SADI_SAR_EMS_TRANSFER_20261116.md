# Qwen3-VL 的 SADI 与 SAR/EMS 后层注意力对照

本轮复用 LLaVA 独立对照的同一批 60 张 COCO val2014 图。Qwen3-VL-8B-Instruct 的三个实验臂是 SVAR 的注意力干预、SADI、SADI＋SAR/EMS 后层注意力。图像顺序为 `rebuttal_holdout`，SHA-256 为 `16d12227b1f838c921c179a3aa86afebb78dc30df043d4912e0b9216c6af7031`。层号和强度在生成前固定。Qwen3 三个版本共用项目提示词 `Describe this image.`、模型的原生聊天模板、BF16、贪心解码和 512 个新 token 的上限。Qwen3 的 SADI 与 SVAR 比较臂按现有 LLaVA 算子移植。 [此前 LLaVA 的同图结果](LLAVA_REBUTTAL_SAR_EMS_MITIGATION_20261116.md#独立-60-图验证) · [SADI 论文](https://arxiv.org/html/2605.24957v1#S4)

Qwen3-VL 文本解码有 36 层，第一轮固定使用与 LLaVA 相同的绝对层号，不按比例调整，也不按这批结果选层。三组各自使用原生带图 `generate`，保留 mRoPE、DeepStack 和 KV 缓存。干预只在文本 eager attention 中执行，使用 Qwen3 原生 Q/K 归一化和旋转后、已更新缓存的 QKV。8 个 KV 头扩展以对齐 32 个查询头，视觉范围由 processor 的连续图像 token 自动定位。WordNet 实体词的词首边界使用 Qwen3 的 BPE 标记。启动后每卡识别 12,656 个可触发实体 token。 [Qwen3-VL 配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json) · [所用 Transformers 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

SADI 在第 5–18 层按作者公式修改预测位置的视觉 logits，系数范围 0.25–0.80。后层增强版与 SADI 共用这个基底。第 18 层按实际投影写入量及 final-norm 词表概率建立目标语义分布 $Q$。第 20 层使用 FFN 之后读出的暂定实体词 $y$，FFN 对 $y$ 有正向提升才触发。第 20 层层前视觉状态与 FFN 输出的余弦经过温度 0.03 的 softmax 得到空间偏好 $P$。第 21、22 层用 $[Q-P]_+$、平方 Hellinger 距离和 FFN 的正向增益共同设定补偿，系数为 5。重分配保持每个头在视觉位置上的总概率，层号从 0 开始。 [原公式](LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md)

## 同图 60 张

| 模型 | 方法 | CHAIRs ↓ | CHAIRi ↓ | 物体召回 ↑ | F1 ↑ | 幻觉／全部提及 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-7B | SVAR | 21.67% | 5.43% | 74.23% | 77.86% | 22／405 |
| LLaVA-1.5-7B | SADI | 15.00% | 2.10% | 71.99% | 77.55% | 12／572 |
| LLaVA-1.5-7B | SADI＋SAR/EMS 后层版 | 15.00% | 2.22% | 71.49% | 77.30% | 13／585 |
| Qwen3-VL-8B-Instruct | SVAR | 40.00% | 10.54% | 77.06% | 73.36% | 47／446 |
| Qwen3-VL-8B-Instruct | SADI | 23.33% | 15.29% | 70.04% | 73.62% | 113／739 |
| Qwen3-VL-8B-Instruct | SADI＋SAR/EMS 后层版 | 16.67% | 1.58% | 68.24% | 72.15% | 11／695 |

Qwen3 上，后层模块相对 SADI 改变 42 张的生成序列，修复 4 张幻觉图，新增 0 张。按图片配对 bootstrap 5,000 次，CHAIRs/i 差分别为 −6.67 和 −13.71 个百分点，95% 区间为 [−13.33, −1.67] 与 [−35.32, 0.03]。物体召回和 F1 分别低 1.81 和 1.47 个百分点。对同图 SVAR，后层版修复 15 张、新增 1 张幻觉图，召回和 F1 分别低 8.82 和 1.21 个百分点。Qwen3 增强版共触发 500 步，其中 54 步的暂定词与干预后的生成 token 一致。

最大变化发生在图像 379162，普通 SADI 在 512 token 里把不存在的球提及 98 次，后层版在 193 token 里没有该幻觉。其余三张修复图是 378482、448288 和 516856。只排除 379162 后，SADI 与后层版的绝对幻觉提及为 15 与 11 次，CHAIRs/i 从 22.03%／2.37% 降至 16.95%／1.61%。

共同剔除 Qwen3 任一方法触及 512 token 上限的 21 张，保留 39 张。SVAR、SADI、后层版的 CHAIRs/i 依次为 41.03%／11.23%、25.64%／6.11%、20.51%／5.08%。SADI 的召回/F1 为 69.79%／72.12%，后层版为 69.36%／72.29%。这组作为长度敏感性分析，全量 60 张保留在主结果中。

同一批图上，对同一图像重采样并先求各模型的模块增量，再用 Qwen3 增量减 LLaVA 增量。CHAIRs 和 CHAIRi 的差异分别为 −6.67（95% 区间 [−13.33, −1.67]）和 −13.83（[−35.45, −0.09]）个百分点。后一结果包含重复幻觉的影响。本轮测量的是固定绝对层号配方在两个模型上的差异。

## 实现与验证

2026-11-17 核对原始记录：Qwen3 三臂各有 60 张唯一图片，均为 `native_qwen3_generate` 单图生成，图像清单一致。此 60 图与后续新增 500 图重合数为 0。后续 500 图使用动态批量推理，因此 60 图与 500 图同时改变了图片集合与推理协议。60 图的 CHAIRs 改善对应 4 张修复、0 张新增；500 图为 20 张修复、17 张新增。60 图仅有 4 对不一致样本，精确双侧 McNemar 检验 p=0.125；原 percentile bootstrap 区间不能据此解释成稳健显著增益。重复幻觉图 379162 对 CHAIRs 仅贡献一张图，其主要影响是 CHAIRi。目前证据支持这批 60 图上的有利点估计，尚不支持稳定提升约 7 个百分点。

第 5 层的合成 GQA 检查通过，复算了 SVAR 和 SADI 的最后查询 logits、softmax、输出，确认更早查询保持一致。真实运行的首批动态视觉范围、WordNet 触发频率和逐步后层调用数得到复核。增强版要求每次触发对应第 21、22 层各一次调用，生成步数必须与输出 token 数对应。第一次日志检查对多个文件使用 `tail -12` 报参数错误，改用 SFTP 读取日志。 [干预代码](../../features/qwen3_sar_ems_attention.py) · [生成入口](../../scripts/generation_time_evidence_drift_qwen3.py)

服务器产物位于 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3/`。配置和图像清单为 `protocol.json` 和 `cohort.json`。三臂的逐图生成和指标位于 `rebuttal_holdout/`，其中 `metrics.first60.sadi-sadi_sar_ems_attention-svar.json` 指向逐图 CHAIR，日志位于 `logs/`。跨模型配对 bootstrap 与未触顶分析保存在 `cross_model_increment.first60.json`。
