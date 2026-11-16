# Qwen3 的 SVAR＋SAR/EMS 后层注意力

用户要求检验此前 SADI 的同一后层干预能否迁移到 SVAR。使用 Qwen3-VL-8B-Instruct，从 COCO 500 图 seed 43 原始随机抽样顺序取前 100 张，排序后两卡各处理 50 张。两组为 SVAR 与 SVAR＋SAR/EMS，逐图使用相同批次、原生聊天模板、`Describe this image.`、BF16、贪心及 512 个新 token 上限。100 图的批次组成改变，因此两组重新生成。

第 5–18 层的基底为 SVAR 图像 logits 增益。新增模块在第 18 层构建 SAR 的投影视觉写入与目标词语义分布，在第 20 层读取 FFN 更新的视觉偏好和候选实体词增益。Hellinger 差异与正的证据差异决定第 21–22 层视觉区域的注意力重分配，强度为 5，温度为 0.03。层号从 0 起算。公式沿用此前 SADI 后层版。[公式与构造](LLAVA_GENERATION_SAR_EMS_ATTENTION_20261116.md)

服务器根目录为 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_svar100/`。抽样顺序、参数和源文件哈希分别保存在 `cohort.json`、`protocol.json`，生成位于 `coco100/`。单图与批量实现的单行输出及干预统计在 smoke 中通过一致性检查，日志为 `logs/smoke.log`。

本次为同一后层配方的模型内基底迁移实验。选取的 100 张属于已完成的 500 图。

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 物体召回 ↑ | 物体 F1 ↑ | 幻觉／全部物体提及 |
|---|---:|---:|---:|---:|---:|
| SVAR | 44.00% | 10.44% | 77.10% | 75.57% | 79／757 |
| SVAR＋SAR/EMS 后层注意力 | 40.00% | 9.28% | 76.64% | 76.38% | 68／733 |

CHAIRs 下降 4.00 个百分点，CHAIRi 下降 1.16 个百分点，F1 上升 0.80 个百分点，物体召回下降 0.47 个百分点。修复 8 张幻觉图，同时新增 4 张，93 张的 token 序列改变。100 张均触发过新增模块，共 1,627 个位置，其中 186 个暂定词与最终生成 token 一致。两层调用数与每图的触发数对应。

按图片配对 bootstrap 5,000 次、seed 43，CHAIRs 差值的 95% 区间为 [−11.00, +2.00] 个百分点，CHAIRi 为 [−3.03, +0.44]，F1 为 [−1.92, +3.75]。三项区间均跨零。这批 100 张上呈现有利的点估计，独立大样本中的稳定性仍待检验。

SVAR 有 0 张达到 512 token，后层引导版有 1 张达到。两组共同未触顶 99 图的敏感性结果为 CHAIRs 43.43%→39.39%，CHAIRi 10.38%→9.18%，F1 75.66%→76.60%。完整 100 张为主结果。

独立重算逐图评分与汇总数值一致。幻觉提及下降最多的图像为 `96769`，从 6 次降至 0 次，低于全部净减少的 11 次。移除该图后，CHAIRs 为 43.43%→40.40%，CHAIRi 为 9.88%→9.38%，F1 为 75.53%→76.14%，点估计方向相同。

评分文件为 `coco100/metrics.first100.svar-svar_sar_ems_attention.json`，逐图评分为同目录下 `chair_per_image.first100.svar-svar_sar_ems_attention.jsonl`。配对完整性与上限敏感性在 `validation.json`，触发核对在 `guidance_audit.json`。200 条描述、两组图像与批次完全匹配，单行一致性、编译和 diff 检查已通过。

本次本地辅助启动脚本初次写入因 PowerShell here-string 结束标记换行缺失而失败，修复后启动。首次新建无 PTY 的 SSH 连接因 stdin EOF 结束，随后使用受控 PTY 连接。实验生成与评分均完成。
