# 当前任务

目标：在四个 COCO4000-512 ENDAC generation 上补齐 object-token baseline，独立比较
MetaToken、SVAR、DHCP、ProjectAway、TokenGrounding 和 InsLen 的检测效果。

约束：

- 样本键固定为 `(image_id, response_token_idx, target_token_id)`，不重新生成或改标签。
- TokenGrounding=`concat(ads_per_layer, cgc_per_layer)`，不单列 ADS/CGC。
- InsLen 固定 K=16、m=4、tau=10、alpha=2，仓库权重
  `0.6*S_cls + 0.4*S_ccs`。
- InsLen object state 使用官方 LLaVA 实现对应的最终 decoder norm；主提取结束后只重算这一项并按精确三元 key 覆盖。
- 六种 baseline 均独立检测，不与 risk、F 或 DGST 特征拼接。
- 复用 key/label/generation 一致的已有特征，只补缺失 payload。

当前工作：

- [x] 接入 TokenGrounding 和 InsLen baseline payload。
- [x] 四 wrapper 捕获 InsLen 所需 instruction states。
- [x] 小样本验证 causal position、层索引和公式。
- [x] 四模型补提并合并统一 features.pkl。
- [x] 四模型 InsLen final-norm 修正与审计完成。
- [x] 独立训练/直接评分完成，seeds 43/44/45 汇总完成。

完整结果：`outputs/COCO4000-512-ENDAC-BASELINES-V2/complete_results.md`。

方案：`outputs/endac_object_baselines_extraction_plan.md`。
进度：`outputs/COCO4000-512-ENDAC-BASELINES-V2/progress.md`。

补充单图分析（2026-09-21）：

- LLaVA image 831 的 `log_s_hpre` raw attention / target distribution / source
  distribution 已按第 18/21/24/27/30/32 层独立取 Top-32 并绘制冷色底图。
- `skateboard`/`handbag` 的独立 Top-16 区域已完成 2×2 交叉 `F_log` 评分。
- 原始 32×576 矩阵、Top-K 索引、逐层/六层汇总及 PNG/PDF 位于
  `outputs/llava_1_5_7b/COCO831-LOGS-PRE-TOPK-ANALYSIS/`。
