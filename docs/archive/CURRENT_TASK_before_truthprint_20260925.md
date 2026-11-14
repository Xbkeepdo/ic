# 当前任务

SAR/EMS 附录 8:2（2026-09-25，已完成）：四模型按精确 object-token key
对齐原 EMS、SAR、视觉写入 I 与五种 T 的现有特征，保持 K=32、tau=0.03；
同一 3200/800 划分及探针设置训练 8 条路线 × 3 seed × 4 模型，共 96 次。
主路线逐种子 AUROC 与既有 8:2 完全一致；完整结果和 LaTeX 表见
`outputs/log_s_sar_ems_appendix_82/`，附录草稿见 `outputs/log_s_appendix/`。

进行中：四模型 Log-S targeted mechanism ablation，固定 COCO ENDAC
3200/400/400 图像划分、`K=32`、`tau=0.03`、seeds 43/44/45。

- [x] 盘点现有 SAR、EMS 和 `I`特征，四模型 sample ID 精确对齐。
- [x] 接入 8 组统一训练与三随机种汇总。
- [x] 完成 `gfchair` 提取去 target semantics、`OT_G`、联合区域 JS
  与 FFN-input source 四类新特征。
- [x] 完成四模型 8 组训练与数值审计；共 96 次训练，零缺失，完整 EMS
  对旧缓存的最大绝对误差均为 0。

完整结果：`outputs/log_s_mechanism_ablation_811/complete_results.md`。

K=64 follow-up（已完成）：保持 8:1:1、`tau=0.03`、seeds 43/44/45 与八组
机制条件不变；四模型共 96 次训练，零缺失，完整 EMS 锚点误差均为 0。
结果：`outputs/log_s_mechanism_ablation_k064_811/complete_results.md`。

K=32 复杂度 follow-up（已完成）：LLaVA/Qwen3 同一组 50 张 COCO 图比较
SAR+EMS、SVAR、DHCP 与 TokenGrounding-MLP；计时包含读图、前向和后处理。
结果：`outputs/complexity_k032_50/{analysis.md,complexity_table.tex}`。

目标：在四个 COCO4000-512 ENDAC generation 上补齐 object-token baseline，独立比较
MetaToken、SVAR、DHCP、ProjectAway、TokenGrounding 和 InsLen 的检测效果。

约束：

- 样本键固定为 `(image_id, response_token_idx, target_token_id)`，不重新生成或改标签。
- TokenGrounding=`concat(ads_per_layer, cgc_per_layer)`，不单列 ADS/CGC。
- 除 MetaToken 外，baseline 使用论文/官方仓库的原生检测与训练方式；
  TokenGrounding 分别报告原仓库 XGB、RF、sklearn MLP 三套网格结果。
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
- [x] 原生训练方法审计完成；替换错误的 TokenGrounding SVAR-MLP 头并重训四模型。

完整结果：`outputs/COCO4000-512-ENDAC-BASELINES-V2/complete_results.md`。

原生训练审计：`outputs/COCO4000-512-ENDAC-BASELINES-V2/native_training_audit.md`。

方案：`outputs/endac_object_baselines_extraction_plan.md`。
进度：`outputs/COCO4000-512-ENDAC-BASELINES-V2/progress.md`。

补充单图分析（2026-09-21）：LLaVA image 831 的 A/T/P 六层 Top-32、
`skateboard`/`handbag` 独立 Top-16 与交叉 `F_log` 位于
`outputs/llava_1_5_7b/COCO831-LOGS-PRE-TOPK-ANALYSIS/`。
- 2026-09-23 按用户修正为主线 `K=32, tau=0.03` risk，并将 P 换算至同一
  tau；重画深蓝底柔化热点的 Top-32/Top-16 图，保存 32 层 risk 曲线、逐层
  CSV 与交叉 F_log 于
  `outputs/llava_1_5_7b/COCO831-LOGS-PRE-K32-TAU003-REFERENCE-STYLE/`。
- 第 27 层另导出 skateboard/handbag 各自的 raw attention、T、P 六张
  独立 Top-16 PNG/PDF，位于上述目录的 `layer27_top16/`。
- 2026-09-25 扩展三模型 COCO4000 Log-S/F_log 热力图：72 个 token、
  54 张模型图像、18 组配对；Qwen3 新增 5 组需复核，InternVL 新增 6 组配对及 11 个单 token。八组配对附录与 13 张单 token 附录位于 `outputs/log_s_paper_heatmap_candidates/{appendix_pairs,appendix_single}/`；图库、逐层指标和取舍见 `outputs/log_s_paper_heatmap_candidates/{README.md,expansion_2026-09-25.md}`。

已完成（2026-09-24）：以逐 patch `F_log` 贡献归一化为新 T，完成四模型
`K={32,64}`、`tau={0.03,0.05,0.07}` 提取与三种子训练；保持原 8:2
测试划分，内部验证选择共享 `K=64,tau=0.03`。完整网格与原 T 对比见
`outputs/flog_target_k_tau_ablation/{summary.md,complete_results.md}`。

已完成（2026-09-25）：同一网格分别用 `softmax_visual(attention*log_s)`、
`softmax_visual(||a_m||*(1+log_s))` 作为 T，四模型提取、三种子验证和测试
网格均完成；结果见 `outputs/softmax_target_k_tau_ablation/report.md`。
