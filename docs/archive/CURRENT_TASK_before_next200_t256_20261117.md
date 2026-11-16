# 当前任务

已完成（2026-11-17）：用户确认的Qwen3、SADI基底、新100张COCO、max batch16、两卡、零起19/20层反馈实验。三臂全部300条描述及评分完成，无GPU续跑任务。

主结果（CHAIRs／CHAIRi／召回／F1）：SADI 29.00／8.10／68.26／73.20%；下一层反馈27.00／6.31／67.64／73.06%；同层反馈32.00／5.93／67.00／71.44%。下一层修复4、新增2幻觉图；同层修复3、新增6。幻觉提及84→75／84，总物体提及1037→1188／1416。同层CHAIRi下降来自分母增加，不能称减少幻觉次数。下一层CHAIRi的95%配对区间[-3.35,-0.25]pp，CHAIRs区间[-7,+3]pp；同层F1下降区间[-3.28,-0.42]pp。

共同未触及512 token的69图是输出条件化补充：SADI／next／same两CHAIR分别27.54/6.36、27.54/6.04、30.43/7.21%；不能用该子集替代主100图。达到512 token图数26/26/28。下一层净少9次幻觉提及，其中图152336贡献6次；移除此图两CHAIR仍改善。独立复核已全部通过，audit.json已保存。同层分母新增379次，前3重复物体词图贡献73.61%；排除前5图CI优势只剩0.13pp。next排除分母前3图后两组分母同为944、幻觉83→75，仍有实际减少。所有排除仅事后敏感性分析。

完整报告：`docs/archive/QWEN3_SADI_LAYER_FEEDBACK100_20261117.md`。服务器输出：`/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_layer_feedback100_b16/`。本地完整下载：`C:/Users/xbo/AppData/Local/Temp/qwen3_layer_feedback100_checked/`。主指标在coco100/metrics.first100.*.json，逐图评分chair_per_image.*.jsonl；validation.layer_feedback100_b16.json记录批次、触发、配对、补充与用时。run_status=complete，评分exit0。

协议：SADI5–18；next用Q18/P18→A19、Q19/P19→A20；same分别19/20探测后局部重算当前位置attention+FFN，不重复KV。每层新预算，强度5/tau0.03/noun+positive-gain固定，未按新100中途效果调整。greedy、BF16、Describe this image.、512新token。与旧5250图零重合，PCG64 seed43从35254候选抽100；集合SHA7ec202db8393734fa395253435613a48d652b1e9da00734118333d4bf7afe00c，21源码哈希已冻结。每方法22匹配批次，32图实际B16；无OOM/异常。

验证：旧2图24步，零强度/空noun四组与SADI逐步logits/token exact；36层KV增长正确，self-attention每步一次；same真实FFN重算及prefill非末位置不变。注意noop绕过replay，不能据此宣称同层重算数值逐位等价。主方法100张均触发，next1854/same1712逐图逐层事件。两增强生成计时约增加10.3%，受长度影响。全程北京时间06:34:11→06:52:59，18分48秒（含加载/验证/评估）。

实现：features/qwen3_sar_ems_layer_feedback.py、scripts/generation_time_qwen3_layer_feedback.py、scripts/verify_qwen3_layer_feedback.py。脚本与冻结清单：scripts/experiment_snapshots/qwen3_layer_feedback100_b16/。本轮源码/报告只同步本地与服务器，未推GitHub。保留原ac已有改动。

此前实验已归档：`docs/archive/CURRENT_TASK_before_layer_feedback100_20261117.md`。旧新200 SVAR增强两CHAIR变差；旧单图500无稳定改善。不要把不同图片、batch、基底的实验混为同条件复现。本轮未运行同图SVAR或原后层21/22配方。

本轮准备的非GPU问题：首次误读根CURRENT_TASK路径、旧交互SSH会话失效，均已修正。实现、协议fixture与GPU验证通过，没有重采样或失败后重跑新100。
