# Qwen3：FFN正增益门控的开发100消融

本消融未通过预设CHAIRs严格下降条件，不扩大到500，不放宽门槛。

只改变一个门控：原前缀首次类别、I/E_l的p_real<.5之外，要求19层FFN的目标logit增益G19(y)>0。不要求旧符号门控的V19(y)<0。局部读出一次并供未变的注意力动作复用。

G19(y)=z_y(N(r19+u19))−z_y(N(r19))，r为注意力后的残差、u为该层实际FFN更新，N为最终RMSNorm。该量是局部logit-lens差值，不是最终目标概率变化；动作最终仍需实际原目标概率下降才接受。

SADI5–18、源／动作19、重算当前20–35、梯度方向、FFN空间偏好和rho=.5D不变。旧100、cap512/maxB16、原22批次及GPU分配，阈值为自然零符号，无新训练。设计使用了之前100及两500的信息，因此这是开发消融。

## 复用与运行

旧V<0且G>0门控在23个风险阳性事件中，只有图52966／step113同时G>0且V>=0。它原来就是单图批次。其余21批／99图的全部门控决定相同，且原图、代码、动作、批次和接受条件一致，因此复用完整轨迹；对受影响的原批次重新生成。新审计重新核对全部谓词及缓存行的数值、元数据和token，不能将99缓存轨迹称作新生成。

原整批队列665180在GPU生成开始前取消，原协议和状态保留`...ffn-promotion-pilot`；新独立root为`...ffn-promotion-cached-pilot`。没有修改冻结旧协议。新driver665357、worker666003均结束。实际新推理14.19秒，含模型加载、64token原生SADI复核和两口径评分共54.78秒。

## 完整100结果

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 召回 ↑ | F1 ↑ | 假提及／总提及 |
|---|---:|---:|---:|---:|---:|
| SADI | 30.00 | 7.9393 | 70.5960 | 74.9497 | 89／1121 |
| 原首次类别引导 | 30.00 | 7.7059 | 70.5960 | 75.1124 | 87／1129 |
| FFN正增益门控 | 30.00 | 7.7818 | 70.5960 | 75.1124 | 87／1118 |

相对SADI：Cs0pp，Ci−0.1576pp（95%配对区间[−0.4291,+0.0086]pp），F1+0.1627pp。没有自动CHAIRs消除或新增。相对原首次类别引导，Ci上升0.0758pp。11提案10保留，12个风险阳性事件被G条件阻止；91无保留动作图逐token同SADI，局部contrast共11次且每次供梯度动作复用。

整体数值与旧V<0且G>0门控相同，但52966的token和文本不同，不能因汇总指标相同就宣称所有输出完全一样。99复用图已验证完全相同，52966的差异属于新生成分支。

cached_caption_pilot、head_feedback、ffn_promotion_cached、caption_denominator四项检查通过；300条／两口径各24有向区间、全部新门控／前缀／物理动作以及99复用／1新生成来源已核对。90个产物和冻结源码SHA取回，压缩包SHA `70c1b80f21e867b38ee7621b2f9087b5758333502134712bcd14bb51f6ef2951`。

## 检查过程中的失败记录

保存简要状态时首次8KB断言失败，未覆盖原状态，压缩已归档内容后通过。汇总诊断一度把“指标相同”误推为100图输出相同，完整token／文本断言发现52966不同；已改为保存真实差异列表及False一致性字段，没有改动生成或评分。CPU门控集成检查另覆盖负／零G拒绝、正G且正V允许、低风险不进入、读出一次复用和钩子恢复。

服务器`/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-ffn-promotion-cached-pilot`；本机` scripts/experiment_snapshots/qwen3_ffn_promotion_cached_pilot/`。入口`run_qwen3_ffn_promotion_cached_pilot.py`，特征`qwen3_probe_ffn_promotion.py`，主审计`audit_qwen3_ffn_promotion_cached.py`。
