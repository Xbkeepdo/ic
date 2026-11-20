# Qwen3：固定反对证据的FFN一致性接受（完成，未通过推进条件）

候选只改变提案的接受条件。SVAR5–18、原I/E_l风险门控、首次token/类别、19层注意力回收及20–35当前查询重算全部保持。

原目标y的最终log概率梯度给出视觉写入的正负作用，原注意力提案据此计算反对证据Q。这里固定这份提案前的Q，不在重算后移动它。令P_before和P_after为原FFN更新及实际注意力重算后FFN更新与同一组视觉hpre的余弦softmax，tau=.03；D=1−sum sqrt(Q*P)。仅在原目标实际log概率下降并且D_after<D_before时接受，否则恢复hidden及20–35当前KV。

Q是反对原目标的视觉证据，不是支持原目标的概率。该接受条件检验FFN更新是否更接近本次干预使用的证据；它没有直接缩放FFN，也没有把FFN目标gain为正等同幻觉。是否减少真实幻觉仍需生成和看图验证。

## 计算与记录

在_budget中暂存原normalizer已经算出的Q；按梯度probe原行索引映射到最终有效提案行。临时hook读取_recompute已经调用的源层MLP输出。只多做一次P_after空间偏好点积，不额外运行模型、MLP或LM-head。每个真正提案只保存Q/P_after两个一维小向量和标量D，可独立复算；不保存完整注意力或hidden。

CPU集成已覆盖：Q映射到乱序/筛选后的图片行、复用一次FFN调用、临时hook清理、严格D接受，以及乱序行对应的hidden和16层当前KV恢复。语法/导入与准备预检通过，122项依赖冻结。

## GPU预检与100协议

原driver669095及workers669661/669662已结束。原生SVAR预检包含关闭、零预算、父拒绝、强制新反馈拒绝和正常模式，逐步比较logits和20–35当前KV。对比原父类的拒绝参考前向次数，检查没有新增model/sourceFFN/LM-head调用。需实际触发新的拒绝分支，不能仅检查无触发时输出一致。

之后旧100、原22有序批次/GPU分片、cap512/maxB16；复制SVAR、原风险版及SADI参考，复用22批E_l，只生成新一臂。语义缓存和所有来源SHA核对。新条件比“概率下降”更严格，所以保留原始log_probability_decreases计数；不能用通用head审计中accepted==probability_decreases的旧假设，也不重写计数来通过检查。

独立audit_qwen3_svar_ffn_alignment.py核对全部前缀类别、物理动作、原概率变化、严格新接受条件、无保留图逐token同SVAR，以及从保存向量重算D_after。另跑cached_caption_pilot验证400条和两口径各48有向区间、caption_denominator，以及显式SVAR主提及比较。

预先条件：相对SVAR Cs/Ci/假提及严格下降，F1损失<=1.5pp、召回<=2pp、共同去最大收益图Ci不更差，并至少一项看图确认的真实纠正，才考虑500；不事后改变门槛。原风险版完整报告，无论它是否更强。GPU预检和完整100评分已完成，结果见下。

root `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-svar-ffn-alignment-pilot`。入口run_qwen3_svar_ffn_alignment_pilot.py，特征qwen3_svar_ffn_alignment.py，预检verify_qwen3_svar_ffn_alignment.py。沿用既有视觉写入/FFN空间偏好组件，不称完整SAM/SAM-V2。周额度最新余85%，目标active。

51份冻结源码、协议和词表已按122依赖中的SHA取回本机快照qwen3_svar_ffn_alignment_pilot；GPU预检已通过。

## GPU预检与完整100结果

新增反馈拒绝实际触发2次、保留0；live也2次因D不减小而拒绝。各64个行/步的完整logits和20–35当前KV精确同原生SVAR。与原父类的拒绝参考相比，model/sourceFFN/LM-head调用分别均为32/36/67，没有新增前向；这是相对已有注意力提案的接受逻辑比较，不是说整个方法相对SVAR没有额外开销。

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 召回 ↑ | F1 ↑ | 假提及／总提及 |
|---|---:|---:|---:|---:|---:|
| SVAR | 45.00 | 9.5710 | 76.8421 | 76.7225 | 87／909 |
| 原风险引导 | 42.00 | 8.7838 | 76.2849 | 76.9634 | 78／888 |
| 固定Q的FFN一致性接受 | 43.00 | 9.1111 | 76.4849 | 76.9011 | 82／900 |

相对SVAR，Cs−2pp，Ci−.4598pp，F1+.1786pp；两项CHAIR差值区间均未严格低于零。相对原风险引导，Cs高1pp、Ci高.3273pp。

虽然平均数字改善，但最大的提及下降图369675贡献−5次，正好等于全量净−5。共同去掉它后Ci从9.1314升至9.1518，未通过预设“去最大收益图Ci不更差”。此图是train car被算成道路car的词义边界；两项自动Cs消除仍是train car和male bird。保留全部结果，不扩大500，不改变门槛。

52提案的原目标概率都下降；D条件拒绝28、保留24。80无保留图逐token同SVAR，只有15条完整序列改变。51次梯度/固定Q读取、51次后偏好读取；从保存向量独立重算D最大误差5.96e−8，最小保留D下降8.03e−5。这说明新条件实现并实际起作用，但没有带来更好的幻觉缓解结果。

三项检查通过：cached_caption_pilot、svar_ffn_alignment、caption_denominator；400条、两口径各48有向区间、全部新接受条件/回滚/类别及显式SVAR提及都核对。106个产物/冻结源码SHA取回，压缩包SHA `c470aa1f2dbaa170e80b19ba767698a6e0931f0d98ba51244c12441fdabbb72f`。新增单臂及评分墙钟6.55分钟。当前方法不晋级。
