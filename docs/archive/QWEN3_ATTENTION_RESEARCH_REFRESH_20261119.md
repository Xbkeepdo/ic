# 注意力干预文献补查与下一假设

本轮只读取公开论文正文与官方摘要，不把论文报告数值当成本项目结果。

| 来源 | 本次核对的内容 | 对本项目的启发 |
|---|---|---|
| [CAI](https://arxiv.org/html/2606.29847) | 按词与视觉特征相似度增强视觉attention logits，并用熵/深度门控；式5为A加abs(A)乘权重。 | 借用有界非负增强形式，以我们的证据/FFN信号替换位置和触发信息。 |
| [ILVAD](https://arxiv.org/html/2605.20965) | 由早期生成的跨层显著激活形成视觉图，增强视觉证据和与证据关联的文本。 | 提醒我们区分提升证据与单纯压低当前目标。先不引入额外早期生成窗口。 |
| [FADE](https://arxiv.org/html/2606.29431) | 在选定层缩小FFN更新，式5把更新乘1−alpha。 | 可作未来辅助；本项目仍以注意力操作为主，不把它换成主要FFN衰减。 |
| [CausalLens官方摘要](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html) | 摘要描述视觉/文本/系统路径与可靠头；[官方仓库](https://github.com/jijy20/CausalLens)的README已核对，PDF/网页直接读取403。 | 尚未核完整公式，不据摘要实现或声称原版复现。 |

## 待检验的项目适配

保留完整SADI5–18，后续19–35进行同原生点的联合注意力调整。Q=Norm(write_norm*(1+log_s))，P=softmax(cos(uFFN,hpre_visual)/.03)，D=Hellinger²(Q,P)。候选空间权重w=Q/(Q+P)在[0,1]，向Q相对P更强的视觉位置分配增强；使用Delta visual_logit=abs(native_visual_logit)*D*w。若Q=P，则D=0，不改变注意力。

这是本项目的新组合，不是CAI原公式或原版复现。固定对照取w=.5/D=1，仅预算对照取w=.5并保留D。gamma=1来自所借算子不额外加尺度，风险w=1与该预算参数是不同概念。

新方向只增强视觉位置，保持对非视觉logits不直接加项；在精确softmax下视觉总质量不下降。用当前原生context缓存计算实际context差，复用原联合尾层回放和原目标实际概率下降接受规则。FFN只提供偏好信号，不直接改FFN。先验证CPU边界与GPU零/拒绝等价性，之后再跑开发100。

目标条件证据与FFN偏离仍为方法主线；物体纠正既可直接发生也可由上游干预间接引起，须如实说明，不将即时身份替换设为额外成功条件。新500仅用于值得继续验证的候选，排除已经用于开发的500样本。
