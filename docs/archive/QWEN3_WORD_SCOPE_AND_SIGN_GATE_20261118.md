# Qwen3：完整词范围诊断与视觉／FFN符号门控

以下诊断和规则在原500图仍生成时固定，只使用旧开发集。原500现已完成，结果见QWEN3_FIRST_CLASS_FRESH500_20261118.md，原方法和设置保持冻结。

## 缓存上的完整词范围诊断

`scripts/analyze_qwen3_word_boundary_scope.py`直接读取旧100图SADI文本、真实输出token偏移和已有I/E_l概率，不重新生成，也不训练模型。用完整的已保存文本识别单词和多词别名，保留连字符词整体，并应用与现有历史类别门控相同的名词还原和最长别名匹配。

| 范围 | 候选 | 保留 | 可标注物体起始位置 | 非物体起始位置 | 幻觉位置 | 真实位置 | p_real<0.5幻觉／真实 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前309词形＋前缀类别门控 | 441 | 209 | 191 | 18 | 29 | 162 | 11／7 |
| 事后完整词＋前缀类别门控 | 441 | 231 | 223 | 8 | 31 | 192 | 11／8 |

完整词范围消除了`man-made`、`ballast`、`bedspread`等首片段误命中，也恢复了`giraffes`、`teddy bear`等多token词。231个保留候选中，按实际分词估计有196个需要至少1步前瞻、25个需要2步、10个需要3步。这是确认词边界所需的最低步数统计，不是运行时间测量。

该结果使用了保存的未来模型文本，尚不是在线算法或缓解效果。已标注物体上的风险触发数没有增加，非物体位置又没有完整缓存检测概率，不能据此声称检测精度或推理成本已经改善。颜色与水果、复合名词、上下文词性仍存在歧义。因此本轮先不实现完整模型前瞻，保留这项诊断作为后续词范围改进的依据。

产物为`outputs/qwen3_cached_word_boundary_scope/analysis.json`，本机在`scripts/experiment_snapshots/qwen3_cached_word_boundary_scope/`。脚本记录全部输入SHA，原209候选及11／7触发计数独立复现。

## 在500图结果前固定的符号门控

原开发集已经出现：目标概率虽然下降，但干预作用在`man-made`等非物体词上；该例19层FFN目标增益为负。因此新增一个不拟合数值阈值的规则：当前词已经通过类别首次和原I/E_l风险门控后，只有局部视觉写入对它的支持为负、FFN对它的目标logit增益为正时，才执行原注意力回收。

令19层注意力后残差为r、FFN更新为u、视觉注意力写入为v，最终归一化和LM头组成读出L。两个局部量为：

\[
V(y)=L_y(r)-L_y(r-v),\qquad G(y)=L_y(r+u)-L_y(r).
\]

门控为`p_real<0.5`且`V(y)<0`且`G(y)>0`。原SADI5–18层、19层动作、负向最终概率梯度证据、FFN空间偏离、回收比例以及拒绝后的20–35层当前KV恢复均不变。这两个读出只算一次，并缓存允许行供原动作使用，避免重复计算。

规则利用的是用户的两种关系：当前视觉写入是否支持目标，以及FFN更新是否把预测推向与此冲突的方向。它使用完整I/E_l检测信号和视觉／FFN读出，不等同于原完整SAM或精确OT risk公式。

实现为`features/qwen3_probe_first_class_conflict.py`，100图入口`scripts/run_qwen3_probe_first_class_conflict_pilot.py`。原driver654401/workers654974/654975已完成。协议记录`independent500_status_at_design=generating`，规则在测试指标前冻结。

旧100图Cs/Ci/F1：SADI30.00/7.9393/74.9498，类别首次30.00/7.7059/75.1124，新增符号30.00/7.7818/75.1124；幻觉提及89／1121、87／1129、87／1118。风险23次，符号阻止13，提案10保留9，立即换词2；缓存读出命中10等于梯度探测10，hidden差0。两次换词仍为颜色词与remote→similar gadget，没有新增可靠纠正病例。300条、两口径48区间及类别／符号／动作／分母审计全部通过。

首次专用审计错误预期统计字段名；修正为实际`conflict_sign_gate`、`require_negative_visual_support`、`require_positive_ffn_gain`、`contrast_readout_cache_hits`后通过，未改或重跑生成。

满足预设的旧100 Cs/Ci不恶化、F1损失<=1.5pp、幻觉减少及审计条件后，启动`scripts/run_qwen3_cached_first_class_conflict500.py`。root为`generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-first-class-conflict-500`，原driver656619及workers656622/656623已结束。仅新增一臂，复用原500三个对照、相同有序批次、E_l及各SHA；不读取测试指标决定参数。与原类别首次版共同报告，不能称为另一批独立测试。

## 论文核对与后续方向

SADI在中间层对视觉attention logits做加性调整；作者的背景显式扣除消融反而增加幻觉。当前回收操作保留原视觉位置、从历史文本移入注意力，区别于该消融。这个结果提示应继续检查上下文破坏，而不是仅增大干预强度。[SADI原文，§4.3](https://arxiv.org/html/2605.24957v1)

AttnReal只从超过注意力平均值阈值的历史输出位置回收注意力，并均匀分给视觉位置。当前实现从所有具有正目标效应的历史位置回收，尚未要求它们是attention sinks；因此后续可检验仅回收sink是否更能保留语言上下文。这是尚未实施的改进假设，不能把当前结果当作完整AttnReal复现。[AttnReal原文，§3.3](https://arxiv.org/html/2503.08342v2)

CRG区分视觉路线与文本路线的目标效应，并对冲突路线作选择性干预。当前新规则比较的是视觉写入与FFN增益，后者是用户直觉的另一组成部分；当前代码没有复现CRG的逐头VRI排名或完整路线门控。[CRG原文，§3–4](https://arxiv.org/html/2605.24024v1)

OPERA针对历史token上的注意力聚合使用惩罚与回溯。它与当前已观察的“同一输出词未改，但后续KV轨迹进入重复”提供了不同的处理方向。下一步可先检查仅保留真正改变当前词的提案能否避免潜在状态扰动；这项推断来自当前病例，不是OPERA已经证明的结论。[OPERA，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_OPERA_Alleviating_Hallucination_in_Multi-Modal_Large_Language_Models_via_Over-Trust_CVPR_2024_paper.html)

周额度最近用13%、余87%，目标保持active。继续跟进同集合新增500图臂，完成后核对四臂、复用来源、两口径指标及逐事件，再决定后续实验。


## 同集合500图已完成

新增符号门控Cs26.20/Ci3.66/F173.78，SADI26.60/3.72/73.96，原类别首次候选24.60/3.43/73.90。符号版仅净少9次幻觉且由一张图贡献全部净减少，两CHAIR区间跨零，暂不采用。五项完整审计通过、101个产物和冻结源码已取回。详见[500图符号消融](QWEN3_FIRST_CLASS_SIGN_ABLATION500_20261118.md)。第二批独立500图已启动复验原候选，固定方法参数。
