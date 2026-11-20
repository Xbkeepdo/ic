# SAR/EMS选择关键视觉位置的对比解码：固定前缀诊断

状态：76项CPU检查、双卡6个位置的24个分支和零重算／全部KV复原检查全部通过。两个已核查无支持物体table、fork在四个策略下都未改变，不推进本方案完整100图。此处没有完整描述或CHAIR效果结论。上一轮完整SADI后六层直接扩展已闭合为负结果，见QWEN3_FULL_SADI_NEXT6_20261120.md。

## 动机和来源

本轮改变视觉干预的用途：在一个只读分支中遮掉关键视觉位置，以原始路径和遮挡路径的输出差异进行对比解码。这样可以检查目标预测对视觉证据的响应，而不是仅增加几层相同的注意力偏置。

[ASCD原文](https://arxiv.org/html/2506.14766v1)通过正向注意力引导和关键视觉token抑制构造对比路径。本轮借用关键视觉遮挡与对比解码思想，保留本项目完整SADI为正路径，用SAR／EMS决定负路径遮挡位置。没有复现作者的离线文本头选择、正向头增强、各层独立关键位置选择或完整双轨缓存；不能称为原版ASCD结果。

## 固定方案

Qwen3-VL-8B-Instruct，BF16/eager。SADI原5–18层保留，以下层号从0开始。复用旧原有序批次6图／4图、相同prompt和前缀，目标是已保存的最终首选token，不用中间层首选词替代。六个位置来自243882和353411；它们已多次用于探索，不是独立样本。

源19层按原生BF16逐视觉位置写入范数和final-norm hpre概率计算：

```text
Q_j = Normalize(write_norm_j × (1 + log_s_j(y)))
P_j = softmax(cos(hpre_j, FFN_update19)/0.03)
```

每种策略都选ceil(0.1M)个视觉位置；相同值用位置顺序打破平局。

| 策略 | 选择分数 |
|---|---|
| attention | 源19层各头实际视觉注意力之和 |
| sar | Q |
| sar_ems | relu(Q−P)，全部为0时使用Q |
| shifted | 与sar_ems同一组位置，沿扁平视觉序列移动floor(M/2) |

选中位置在只读20–35层当前query中置logit为负无穷，随后对全部上下文softmax，故视觉总注意力会变化。所有策略使用同样位置数量、层范围和对比强度；本轮只比较位置选择，不另加D强度缩放。

```text
z_contrast = 2 z_native − z_erased
只允许 z_native(y) ≥ max(z_native) + log(0.1) 的候选
```

原正路径的全部历史KV保留。遮挡分支仅借用当前查询尾部计算，结束后恢复当前20–35层KV和视觉槽；不提交负路径状态。每次试验核对完整36层缓存哈希，零遮挡重算要求hidden、logits、全部KV与原SADI精确相同。分支沿同一个原前缀进行，未做完整描述生成。

## 复用与检查范围

原36层309列语义缓存仅取19层log_s，核原batch IDs、span、prompt和文件SHA。没有重新投影视觉LM-head、crop编码或训练分类器；动态注意力和FFN状态来自真实前缀回放。两卡各原图encoder一次。CPU检查覆盖四种选择数量、错位对应关系、对比公式、零对比和相同两路径；不把CPU通过当作缓解有效。

入口scripts/probe_qwen3_signal_erasure_contrast.py；算子features/qwen3_signal_erasure_contrast.py。远端outputs/qwen3_signal_erasure_contrast_probe_20261120，父743334/start98153556、worker743335／743336。协议先于本轮分支输出固定；尚未使用保留500。

只有诊断出现值得完整生成验证的物体候选变化，才准备完整100图实验；此处的即时换词、相对概率或删除名词均不能单独证明缓解幻觉。

## 诊断结果与后续

| 图像／step | 原最终候选 | attention | SAR | SAR/EMS | 错位 |
|---|---|---|---|---|---|
| 243882／4 | person | close | close | close | person |
| 243882／7 | book | book | book | book | book |
| 243882／27 | table | table | table | table | table |
| 353411／24 | sandwich | large | sandwich | sandwich | sandwich |
| 353411／97 | fork | fork | fork | fork | fork |
| 353411／114 | person | person | person | person | person |

表中是同一个固定前缀下的单个下一token，不是完整新句子。close、large的词性变化不能据此认定物体遗漏或纠错。图243882的table和353411的fork均有此前原图核查记录。

SAR/EMS遮挡分支对table的原目标log probability降低0.04020，对fork降低0.09647；这会使`2z_native−z_erased`朝强化这些目标的方向移动，而不是提供期望的抑制方向。SAR单独遮挡亦相同，分别降低0.07879和0.11338。不能假设遮掉语义上支持当前目标的视觉位置就一定产生“更幻觉”的分支：目标本身无支持时，视觉隐藏状态的语义读出仍可能响应错误类别。

该小诊断没有证明一般规律，但足以拒绝把这版直接扩大到100／500。下一项固定前缀诊断比较由Q和P构造的注意力路径，而非继续增加同类遮挡强度。其方案另见QWEN3_SIGNAL_PREFERENCE_CONTRAST_20261120.md。

两worker原图编码各1次，新增语义LM-head投影均0，回放耗时12.17／22.94秒（不含建模加载），父驱动含加载约38.06秒。父和两个worker均已结束，pipeline返回[0,0]。本机`scripts/experiment_snapshots/qwen3_signal_erasure_contrast_probe`保存12个远端产物、120,934字节，逐文件SHA已验证。
