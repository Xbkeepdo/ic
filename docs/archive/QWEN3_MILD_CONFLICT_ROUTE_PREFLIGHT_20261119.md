# 完整SADI后的文本路线抑制：幅度减半预检与100图协议

上一轮0–0.5文本系数在开发100图降低了自动CHAIR，但损失真实召回并产生长重复。本次仅检验抑制幅度是否过大，保持模型、SADI层、读出、梯度、头排序规则、候选范围、接受与KV恢复机制不变。不将开发集幅度探索称作独立验证。

## 唯一方法改动

沿用完整SADI的0-based5–18层，在19–35层加入CRG式Conflict-B文本路线干预。VRI排序、Q/P信号排序、信号移动16/32头的错位对照保持原规则。对原本选择的头：

```text
g_mild = 1 − 0.5 × (1 − g_strong)
g_strong ∈ [0,0.5]  →  g_mild ∈ [0.5,0.75]
O'_h = O_h + (g_mild,h − 1) × O_text,h
```

未选择头仍为1。这里减半的是从文本路线移除的比例，未把父类gamma直接设为0.5。这个父类的gamma仅允许0或1。新子类显式保存原系数、实际系数、实际native扰动范数及重新计算的线性效应。排序规则保持，并不声称生成分歧后的候选、头或总剂量在两种幅度间完全相同。

我们的目标条件视觉写入分布Q与FFN偏好P仍用于风险R_h=Σ_j aVis_hj P_j/(Q_j+P_j)，它决定头排序，不在本次重新拟合。保留k=8、温度0.03、309词元及首次词元/前缀类别范围；不用学习门控和GT。一个native点反向查询供17层共用，重放实际各层文本路线，最终原目标log概率下降才接受，否则恢复hidden及current20–35KV，历史KV不改。该适配不等于完整CRG论文实现或next层动态刷新。

## 验证

CPU独立验证通过：移除比例0精确恒等、比例1精确恢复强版本；0.5保持头选择和次序，局部系数多重集一致；源payload不变；重新计算的效应符合独立线性目标。最大效应误差7.3159e−7，0次CUDA初始化。

B2/B16真实32token预检通过，冻结462项输入，request SHA：`c8b5ec1c1df7710bea55ffc8690b7ad6eaa915f013faf6a5ae08c4b947419379`。scale0、关闭、gamma0、实际zero-route及强制拒绝时tokens、logits及全部36层当前KV精确等于SADI；scale1精确等于上一轮强版本；flat-signal精确等于减半VRI。历史KV、单次图像编码、0次MLP与OT检查通过，全部native gradient hidden差0，峰值小于30GiB。

正常减半B2三排序均2提案、2保留；B16均22提案、21保留。这只是实现覆盖，未用于判断效果。预检父717813及workers717816/717817均实际missing，随后才启动100图。

## 100图协议

Qwen3-VL-8B-Instruct，旧开发100图，原22有序组、最大batch16、512新token、Describe this image.、BF16/eager/greedy、两4080SUPER。三个新臂mild_vri/mild_signal/mild_roll各完整新生成100图；六个对照SADI/SVAR/causal_fixed/route_vri/route_signal/route_roll逐字复用，共报告九组。不使用旧9图无阳性证书。

主比较是减半signal相对减半VRI/roll，幅度消融是相对同排序强版本。完整100图sentence为主，legacy保留；自动运行机制、独立数值、配对、分母四项审核。信号需要超越对应VRI/roll的两CHAIR及错误计数，SADI参考下F1损失≤1.5pp、召回损失≤2pp，并排查重复/词义和原图内容，才考虑预留seed2026111903独立500。测试集不重抽、不用于调参。

本次精简changed_text_cases里每图分数中重复嵌入的统计数组，保留数值、文字和简短因果事件；完整统计仍在原生成及评估中。它是分析文件格式调整，未改生成、CHAIR评分、原图或配对比较。

远端`outputs/qwen3_mild_conflict_route_pilot_20261119`。父717866/start93102442、workers717869/717870/start93102534已核实际运行。运行记录后续以/proc与产物为准。当前不报告未完成实验的部分指标。
