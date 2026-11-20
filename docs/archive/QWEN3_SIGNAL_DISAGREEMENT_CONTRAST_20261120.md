# FFN相对目标视觉证据的过度偏好：固定前缀诊断

状态：76项CPU检查、两卡6位置／24个分支、零路径与完整36层KV恢复、独立FP32候选重算全部通过。excess与uniform在六个实际argmax上完全相同，未发现相对匹配控制的额外候选纠正，不推进完整10／100或500。此处不是完整描述效果。

## 动机与固定方法

前一完整100图中，直接把整个P_FFN作为反事实负路径prior，与同强度uniform在逐图假对象及提及数上完全相同。这里进一步检验：只放大FFN相对当前目标视觉证据Q过度偏好的部分，是否能产生更合适的对比方向。该构造利用两者差异，不改变原SADI，也不增加强度或层数。

Qwen3-VL-8B-Instruct、BF16/eager，原6／4图有序组，243882和353411的6个已有前缀。层号0-based；SADI5–18完整，源19，负尾部20–35。Q为原生BF16逐patch写入范数乘(1+log_s)的归一化分布；P为实际FFN19更新与视觉hpre余弦温度0.03的softmax。四个prior为：

```text
R_excess  = Normalize(relu(P−Q))
R_deficit = Normalize(relu(Q−P))
R_uniform = 1/M
R_rolled  = Roll(R_excess, floor(M/2))
```

零质量时归一化回uniform。共同eta=.5D，D=1−sum(sqrt(QP))，不对每种prior另算或另调强度。各头视觉条件注意力与prior混合，局部视觉总量保留、非视觉不变；正路径native完整SADI，输出`2native−negative`，alpha1/beta.1。FFN只读不编辑，负分支不提交KV，历史及当前KV逐次恢复。

原309列19层log_s直接复用已核SHA缓存，无新增视觉LM-head投影，无crop或标签训练。所有候选身份来自实际最终原生首选词；用预先固定的已探索6位置强制诊断，不执行旧MLP门控，不作独立验证声明。协议在本轮分支输出前冻结，并明确要求相对uniform／错位有可用变化才考虑完整生成。

## 结果

| 图像／step | 原候选 | excess | deficit反方向 | uniform | rolled excess |
|---|---|---|---|---|---|
| 243882／4 | person | person | person | person | person |
| 243882／7 | book | book | book | book | book |
| 243882／27 | table | table | table | table | table |
| 353411／24 | sandwich | sandwich | partially | sandwich | large |
| 353411／97 | fork | nap | nap | nap | nap |
| 353411／114 | person | person | person | person | person |

表中为真实argmax和其token文本，不用不稳定topk首项代替。无支持的table仍不变；fork四种都变为nap，不能把它当作excess的独立作用。旧完整10图已证明nap后可以续写回fork，因此此表不能提前解释成完整纠错。partially／large也只是当前单token，没有新后文，不能据此认定误删真实三明治。

独立numpy FP32按保存的全部允许候选及两路径logits重算24个选择，全一致；Q/P归一化、D/eta和四分支20–35真实调用均核对。CPU检查还覆盖excess的逐元素独立构造、交换Q/P得到反方向、错位可逆、相等分布回uniform；CUDA未初始化。

两卡原encoder各一次（原6／4图整组），新增语义head调用0。worker回放12.50／21.30秒，不含模型构建；含加载父墙钟36.06秒。父745079/start98564060、子745080／745081全结束，pipeline[0,0]。本机`scripts/experiment_snapshots/qwen3_signal_disagreement_contrast_probe`取回12产物207,280字节核SHA，另有`independent_disagreement_audit.json`。

本轮没有证据支持扩大该构造，保留为负诊断，不从6个位置断言一般无效。下一步可检查偏好表示本身：当前P比较FFN更新与视觉hpre；另一个待检验的定义是比较查询FFN更新与各视觉位置经过同一FFN得到的更新。后者能在原prefill hook中只读取得，不需要额外FFN前向，仍只用于注意力分支。这个替代表示尚未实现或验证。

入口`scripts/probe_qwen3_signal_disagreement_contrast.py`，新prior函数`features/qwen3_signal_disagreement_contrast.py`，attention仍复用冻结`qwen3_signal_preference_contrast.py`。根`outputs/qwen3_signal_disagreement_contrast_probe_20261120`。独立500未消费。
