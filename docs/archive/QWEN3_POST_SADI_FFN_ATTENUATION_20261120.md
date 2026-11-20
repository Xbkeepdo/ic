# 完整 SADI 后的注意力指导与 FFN 标量衰减

六个旧固定前缀、六臂共 36 次检查已完成，没有形成值得扩大为完整描述的候选。保留完整 SADI 代码层号 5–18，源层 19 读取证据和 FFN，后续注意力作用于 20–35 层，未截短原 SADI。

整体 FFN 输出衰减借鉴 [FADE 的式 5](https://arxiv.org/html/2606.29431v4#S3.SS4)，不是作者方法的完整复现。本轮在原生源层用 $r+(1-\gamma)u$ 替代 $r+u$，不增加源层 MLP 调用。视觉注意力保持每头视觉总量，以 $\eta=0.5D$ 向目标证据 $Q$ 混合，$D=1-\sum\sqrt{QP}$；P 为原 FFN 更新与视觉 hpre 的余弦 softmax，温度 0.03。

六臂为仅注意力、仅 FFN 衰减、联合、注意力＋固定衰减、错位 Q＋同样衰减、仅固定衰减。信号衰减取 $\gamma=0.5D$；固定系数在本轮输出产生前由旧六个位置的均值确定，为 0.162719264626503。这只匹配六位置的名义总衰减，不称完整生成的实际作用量严格匹配。所有主选择使用同一原生 beta=0.1 候选集，也另存未加掩码的实际首选词。

| 目标位置 | 真实／错误 | 19 层 FFN 局部目标 logit 增益 | 仅注意力 | 仅 FFN | 联合 | 联合固定 | 联合错位 | 仅固定 FFN |
|---|---|---:|---|---|---|---|---|---|
| 243882/4 person | 真实 | −0.0422 | person | person | person | person | close | person |
| 243882/7 book | 真实 | −0.5378 | book | book | book | book | book | book |
| 243882/27 table | 错误 | −0.4999 | table | table | table | table | table | table |
| 353411/24 sandwich | 真实 | −1.3388 | sandwich | sandwich | sandwich | sandwich | sandwich | sandwich |
| 353411/97 fork | 错误 | +0.4120 | fork | fork | fork | fork | fork | fork |
| 353411/114 person | 真实 | +1.3872 | person | person | person | person | person | person |

有／无原生候选掩码的实际首选词在这 36 项上全部相同，不能将失败归因于掩码排除了替代词。主选择的 36 个 argmax 已由保存的全部 eligible FP32 logits 独立复算；未宣称对未保存的完整词表 logits 作独立验证。

仅 FFN 衰减使 table 最终原词 logp 上升 0.07258，而 fork 下降 0.03546；联合后分别上升 0.00906、0.01218。19 层原 FFN 对这两个错误目标的局部作用符号相反，单纯缩小更新并没有共同的抑制方向。局部 logit-lens 增益也不是最终概率效应；本轮仅提供两个错误位置的证据，不推断所有 FFN 或所有模型都具有该规律。

原生目标/logp、Q/P 与旧双路径缓存精确一致。post-attention 的真实残差与实际 FFN 更新之和逐值等于原 19 层输出；零衰减／零注意力的 hidden、logits 和全部 36 层 KV 精确匹配。所有非零分支恢复源 anchor 和全部当前 KV，其他 batch 行精确保持原生状态。每卡图像编码一次，新增视觉语义 head 和源 MLP 调用均为零。

CPU 15 项检查及双卡流水线均通过，没有计算失败或恢复。两卡 worker 为 16.04/27.02 秒，含模型加载的墙钟约 42.05 秒。远端根 `outputs/qwen3_post_sadi_ffn_attenuation_probe_20261120`，父 PID 745929/start_ticks 98936647。14 项远端产物共 130,816 字节已下载到本机 `scripts/experiment_snapshots/qwen3_post_sadi_ffn_attenuation_probe` 并核 SHA，另存 `independent_attenuation_audit.json`。

这不是新的完整描述结果，没有新增 CHAIR/F1，也未使用独立 500。下一项检验改为读取原 exact-risk 的 EMD 运输计划，用其条件转移关系实际搬移视觉注意力；先核实仓库仅用计划计算标量风险，尚无该计划指导生成的 Qwen3 实现，再做独立小检查。
