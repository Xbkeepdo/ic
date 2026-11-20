# 完整 SADI 后的证据路径与 FFN 偏好路径对比

状态：六个固定前缀、五条注意力路径、六组评分共 36 个决策已完成；CPU 101 项及全部 KV 恢复检查通过。所有首选词均与原生目标相同，没有推进完整描述或独立 500 图。

保留 SADI 代码层号 5–18，在其后比较两条由信号指导的注意力路径。此前仅改变负分支的视觉 FFN 输出正差版本在完整 10 图没有优于匹配控制，低 CHAIRi 来自重复正确提及，见 `QWEN3_VISUAL_FFN_PREFERENCE_20261120.md`。本轮将视觉证据也用于正分支，而不是仅用于计算负分支强度。

## 固定方案

Qwen3-VL-8B-Instruct，BF16/eager。原六个开发位置、原 6/4 有序批次，不新增样本。源层 19 读取实际注意力写入、目标语义权重及 FFN 更新；只读重算 20–35 层当前查询，结束后恢复全部当前 KV，不写入未来生成状态。

证据 $Q$ 沿用写入范数乘 $(1+\log s)$ 后归一化；$P$ 为视觉 hpre 与当前查询 FFN 更新的余弦 softmax，温度 0.03；$P_{uv}$ 为视觉 FFN 输出与查询 FFN 更新的余弦 softmax。五条视图对应 $Q,P,P_{uv},U,\operatorname{roll}(P)$，其中 $U$ 为均匀分布。每头视觉注意力在其原视觉总量内与相应先验混合，所有视图共用

$$\eta=0.5\left(1-\sum_j\sqrt{Q_jP_j}\right).$$

五条路径只各计算一次，复用于六组评分。主评分为 $2z_Q-z_P$；对照分别为 $2z_Q-z_{uv}$、$2z_U-z_P$、$2z_Q-z_U$、$2z_Q-z_{roll(P)}$ 和 $z_Q$。所有评分共用原生 SADI 概率不低于原生最大概率 0.1 倍的候选集，不使用各正分支自己的候选集。

这是两条后续注意力路径的机制检查。$\eta=0$ 的重算与原生路径精确相同；将对比系数设为零则只剩正路径，不能把它误称为关闭全部干预。

## 结果

| 图像／位置 | 原生目标 | SAR−FFN | SAR−UV | 均匀−FFN | SAR−均匀 | SAR−错位 FFN | 仅 SAR |
|---|---|---|---|---|---|---|---|
| 243882／4 | person | person | person | person | person | person | person |
| 243882／7 | book | book | book | book | book | book | book |
| 243882／27 | table | table | table | table | table | table | table |
| 353411／24 | sandwich | sandwich | sandwich | sandwich | sandwich | sandwich | sandwich |
| 353411／97 | fork | fork | fork | fork | fork | fork | fork |
| 353411／114 | person | person | person | person | person | person | person |

表中 “SAR−FFN” 简写对应 $2z_Q-z_P$，并非特征数值相减。错误 table 和 fork 在六种评分下均未改变。相同原生候选集内，fork 条件分数由仅 SAR 的 0.49952 变为 SAR−FFN 的 0.44437，但 SAR−均匀和 SAR−错位也同为 0.44437；table 在 SAR−FFN 与仅 SAR 下均为 0.77730，SAR−均匀及 SAR−错位为 0.75491。没有显示出完整双信号相对控制的独有纠错候选。条件分数不是校准的幻觉概率。

本轮只拒绝将这版构造直接扩大为完整描述试验，不将六个开发位置推断为一般无效结论。主线仍保留完整 SADI 后增加信号指导；不回到两者各占一半层的限制。

## 检查、时间与产物

原生目标与 logp 匹配旧固定前缀；源 Q/P/P_uv 与旧数组逐值相同，原 prefill 读取的视觉 FFN 输出与保存的 BF16 数组逐值相同。语义概率直接取原 309 列缓存，新增视觉 LM-head 投影为零。每卡图像编码一次、源视觉 FFN hook 一次，没有新增源层 MLP 前向。

零路径 hidden/logits 精确匹配，所有五视图和全部六位置的 36 层完整 KV 哈希均保持不变；各视图实际覆盖 20–35 层。CPU 101 项覆盖原生候选掩码、独立逐元素评分、相同视图及零对比的含义。保存的 FP32 视图 logits 由 NumPy 独立重算全部 36 组实际 argmax，全部通过。

两 worker 耗时 13.58/22.52 秒，含建模管线墙钟 36.05 秒，无失败恢复。根目录 `outputs/qwen3_dual_attention_contrast_probe_20261120`，父 PID 745725/start_ticks 98855137，子进程身份见 `workers_launch.json`。管线返回 [0,0]，父子均结束。本机 `scripts/experiment_snapshots/qwen3_dual_attention_contrast_probe` 保存并核 SHA 的 14 项远端产物共 251,258 字节，另有 `independent_dual_attention_audit.json`。

入口 `scripts/probe_qwen3_dual_attention_contrast.py`、评分函数 `features/qwen3_dual_attention_contrast.py`。协议、源码、原始数值和审计保留；独立 500 图仍未消费。
