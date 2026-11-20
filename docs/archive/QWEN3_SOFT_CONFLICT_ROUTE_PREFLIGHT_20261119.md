# 完整SADI后的中性零剂量文本路线干预：预检与开发协议

## 问题与改动

保留SADI代码中0-based5–18层，在19–35层加入模块。此前Q/P信号只决定头排序，接近中性也施加预定0–0.5或0.5–0.75文本系数；强/半幅度100图均漏掉真实物体。此次只检验一种新规则：由信号的正偏离决定文本移除量，避免信号微弱时仍施加强干预。不改变SADI、候选词范围、梯度目标、实际文本路线算子或最终接受条件。

沿用309词元、首次token/首次规范类别筛选。每个原生点一次反向得到17层敏感度，在每层Conflict-B（视觉效应<0且非视觉文本效应>0）内，固定选择最小VRI的至多8头。VRI以保存的FP32方向效应转Python双精度计算，索引打破并列。

目标条件视觉证据Q=Norm(write_norm*(1+log_s_finalnorm_hpre(y)))，FFN偏好P=softmax(cos(native_FFN,hpre_visual)/0.03)。R_h=Σ_j conditional_visual_attention_hj P_j/(Q_j+P_j)。这是偏好相对证据的量，不是幻觉概率。

固定新公式：d_h=0.5[2R_h−1]_+=[R_h−0.5]_+，g_h=1−d_h，最大移除0.5；未选中头g=1。Q=P时R=0.5，各头精确零剂量；R=0.5反过来不代表Q=P。三种模式：signal按各头d作用；flat给所选头相同的平均d；roll在同一所选集合按VRI次序循环移动半个集合的d。signal和roll的局部系数多重集完全匹配，flat仅平均移除量匹配，不声称投影扰动范数或后续轨迹匹配。

实际重放本层当前query的文本路线O'_h=O_h+(g_h−1)O_text,h，不重新归一化attention。本层视觉路线不直接修改；后层query可变化。所有g在同一原生点固定。最终原目标log概率下降才保留，否则恢复hidden/current20–35KV；历史KV不改。这不是逐层刷新next，也不是完整CRG复现。

## 缓存CPU诊断

先固定公式与对照，再查已封存100图84个共同、未接受任何前序动作的原生点，沿用上次bitwise效应一致与SADI前缀目标一致的成员，不将后续分歧点当反事实。signal/roll原始R在84点逐值相同。

80个自动real点，8344个所选头中2102个有正剂量，平均移除0.027023，最大0.479872；没有整点全零。2个自动假点201个所选头，67个正剂量，平均0.038120。后两例为train station场所/真实desk类别问题，不是已确认物理幻觉。该诊断只支持强度明显变化和覆盖，不能估计CHAIR或声称更准的检测。stove和chair等真词仍可被触发。

原始CPU规则协议SHA：9b95d0e19fc2fba496ff5e7dd7331b4f2058e15157b2ea4f936873e35d4b0c1f。没有新模型调用/拟合/阈值搜索。

## 数值与真实GPU预检

CPU独立四头算术、直接二维线性目标、Q=P中性与低R恒等、最大剂量0.5、非Conflict-B不改、最多8头、signal/roll多重集、flat平均剂量、原payload/文本路线不改均通过。实际张量独立目标最大误差3.4237e−7；0次CUDA初始化。

两卡B2/B16、32token预检通过，464冻结输入。request SHA：c5a1f6988c90b451a0424bc0f7b1fcea0ff69625341158112b5973ed94d51548。关闭/gamma0/实际zero/强制拒绝/中性signal/中性flat均精确恢复SADI的tokens、logits和36层当前KV。历史KV张量、单encoder、0学习探针/OT、native梯度隐藏差0均通过。B2正常signal/flat/roll提案均2、保留2/0/1；B16均20、保留13/15/12。这不是缓解效果证据。峰23,611,235,328字节<30GiB。

预检父718238/start93576564，workers718241/718242/start93576600均实际missing。CPU6项、GPU15项本机逐SHA核过，新6源与本机相同。

## 100图协议

Qwen3-VL-8B-Instruct，旧开发100图、原22有序组、最大batch16、512新token、Describe this image.、BF16/eager/greedy、两4080SUPER。soft_signal/soft_flat/soft_roll各100张全新生成，六个对照SADI/SVAR/causal_fixed/mild_vri/mild_signal/mild_roll字节复用，共9臂。三新臂都计算Q/P并共享RAM图像语义缓存，无额外encoder；不保存全词表或hidden矩阵。

运行入口仅用子类适配route_mode→dose_mode，并纠正继承统计中的排名字段：本方案使用信号调剂量、不改VRI头排序。预检计算子类及其父实现未修改；入口没有新增数值方法。原父排名标记仅保留为parent_legacy_rank_flag。

先全量sentence/legacy及机制、独立数值、配对、分母四项审核，再查所有fix/new/top3错误/召回变化及signal对flat/roll的规范假类变化。开发筛选需signal超越两个剂量对照的两CHAIR与假计数，相对SADI的F1损失≤1.5pp、召回损失≤2pp，并有原图支持。只有有效才使用预留seed2026111903独立500；不重抽或按验证调参。

远端outputs/qwen3_soft_conflict_route_pilot_20261119。父718304/start93628208，workers718307/718308/start93628295已核S/R，两卡实际运行。正式分数尚未读取。官方额度已用37%余63%。

## 相关工作与区别

[CRG](https://arxiv.org/html/2605.24024v1)的视觉/文本路线分解与冲突头筛选是算子来源，本项目的Q/P剂量与后SADI层段是适配。

[Visual-Grounding-Guided Head Suppression](https://aclanthology.org/2026.aacl-long.152/)强调区分何时干预和干预哪些头，用区域反事实识别视觉敏感头；此处未执行该离线识别，不能称为复现。

[VGD](https://arxiv.org/html/2607.27823v1)使用带符号的跨头特征和轻量验证器，再回滚并抑制风险物体及同义词。其启发是保护有根据的内容；本方案不训练该验证器、不执行该词级重生成流程。

[VGA](https://arxiv.org/html/2511.20032v1)用视觉语义引导和覆盖更新加入视觉value，并平衡不同头。项目曾在LLaVA做过24图迁移，本轮没有重复该试验或把它当成Qwen3结果。
