# 完整SADI后的区域引导：保留条件消融

状态：CPU谓词、两卡GPU等价性、完整10图续写、双评估及质量复算均已完成。放开保留条件确实改变了后续文本，但没有减少错误图或错误提及；本消融不进入100／500。

## 检验的假设

旧区域引导只在当前首选token立即改变且原目标logp下降时提交hidden/KV。它因此拒绝了可能保持当前词、但影响后续生成的视觉写入调整。本轮只隔离这个保留条件，不要求每次调整立即完成物体身份纠正，也不假定去掉条件一定改善。

完整SADI保留0-based5–18，源19、动作20–35。固定108维SAM-V2门控、阈值0.418672114610672、309词首token和首次类别规则沿用。对当前最终候选y，Q由原生视觉写入范数乘(1+log_s_y)归一化；P为本层FFN更新对视觉hpre的余弦softmax，温度0.03；D=1−sum(sqrt(QP))。目标类别别名语义概率的峰值选择3×3 ROI，视觉logits增强5D，逐头修正logsumexp保持原视觉质量；不编辑FFN，不编码crop或重建细前缀。

| 版本 | 提案保留条件 | 定位与预算 |
|---|---|---|
| 旧unverified | 新首选不同且原目标logp下降 | 旧完整输出复用，原ROI、5D |
| probability | 原目标logp下降，允许首选不变 | 原ROI、5D |
| direct | 非零提案均保留 | 原ROI、5D |
| shifted | 同direct | 同一实际D，ROI在原网格平移半幅 |

所有行先在同一未修改原生batch状态上计算提案，再统一提交被保留行的当前KV20–35；过去KV和缓存长度不变。关闭／零预算／强制拒绝均不得提交。保留次数仅表示执行，不能用原词概率下降代替正确性判断。

## 冻结设置与验证范围

沿用原开发100中的两个完整有序组，共10图，B/P6/206与4/314；同样Qwen3、BF16/eager/greedy、Describe this image.及cap512。该10图和多个难例已反复观察，不是独立验证集。

features/qwen3_region_commit.py继承未改动的旧控制器，独立覆盖提交判断；未修改旧源、旧结果或正在进行的锚点100实验。CPU验证72个保留条件组合通过，覆盖首选改变／不变、目标logp下降／相等／上升、零预算和强制拒绝。此检查仅证明条件语义，不证明GPU数值或效果。

GPU依次检查原SADI128、旧parent unverified128、兼容wordchange128精确匹配parent，再做disabled／zero／reject128精确SADI。比较token、逐步logits、36层currentKV及完整终局KV。之后三个新臂各生成完整512描述；控制器检查历史前缀和实际KV提交，统计same-token commit，不能把未覆盖的分支当作已验证。

SADI/SVAR原始10图控制按字节复用。旧unverified取原100内对应的完整组，JSON记录逐值保持并固定来源SHA。三新臂不复用改变后轨迹的风险分数，实际逐步重新计算。图像编码及静态语义在相同B/P组内经严格检查后共享。

## 当前执行与来源

根outputs/qwen3_region_commit10_20261120，入口scripts/run_qwen3_region_commit10.py，分析scripts/analyze_qwen3_region_commit10.py，质量复核scripts/analyze_qwen3_clva_quality.py。新父742655/start97744702，首个子742658；另一子仅在原锚点100的GPU1子741829/start97526209结束且成功后启动，身份保存workers_launch.json。GPU0已经完成原100中的50图后才用于新预检，未抢占原任务。

新GPU启动前，launcher误读旧根不存在的independent_summary.json而被守卫拦下，无新子进程。核查发现旧主评分失败记录保留为analysis_complete(passed=false)，独立恢复评分在analysis_recovery_complete(passed=true)。已逐一核成功恢复、provenance、原协议及10个caption文件SHA，生成source_recovery_certificate.json并在启动前验证。没有覆盖旧失败记录或放松完成条件；新算法、协议与输入未改。

先完整报告三新臂对SADI、旧unverified及错位控制的两种CHAIR、错误提及、召回/F1和重复变化，复查原图及完整描述。若出现值得继续检验的物理物体纠错，再进入固定原100；不凭10图进入独立500。不以改变概率或词片段代替效果，也不把seat／颜色别名差异、正确物体循环计为指导成功。

## 完整10图结果

两个GPU worker分别220.19／176.57秒，均返回0；GPU1等待原100的确切进程结束后才启动。原parent unverified和新wordchange128的tokens、logits、36层current及完整KV精确匹配；disabled／zero／reject128全部精确SADI。CPU主分析和重复复算均返回0。以下是sentence完整10图，单位%。

| 方法 | CHAIRs↓ | CHAIRi↓ | 物体召回↑ | 物体F1↑ | 错误／总提及 | 去重正确类别 | 重复正确提及 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SADI | 50.00 | 20.93 | 75.83 | 77.71 | 18／86 | 18 | 50 |
| 旧unverified | 50.00 | 20.93 | 75.83 | 77.71 | 18／86 | 18 | 50 |
| probability | 50.00 | 20.93 | 75.83 | 77.71 | 18／86 | 18 | 50 |
| direct | 50.00 | 20.69 | 77.08 | 79.07 | 18／87 | 19 | 50 |
| shifted | 50.00 | 18.11 | 77.08 | 79.07 | 23／127 | 19 | 85 |
| SVAR | 60.00 | 7.23 | 71.25 | 70.29 | 6／83 | 17 | 60 |

probability的7次提案保留4次，其中3次首选不变且当前KV确实变化。direct保留7次，其中6次为非零same-token commit；shifted保留6次，均为非零same-token commit。因此新保留分支已被实际覆盖，不能因没有纠错而推断代码未触发。

direct相对SADI增加了353411中可见的table提及，得到1个正确图像类别和F1 +1.36pp。这是有用的额外事实，但shifted同样出现；两者仍保留无支持的fork。probability虽改变353411、575293的句子，没有改善四个指标。direct的Ci小降来自这1个正确提及，不是错误减少；它也没有新增长循环。

shifted在243882仍写无支持table，并多5次错误提及；正确重复却增加35，导致Ci反而更低。不能按它的Ci数值选择该错位版本。完整文本与原图复查覆盖所有发生文字变化的3图353411、575293、243882，记录在changed_caption_review.json和visual_review.json。

该消融说明，旧的立即换词条件并不是前一版失败的唯一原因；允许中间状态影响后续生成也未消除已知fork／table错误。保留为实现和因果范围证据，不据旧10图宣称总体有效或无效，不再为这一保留条件单独花100图预算。

本轮新完整描述30份，原SADI/SVAR/旧区域控制复用，正式3臂每臂encoder额外调用0（每组仅原生前缀编码1次、组内共享），无crop编码／细前缀重建。三臂生成GPU秒数分别73.21／74.62／74.19，连同模型加载、等候原GPU1和验证／评分总墙钟5分23秒。本机87个远端产物共2,752,755字节，逐项SHA核验通过。
