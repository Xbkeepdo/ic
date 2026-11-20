# 视觉证据与FFN偏好构造双注意力路径：固定前缀诊断

状态：56项CPU检查、两卡6位置／24个对比选择、真实零混合和全部KV恢复检查已完成。native−P在fork位置改为nap，进入完整10图筛查；Q−P未改变两个无支持物体。前一项遮挡对比诊断已闭合，不重跑其遮挡分支。此处没有完整描述的缓解效果结论。

本轮继续保留完整SADI5–18，用同一个最终候选token读取源19层的Q_SAR和P_FFN。针对前一项“遮挡反而强化不支持目标”的结果，直接比较视觉写入支持和FFN视觉偏好对应的路径。思想参考ASCD的正负注意力路径，不使用作者完整头选择或其算子，不称原版复现。[ASCD方法](https://arxiv.org/html/2506.14766v1)。

设源19层D=1−sum(sqrt(QP))，eta=0.5D。尾部每层原本实际注意力为A，头h在视觉位置的总量为m_h。分支在20–35层逐层使用：

```text
A'_hj = (1−eta) A_hj + eta m_h R_j       j为视觉位置
A'_hk = A_hk                            k为非视觉位置
```

R分别为Q、P、均匀分布及沿扁平视觉位置移动floor(M/2)的P。每条路径均从原生同一源19状态起算，各层注意力使用该路径实际经过前层修改后的当前状态；R与eta固定来自原生19层。各层在其当前状态下保留视觉总注意力，分支之间后续状态不同，因此不称全路径的实际视觉量严格匹配。没有编辑FFN；FFN提供P和D。

固定alpha1、beta0.1，对比为`2z_positive−z_negative`，候选限制由positive路径的max logit＋log(beta)决定。比较native−P、Q−P、Q−uniform、Q−rolledP，另报告不对比的Q路径首选词。错位与均匀参照用于检查空间信号，强度不另调。D=0时eta=0，所有路径回到原SADI。

复用已有6／4原组、同六个已探索位置、原309列19层log_s缓存，新增视觉LM-head投影0。负／正分支都只作诊断，所有20–35当前KV恢复，逐次核完整36层缓存不变，零混合的hidden／logits／KV要求精确原生。保存全部允许候选及正负logits，可独立复算对比选择；保存Q/P核D和实际层调用。不是完整生成，不能把即时候选变化当成CHAIR改善。

入口`scripts/probe_qwen3_signal_preference_contrast.py`，算子`features/qwen3_signal_preference_contrast.py`；根`outputs/qwen3_signal_preference_contrast_probe_20261120`。协议在当前分支结果前冻结，未使用预留500。若两个无支持目标仍无合适方向，或损害真实内容，则不推进这版完整生成。

## 六位置结果

以下均为实际argmax，按词表最低ID解决并列。保存的topk展示不保证相同的并列顺序。

| 图像／step | 原最终候选 | native−P | Q−P | Q−uniform | Q−rolledP |
|---|---|---|---|---|---|
| 243882／4 | person | person | person | person | person |
| 243882／7 | book | book | book | book | book |
| 243882／27 | table | table | table | table | table |
| 353411／24 | sandwich | sandwich | sandwich | sandwich | sandwich |
| 353411／97 | fork | nap | fork | fork | fork |
| 353411／114 | person | person | person | person | person |

native−P保留四个原真实目标，在已核查无fork的图353411上改为nap首subtoken，有可能续成napkin，但尚未生成后文，不能提前称物体纠正。Q−P和两个控制保留fork，Q路径并未带来预期增益。新完整10图只继续native正路径对比，分离是否需要SAR/EMS的D缩放，以及P的空间位置。

独立CPU按保存的全部允许候选及FP32正负logits，复算24个实际argmax均一致；Q/P归一化、D和eta、四路径16层真实调用全核对。首次审核误将topk首项当作argmax，在person位置两项并列时触发；实际argmax为person，topk首项为book。已保存`independent_preference_audit.json`记录两个显示并列，不改原topk、分支或方法。对比分支fork→nap的实际token ID25859通过独立复算。

父743592/start98187170及worker743593／743594均结束，pipeline返回[0,0]。两worker实际回放13.13／19.91秒（不含加载），含加载父驱动34.05秒。原图encoder各1次，新增语义head调用0；没有新完整描述。本机`scripts/experiment_snapshots/qwen3_signal_preference_contrast_probe`取回12个远端产物204,377字节并核SHA，另存独立CPU审核。

## 完整10图后续

根`outputs/qwen3_live_preference_contrast10_20261120`，父743745/start98247430。新入口`run_qwen3_live_preference_contrast10.py`和控制器`qwen3_live_preference_contrast.py`；源19、只读尾部20–35不变，完整SADI5–18保留。四臂p_drift、p_fixed、uniform_drift、rolled_drift，分别使用P与eta=.5D、P与eta=.5、uniform与eta=.5D、半序列错位P与eta=.5D；均native正路径、alpha1/beta.1。

在线目标范围为已冻结309个COCO物体首token，包含重复词，不用旧MLP或历史类别过滤；prefill不重算。和原next6全WordNet12,656词范围不同，必须分开描述。视觉编码和19层静态语义只在原B/P组内共享，首次复用核逐值hidden、tokens及权重身份。每个生成step选中图片共同重算一次完整B的尾部，负路径只替换当前KV并在返回前恢复。模型输出logits作对比，原生生成历史KV保留，FFN本身不编辑。

正式40份新描述前，两卡各自要求native128与旧缓存匹配；disabled128、真实非零反事实配合alpha0、真实eta0重算均需tokens、logits、36层current和完整KV精确SADI，再保持模型加载继续512生成。SADI／SVAR20份控制复用；生成后先全10两评估和5000次bootstrap独立复算，再做完整文本／原图／重复核查，不按单个nap首token决定晋级。

两卡128预检已全部通过，实际zero-alpha和zero-amount分别覆盖29／19个目标事件，disabled0事件；匹配含每步logits／36层current及终局完整KV。父743745、worker743746／743747在核对start_ticks后继续运行四臂正式512；两卡预检后保持模型和视觉缓存，没有重载或另开GPU任务。

## 完整10图结果

四臂各10份完整描述已生成，主sentence和辅助legacy评分、独立5000次配对bootstrap复算、质量分析均完成。以下均为百分数；10图已用于探索，不是独立测试。

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 召回 ↑ | F1 ↑ | 幻觉／全部物体提及 |
|---|---:|---:|---:|---:|---:|
| 完整SADI | 50.00 | 20.93 | 75.83 | 77.71 | 18／86 |
| SVAR | 60.00 | 7.23 | 71.25 | 70.29 | 6／83 |
| SADI＋P偏好对比，eta=.5D | 50.00 | 20.99 | 70.83 | 73.54 | 17／81 |
| SADI＋P偏好对比，eta=.5 | 30.00 | 18.75 | 70.83 | 75.33 | 15／80 |
| SADI＋uniform对比，eta=.5D | 50.00 | 20.99 | 70.83 | 73.54 | 17／81 |
| SADI＋rolledP对比，eta=.5D | 40.00 | 14.68 | 70.83 | 73.90 | 16／109 |

P偏好+.5D与uniform+.5D的全部10张生成token和文本相同，未体现P空间选择的增益。固定.5版本相对SADI，Cs−20.00pp、Ci−2.18pp、F1−2.37pp；配对95%区间分别[−50.00,0.00]、[−8.56,1.00]、[−12.14,4.29]pp。不能从两张指标定义的修正图推断稳定增益。

固定P的15次假提及仍高于SVAR的6次；较低Cs不能替代Ci或提及数比较。错位+.5D的Ci较低，主要伴随正确重复提及50→77、总提及86→109；不能据此把它当作更好的空间定位。

### 完整文本与原图

重新查看原JPEG353411、377003和575293，全部复用本机原图，不修改自动标签。新方法相对SADI的全部非零错误类别／去重真实类别变化均由这三图覆盖；其余仅文字变化的435783／517674未被解释为物体纠错。

- **353411**：餐巾上可见勺子，原图未见描述中的叉子。P+.5D虽在step97将fork改为nap，后文续成“a napkin with a fork on it”，最终仍包含fork；这说明即时替词没有完成缓解。uniform+.5D生成同样文本。固定P在更早的step24把sandwich首选改为large，后文仍正确描述sandwich，并最终不再说fork；但rolledP+.5D生成同样的整张描述，尚不能把这次纠正归因于正确的P空间结构。新句子的glass table surface会被CHAIR把glass映射为cup，不能当作明确找到了右侧真实金属杯。
- **377003**：照片中的躺椅、小圆桌、盆栽和沙发都可见。四新版本均把recliner chair改成recliner，potted plant改成plant in a black pot；自动去重真实类别少2，召回下降，但物体仍在文本中。table假标签从2次降1次，图中有真实边桌，不作为移除凭空桌子的证据。
- **575293**：照片确有路边旧沙发，seat cushion属于沙发部件。固定P删去了原文“seat cushion is missing”及相应seat→chair自动标签；不能把该标签消失描述为纠正一把独立幻觉椅子。

本轮固定P保留为需检查的部分EMS候选，尚无SAR缩放或P空间优于对照的证据。下一步应在完整开发100图比较同样eta=.5的P、uniform、rolledP，保持SADI不变；可复用本轮P_fixed的10份原组输出，其余90份新生成。三臂的固定强度对照需要在100生成前冻结，不先进入独立500，不按当前单例调eta／alpha／beta。

### 审计恢复与产物

原分析已完成两口径评分，但独立审核要求的`copied_controls`列表未写入协议，触发KeyError。冻结协议和全部caption／metrics保留不改。独立`audit_qwen3_live_preference_contrast10_recovery.py`先核SADI/SVAR四份控制文件与原源逐字节相同，再以有收据的只读内存计划补充该列表，运行同一个独立数值审核；原始协议SHA仍保留，恢复请求记录全部caption和指标SHA。评分、bootstrap和模型输出均不重做。恢复首次从SSH home直接调用，旧审计相对依赖路径找不到；随后从仓库cwd调用同一个冻结脚本通过，另存失败说明。

pipeline、analysis_recovery_complete、independent_summary和quality_complete均passed；原analysis_complete=false、driver_failure和failureNone日志保留。60份含控制输出、两口径共240个有向配对指标区间独立复算一致。生成加预检父墙钟318.07秒（5分18秒），两worker302.86／181.32秒，不含各自模型构建；最终恢复与质量完成距启动474.31秒。每卡原图编码1次、共享命中7次，四正式臂新增语义head调用均0。

本机`scripts/experiment_snapshots/qwen3_live_preference_contrast10`保存81个远端产物、2,000,164字节并核SHA，另存原图复查记录；本轮父和两个worker均已结束，当前没有GPU任务。目标继续active，独立500仍未消费。
