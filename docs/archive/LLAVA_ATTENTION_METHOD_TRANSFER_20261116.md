# LLaVA 生成时注意力方法的投影视觉写入迁移

本轮把 SAR 的投影视觉写入接入已有缓解方法的生成干预。分别匹配各方法的原始选头、原始选 patch 或原始区域覆盖更新。用同一 24 张 COCO val2014 开发图、LLaVA-1.5-7B、FP16、`Describe this image.`、贪心解码及 512 个新 token。原始对照按作者公开代码移植到现有运行器，图片、提示词和 Transformers 版本采用项目的统一设置。层编号从 0 开始。同一张图的各方法复用提示 KV，已存在的 SVAR 生成直接用于评分。

## 已实现的迁移

- **SPIN** 在每层按图像位置的 pre-softmax logit 和从高到低选 30 个头。其余 2 个头的完整注意力输出在输出投影前乘 0.08。移植只修改最后一个 prompt 查询及每个解码查询，沿用作者代码。作者的 FP16 标准化和 softmax 排序在本地化简为 FP32 对原始和排序，可能在数值近似平分时产生差异。投影写入版比较 $\|W_{O,h}\sum_{m\in\mathrm{image}}A_{hm}v_{hm}\|_2$，层、选头数和系数相同。输出投影的 Gram 矩阵整个进程只计算一次。 [论文](https://aclanthology.org/2025.emnlp-main.631/) · [作者代码](https://github.com/YUECHE77/SPIN/blob/main/attentionSPIN.py)
- **SPARC** 用第 20 层头均视觉注意力相对 EMA 的增幅选 patch。按作者代码用当前注意力系数 0.9，上一 EMA 系数 0.1，选相对增幅至少 1.5 的 patch。从下一查询开始，在第 0–31 层把这些 patch 的 Value 缓存乘 1.1，重复选中时累乘。投影写入版在各自轨迹的每步保留同一选区规则给出的配额，用输出投影后写入模长的相对增幅排序。写入依据使用未放大的初始视觉 Value，避免放大本身成为再选择的依据。两组输出轨迹分歧后各自累计选区数可能不同。 [论文](https://proceedings.mlr.press/v267/jung25c.html) · [作者代码](https://github.com/mingi000508/SPARC/blob/bad02ff858b986d61fcae17c3be1449430200a97/attn_util.py#L151-L225)
- **VGA** 从图像前缀得到作者使用的视觉概率和初始引导图，按原规则在第 2–15 层对注意力头加入加权视觉 Value。生成出满足作者词首规则的 token $y$ 后，以 $p_m(y)$ 定位已描述的区域，再降低该区下一步的引导权重。投影写入版以 $p_m(y)\|\sum_h A_{hm}W_{O,h}v_{hm}\|_2$ 替换覆盖图，$A_{hm}$ 取自实际预测 $y$ 的第 18 层查询。其余引导和覆盖更新沿用作者设置。两臂共用图像前缀及视觉词表概率缓存，均保留同样的零质量回退。 [论文](https://arxiv.org/html/2511.20032) · [作者代码](https://github.com/beta-nlp/VGA/blob/main/vcd_utils/greedy_sample.py#L118-L138)
- **SVAR＋后层 SPIN** 是在首轮 24 图结果出来后加做的开发对照。沿用第 5–18 层的 SVAR 图像 logit 增益，仅在第 19–31 层按 SPIN 规则压制后两头。原 SPIN 打分和投影写入打分各一臂，压制强度同为 0.08。比较用于检查投影选头对更强的直接解码基底是否仍有价值。 [SVAR 论文](https://arxiv.org/html/2411.16724#S4)

SPIN 和 SPARC 的替换检验当前查询的视觉写入，VGA 的替换还使用实际目标 token 的区域语义。这轮没有加入原版 EMS 的最优传输风险，也没有按最终描述重新选句。三种迁移均不增加每个 token 的模型前向次数。

另外核对了 AllPath 的路径分组与预选头发布物、EVAS 的浅层视觉广播和 VGD 的物体级回退。AllPath 的公开 CHAIR 头排名来自 500 张探测图，其中 3 张与项目开发 100 图重合（3865、72811、127119），1 张与待验证的 60 图重合（314026）。后续使用该排名时可从所有方法同时排除这些图。EVAS 的公开 CHAIR 指令使用 5-beam。VGD 的选择性回退要拟合并校准模型专用核验器。本轮实测采用可在现有贪心解码中一处替换的三种机制。 [AllPath](https://arxiv.org/html/2511.17254#S4) · [AllPath 探测集](https://github.com/SooLab/AllPath/blob/c8792510f12886848b3f7585db0246af9ac16013/more_benchmarks.py) · [EVAS](https://github.com/itsqyh/Shallow-Focus-Deep-Fixes#chair) · [VGD](https://arxiv.org/html/2607.27823v1#S4)

## 同图开发筛选

CHAIRi 由汇总物体提及计算。召回和 F1 取逐图平均。所有已生成的完整结果纳入，包括达到 512 上限的描述。

| 方法 | CHAIRs ↓ | CHAIRi ↓ | 物体召回 ↑ | 物体 F1 ↑ | 幻觉／全部提及 | 解码秒／图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVAR | 20.83% | 4.52% | 70.62% | 77.59% | 8／177 | 4.78 |
| SPIN | 41.67% | 15.28% | 77.92% | 75.59% | 22／144 | 5.15 |
| SPIN＋投影写入 | 37.50% | 10.29% | 75.97% | 79.52% | 14／136 | 5.13 |
| SPARC | 41.67% | 11.31% | 77.85% | 79.07% | 19／168 | 7.45 |
| SPARC＋投影写入 | 50.00% | 12.50% | 77.22% | 75.73% | 20／160 | 7.67 |
| VGA | 37.50% | 16.36% | 74.17% | 75.99% | 27／165 | 5.38 |
| VGA＋目标词投影写入 | 33.33% | 11.93% | 70.90% | 74.20% | 21／176 | 6.25 |
| SVAR＋后层 SPIN | 29.17% | 8.51% | 74.58% | 79.30% | 16／188 | 5.95 |
| SVAR＋后层 SPIN 投影写入 | 29.17% | 4.21% | 75.63% | 79.72% | 8／190 | 6.12 |

单独迁移的三组中，SPIN 和 VGA 的两项 CHAIR 点估计和绝对幻觉提及比各自原方法低。其物体召回分别下降 1.95 和 3.26 个百分点。SPIN 的 F1 提高 3.93 个百分点，VGA 的 F1 下降 1.79 个百分点。三组都没有在两项 CHAIR 上低于同图 SVAR。

SPIN 投影版修复 3 张并新增 2 张幻觉图。VGA 投影版修复 2 张并新增 1 张。配对 bootstrap 5,000 次，两组的 CHAIRs/i 差值的 95% 区间均跨零。SPARC 投影版修复 0 张并新增 2 张。

SVAR＋后层 SPIN 加投影写入后，两臂有幻觉的图片集合相同，但幻觉提及从 16 次降至 8 次，CHAIRi 差 −4.30 个百分点（95% 配对区间 [−9.51, 0.37]）。相比同图 SVAR，投影版新引入 2 张幻觉图，幻觉提及总量相同。CHAIRi 低 0.31 个百分点由分母从 177 增至 190 产生，召回／F1 高 5.00／2.13 个百分点。两臂与 SVAR 都无 512 上限样本。规则与代码哈希冻结后，在完整开发集 100 张补齐两臂，剩余 76 张单独评估。

## 后层 SPIN 扩展结果

| 图像 | 方法 | CHAIRs ↓ | CHAIRi ↓ | 物体召回 ↑ | 物体 F1 ↑ | 幻觉／全部提及 | 解码秒／图 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全开发集 100 | SVAR | 21.00% | 5.01% | 71.86% | 78.17% | 34／679 | 5.37 |
| 全开发集 100 | SVAR＋后层 SPIN | 25.00% | 6.13% | 69.64% | 75.16% | 43／701 | 5.59 |
| 全开发集 100 | SVAR＋后层 SPIN 投影写入 | 23.00% | 3.98% | 71.35% | 77.03% | 35／880 | 6.42 |
| 后续 76 | SVAR | 21.05% | 5.18% | 72.25% | 78.36% | 26／502 | 5.56 |
| 后续 76 | SVAR＋后层 SPIN | 23.68% | 5.26% | 68.09% | 73.85% | 27／513 | 5.48 |
| 后续 76 | SVAR＋后层 SPIN 投影写入 | 21.05% | 3.91% | 70.00% | 76.18% | 27／690 | 6.51 |

全 100 张中，对于相同的 SVAR＋后层 SPIN 干预，改投影选头修复 8 张、新增 6 张，CHAIRs 差 −2.00 个百分点（95% 配对区间 [−9.00, 5.00]），CHAIRi 差 −2.16（[−4.67, 0.33]）。比原 SVAR 修复 7 张、新增 9 张，两项 CHAIR 差分别是 +2.00（[−6.00, 10.00]）和 −1.03（[−3.21, 1.14]）个百分点。相对原 SVAR 幻觉提及 34→35，总提及 679→880，平均长度 108.59→127.41 token。更低的 CHAIRi 来自分母增大，物体召回和 F1 仍比原 SVAR 低。

后续 76 张中，投影版对两种参照的 CHAIRi 都较低，但绝对幻觉提及与后层 SPIN 相同、比原 SVAR 多 1 次。投影版的真实物体提及比 SVAR 多 187 次，按图像和物体类别去重的真实提及却从 159 项降至 151 项，分母增长集中在重复描述。物体召回／F1 相对原 SVAR 低 2.25／2.17 个百分点。排除任一方法触及 512 上限的 7 张后，剩余 69 张的 CHAIRs/i 依次为 SVAR 21.74%／5.87%、后层 SPIN 26.09%／6.43%、投影版 21.74%／5.16%。相应幻觉提及为 25、27、24 次。两项 CHAIR 同时下降并保住覆盖的候选尚未确认，未进入封存 60 图。

## 文件与验证

结果在服务器 `/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_llava/pilot/`。24 图指标存于 `metrics.first24.sparc-sparc_write-spin-spin_projected-svar.json`、`metrics.first24.svar-vga-vga_write.json` 和 `metrics.first24.svar-svar_spin-svar_spin_projected.json`。100 图指标为 `metrics.first100.svar-svar_spin-svar_spin_projected.json`，后续 76 图指标为 `metrics.first100.subsetbd913e72cd2d.svar-svar_spin-svar_spin_projected.json`。同名 `chair_per_image.*.jsonl` 保存逐图分数。筛选图片顺序哈希为 `3666044b7085c376183a7f2eeb763ff6cd7c25e596c1bfe0c189bcb63b8aa413`。完整开发集扩展的清单与配置在输出根目录的 `transfer_svar_spin_dev_remaining76.json` 和 `transfer_svar_spin_dev100_plan.json`。代码见 [SPIN](../../features/llava_spin.py)、[SPARC](../../features/llava_sparc.py)、[VGA](../../features/llava_vga.py) 和[共享生成入口](../../scripts/generation_time_evidence_drift_llava.py)。

本地 `git clone --depth 1 https://github.com/YUECHE77/SPIN.git` 对 443 端口连接超时，改用公开源文件逐行核对。服务器检查覆盖 SPIN 投影范数和最后一行缩放、SPARC 逐步 Value 缓存写入、VGA 的词首门控和区域引导更新。VGA 初次交互式合成测试直接传输词首字符，断言失败。用 Unicode 码点构造该字符后，原版与投影版的更新均通过。
