# 当前任务摘要

整理：2026-09-20。既有融合结果见 `outputs/iclr_512_endac_attention_i_fusion/summary.md`。

## 已完成：semantic probability 与固定 risk 融合

- GF raw/norm 概率的 sqrt、log、mass-preserving、asinh-sqrt 曲线见 `outputs/norm_semantic_transform_curves_by_label/`；Qwen2.5 raw近零源于未过 final RMSNorm 的 logit 尺度漂移，审计见 `../gfchair/outputs/qwen25_semantic_probability_audit/`。
- raw-logit risk 融合见 `outputs/fixed_wcost_norm_semantic_transform_fusion/summary.md`。
- 指定 `hpre_softmax_prob_gauss/source=softmax/w=.2` risk 与四种 norm F 的48次训练见 `outputs/fixed_wcost_softmax_prob_norm_semantic_transform_fusion/summary.md`；最佳 AUC 为 LLaVA sqrt .9182、InternVL sqrt .8891、Qwen2.5 AS .8500、Qwen3 AS .9066。全量目标精确对齐，单层128、训练集标准化、无BN、seeds43–45，审计通过。
- 该 risk 的测试集 HALL/REAL 逐层均值±SEM 曲线见 `outputs/fixed_wcost_softmax_prob_risk_curves_by_label/`；四模型均为 HALL 均值更高。

## 已完成：semantic softmax 的 Gauss-MAD CDF gate 曲线

- 对 semantic softmax probability 做 Gauss-MAD 标准化和正态 CDF，按视觉 token 求和后绘制四模型标签均值与 SEM；见 `outputs/softmax_gauss_mad_cdf_sum_curves_by_label/summary.md`。
- 已保存字段为 `sigmoid(z)`，脚本用 `logit` 恢复 z 后计算 CDF。测试集上除 Qwen2.5 外，幻觉样本的层平均值更低；Qwen2.5 方向相反且 SEM 较大。

## 已完成：近期 weighted-cost 融合

- 固定 risk 的 7 组 raw semantic / I 组合见 `outputs/fixed_wcost_raw_semantic_i_combinations/summary.md`；36 risk × 4 种 `I×semantic` 结果见 `outputs/weighted_cost_i_semantic_product_fusion/`。

## 已完成：weighted-cost risk 与 ENDAC attention / I 融合

- 将四模型 `WCOST-ALL` 的 24 种 risk 和 `WCOST-RAWATTN` 的 12 种 risk，分别拼接 `all_visual/{raw,norm}/{semantic_only,semantic_attention}` 与逐层 `I`。每模型 36 risk × 5 附加特征 × 3 seeds = 540 次训练；单隐藏层宽 128，训练集逐列 z-score，无 BatchNorm。
- 脚本 `scripts/train_weighted_cost_attention_i_fusion.py` 完成四模型各 540/540 次训练；共 2160 份 metrics 均有限，scaler 有效，配置均为单隐藏层 128、标准化、无 BN。各模型完整表和跨模型 AUC 前五名位于 `outputs/weighted_cost_attention_i_fusion/`。
- 180 个四模型统一配置与 native SVAR 的 Real AUC 对照见 `common_config_vs_svar_native.md/csv`：无配置四模型全部超过 SVAR；若“略低”定义为不超过 1 AUC 百分点，共 3 个，均为 `WCOST-ALL / softmax / hpre_softmax_prob_gauss / I` 的 w=0.2/0.4/0.6。最稳健为 w=0.4，四模型差值依次 -0.0074/-0.0079/-0.0028/+0.0122。

## 已完成：加权 cost、三种源分布与 raw-attention 目标分布

- 已确认 `P_pam=softmax_m(<f_t,a_m>/||a_m||)` 是有符号标量投影；`P_cam=softmax_m(cos(f_t,a_m)/0.07)`。四个 cost 权重为 0.2/0.4/0.6/0.8。
- 代码和配置已加入，LLaVA 的 pam 单图冒烟成功（8 个目标、32 层、全部风险有限且 P 归一）；数值/重建测试通过。
- 每模型一次前向共算三源，四个 w 共用语义 EMD，单图风险最大差 3.6e-7。`WCOST-ALL` 四模型均完成 72/72 次训练；`WCOST-RAWATTN` 将 T 换为归一化 raw attention，四模型均完成 36/36 次训练。结果在各模型对应目录的 `weighted_cost_probe/`。

## 新增：softmax_target_penalty 单隐藏层宽度选择

- 四模型分别对 risk-only、risk+attention、risk+I 调单隐藏层宽度 32/64/128/256/512；从原 3200 张训练图像固定抽 640 张做验证，仅按验证 Hall AUPR 选宽度，之后在完整 3200 张训练图像重训并评估原 800 张测试图像。仍用训练集逐列 z-score、无 BatchNorm、seeds 43–45。
- 180 个候选训练与 36 个最终训练均完成；候选网格、选定宽度、测试结果和旧宽度128对照见 `outputs/iclr_512_softmax_target_penalty_width_tuning/summary.md`。已核验 216/216 指标、scaler、checkpoint 无 BN，以及四模型一致且无交叉的 2560/640/800 图像划分。复现脚本为 `scripts/tune_iclr_512_softmax_target_penalty_width.py` 与 `scripts/summarize_iclr_512_softmax_target_penalty_width.py`。

## 新增：输入标准化、无 BatchNorm 的单/三隐藏层 MLP

- 用户要求在原四模型 risk、risk+attention、risk+I 上逐列训练集 z-score、移除 BatchNorm，比较单隐藏层 128 与三隐藏层 128/64/32；两种结构 × 四模型 × 四 risk × 三特征组合 × 三种子，共 288 次训练完成。四种 risk 分别列出的主表、旧训练参照与逐分支 CSV 在 `outputs/iclr_512_endac_attention_i_fusion_standardized_no_bn/summary.md`；不要用跨 risk 平均代替分支结果。
- 训练脚本 `scripts/train_iclr_512_fusion_standardized_no_bn.py`，汇总脚本 `scripts/summarize_iclr_512_fusion_standardized_no_bn.py`；核心探针增加默认保持旧行为的 `standardize_input`、`batch_norm` 开关。每次运行保存训练集均值/scale。已核验 288/288 指标有限、scaler 合法、checkpoint 无 BatchNorm 参数、配置结构正确，脚本编译通过。
- 新旧实验同时改变输入标准化和 BatchNorm，旧结果仅作参照；单层与三层新实验的对照满足同数据、同标准化和同无 BN 条件。

## 已完成：同一 COCO4000-512 generation 的完整对照

- LLaVA、InternVL、Qwen2.5 使用 `outputs/512/outputs/<model>/COCO4000-512/generations.json`；Qwen3 的 GF generation 与原 ICLR 相同，复用其 risk。四模型与 GF 的完整 4000 条 generation 和标签一致，目标级匹配率均为 100%，无前缀冲突。
- 前三个模型的 risk/MetaToken/SVAR 产物在 `outputs/<model>/COCO4000-512-ICLR-RISK/`；Qwen3 新 baseline 在 `outputs/qwen3_vl_8b/COCO4000-512-ICLR-BASELINES/`。全部采用严格图像级 3200/0/800 split；源 512 split 有重叠，未复用。
- 训练完成：四模型 × 四种 risk × {risk、risk+attention、risk+I} × 三种子，共 144 次；MetaToken-LR/GB、SVAR native 与 MetaToken/SVAR shared MLP 对照全部生成三种子汇总。融合结果位于 `outputs/iclr_512_endac_attention_i_fusion/`，总表 `summary.md` 含全组合指标及 baseline。
- 提取配置：`configs/model_configs_coco4000_512_iclr_fusion_baselines.yaml`，Qwen3 baseline-only 配置：`configs/model_configs_coco4000_512_iclr_qwen3_baselines.yaml`；复现脚本：`scripts/train_iclr_endac_attention_i_fusion.py`。已检查 144 份融合指标存在且有限、四份对齐审计 100% 匹配、baseline 汇总完整、脚本编译通过。

## 环境与已知检查结果

- 原服务器需显式设置 `PYTHON_BIN=/opt/conda/private/envs/vicr/bin/python`、`NLTK_DATA=/home/apulis-dev/userdata/nltk_data`；当时默认 Python 缺 PyYAML。换环境后先核实路径。
- 历史全量检查：`CUDA_VISIBLE_DEVICES='' /opt/conda/private/envs/vicr/bin/python -m unittest discover -s tests -p 'test_*.py'`，248 项中 15 failures、5 errors。
- 失败主要涉及 manifest 开关、旧 VPend 断言、unified/fj01 语义差异和已关闭 sweep 的旧预期。本次未修复或复跑，不能宣称全套通过。
- 当时 spatial-cost 数值/序列化 22 项、wrapper/prompt-target plumbing 6 项及编译、shell 语法检查通过；详细历史命令和错误摘要见归档。
- 本地有多个未提交改动，编辑前检查相关状态并保留；历史 stash `codex-preserve-before-spatial-cost-pull-20260816` 是否仍存在需按需核实。

## 历史与维护

- [完整原始记录](archive/CURRENT_TASK.before-context-optimization-20260912T111400Z.md)逐字节保留；按任务词或日期 `rg -n` 检索，再读附近段落。
- 此处仅维护当前目标、未完成事项和关键路径；完成细节进入 `docs/archive/` 或实验报告。
