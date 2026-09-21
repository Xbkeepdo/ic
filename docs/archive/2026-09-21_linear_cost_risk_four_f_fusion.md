# Linear-cost risk × 四种 F 融合实验

## 实验设置

- 六个模型，每模型 4 个 risk-only、4 个 F-only、16 个 `risk + F`。
- 种子 43、44、45，共 432 个训练头。
- 单隐藏层 128，训练集列标准化，无 BatchNorm，dropout 0.3。
- 预定义 3200/800 图像切分；train 训练，val+test 严格测试。
- 完整结果：`outputs/linear_cost_log_s_risk_four_f_fusion/summary.md`。

## 各模型最高 Real AUROC 组合

| 模型 | 组合 | Real AUROC | HALL AUPR |
|---|---|---:|---:|
| Qwen2.5-VL-7B | `log_s_hpre + log_residual_fusion` | 0.8516±0.0010 | 0.4430±0.0138 |
| LLaVA-1.5-7B | `log_s_hpre + sqrt_residual_fusion` | 0.9173±0.0003 | 0.7640±0.0023 |
| Qwen3-VL-8B | `log_s_hpre + asinh_sqrt_fusion` | 0.9051±0.0008 | 0.7352±0.0057 |
| InternVL2.5-8B | `log_s_hmid + log_residual_fusion` | 0.8880±0.0004 | 0.6588±0.0033 |
| MiniGPT4-7B | `log_s_hpre + log_residual_fusion` | 0.9369±0.0012 | 0.7761±0.0035 |
| Shikra-7B | `original_hmid + log_residual_fusion` | 0.8690±0.0009 | 0.6763±0.0049 |

跨六模型最稳的共同组合为 `log_s_hpre + log_residual_fusion`：平均模型内排名 3.00，模型 AUROC 的描述性均值为 0.8920。

96 个模型/组合案例中，融合相对匹配 risk 的 AUROC 全部提高；相对匹配 F 的 AUROC 在 95/96 个案例提高。HALL AUPR 相对两类匹配基线均为 96/96 提高。

## 验证

- 六模型紧凑特征 shape、有限值和切分互斥检查通过。
- 四个已有模型的四种 F 与旧 compact 文件逐元素一致。
- 432 份 metrics、432 个 scaler、432 个 checkpoint 均存在。
- 每模型 72 份 metrics；模型级汇总各 48 行；全局汇总 288 行。
- 汇总 AUROC 已从逐 seed metrics 独立重算并逐项一致。
- 训练及进度脚本通过 `py_compile`；四种 F 通过小型数值检查。
