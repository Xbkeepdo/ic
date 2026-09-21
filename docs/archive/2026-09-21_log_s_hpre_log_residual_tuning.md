# log_s_hpre + log_residual_fusion 单隐藏层搜参

## 协议

- 固定输入：`concat(log_s_hpre, log_residual_fusion)`。
- 单隐藏层，搜索宽度、dropout、学习率、weight decay、batch size、early-stopping patience。
- 38 个候选用 seed 43 初筛；内部验证前 8 名补跑 seeds 44/45。
- 原训练分区按图像固定 80/20 切分；严格测试分区不参与参数选择。
- 最佳参数在完整训练分区重训 seeds 43/44/45。
- 训练集列标准化，无 BatchNorm，Linear 使用 Kaiming uniform ReLU 初始化。

## 独立选参严格测试结果

| 模型 | 选中配置 | Real AUC | ΔAUC | HALL AUPR | ΔAUPR |
|---|---|---:|---:|---:|---:|
| Qwen2.5-VL | width128, dropout0.2 | 0.8521±0.0025 | +0.0006 | 0.4434±0.0131 | +0.0003 |
| LLaVA-1.5 | width512, batch128, dropout0.3, lr3e-4 | 0.9169±0.0001 | +0.0019 | 0.7627±0.0021 | +0.0058 |
| Qwen3-VL | width128, lr2e-3 | 0.9020±0.0017 | -0.0028 | 0.7289±0.0094 | -0.0067 |
| InternVL2.5 | width128, batch128 | 0.8871±0.0010 | +0.0000 | 0.6685±0.0063 | +0.0020 |
| MiniGPT4 | width128, wd1e-3 | 0.9371±0.0009 | +0.0003 | 0.7770±0.0028 | +0.0008 |
| Shikra | width32 | 0.8582±0.0007 | +0.0012 | 0.6619±0.0036 | +0.0156 |

模型独立选参的平均 ΔAUC 为 +0.0002，平均 ΔHALL-AUPR 为 +0.0030；Qwen3 的验证集选择没有迁移到严格测试集。

## 统一配置

六模型内部验证初筛的最佳共同配置为 width128、batch256、dropout0.3、lr1e-3、wd1e-4、patience10。严格测试平均 ΔAUC 为 +0.0001，平均 ΔHALL-AUPR 为 +0.0006。

## 输出与验证

- 独立结果：`outputs/log_s_hpre_log_residual_fusion_tuning/summary.md`
- 统一结果：`outputs/log_s_hpre_log_residual_fusion_tuning/shared_summary.md`
- 共同配置排名：`outputs/log_s_hpre_log_residual_fusion_tuning/shared_selection.md`
- 六模型 342 个独立搜参/最终头与 18 个统一配置最终头完整存在。
- 所有配置均为单隐藏层、无 BatchNorm、训练集标准化；指标有限且在 `[0,1]`。
- 所有内部 fit/validation/strict-test 图像集合互斥。
