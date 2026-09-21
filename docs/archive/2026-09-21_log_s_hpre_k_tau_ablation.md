# log-s hpre 的 K/tau 消融（2026-09-21）

## 范围

- 模型：Qwen2.5-VL-7B、LLaVA-1.5-7B、Qwen3-VL-8B、InternVL2.5-8B。
- 网格：`K={16,32,64,128}`，`tau={.02,.03,.04,.05,.06,.07,.08}`，共28组。
- 每组训练 risk-only 和 risk+F_log，使用 seeds 43/44/45。
- MLP固定：单隐藏层128、batch256、dropout0.3、LR1e-3、WD1e-5、patience10、标准化、无BN。
- 原训练集内部固定80/20图片级划分用于选参，固定800图测试集只评估最终共享设置。

## 结果

- 提取、内部验证、共享选择和最终测试全部完成。
- 共享设置：`K=128, tau=0.03`。
- 四模型平均验证 AUROC `0.8965`，HALL-AUPR `0.6511`；相对原基准 `K=64,tau=.07` 在验证集上四模型均提高。
- 最终测试 risk+F_log 相对原基准：平均 AUROC `+0.0007`，平均 HALL-AUPR `+0.0035`。
- 完整报告：`outputs/log_s_hpre_k_tau_ablation/complete_results.md`。
- 完整验证数据：`outputs/log_s_hpre_k_tau_ablation/validation_all.csv`。
- 最终对比：`outputs/log_s_hpre_k_tau_ablation/final_comparison.csv`。
- 后续补充了四模型全部28组严格测试集网格；报告见`test_grid_results.md`，原始数据见`test_grid_all.csv`。该网格不参与选参。

## 验证

- `K=64,tau=.07` 单图与旧 risk 最大绝对误差 `1.19e-7`。
- 5项数值检查、`py_compile`、相关 `git diff --check` 均通过。
