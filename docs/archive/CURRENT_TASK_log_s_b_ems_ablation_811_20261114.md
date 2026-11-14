# 当前任务

目标：在 COCO 4000 图像的随机种子 43/44/45 的 8:1:1 设置下，比较 LLaVA-1.5-7B、Qwen3-VL-8B 的 [B,EMS] 与原版 [SAR,EMS]。B_l=∑_m I_lm log(1+C p_lm)=SAR_l-I_l。图片划分和 MLP 初始化都随种子改变；其余特征、标签、训练器和 K=32、tau=0.03 固定。

实验已完成。代码 scripts/run_log_s_b_ems_ablation_811.py 通过语法检查；B 用原 SAR 和 I 同源紧凑缓存相减，且与 final-norm 语义缓存的 SAR 逐元素核对。LLaVA 与 Qwen3 各三个种子的训练分别在本地 GPU 0/1 完成。完整报告 outputs/log_s_b_ems_ablation_811/complete_results.md，逐种子 per_seed.csv。六组比较的检测器配置、train/val/test token 数和 val/test 标签逐项一致，测试分数有限。原版相对 B+EMS 的 AUROC / Real AUPR 提升：LLaVA +0.48 / +0.17 百分点，Qwen3 +1.05 / +0.40 百分点。原版结果和 split 来自 outputs/log_s_ems_i_mean_svar_seeded_811_43_45/。
