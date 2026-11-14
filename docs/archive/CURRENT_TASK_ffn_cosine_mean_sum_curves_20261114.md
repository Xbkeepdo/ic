# FFN 视觉 cosine 均值与求和的 Real/Hall 曲线（2026-11-14）

目标：用 COCO4000 全量可用 ENDAC object token，对 LLaVA-1.5-7B 和 Qwen3-VL-8B 绘制 FFN 与视觉 hpre 的 cosine 均值及求和的逐层 Real/Hall 曲线。

实现：`scripts/plot_ffn_cosine_mean_sum_by_label.py`。分别读取已有 `outputs/ffn_cosine_i_s_811/{model}/compact_features.npz` 和 `outputs/ffn_cosine_sum_i_s_811/{model}/compact_features.npz`，按 `(image_id,response_token_idx,target_token_id,label)` 严格核对。使用全部 12,686 个 LLaVA object token 和 11,751 个 Qwen3 object token。曲线点为 token 加权均值；95% CI 使用图片聚类的标准误近似。

输出：`outputs/ffn_cosine_mean_sum_curves_by_label/ffn_cosine_mean_sum_real_hall_all4000.{png,pdf,svg}`，同目录 `curves.csv` 与 `summary.md`。

观察：LLaVA 的 Real−Hall 最大均值差出现在第 30 层，`F_mean` 为 `+0.0370`，`F_sum` 为 `+21.3273`；Qwen3 的最大正差出现在第 16 层，分别为 `+0.0646` 与 `+17.3284`。Qwen3 最后一层差值略为负，均值曲线和求和曲线形状相近。

验证：脚本 `py_compile` 通过；均值/求和的全部 object token 键、标签、图片 ID、数值有限性与 `F_sum=M×F_mean` 一致性检查通过；SVG/PDF/PNG 与逐层 CSV 已生成并目视核查图形。
