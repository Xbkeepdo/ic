# SVAR 原始生成期干预的 LLaVA 100 图复现

完成日期：2026-11-14。

目标：按用户纠正，运行 SVAR 论文自己的幻觉缓解方法，而非仅使用 SVAR 检测器门控其他方法；在上轮相同的 100 张图片与原始贪心描述上比较。

论文与代码：Jiang 等人 CVPR 2025 *Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens*；作者仓库 `ZhangqiJiang07/middle_layers_indicating_hallucinations`，检查提交 `1467fe07fefd4f15ed1ac3560455dff5b53dbc08` 的 `modify_attention.py`。方法为每步对图像 token 注意力 logits 加上跨头绝对值均值的 `alpha=0.5` 倍，随后做 softmax。论文写第 5–18 层，作者代码 `range(5,18)` 实为第 5–17 层，两种均运行。

完成：[报告](../../outputs/generation_time_svar_official_seed43/summary.md)；[实现](../../scripts/generation_time_svar_official_seed43.py)。论文范围 CHAIRs 48%→23%、CHAIRi 14.75%→4.82%，物体召回 80.54%→71.65%，物体 F1 77.31%→77.91%。代码范围 CHAIRs 23%、CHAIRi 6.92%、召回 66.65%、F1 73.76%。全部同 100 图配对评估，5000 次图片级 bootstrap；没有用测试标签调节方法。

验证：零系数时移植版与原始 LLaVA 首张图逐 token 一致；两个 100 图 JSONL 各有 100 个不同图片 ID，所有指导层每图均被调用，100 行逐图 CHAIR 齐全，脚本编译通过。生成命令：`python scripts/generation_time_svar_official_seed43.py generate --limit 100 --device cuda:0 --variants paper_layers_5_18`，另一范围在 `cuda:1` 同时运行；评估命令：`NLTK_DATA=/home/apulis-dev/userdata/nltk_data python scripts/generation_time_svar_official_seed43.py evaluate --limit 100`，其中 `python` 为 `/opt/conda/private/envs/vicr/bin/python`。

相关失败：系统没有 `pdftotext`，首次用 `curl ... && pdftotext` 提取论文文本时返回命令不存在；改用环境已安装的 `pypdf` 提取并核对论文公式、层范围和参数，后续实验全部成功。
