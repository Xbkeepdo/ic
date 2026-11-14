# 三篇论文方法的检测门控 LLaVA 100 图试验

完成日期：2026-11-14。

目标：联网查找已发表顶会且作者代码公开的方法，迁移到 SAR+EMS / SVAR 检测门控的生成时干预，并在种子 43、LLaVA-1.5-7B、前 100 张 COCO 测试图上配对比较。

完成内容：选定 VCD（CVPR 2024）、DeCo（ICLR 2025）、ICD（Findings of ACL 2024），实现三个论文核心解码公式在第一个检测高风险物体词位置的一步贪心适配。分别接入 SAR+EMS 与原生 SVAR 检测器，共六组，每组完整生成 100 张图。SVAR 旧缓存 257 个物体词精确对齐，13 个不对齐词重新提取；阈值只取验证集。统一完成 CHAIRs、CHAIRi、物体召回、逐图变化与图片级配对 bootstrap。

结果：[正式报告](../../outputs/generation_time_paper_transfer_seed43/summary.md)。原始贪心 CHAIRs 48%、CHAIRi 14.75%；SAR+EMS + DeCo 为 47% / 14.82%，其余方法没有降低 CHAIRs，六组均未降低 CHAIRi。SAR+EMS 触发 22 张，SVAR 12 张；唯一整图修复发生在 SAR+EMS + DeCo。结果只代表单步适配，不是论文全程解码复现。

实现：[脚本](../../scripts/generation_time_paper_transfer_seed43.py)。验证：脚本编译通过；六个 JSONL 各有 100 个不同图片 ID；所有触发位置的所选 token 与生成 token 一致；100 行逐图评估齐全。实验命令：`/opt/conda/private/envs/vicr/bin/python scripts/generation_time_paper_transfer_seed43.py generate --limit 100 --device cuda:0`；评估命令：`NLTK_DATA=/home/apulis-dev/userdata/nltk_data /opt/conda/private/envs/vicr/bin/python scripts/generation_time_paper_transfer_seed43.py evaluate --limit 100`。

调试记录：首次运行因为脚本入口缺少仓库根目录加入 `sys.path` 而报 `ModuleNotFoundError: detection`，已修正；首次 DeCo 前向把 `lm_head` 错取自 `LlamaModel`，出现 `AttributeError`，已改为取 LLaVA 顶层 `lm_head` 并重跑成功。没有遗留失败样本。
