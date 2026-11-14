#!/usr/bin/env python3
"""Put the original 8:2 Log-S and baseline CLEVR/AMBER AUROCs in one table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_qa_811_ours_vs_82_baselines import (  # noqa: E402
    BASE, CATEGORIES, METHODS, MODELS, OLD, old_amber, old_clevr,
)


def main() -> None:
    clevr, amber = old_clevr(), old_amber()
    header = ["模型", "任务", "8:2 测试题数", "Log-S 8:2",
              *(f"{method} 8:2" for method in METHODS)]
    lines = [
        "# QA 8:2：Log-S 与 baseline 的 AUROC 对照",
        "",
        "所有数值均为 **AUROC（%）**，使用同一 8:2 测试划分。Log-S 显示原报告的三随机种"
        "均值±标准差，baseline 保留原报告数值。CLEVR 为 `prompt_last_token | "
        "answer_correctness_all`；AMBER 为 `prompt_last_token | answer_correctness_all`，"
        "并按 existence / attribute / relation 分项统计幻觉 AUROC。",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for _, display in MODELS:
        cases = [("CLEVR-Exist-9K", 1800, clevr[display])]
        cases += [(f"AMBER {category}", count, amber[display][category])
                  for category, count in zip(CATEGORIES, (987, 1500, 344))]
        for task, count, values in cases:
            cells = [display, task, str(count),
                     f"{values['Log-S']:.2f}±{values['Log-S std']:.2f}"]
            cells += [f"{values[method]:.2f}" if method in values else "—"
                      for method in METHODS]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "CLEVR 的 MetaToken 为最后层 LR、无 StandardScaler；AMBER 的 MetaToken "
        "为原报告的标准化配置。AMBER 原 8:2 三类表没有 DHCP 分项，故标为“—”。",
        "",
        "来源：[CLEVR 原对比表](clevr_answer_correctness_prompt_last_comparison.md)、"
        "[AMBER 三类原表](clevr_amber3_auc_tables.tex)。",
    ]
    output = BASE / OLD / "all_82_ours_baselines_clevr_amber3.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(MODELS) * 4} rows: {output}")


if __name__ == "__main__":
    main()
