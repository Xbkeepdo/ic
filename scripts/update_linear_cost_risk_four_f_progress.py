#!/usr/bin/env python3
"""Write a compact Markdown progress page for the risk x four-F experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/linear_cost_log_s_risk_four_f_fusion"
MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
    "minigpt4_7b",
    "shikra_7b",
)
PER_MODEL = 72


def bar(done: int, total: int, width: int = 30) -> str:
    filled = round(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def write_progress() -> int:
    counts = {
        model: len(list((OUT / model).glob("*/seed*/metrics.json")))
        for model in MODELS
    }
    total = sum(counts.values())
    expected = PER_MODEL * len(MODELS)
    lines = [
        "# Linear-cost risk × four F 训练进度",
        "",
        f"> 自动覆盖更新时间：**{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}**",
        "",
        f"## 总进度 `{bar(total, expected)}` {100 * total / expected:.2f}%",
        "",
        f"已完成 **{total} / {expected}** 个训练头。每个模型为 24 组特征 × 3 个种子。",
        "",
        "| 模型 | 进度条 | 训练头 | 百分比 | 状态 |",
        "|---|---|---:|---:|---|",
    ]
    for model, done in counts.items():
        status = "完成" if done == PER_MODEL and (OUT / model / "summary.md").exists() else "训练中"
        lines.append(
            f"| `{model}` | `{bar(done, PER_MODEL)}` | {done}/{PER_MODEL} | "
            f"{100 * done / PER_MODEL:.2f}% | {status} |"
        )
    lines += [
        "",
        "## 实验规模",
        "",
        "- 4 个 risk-only 基线。",
        "- 4 个 F-only 基线。",
        "- 16 个 risk + F 拼接组合。",
        "- 种子：43、44、45。",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / ".progress.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(OUT / "progress.md")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        done = write_progress()
        if args.once or done == PER_MODEL * len(MODELS):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
