#!/usr/bin/env python3
"""Write Markdown progress for log_s_hpre + log-residual tuning."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_hpre_log_residual_fusion_tuning"
MODELS = (
    "qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b",
    "internvl_2_5_8b", "minigpt4_7b", "shikra_7b",
)
PER_MODEL = 57


def bar(done: int, total: int, width: int = 30) -> str:
    filled = round(width * done / total)
    return "█" * filled + "░" * (width - filled)


def update() -> int:
    counts = {
        model: len(list((OUT / model / "validation").glob("*/seed*/metrics.json")))
        + len(list((OUT / model / "final").glob("*/seed*/metrics.json")))
        for model in MODELS
    }
    expected = PER_MODEL * len(MODELS)
    done = sum(counts.values())
    lines = [
        "# log_s_hpre + log_residual_fusion 搜参进度", "",
        f"> 自动覆盖更新时间：**{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}**", "",
        f"## 总进度 `{bar(done, expected)}` {100 * done / expected:.2f}%", "",
        f"已完成 **{done} / {expected}** 个训练头。", "",
        "| 模型 | 进度条 | 训练头 | 百分比 | 状态 |",
        "|---|---|---:|---:|---|",
    ]
    for model, count in counts.items():
        complete = count == PER_MODEL and (OUT / model / "summary.md").exists()
        lines.append(
            f"| `{model}` | `{bar(count, PER_MODEL)}` | {count}/{PER_MODEL} | "
            f"{100 * count / PER_MODEL:.2f}% | {'完成' if complete else '搜索中'} |"
        )
    lines += ["", "每个模型：38 个单种子初筛 + 前 8 名补跑 2 个种子 + 最终 3 种子测试。"]
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / ".progress.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(OUT / "progress.md")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        done = update()
        if args.once or done == PER_MODEL * len(MODELS):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
