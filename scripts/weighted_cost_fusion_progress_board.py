#!/usr/bin/env python3
"""Atomically overwrite the weighted-cost attention/I fusion progress board."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/weighted_cost_attention_i_fusion"
BOARD = RUN / "progress.md"
TOTAL = 540
MODELS = (
    ("LLaVA-1.5-7B", "llava_1_5_7b", "本地 GPU 0"),
    ("Qwen2.5-VL-7B", "qwen2_5_vl_7b", "本地 GPU 1"),
    ("InternVL2.5-8B", "internvl_2_5_8b", "31817 GPU 0"),
    ("Qwen3-VL-8B", "qwen3_vl_8b", "31817 GPU 1"),
)


def bar(done: int, total: int) -> str:
    done = min(max(done, 0), total)
    filled = round(16 * done / total)
    return f"`{'█' * filled}{'░' * (16 - filled)}` {done}/{total} ({done / total:.1%})"


def model_row(name: str, slug: str, device: str) -> tuple[str, bool]:
    directory = RUN / slug
    progress_path = directory / "progress.json"
    summary = directory / "summary.md"
    completed = sum(1 for _ in directory.glob("*/*/seed*/metrics.json")) if directory.exists() else 0
    stage = "载入与对齐"
    current = "—"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        completed = max(completed, int(progress.get("completed", 0)))
        stage = "完成" if progress.get("stage") == "complete" else "训练中"
        if progress.get("risk"):
            current = f"{progress['risk']} / {progress.get('added_feature', '—')} / seed {progress.get('seed', '—')}"
    if summary.exists() and completed == TOTAL:
        stage = "完成"
    log = directory / "train.log"
    updated = "—"
    if log.exists():
        updated = datetime.fromtimestamp(log.stat().st_mtime).strftime("%m-%d %H:%M")
        if stage != "完成" and time.time() - log.stat().st_mtime > 1800:
            stage += "（日志超 30 分钟未更新）"
    complete = stage == "完成" and completed == TOTAL
    return (
        f"| {name} | {device} | {stage} | {bar(completed, TOTAL)} | {current} | {updated} |",
        complete,
    )


def write(path: Path) -> bool:
    rows = [model_row(*model) for model in MODELS]
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    content = (
        "# Weighted cost risk + ENDAC attention / I 进度\n\n"
        f"**刷新时间：{stamp}**（运行时每 30 秒覆写）\n\n"
        "| 模型 | 设备 | 阶段 | 单层 MLP 训练 | 当前组合 | 最近日志 |\n"
        "|---|---|---|---:|---|---|\n"
        + "\n".join(row for row, _ in rows)
        + "\n\n每模型共 540 次训练：36 种 weighted-cost risk × 5 种附加特征 × 3 个种子。"
        "附加特征为 all_visual 的 raw/norm semantic_only、raw/norm semantic_attention，以及 I。\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)
    return all(complete for _, complete in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BOARD)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        complete = write(args.output.resolve())
        if not args.watch or complete:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
