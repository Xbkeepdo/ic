#!/usr/bin/env python3
"""Write a live progress board for the I × semantic weighted-cost fusion."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/weighted_cost_i_semantic_product_fusion"
BOARD = RUN / "progress.md"
TOTAL = 432
MODELS = (
    ("LLaVA-1.5-7B", "llava_1_5_7b", "本地 GPU 0"),
    ("InternVL2.5-8B", "internvl_2_5_8b", "31069 GPU"),
    ("Qwen2.5-VL-7B", "qwen2_5_vl_7b", "本地 GPU 1"),
    ("Qwen3-VL-8B", "qwen3_vl_8b", "32510 GPU"),
)


def bar(done: int) -> str:
    done = min(max(done, 0), TOTAL)
    filled = round(16 * done / TOTAL)
    return f"`{'█' * filled}{'░' * (16 - filled)}` {done}/{TOTAL} ({done / TOTAL:.1%})"


def row(name: str, slug: str, device: str) -> tuple[str, bool]:
    root = RUN / slug
    progress_path = root / "progress.json"
    done = sum(1 for _ in root.glob("*/*/seed*/metrics.json")) if root.exists() else 0
    stage, current = "载入与对齐", "—"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        done = max(done, int(progress.get("completed", 0)))
        stage = "完成" if progress.get("stage") == "complete" else "训练中"
        if progress.get("risk"):
            current = f"{progress['risk']} / {progress.get('added_feature')} / seed {progress.get('seed')}"
    complete = done == TOTAL and (root / "summary.md").exists()
    if complete:
        stage, current = "完成", "—"
    updated = "—"
    log = root / "train.log"
    if log.exists():
        updated = datetime.fromtimestamp(log.stat().st_mtime).strftime("%m-%d %H:%M")
    return f"| {name} | {device} | {stage} | {bar(done)} | {current} | {updated} |", complete


def write() -> bool:
    rows = [row(*model) for model in MODELS]
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    text = (
        "# I × semantic feature 与 weighted-cost risk 融合进度\n\n"
        f"**刷新时间：{stamp}**（运行时每 30 秒覆写）\n\n"
        "| 模型 | 设备 | 阶段 | 训练 | 当前组合 | 最近日志 |\n"
        "|---|---|---|---:|---|---|\n"
        + "\n".join(value for value, _ in rows)
        + "\n\n每模型 432 次：36 risks × 4 个逐层乘积特征 × 3 seeds。\n"
    )
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    temporary = BOARD.with_name(f".{BOARD.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, BOARD)
    return all(complete for _, complete in rows)


if __name__ == "__main__":
    while True:
        if write():
            break
        time.sleep(30)
