#!/usr/bin/env python3
"""Refresh the softmax-target experiment's Markdown progress file."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
GF_OUT = ROOT.parent / "gfchair/outputs/coco4000_512_softmax_target_k_tau_ablation_811"
OUT = ROOT / "outputs/softmax_target_k_tau_ablation"
MODELS = ("llava_1_5_7b", "qwen2_5_vl_7b", "internvl_2_5_8b", "qwen3_vl_8b")


def read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def bar(done: int, total: int, width: int = 16) -> str:
    filled = round(width * min(1.0, done / max(total, 1)))
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def render() -> bool:
    lines = [
        "# Softmax target 消融进度", "",
        "两种 T 在同一次前向中提取；每种 T 有 6 组 K/tau，分别训练 Risk 和 Risk+F_log，各 3 seeds。", "",
        "| 模型 | 提取图片 | 内部验证 | 测试网格 |",
        "|---|---:|---:|---:|",
    ]
    finished = True
    for model in MODELS:
        gf = GF_OUT / model
        target = OUT / model
        progress = read(gf / "progress.json")
        total = int(progress.get("total", 0))
        done = int(progress.get("completed", 0))
        extracted = (gf / "summary.json").exists()
        if extracted:
            done = total
        else:
            finished = False
        validation = read(target / "progress.json")
        val_done = int(validation.get("completed", 0))
        if (target / "validation_summary.csv").exists():
            val_done = 72
        grid = read(target / "test_grid_progress.json")
        grid_done = int(grid.get("completed", 0))
        if (target / "test_grid_summary.csv").exists():
            grid_done = 72
        if grid_done < 72:
            finished = False
        lines.append(
            f"| {model} | {bar(done, total or 1)} | {bar(val_done, 72)} | {bar(grid_done, 72)} |"
        )
    lines.extend(["", "共享选参：" + ("已完成" if (OUT / "shared_selection.json").exists() else "等待四模型验证"),
                  "完整结果：" + ("已完成" if (OUT / "complete_results.md").exists() else "等待测试网格"), ""])
    content = "\n".join(lines)
    path = OUT / "progress.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)
    return finished and (OUT / "summary.md").exists()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        if render():
            break
        time.sleep(30)
