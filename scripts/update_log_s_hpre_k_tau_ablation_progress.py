#!/usr/bin/env python3
"""Atomically refresh the four-model K/tau ablation progress page."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
EXTRACTED = GF_ROOT / "outputs/coco4000_512_log_s_hpre_k_tau_ablation_811"
OLD_EXTRACTED = GF_ROOT / "outputs/coco4000_512_linear_cost_log_s_risk_811"
OUT = ROOT / "outputs/log_s_hpre_k_tau_ablation"
MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
)
VALIDATION_HEADS = 28 * 2 * 3
FINAL_HEADS = 2 * 3


def bar(done: int, total: int, width: int = 24) -> str:
    ratio = min(max(done / total, 0.0), 1.0) if total else 0.0
    filled = round(width * ratio)
    return "█" * filled + "░" * (width - filled)


def extraction(model: str) -> tuple[int, int, str]:
    if (EXTRACTED / model / "summary.json").exists():
        value = json.loads((EXTRACTED / model / "summary.json").read_text())
        return int(value["images"]), int(value["images"]), "完成"
    expected_path = OLD_EXTRACTED / model / "summary.json"
    total = int(json.loads(expected_path.read_text())["images"]) if expected_path.exists() else 0
    part_paths = sorted((EXTRACTED / model).glob("progress.part*.json"))
    if part_paths:
        newest_by_assignment = {}
        for path in part_paths:
            value = json.loads(path.read_text())
            assignment = int(value["part_index"])
            candidate = (path.stat().st_mtime_ns, value)
            if assignment not in newest_by_assignment or candidate[0] > newest_by_assignment[assignment][0]:
                newest_by_assignment[assignment] = candidate
        parts = [newest_by_assignment[key][1] for key in sorted(newest_by_assignment)]
        return (
            sum(int(value["completed"]) for value in parts),
            sum(int(value["total"]) for value in parts),
            f"{len(parts)}路分片提取中",
        )
    progress_path = EXTRACTED / model / "progress.json"
    if progress_path.exists():
        value = json.loads(progress_path.read_text())
        return int(value["completed"]), int(value["total"]), "提取中"
    return 0, total, "等待"


def training(model: str) -> tuple[int, int, str]:
    model_out = OUT / model
    validation = len(list((model_out / "validation").glob("*/*/seed*/metrics.json")))
    final = len(list((model_out / "final_test").glob("*/*/seed*/metrics.json")))
    selected = (OUT / "shared_selection.json").exists()
    if final == FINAL_HEADS and (model_out / "final_test_summary.json").exists():
        return VALIDATION_HEADS + FINAL_HEADS, VALIDATION_HEADS + FINAL_HEADS, "完成"
    if selected:
        return validation + final, VALIDATION_HEADS + FINAL_HEADS, "最终测试"
    if validation:
        return validation, VALIDATION_HEADS + FINAL_HEADS, "内部验证"
    return 0, VALIDATION_HEADS + FINAL_HEADS, "等待提取"


def write() -> bool:
    extract_rows = [(model, *extraction(model)) for model in MODELS]
    train_rows = [(model, *training(model)) for model in MODELS]
    e_done, e_total = sum(row[1] for row in extract_rows), sum(row[2] for row in extract_rows)
    t_done, t_total = sum(row[1] for row in train_rows), sum(row[2] for row in train_rows)
    lines = [
        "# Log-s hpre 的 K/tau 消融进度", "",
        f"> 自动覆盖更新时间：**{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}**", "",
        "参数：`K={16,32,64,128}`，`tau={0.02,...,0.08}`；模型为Qwen2.5、LLaVA、Qwen3、InternVL。", "",
        f"## 特征提取 `{bar(e_done, e_total)}` {100*e_done/e_total if e_total else 0:.2f}%", "",
        "| 模型 | 进度 | 图片 | 状态 |", "|---|---|---:|---|",
    ]
    for model, done, total, status in extract_rows:
        lines.append(f"| `{model}` | `{bar(done,total)}` | {done}/{total or '?'} | {status} |")
    lines += [
        "", f"## 检测训练 `{bar(t_done, t_total)}` {100*t_done/t_total:.2f}%", "",
        "每模型内部验证168个头，选出共享设置后最终测试6个头。", "",
        "| 模型 | 进度 | 训练头 | 状态 |", "|---|---|---:|---|",
    ]
    for model, done, total, status in train_rows:
        lines.append(f"| `{model}` | `{bar(done,total)}` | {done}/{total} | {status} |")
    lines += [
        "", "## 协议", "",
        "- 28组完整网格共同提取，一张图片只运行一次VLM前向。",
        "- 每组训练risk-only与risk+F_log，固定原始单隐藏层配置和seeds 43/44/45。",
        "- 共享K/tau仅由原训练集内部按图片80/20验证选择；固定测试集只评估最终共享设置。",
        "- 基准为`K=64, tau=0.07`。",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / ".progress.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(OUT / "progress.md")
    return t_done == t_total and (OUT / "summary.md").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    while True:
        complete = write()
        if args.once or complete:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
