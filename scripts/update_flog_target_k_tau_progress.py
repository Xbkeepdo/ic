#!/usr/bin/env python3
"""Refresh the F_log target ablation progress board from extraction/training caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
GF = ROOT.parent / "gfchair/outputs/coco4000_512_flog_target_k_tau_ablation_811"
OUT = ROOT / "outputs/flog_target_k_tau_ablation"
MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B", "本机 GPU 1 / 32081"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B", "本机 GPU 0"),
    ("internvl_2_5_8b", "InternVL2.5-8B", "本机 GPU 1 / 32081"),
    ("qwen3_vl_8b", "Qwen3-VL-8B", "本机 GPU 0 / 32081"),
)
TRAINING_TOTAL = 6 * 2 * 3


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def progress(value: int, total: int) -> str:
    if total <= 0:
        return "等待"
    count = min(10, round(10 * value / total))
    return f"{'█' * count}{'░' * (10 - count)} {value}/{total} ({100 * value / total:.1f}%)"


def render() -> bool:
    lines = [
        "# F_log 作为 T 的 K/tau 消融进度", "",
        "- 网格：`K={32,64}`，`tau={0.03,0.05,0.07}`。",
        "- 每组训练 Risk-only 与 Risk+F_log，各 3 个种子。",
        "- 本机提取 Qwen3、InternVL；远端 32081 提取 Qwen2.5、LLaVA。训练由两台机器分担。", "",
        "| 模型 | 提取 | 内部验证 | 测试网格 | 状态 |",
        "|---|---|---|---|---|",
    ]
    complete = True
    for model, display, _worker in MODELS:
        gf_model = GF / model
        td_model = OUT / model
        extraction = read_json(gf_model / "progress.json")
        validation = read_json(td_model / "progress.json")
        test_grid = read_json(td_model / "test_grid_progress.json")
        extraction_done = (gf_model / "summary.json").exists()
        validation_done = (td_model / "validation_summary.csv").exists()
        test_done = (td_model / "test_grid_summary.csv").exists()
        extraction_text = "完成" if extraction_done else progress(
            int(extraction.get("completed", 0)), int(extraction.get("total", 0))
        )
        validation_text = "完成" if validation_done else progress(
            int(validation.get("completed", 0)), TRAINING_TOTAL
        ) if extraction_done else "等待提取"
        test_text = "完成" if test_done else progress(
            int(test_grid.get("completed", 0)), TRAINING_TOTAL
        ) if validation_done else "等待验证"
        status = "完成" if test_done else "运行中" if extraction.get("completed") or validation.get("completed") else "等待"
        lines.append(f"| {display} | {extraction_text} | {validation_text} | {test_text} | {status} |")
        complete &= test_done
    lines.extend([
        "", "## 输出", "",
        "- [实验定义](README.md)",
        "- [共享参数选择](shared_selection.md)",
        "- [完整测试网格与原 target 对比](complete_results.md)",
        "- [共享参数最终结果](summary.md)",
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / f".progress.{os.getpid()}.tmp"
    temp.write_text("\n".join(lines) + "\n")
    os.replace(temp, OUT / "progress.md")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        done = render()
        if not args.watch or done:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
