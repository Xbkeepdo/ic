#!/usr/bin/env python3
"""Atomically overwrite a Markdown progress board for weighted-cost runs."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "outputs/weighted_cost_projection/progress.md"
EXTRACT = re.compile(r"Extracting features:\s*\d+%\|[^\r\n]*?\|\s*(\d+)/(\d+)")
LABEL = re.compile(r"COCO-CHAIR labeling:\s*\d+%\|[^\r\n]*?\|\s*(\d+)/(\d+)")
CASES = 3 * 2 * 4 * 3
MODELS = (
    ("LLaVA-1.5-7B", "llava_1_5_7b", "本地 GPU 0", "outputs/weighted_cost_projection/logs/llava_1_5_7b-all.log"),
    ("Qwen2.5-VL-7B", "qwen2_5_vl_7b", "本地 GPU 1", "outputs/weighted_cost_projection/logs/qwen2_5_vl_7b-all.log"),
    ("InternVL2.5-8B", "internvl_2_5_8b", "31817 GPU 0", "outputs/internvl_2_5_8b/coco4000_512_wcost_all.log"),
    ("Qwen3-VL-8B", "qwen3_vl_8b", "31817 GPU 1", "outputs/qwen3_vl_8b/COCO4000-512-WCOST-ALL/pipeline.log"),
)


def bar(done: int, total: int) -> str:
    if total <= 0:
        return "`░░░░░░░░░░░░` —"
    done = min(max(done, 0), total)
    filled = round(12 * done / total)
    return f"`{'█' * filled}{'░' * (12 - filled)}` {done}/{total} ({done / total:.1%})"


def latest(pattern: re.Pattern[str], data: str) -> tuple[int, int]:
    found = pattern.findall(data)
    return tuple(map(int, found[-1])) if found else (0, 0)


def tail(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 262144))
        return handle.read().decode("utf-8", "replace")


def row(entry: tuple[str, str, str, str], *, raw_attention: bool) -> tuple[str, bool]:
    name, slug, device, log_name = entry
    run_name = "COCO4000-512-WCOST-RAWATTN" if raw_attention else "COCO4000-512-WCOST-ALL"
    run = ROOT / "outputs" / slug / run_name
    log = run / "pipeline.log" if raw_attention else ROOT / log_name
    data = tail(log)
    ld, lt = latest(LABEL, data)
    ed, et = latest(EXTRACT, data)
    if (run / "labeling.json").exists():
        ld, lt = 4000, 4000
    results = run / "weighted_cost_probe"
    trained = sum(1 for _ in results.glob("*/*/seed*/metrics.json")) if results.exists() else 0
    cases = 36 if raw_attention else CASES
    complete = (results / "summary.md").exists() and trained == cases
    if complete:
        stage = "完成"
    elif trained:
        stage = "训练中"
    elif "[Extractor] Done." in data:
        stage = "提取完成，待训练"
    elif et:
        stage = "提取中"
    elif lt and ld < lt:
        stage = "打标签中"
    elif ld == 4000:
        stage = "载入模型 / 待提取"
    else:
        stage = "待启动"
    if complete or trained or "[Extractor] Done." in data:
        ed = et
    updated = "—"
    if log.exists():
        updated = datetime.fromtimestamp(log.stat().st_mtime).strftime("%m-%d %H:%M")
        if not complete and time.time() - log.stat().st_mtime > 1800:
            stage += "（日志超 30 分钟未更新）"
    return (
        f"| {name} | {device} | {stage} | {bar(ld, lt)} | {bar(ed, et)} | "
        f"{bar(trained, cases)} | {updated} |",
        complete,
    )


def write_board(path: Path, *, raw_attention: bool = False) -> bool:
    rows = [row(entry, raw_attention=raw_attention) for entry in MODELS]
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    title = "raw attention 目标分布" if raw_attention else "加权 cost"
    gate_count = 1 if raw_attention else 2
    cases = 36 if raw_attention else CASES
    content = (
        f"# COCO4000-512 {title}实验进度\n\n"
        f"**刷新时间：{stamp}**（运行时每 30 秒覆写）\n\n"
        "| 模型 | 设备 | 阶段 | 打标签 | 特征提取 | 三种子训练 | 最近日志（本机时区） |\n"
        "|---|---|---|---:|---:|---:|---|\n"
        + "\n".join(value for value, _ in rows)
        + f"\n\n每模型一次捕获共同提取 softmax、pam、cam。训练共 {cases} 次="
        f"3 种源分布 × {gate_count} 种目标分布 × 4 个 cost 权重 × 3 个种子。"
        "提取分母是有可提取目标的图像数，可能少于 4000。\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
    return all(complete for _, complete in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BOARD)
    parser.add_argument("--raw-attention", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        complete = write_board(args.output.resolve(), raw_attention=args.raw_attention)
        if not args.watch or complete:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
