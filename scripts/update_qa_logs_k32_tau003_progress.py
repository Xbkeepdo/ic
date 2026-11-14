#!/usr/bin/env python3
"""Atomically refresh the formal Log-S QA benchmark progress page."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.pickle_journal import iter_pickle_records


QA_ROOT = ROOT / "outputs" / "qa_benchmarks"
PREPARED_ROOT = Path("/home/apulis-dev/userdata/datasets/qa_benchmarks")
OUTPUT_NAME = "LOGS-K32-TAU003-NATIVE6-FULL"
REPORT_ROOT = QA_ROOT / OUTPUT_NAME
LOG_ROOT = QA_ROOT / "_logs" / OUTPUT_NAME
_PART_COUNT_CACHE: dict[str, tuple[int, int, int]] = {}
_BASELINE_PAYLOAD_COUNT_CACHE: dict[str, int] = {}
BASELINES = {
    "metatoken",
    "svar",
    "dhcp",
    "projectaway",
    "token_grounding",
    "instruct_lens",
}
MODELS = (
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B", 56),
    ("llava_1_5_7b", "LLaVA-1.5-7B", 64),
    ("qwen3_vl_8b", "Qwen3-VL-8B", 72),
    ("internvl_2_5_8b", "InternVL2.5-8B", 64),
)
DATASETS = (
    ("pope", "POPE", 2),
    ("clevr_exist_9k", "CLEVR-Exist-9K", 2),
    ("amber_discriminative", "AMBER-Discriminative", 1),
)


def _bar(done: int, total: int, width: int = 24) -> str:
    ratio = min(max(done / total, 0.0), 1.0) if total else 0.0
    filled = round(width * ratio)
    return "█" * filled + "░" * (width - filled)


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _expected_questions(dataset: str) -> int:
    return _line_count(PREPARED_ROOT / dataset / "questions.jsonl")


def _feature_count(run_dir: Path, total: int) -> int:
    summary = run_dir / "qa_feature_summary.json"
    if summary.is_file():
        try:
            return min(int(json.loads(summary.read_text())["num_questions"]), total)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if (run_dir / "features.pkl").is_file():
        return total
    parts = list((run_dir / "features.parts").glob("part*.pkl"))
    count = 0
    for path in parts:
        try:
            stat = path.stat()
            cache_key = str(path)
            cached = _PART_COUNT_CACHE.get(cache_key)
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
            if cached is not None and cached[:2] == signature:
                count += cached[2]
                continue
            row_count = sum(
                1
                for _ in iter_pickle_records(
                    path, ignore_trailing_record=True
                )
            )
            _PART_COUNT_CACHE[cache_key] = (*signature, row_count)
            count += row_count
        except (EOFError, OSError, pickle.PickleError, TypeError):
            continue
    return min(count, total)


def _baseline_payload_count(run_dir: Path) -> int:
    baseline_dir = run_dir / "baseline" / "answer_correctness_all"
    cache_key = str(baseline_dir)
    cached = _BASELINE_PAYLOAD_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parts = sorted((baseline_dir / "feature_parts").glob("part*.pkl"))
    if not parts:
        parts = (
            [baseline_dir / "features.pkl"]
            if (baseline_dir / "features.pkl").is_file()
            else []
        )
    if not parts:
        return 0
    try:
        if parts[-1].name == "features.pkl":
            with parts[-1].open("rb") as handle:
                rows = pickle.load(handle)
            row = rows[0] if rows else None
        else:
            row = next(
                iter_pickle_records(parts[-1], ignore_trailing_record=True),
                None,
            )
        if row is None:
            return 0
        positions = row.get("positions")
        if isinstance(positions, dict):
            count = sum(
                len(BASELINES.intersection(value.get("baselines", {})))
                for value in positions.values()
                if isinstance(value, dict)
            )
        else:
            count = len(BASELINES.intersection(row.get("baselines", {})))
        _BASELINE_PAYLOAD_COUNT_CACHE[cache_key] = count
        return count
    except (EOFError, OSError, pickle.PickleError, TypeError):
        return 0


def _main_training_count(run_dir: Path) -> int:
    return len(list((run_dir / "results").glob("*/*/*/seed_*/result.json")))


def _baseline_training_count(run_dir: Path) -> int:
    completed: set[tuple[str, str, str]] = set()
    for protocol in ("object_hallucination_yes_only", "answer_correctness_all"):
        result_root = run_dir / "baseline" / protocol / "results"
        for path in result_root.glob("**/seed*/*.json"):
            if "summary" in path.name or "shared_torch_mlp" in path.name:
                continue
            seed_dir = next(
                (parent.name for parent in path.parents if parent.name.startswith("seed")),
                "",
            )
            position = next(
                (
                    value
                    for value in (
                        "prompt_last_token",
                        "question_object_pre_token",
                    )
                    if value in path.parts
                    or f"native_paper_{value}" in path.parts
                ),
                "prompt_last_token",
            )
            if seed_dir:
                completed.add((protocol, position, seed_dir))
    return len(completed)


def _comparison_count(run_dir: Path) -> int:
    return len(list((run_dir / "comparison").glob("*/*qa_comparison_3seed.md")))


def _failure_count(run_dir: Path) -> int:
    return _line_count(run_dir / "extraction_failures.jsonl")


def _log_is_active(model: str, dataset: str) -> bool:
    paths = [LOG_ROOT / f"{model}__{dataset}.log"]
    paths.extend(LOG_ROOT.glob(f"{model}__{dataset}__shard*.log"))
    return any(
        path.is_file() and time.time() - path.stat().st_mtime < 300.0
        for path in paths
    )


def _state(
    *,
    model: str,
    dataset: str,
    extracted: int,
    total: int,
    main_training: int,
    baseline_training: int,
    comparisons: int,
) -> str:
    if comparisons >= 2:
        return "完成"
    if baseline_training:
        return "Baseline 训练"
    if main_training:
        return "主方法探针训练"
    if extracted >= total and total:
        return "等待/开始训练"
    if extracted:
        return "特征提取中" if _log_is_active(model, dataset) else "已落盘，可续跑"
    return "排队中"


def _snapshot() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    totals = {dataset: _expected_questions(dataset) for dataset, _name, _p in DATASETS}
    for model, display_model, feature_dim in MODELS:
        for dataset, display_dataset, position_count in DATASETS:
            run_dir = QA_ROOT / model / OUTPUT_NAME / dataset
            total = totals[dataset]
            extracted = _feature_count(run_dir, total)
            main_training = _main_training_count(run_dir)
            baseline_training = _baseline_training_count(run_dir)
            comparisons = _comparison_count(run_dir)
            expected_main = 2 * position_count * 3
            rows.append(
                {
                    "model": model,
                    "model_display": display_model,
                    "dataset": dataset,
                    "dataset_display": display_dataset,
                    "feature_dim": feature_dim,
                    "positions": position_count,
                    "extracted": extracted,
                    "total": total,
                    "baseline_payloads": _baseline_payload_count(run_dir),
                    "main_training": main_training,
                    "expected_main": expected_main,
                    "expected_baseline": position_count * 2 * 3,
                    "baseline_training": baseline_training,
                    "comparisons": comparisons,
                    "failures": _failure_count(run_dir),
                    "state": _state(
                        model=model,
                        dataset=dataset,
                        extracted=extracted,
                        total=total,
                        main_training=main_training,
                        baseline_training=baseline_training,
                        comparisons=comparisons,
                    ),
                }
            )
    return rows


def write() -> bool:
    rows = _snapshot()
    extracted = sum(int(row["extracted"]) for row in rows)
    total = sum(int(row["total"]) for row in rows)
    trained = sum(
        int(row["main_training"])
        + int(row["baseline_training"])
        + int(row["comparisons"])
        for row in rows
    )
    training_total = sum(
        int(row["expected_main"])
        + int(row["expected_baseline"])
        + 2
        for row in rows
    )
    complete_runs = sum(int(row["comparisons"]) >= 2 for row in rows)
    lines = [
        "# Log-S K32 τ0.03 · QA benchmark 正式进度",
        "",
        "> 本文件由脚本每 30 秒原子覆写；只统计 `LOGS-K32-TAU003-NATIVE6-FULL`，不统计 smoke 或早先中断目录。",
        f"> 最近更新：**{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}**",
        "",
        "固定方案：`log_s_hpre Risk + F_log`，`τ=0.03`，总 `K=32`（source Top-16 ∪ target Top-16），精确 EMD；正式结果排除 DHCP，TokenGrounding 只保留其 MLP 网格。",
        "提取方式：每个位置都由主方法与 6 个 baseline 特征族共用该位置的一次 LVLM 输出；两个位置分别存储。当前结果训练主方法与 5 个非 DHCP baseline。每张卡持续追加一个 pkl，结束后统一合并。",
        "存储说明：DHCP 特征虽随联合前向保留在 pkl 中，但不训练、不进入当前 comparison；当前正式配置关闭内容 SHA256。两套标签协议共享一次标签无关的 baseline 特征计算。",
        "",
        f"## 特征提取 `{_bar(extracted, total)}` {100 * extracted / total if total else 0:.2f}%",
        "",
        f"已落盘 **{extracted:,}/{total:,}** 个 QA 样本；完成全部提取、训练与对比的组合为 **{complete_runs}/12**。",
        "",
        f"## 训练与对比 `{_bar(trained, training_total)}` {100 * trained / training_total if training_total else 0:.2f}%",
        "",
        "训练计数包括主方法的每个 seed 结果、两套标签协议下的 native-baseline seed 结果，以及两张最终对比表。",
        "",
        "## 分项进度",
        "",
        "| 模型 | 数据集 | 位置 | 特征维度 | 提取进度 | 样本 | baseline 位置×方法 | 主方法头 | baseline runs | 对比表 | 失败 | 阶段 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        done = int(row["extracted"])
        row_total = int(row["total"])
        lines.append(
            f"| {row['model_display']} | {row['dataset_display']} | "
            f"{row['positions']} | {row['feature_dim']} | "
            f"`{_bar(done, row_total, 16)}` | {done:,}/{row_total:,} | "
            f"{row['baseline_payloads']}/{int(row['positions']) * 6} | "
            f"{row['main_training']}/{row['expected_main']} | "
            f"{row['baseline_training']}/{row['expected_baseline']} | {row['comparisons']}/2 | "
            f"{row['failures']} | {row['state']} |"
        )
    lines += [
        "",
        "## 位置与标签协议",
        "",
        "- POPE、CLEVR-Exist-9K：`prompt_last_token` 与 `question_object_pre_token`。",
        "- AMBER-Discriminative：仅 `prompt_last_token`，因为没有可靠的精确 object span。",
        "- 标签：`object_hallucination_yes_only` 与 `answer_correctness_all`。",
        "- 当前对比 Baseline：MetaToken、SVAR、ProjectAway、TokenGrounding（仅 MLP）、InsLens；DHCP 已排除。",
        "- `prompt_last_token` 与 `question_object_pre_token`：主方法和 6 个 baseline 特征族都在各自真实因果位置联合提取；当前 5 个非 DHCP baseline 按位置分别训练。",
        "",
        "## 路径",
        "",
        f"- 正式输出名：`{OUTPUT_NAME}`",
        f"- 运行日志：[`../_logs/{OUTPUT_NAME}`](../_logs/{OUTPUT_NAME})",
        "- 每个组合的输出：`../<model>/LOGS-K32-TAU003-NATIVE6-FULL/<dataset>/`",
    ]
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    target = REPORT_ROOT / "progress.md"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return complete_runs == len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    while True:
        complete = write()
        if args.once or complete:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
