#!/usr/bin/env python3
"""Evaluate Popular-only POPE ↔ CLEVR Yes-only transfer on fixed 8:2/8:1:1 splits."""

from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from scripts import run_object_position_transfer_comparison as core


ORIGINAL = core.OUT
OUT = core.BASE / "pope_popular_clevr_yes_only_transfer_82_811"
SCHEME = "yes_only_pope"
SUBSET = "popular"


def _load_master(model: str, dataset: str) -> list[dict]:
    with (ORIGINAL / "prepared" / model / f"{dataset}.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    if dataset == core.POPE:
        rows = [row for row in rows if row["source_split"] == SUBSET]
        if len(rows) != 3000:
            raise ValueError(f"Expected 3,000 POPE Popular records, got {len(rows)}")
    return rows


def _reuse_clevr_checkpoints(split_mode: str, model: str) -> None:
    for method in ("svar", "ours"):
        source = ORIGINAL / SCHEME / split_mode / model / core.CLEVR / method
        destination = OUT / SCHEME / split_mode / model / core.CLEVR / method
        if not destination.exists():
            shutil.copytree(source, destination)


def _fmt(rows: list[dict], field: str) -> str:
    scores = 100 * np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if len(scores) != len(core.SEEDS):
        raise ValueError(f"Expected three seeds for {field}; got {len(scores)}")
    return f"{scores.mean():.2f} ± {scores.std():.2f}"


def _selected(rows: list[dict], split: str, model: str, source: str,
              target: str, method: str) -> list[dict]:
    result = [row for row in rows if (row["split_mode"], row["model"],
              row["source"], row["target"], row["method"]) ==
              (split, model, source, target, method)]
    if len(result) != len(core.SEEDS):
        raise ValueError(f"Missing evaluation: {split}/{model}/{source}/{target}/{method}")
    return result


def _write_summary(rows: list[dict], audits: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "cohort_audits.json").write_text(json.dumps(audits, indent=2) + "\n")

    lines = [
        "# POPE Popular ↔ CLEVR Yes-only transfer", "",
        "只保留模型回答 Yes 的题目：GT=Yes 为 Real，GT=No 为 Hall。POPE 限定 Popular 子集（原始 3,000 题），CLEVR 维持之前的完整 Yes-only 队列。SVAR 使用物体位置；SAR+EMS 使用答案位置 SAR 与物体位置 EMS。", "",
        "沿用此前固定图片级 8:2 与 8:1:1 划分及种子 43/44/45。8:1:1 使用来源验证集，8:2 不使用验证集。跨域使用来源标准化、模型和阈值，目标域不调参。CLEVR 检测器及其检查点与先前全 POPE 实验相同，仅 POPE 检测器按 Popular 重训。单位为 %，均值 ± 总体标准差。", "",
    ]
    name = {core.POPE: "POPE Popular", core.CLEVR: "CLEVR"}
    for split in ("82", "811"):
        lines.extend([f"## {split}", "",
                      "| 模型 | 方法 | 训练 → 测试 | 测试 N (Real/Hall) | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
                      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for model in core.MODELS:
            for source, target in ((core.POPE, core.POPE), (core.CLEVR, core.CLEVR),
                                   (core.POPE, core.CLEVR), (core.CLEVR, core.POPE)):
                for method in ("svar", "ours"):
                    selected = _selected(rows, split, model, source, target, method)
                    counts = {(int(row["n"]), int(row["real_n"]), int(row["hall_n"]))
                              for row in selected}
                    if len(counts) != 1:
                        raise ValueError("Test cohort changes across seeds")
                    n, real, hall = next(iter(counts))
                    label = "SVAR" if method == "svar" else "SAR+EMS"
                    metrics = " | ".join(_fmt(selected, field) for field in core.METRICS)
                    lines.append(f"| {model} | {label} | {name[source]} → {name[target]} | "
                                 f"{n} ({real}/{hall}) | {metrics} |")
        lines.append("")

    with (ORIGINAL / "per_seed_results.csv").open(newline="") as handle:
        old = [row for row in csv.DictReader(handle) if row["scheme"] == SCHEME]
    lines.extend(["## Popular 训练与三子集合并训练：相同 CLEVR 测试集", "",
                  "以下比较仅改变 POPE 来源训练集，CLEVR 目标测试集完全相同。三子集合并训练的数值来自先前实验。", "",
                  "| 划分 | 模型 | 方法 | 训练 POPE 全部子集 AUROC | 仅训练 Popular AUROC | 全部子集 Hall AUPR | Popular Hall AUPR |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for split in ("82", "811"):
        for model in core.MODELS:
            for method in ("svar", "ours"):
                old_rows = _selected(old, split, model, core.POPE, core.CLEVR, method)
                new_rows = _selected(rows, split, model, core.POPE, core.CLEVR, method)
                if {(r["n"], r["real_n"], r["hall_n"]) for r in old_rows} != \
                   {(str(r["n"]), str(r["real_n"]), str(r["hall_n"])) for r in new_rows}:
                    raise ValueError("CLEVR target changed between all-POPE and Popular comparisons")
                label = "SVAR" if method == "svar" else "SAR+EMS"
                lines.append(f"| {split} | {model} | {label} | {_fmt(old_rows, 'auroc')} | "
                             f"{_fmt(new_rows, 'auroc')} | {_fmt(old_rows, 'hall_aupr')} | "
                             f"{_fmt(new_rows, 'hall_aupr')} |")
    lines.extend(["", "## 限制", "",
                  "POPE Popular 的 Yes-only 测试集 Hall 样本很少；尤其 8:1:1 的 LLaVA/Qwen3 分别仅有 13/10 个 Hall。Hall AUPR 应与该测试集的 Hall 占比一起解读。CLEVR → POPE Popular 与先前三子集合并的目标队列不同，不能直接把二者差值解释为训练子集带来的效果。", ""])
    (OUT / "complete_results.md").write_text("\n".join(lines))


def main() -> None:
    torch.set_num_threads(1)
    core.OUT = OUT
    masters = {(model, dataset): _load_master(model, dataset)
               for model in core.MODELS for dataset in (core.POPE, core.CLEVR)}
    all_rows, all_audits = [], []
    for split in ("82", "811"):
        for model in core.MODELS:
            cohorts = {}
            for dataset in (core.POPE, core.CLEVR):
                groups, audit = core.cohort(masters[(model, dataset)], model=model,
                                            dataset=dataset, scheme=SCHEME, split_mode=split)
                cohorts[dataset] = (groups, audit)
                all_audits.append(audit)
                if dataset == core.CLEVR:
                    _reuse_clevr_checkpoints(split, model)
                for method in ("svar", "ours"):
                    core.train(model, dataset, SCHEME, split, method, groups, audit)
            all_rows.extend(core.evaluate(SCHEME, split, model, core.POPE,
                                          core.CLEVR, cohorts))
            print(f"Evaluated Popular/CLEVR {split}/{model}", flush=True)
    _write_summary(all_rows, all_audits)
    print(OUT / "complete_results.md", flush=True)


if __name__ == "__main__":
    main()
