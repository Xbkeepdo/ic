#!/usr/bin/env python3
"""Collect tau=0.07 JS vs fixed OT image-bootstrap results for COCO 8:1:1 and 8:2."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs"
SOURCES = {
    "811": ROOT / "log_s_js_tau007_variants_seeded_811_43_45/bootstrap_3000/bootstrap_auroc_ci.csv",
    "82": ROOT / "log_s_ot_js_tau007_seeded_82_43_45/bootstrap_3000/bootstrap_auroc_ci.csv",
}
METHODS = (
    ("union16", "Top16 并集"),
    ("union16_rq", "Top16 并集 + R_q"),
    ("union32", "Top32 并集"),
    ("union32_rq", "Top32 并集 + R_q"),
    ("rank16", "独立 Top16 排名"),
    ("rank16_rq", "独立 Top16 排名 + R_q"),
)
MODELS = (("llava_1_5_7b", "LLaVA-1.5-7B"), ("qwen3_vl_8b", "Qwen3-VL-8B"))


def load(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["model"], row["comparison"]): row for row in rows if row["scope"] == "three_seed_mean"}


def values(row: dict) -> str:
    delta, low, high = (100 * float(row[key]) for key in ("delta_auc", "ci_low", "ci_high"))
    mark = "†" if low > 0 or high < 0 else ""
    return f"{delta:+.2f} [{low:+.2f}, {high:+.2f}]{mark}"


def main() -> None:
    data = {split: load(path) for split, path in SOURCES.items()}
    expected = {(model, case) for model, _ in MODELS for case, _ in METHODS}
    if any(set(rows) != expected for rows in data.values()):
        raise ValueError("missing model or JS case in bootstrap reports")
    lines = [
        "# COCO：OT 与 tau=0.07 JS 的显著性对比",
        "",
        "差值为 OT−JS 的 real-positive 测试 AUROC 百分点；正值表示 OT 更高。种子 43/44/45 同时作用于图片划分和 MLP 初始化。8:1:1 每种子测试 400 张图片，8:2 每种子测试 800 张图片。两种协议均使用 3,000 次配对图片级 Bootstrap；每轮同一图片的所有 object token 共用权重，重叠测试图片在三种子间同步抽样。95% CI 为未做六方案多重比较校正的百分位区间，条件于已训练检测器及三个固定划分。† 表示该区间不跨零。",
        "",
        "OT 和 R_q 均沿用原 K=32、tau=0.03；JS 仅将视觉源分布 P 改成 tau=0.07，目标分布 T 与 SAR 不变。六组 JS 各协议均按原配置重新训练。",
        "",
    ]
    for model, display in MODELS:
        lines += [
            f"## {display}",
            "",
            "| JS 构造 | 8:1:1 OT / JS AUROC | 8:1:1 OT−JS [95% CI] | 8:2 OT / JS AUROC | 8:2 OT−JS [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
        for case, name in METHODS:
            lhs = data["811"][(model, case)]
            rhs = data["82"][(model, case)]
            def pair(row: dict) -> str:
                return f"{100*float(row['ot_auc']):.2f} / {100*float(row['js_auc']):.2f}"
            lines.append(f"| {name} | {pair(lhs)} | {values(lhs)} | {pair(rhs)} | {values(rhs)} |")
        lines.append("")
    lines += [
        "Bootstrap 区间仅描述固定三次训练结果的测试图片抽样不确定性，不包含重新划分或重新训练；† 为逐项未校正判断，不应解释为六组搜索后的整体显著性。逐种子结果及原始逐轮抽样分数见两个来源目录的 bootstrap_3000 文件夹。",
        "",
    ]
    (ROOT / "log_s_js_tau007_significance_811_82.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
