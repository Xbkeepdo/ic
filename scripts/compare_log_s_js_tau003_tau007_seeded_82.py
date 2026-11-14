#!/usr/bin/env python3
"""Compare six JS variants at tau=0.03 and 0.07 on identical seeded 8:2 tests."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1] / "outputs"
OLD = ROOT / "log_s_ot_js_all_variants_seeded_82_43_45"
NEW = ROOT / "log_s_ot_js_tau007_seeded_82_43_45"
NAMES = {
    "union16": "Top16 union",
    "union16_rq": "Top16 union + R_q",
    "union32": "Top32 union",
    "union32_rq": "Top32 union + R_q",
    "rank16": "Top16 rank",
    "rank16_rq": "Top16 rank + R_q",
}
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)


def load(path: Path) -> dict[tuple[str, int, str], dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["model"], int(row["seed"]), row["case"]): row for row in rows}


def main() -> None:
    old, new = load(OLD / "per_seed.csv"), load(NEW / "per_seed.csv")
    if set(old) != set(new):
        raise ValueError("tau sweeps have different model/seed/case rows")
    for key in old:
        if old[key]["test_tokens"] != new[key]["test_tokens"]:
            raise ValueError(f"{key}: unequal test tokens")
    lines = [
        "# JS 源分布温度：0.03 vs 0.07（COCO 8:2）",
        "",
        "两次实验的 43/44/45 种子、图片划分、目标分布 T、SAR、MLP 配置与测试 object token 相同。仅 JS 的视觉源分布 P 温度改变；OT 与 R_q 均固定为原 tau=0.03，以隔离 JS 温度效应。下表为三种子测试 AUROC 均值 ± 总体标准差，单位 %；差值为 0.07−0.03 百分点。",
        "",
        "| 模型 | JS 构造 | tau=0.03 AUROC | tau=0.07 AUROC | 差值 (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        for case, name in NAMES.items():
            lhs = [100 * float(old[(model, seed, case)]["auroc"]) for seed in SEEDS]
            rhs = [100 * float(new[(model, seed, case)]["auroc"]) for seed in SEEDS]
            lines.append(
                f"| {model} | {name} | {mean(lhs):.2f} ± {pstdev(lhs):.2f} | "
                f"{mean(rhs):.2f} ± {pstdev(rhs):.2f} | {mean(rhs)-mean(lhs):+.2f} |"
            )
    lines += ["", "τ=0.07 对固定 OT 的配对图片级 Bootstrap 见 bootstrap_3000/bootstrap_auroc_ci.md。", ""]
    (NEW / "tau003_vs_tau007.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
