#!/usr/bin/env python3
"""Combine the four-model attention-vs-a_m target ablation reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "llava_1_5_7b",
    "qwen2_5_vl_7b",
    "internvl_2_5_8b",
    "qwen3_vl_8b",
)
MODEL_NAMES = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}
METRICS = (
    "accuracy_real_threshold",
    "accuracy_hall_threshold",
    "auroc",
    "real_f1",
    "real_recall",
    "real_aupr",
    "hall_f1",
    "hall_recall",
    "hall_aupr",
)


def result_root(model: str, split: str) -> Path:
    if split == "82":
        name = f"{model}_log_s_hpre_amnorm_target_82_ablation"
    else:
        name = (
            "llava_log_s_hpre_amnorm_target_811_ablation"
            if model == "llava_1_5_7b"
            else f"{model}_log_s_hpre_amnorm_target_811_ablation"
        )
    return ROOT / "outputs" / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("811", "82"), default="811")
    args = parser.parse_args()
    out = ROOT / "outputs" / f"log_s_hpre_amnorm_target_{args.split}_ablation_all"
    out.mkdir(parents=True, exist_ok=True)
    rows, comparisons, missing = [], [], []
    for model in MODELS:
        path = result_root(model, args.split) / "complete_results.json"
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        if payload.get("missing"):
            missing.extend(payload["missing"])
        for row in payload["rows"]:
            rows.append({"model": model, "model_display": MODEL_NAMES[model], **row})
        for comparison in payload["comparisons"]:
            comparisons.append({
                "model": model,
                "model_display": MODEL_NAMES[model],
                **comparison,
            })
    protocol = (
        "shared 8:1:1 split; K=32; tau=0.03; seeds 43/44/45; dual validation thresholds"
        if args.split == "811"
        else "strict 8:2 split; no validation; train-loss checkpoint; dual training thresholds"
    )
    output = {
        "protocol": protocol,
        "rows": rows,
        "comparisons": comparisons,
        "missing": missing,
    }
    (out / "complete_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    if rows:
        fields = list(rows[0])
        with (out / "complete_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        f"# Four-model log-s target ablation ({args.split}): attention vs a_m norm",
        "",
        "- Old: `T = renorm(attention_weight * s_m)`.",
        "- New: `T = renorm(||a_m||_2 * s_m)`.",
        "- `s_m=log(1+C p_m)`, K=32, tau=0.03; source, cost and penalty are unchanged.",
        (
            "- Shared 3200/400/400 image split; seeds 43/44/45."
            if args.split == "811"
            else "- Strict 3200/800 image split without validation; seeds 43/44/45."
        ),
        (
            "- Real and Hall thresholds independently maximize their validation F1."
            if args.split == "811"
            else "- Checkpoint minimizes training loss; Real and Hall thresholds independently maximize their training F1."
        ),
        "",
    ]
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        lines.extend([
            f"## {MODEL_NAMES[model]}",
            "",
            "| Feature | AUROC | Real F1 | Real AUPR | Hall F1 | Hall Recall | Hall AUPR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in selected:
            values = []
            for metric in ("auroc", "real_f1", "real_aupr", "hall_f1", "hall_recall", "hall_aupr"):
                values.append(
                    f"{100*row[f'{metric}_mean']:.2f} ± {100*row[f'{metric}_std']:.2f}"
                )
            lines.append(f"| {row['display']} | " + " | ".join(values) + " |")
        lines.append("")
    if comparisons:
        lines.extend([
            "## New target minus old target",
            "",
            "| Model | Feature | ΔAUROC | ΔReal F1 | ΔReal AUPR | ΔHall F1 | ΔHall AUPR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in comparisons:
            values = [
                100 * row[f"delta_{metric}"]
                for metric in ("auroc", "real_f1", "real_aupr", "hall_f1", "hall_aupr")
            ]
            lines.append(
                f"| {row['model_display']} | {row['feature']} | "
                + " | ".join(f"{value:+.2f}" for value in values)
                + " |"
            )
    if missing:
        lines.extend(["", "## Missing", "", *[f"- `{path}`" for path in missing]])
    (out / "complete_results.md").write_text("\n".join(lines) + "\n")
    print(f"models={len({row['model'] for row in rows})} rows={len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
