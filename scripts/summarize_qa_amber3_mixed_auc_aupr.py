#!/usr/bin/env python3
"""Compare original 8:2 Log-S with 8:1:1 baselines on AMBER's three types."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from detection.baselines import evaluate_detection_scores  # noqa: E402
from scripts.summarize_qa_targeted_mixed_classification import (  # noqa: E402
    BASE, CATEGORIES, NEW, SEEDS, _matrix, _own_path, _records, _scores, _variants,
)

MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
)
METHODS = (
    ("MetaToken-LR", "MetaToken", "VISAPP'25"),
    ("DHCP", "DHCP", "MM'25"),
    ("SVAR", "SVAR", "CVPR'25"),
    ("ProjectAway", "ProjectAway", "ICLR'25"),
    ("TokenGrounding-MLP", "TokenGrounding", "CVPR'26"),
    ("InsLens", "InsLen", "ICML'26"),
    ("Log-S", "Ours", ""),
)
METRICS = ("auroc", "real_aupr")
COUNTS_82 = {"existence": 987, "attribute": 1500, "relation": 344}
COUNTS_811 = {"existence": 498, "attribute": 766, "relation": 173}
STEM = "amber3_mixed_82_ours_811_baselines_auc_real_aupr"


def _add(values: dict, model: str, category: str, method: str,
         metric: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"Invalid {model}/{category}/{method}/{metric}: {value}")
    values.setdefault((model, category, method, metric), []).append(value * 100)


def collect() -> dict:
    values = {}
    dhcp = json.loads((BASE / NEW / "amber3_dhcp_811.json").read_text())
    for model, _ in MODELS:
        records = _records(model)
        labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
        masks = {category: np.asarray([row["source_split"] == category for row in records])
                 for category in CATEGORIES}
        if {category: int(mask.sum()) for category, mask in masks.items()} != COUNTS_811:
            raise ValueError(f"Invalid 8:1:1 AMBER category counts: {model}")
        root = (BASE / model / NEW / "amber_discriminative" / "baseline"
                / "answer_correctness_all" / "prompt_last_token")
        first = json.loads((root / "seed_43" / "result.json").read_text())
        saved = torch.load(first["methods"]["svar"]["checkpoint"],
                           map_location="cpu", weights_only=False)
        matrices = {
            "metatoken": _matrix(records, "metatoken"),
            "token_grounding": _matrix(records, "token_grounding"),
            "svar": _matrix(records, "svar", start=int(saved["layer_start"]),
                            end=int(saved["layer_end_exclusive"])),
        }
        for seed in SEEDS:
            own = json.loads(_own_path(model, "amber_discriminative",
                                      "answer_correctness_all", "prompt_last_token",
                                      seed).read_text())
            if (own["counts"]["test"] != sum(COUNTS_82.values())
                    or own["split_protocol"] != "strict_82_no_validation"
                    or own["threshold_selection"] != "train_f1"):
                raise ValueError(f"Unexpected 8:2 Log-S run: {model}/seed_{seed}")
            for category in CATEGORIES:
                result = own["test_groups"]["source_split"][category]
                count = result["real"]["support"] + result["hallucination"]["support"]
                if count != COUNTS_82[category]:
                    raise ValueError(f"Invalid 8:2 category count: {model}/{category}")
                _add(values, model, category, "Log-S", "auroc", result["auroc"])
                _add(values, model, category, "Log-S", "real_aupr",
                     result["real_aupr"])

            baseline = json.loads((root / f"seed_{seed}" / "result.json").read_text())
            if (baseline["counts"]["test"] != len(records)
                    or baseline["threshold_selection"] != "validation_real_f1"):
                raise ValueError(f"Unexpected 8:1:1 baseline run: {model}/seed_{seed}")
            variants = _variants(baseline)
            for method, _, _ in METHODS[:-1]:
                variant = variants[method]
                if method == "DHCP":
                    cached = dhcp[model]["seeds"][str(seed)]
                    if abs(cached["threshold_real"] - variant["threshold"]) > 1e-12:
                        raise ValueError(f"DHCP threshold differs: {model}/seed_{seed}")
                    grouped = cached["categories"]
                else:
                    scores = _scores(method, variant, matrices, records)
                    overall = evaluate_detection_scores(labels, scores,
                                                        variant["threshold"],
                                                        positive_class="real")
                    for metric in ("auc", "aupr"):
                        actual = overall["real_positive"][metric]
                        expected = variant["test_metrics"]["real_positive"][metric]
                        if abs(actual - expected) > 1e-5:
                            raise ValueError(f"Overall {metric} differs: "
                                             f"{model}/{method}/seed_{seed}")
                    grouped = {
                        category: evaluate_detection_scores(
                            labels[mask], scores[mask], variant["threshold"],
                            positive_class="real"
                        ) for category, mask in masks.items()
                    }
                for category in CATEGORIES:
                    result = grouped[category]
                    count = sum(map(sum, result["confusion_matrix_hallucination_positive"]))
                    if count != COUNTS_811[category]:
                        raise ValueError(f"Invalid baseline category count: "
                                         f"{model}/{method}/{category}")
                    _add(values, model, category, method, "auroc",
                         result["real_positive"]["auc"])
                    _add(values, model, category, method, "real_aupr",
                         result["real_positive"]["aupr"])
        print(f"{model}: 3 seeds verified", flush=True)

    expected = len(MODELS) * len(CATEGORIES) * len(METHODS) * len(METRICS)
    if len(values) != expected or any(len(series) != len(SEEDS) for series in values.values()):
        raise ValueError(f"Incomplete AMBER metric grid: {len(values)} != {expected}")
    return {key: {"mean": statistics.mean(series), "std": statistics.pstdev(series)}
            for key, series in values.items()}


def _cell(summary: dict, model: str, category: str, method: str,
          metric: str, rank: int) -> str:
    value = summary[model, category, method, metric]
    mean = f"{value['mean']:.2f}"
    if rank == 0:
        mean = rf"\textbf{{{mean}}}"
    elif rank == 1:
        mean = rf"\underline{{{mean}}}"
    if method == "Log-S":
        mean += (r"{\scriptsize$^{\textcolor{DeepPurple}{\bm{\pm"
                 + f"{value['std']:.2f}" + r"}}}$}")
    return mean


def _rank(summary: dict, model: str, category: str, metric: str) -> dict:
    ordered = sorted((method for method, _, _ in METHODS),
                     key=lambda method: summary[model, category, method, metric]["mean"],
                     reverse=True)
    return {method: index for index, method in enumerate(ordered)}


def render_tex(summary: dict) -> str:
    parts = []
    for category in CATEGORIES:
        lines = [
            r"\begin{table*}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{1.0pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\caption{",
            f"Comparison of answer correctness detection on AMBER-Discriminative {category} questions "
            "at the prompt-last token across four LVLMs.",
            "AUROC and Correct-AUPR are reported in percent.",
            f"Ours uses the original 8:2 test set ({COUNTS_82[category]:,} questions); "
            f"baselines use the 8:1:1 test set ({COUNTS_811[category]:,} questions).",
            "The highest and second-highest displayed means are shown in bold and underlined, respectively.",
            r"}", r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{@{}llcc@{\hspace{5pt}}cc@{\hspace{5pt}}cc@{\hspace{5pt}}cc@{}}",
            r"\toprule",
            r"\multirow{2}{*}{\textbf{Method}}",
            r"& \multirow{2}{*}{\textbf{Venue}}",
        ]
        for index, (_, display) in enumerate(MODELS):
            trailer = r"@{\hspace{5pt}}" if index < 3 else r"@{}"
            lines.append(rf"& \multicolumn{{2}}{{c{trailer}}}{{\textbf{{{display}}}}}")
        lines[-1] += r" \\"  # end first header row
        for start in (3, 5, 7, 9):
            lines.append(rf"\cmidrule(lr){{{start}-{start + 1}}}")
        lines += ["&", "& " + " & ".join(
            (r"\textbf{AUROC}$\uparrow$", r"\textbf{AUPR}$\uparrow$") * len(MODELS)
        ) + r" \\", r"\midrule"]
        for method, display, venue in METHODS:
            if method == "Log-S":
                lines.append(r"\rowcolor{ShallowPurple}")
                lines.append(r"\textbf{Ours}")
            else:
                lines.append(display)
            lines.append("& " + venue)
            cells = []
            for model, _ in MODELS:
                for metric in METRICS:
                    rank = _rank(summary, model, category, metric)[method]
                    cells.append(_cell(summary, model, category, method, metric, rank))
            lines.append("& " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"}",
                  rf"\label{{tab:amber-{category}-mixed-auc-real-aupr}}",
                  r"\end{table*}"]
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


def render_md(summary: dict) -> str:
    lines = [
        "# AMBER 三类：Log-S 8:2 与 baseline 8:1:1 的 AUROC / Correct-AUPR",
        "",
        "`prompt_last_token | answer_correctness_all`；四模型、seeds 43/44/45。"
        "数值为测试集百分数 mean±population std；AUPR 将正确回答（Real）作为正类。",
        "",
        "Log-S 使用旧 8:2 的完整测试集；baselines 使用 8:1:1 的新测试集。"
        "两侧测试题不同，不能作为同题配对提升。MetaToken 仅报告 LR；"
        "各类别使用整体 AMBER 的模型和阈值，未按类别重新训练或选点。",
    ]
    for category in CATEGORIES:
        lines += ["", f"## {category}（Log-S {COUNTS_82[category]} 题；baselines "
                  f"{COUNTS_811[category]} 题）", ""]
        lines += ["| Method | " + " | ".join(display + " AUROC / Correct-AUPR"
                                         for _, display in MODELS) + " |",
                  "|---|" + "---:|" * len(MODELS)]
        for method, display, _ in METHODS:
            cells = []
            for model, _ in MODELS:
                fields = []
                for metric in METRICS:
                    value = summary[model, category, method, metric]
                    formatted = f"{value['mean']:.2f}"
                    if method == "Log-S":
                        formatted += f"±{value['std']:.2f}"
                    fields.append(formatted)
                cells.append(" / ".join(fields))
            lines.append("| " + display + " | " + " | ".join(cells) + " |")
    lines += ["", f"[LaTeX 表]({STEM}.tex) · [汇总 CSV]({STEM}.csv)"]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = collect()
    output = BASE / NEW
    (output / f"{STEM}.tex").write_text(render_tex(summary))
    (output / f"{STEM}.md").write_text(render_md(summary))
    fields = ["category", "model", "method", "split", "test_questions",
              "auroc_mean_pct", "auroc_std_pct", "real_aupr_mean_pct",
              "real_aupr_std_pct"]
    with (output / f"{STEM}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category in CATEGORIES:
            for model, _ in MODELS:
                for method, _, _ in METHODS:
                    writer.writerow({
                        "category": category, "model": model, "method": method,
                        "split": "8:2" if method == "Log-S" else "8:1:1",
                        "test_questions": (COUNTS_82 if method == "Log-S"
                                           else COUNTS_811)[category],
                        **{f"{metric}_{stat}_pct": summary[model, category, method, metric][stat]
                           for metric in METRICS for stat in ("mean", "std")},
                    })
    print(f"Wrote three tables and 84 summary rows: {output / (STEM + '.tex')}")


if __name__ == "__main__":
    main()
