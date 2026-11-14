#!/usr/bin/env python3
"""Plot Risk and F layer curves for Real/Hall targets on three LVLMs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/log_s_hpre_k_tau_ablation"
OUTPUT = ROOT / "outputs/log_s_hpre_risk_f_curves_by_label"
MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
)
FOUR_MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
)
FEATURES = (
    ("log_s_hpre_k032_tau003", "Risk"),
    ("f_log", r"$F_{\log}$"),
)
GROUPS = (
    (0, "Hallucination", "#D55E00"),
    (1, "Non-hallucination", "#0072B2"),
)


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    return mean, std, std / np.sqrt(values.shape[0])


def load_curves() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summaries: list[dict] = []
    for model, display in MODELS:
        saved = np.load(SOURCE / model / "compact_features.npz", allow_pickle=False)
        labels = np.asarray(saved["label"], dtype=np.int8)
        strict_test_mask = np.asarray(saved["strict_test_mask"], dtype=bool)
        if labels.ndim != 1 or strict_test_mask.shape != labels.shape:
            raise ValueError(f"{model}: invalid labels/test mask")

        for feature_key, feature_name in FEATURES:
            feature = np.asarray(saved[feature_key], dtype=np.float64)
            if feature.ndim != 2 or feature.shape[0] != len(labels):
                raise ValueError(f"{model}/{feature_key}: invalid shape {feature.shape}")
            if not np.isfinite(feature).all():
                raise ValueError(f"{model}/{feature_key}: non-finite values")

            for scope, selected in (
                ("all", np.ones(len(labels), dtype=bool)),
                ("test", strict_test_mask),
            ):
                scoped_labels = labels[selected]
                scoped_feature = feature[selected]
                for label_value, label_name, _ in GROUPS:
                    values = scoped_feature[scoped_labels == label_value]
                    if len(values) < 2:
                        raise ValueError(f"{model}/{feature_key}/{scope}/{label_name}: too few rows")
                    mean, std, sem = mean_sem(values)
                    summaries.append(
                        {
                            "scope": scope,
                            "model": model,
                            "model_display": display,
                            "feature": feature_key,
                            "feature_display": feature_name,
                            "label": label_name,
                            "count": len(values),
                            "layer_average": float(mean.mean()),
                        }
                    )
                    for layer, (mu, sigma, error) in enumerate(zip(mean, std, sem), start=1):
                        rows.append(
                            {
                                "scope": scope,
                                "model": model,
                                "model_display": display,
                                "feature": feature_key,
                                "feature_display": feature_name,
                                "label": label_name,
                                "layer": layer,
                                "count": len(values),
                                "mean": float(mu),
                                "std": float(sigma),
                                "sem": float(error),
                            }
                        )
    return rows, summaries


def plot(scope: str, rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.3))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.10, top=0.82, wspace=0.19, hspace=0.18)
    for column, (model, display) in enumerate(MODELS):
        axes[0, column].set_title(display, fontsize=12, fontweight="semibold", pad=9)
        for row_index, (feature_key, feature_name) in enumerate(FEATURES):
            axis = axes[row_index, column]
            for _, label_name, color in GROUPS:
                selected = [
                    row for row in rows
                    if row["scope"] == scope
                    and row["model"] == model
                    and row["feature"] == feature_key
                    and row["label"] == label_name
                ]
                layers = np.asarray([row["layer"] for row in selected])
                mean = np.asarray([row["mean"] for row in selected])
                sem = np.asarray([row["sem"] for row in selected])
                axis.plot(layers, mean, color=color, linewidth=2.05, label=label_name)
                axis.fill_between(layers, mean - sem, mean + sem, color=color, alpha=0.17, linewidth=0)
            axis.set_xlim(1, int(layers[-1]))
            axis.margins(y=0.08)
            axis.grid(axis="y", alpha=0.22, linewidth=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(labelsize=9)
            if column == 0:
                axis.set_ylabel(feature_name, fontsize=12, fontweight="semibold")
            if row_index == 1:
                axis.set_xlabel("Decoder layer", fontsize=10)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=10,
    )
    scope_title = "Fixed 8:2 test split" if scope == "test" else "All aligned object tokens"
    fig.suptitle(f"{scope_title} (mean ± SEM)", fontsize=11, y=0.905)
    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUTPUT / f"risk_f_curves_{scope}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_log1p_f_all_curves(models: tuple[tuple[str, str], ...] = MODELS) -> list[dict]:
    """Load all samples, applying log(1 + F) before aggregation."""
    rows: list[dict] = []
    for model, display in models:
        saved = np.load(SOURCE / model / "compact_features.npz", allow_pickle=False)
        labels = np.asarray(saved["label"], dtype=np.int8)
        arrays = (
            ("log_s_hpre_k032_tau003", "Risk", np.asarray(saved["log_s_hpre_k032_tau003"], dtype=np.float64)),
            ("log1p_f_log", r"$\log(1+F_{\log})$", np.log1p(np.asarray(saved["f_log"], dtype=np.float64))),
        )
        for feature_key, feature_name, feature in arrays:
            if not np.isfinite(feature).all():
                raise ValueError(f"{model}/{feature_key}: non-finite values after transform")
            for label_value, label_name, _ in GROUPS:
                values = feature[labels == label_value]
                mean, std, sem = mean_sem(values)
                for layer, (mu, sigma, error) in enumerate(zip(mean, std, sem), start=1):
                    rows.append(
                        {
                            "scope": "all",
                            "model": model,
                            "model_display": display,
                            "feature": feature_key,
                            "feature_display": feature_name,
                            "label": label_name,
                            "layer": layer,
                            "count": len(values),
                            "mean": float(mu),
                            "std": float(sigma),
                            "sem": float(error),
                        }
                    )
    return rows


def plot_log1p_f_all(rows: list[dict]) -> None:
    feature_specs = (
        ("log_s_hpre_k032_tau003", "Risk"),
        ("log1p_f_log", r"$\log(1+F_{\log})$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.3))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.10, top=0.82, wspace=0.19, hspace=0.18)
    for column, (model, display) in enumerate(MODELS):
        axes[0, column].set_title(display, fontsize=12, fontweight="semibold", pad=9)
        for row_index, (feature_key, feature_name) in enumerate(feature_specs):
            axis = axes[row_index, column]
            for _, label_name, color in GROUPS:
                selected = [
                    row for row in rows
                    if row["model"] == model
                    and row["feature"] == feature_key
                    and row["label"] == label_name
                ]
                layers = np.asarray([row["layer"] for row in selected])
                mean = np.asarray([row["mean"] for row in selected])
                sem = np.asarray([row["sem"] for row in selected])
                axis.plot(layers, mean, color=color, linewidth=2.05, label=label_name)
                axis.fill_between(layers, mean - sem, mean + sem, color=color, alpha=0.17, linewidth=0)
            axis.set_xlim(1, int(layers[-1]))
            axis.margins(y=0.08)
            axis.grid(axis="y", alpha=0.22, linewidth=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(labelsize=9)
            if column == 0:
                axis.set_ylabel(feature_name, fontsize=12, fontweight="semibold")
            if row_index == 1:
                axis.set_xlabel("Decoder layer", fontsize=10)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=10,
    )
    fig.suptitle("All aligned object tokens (mean ± SEM)", fontsize=11, y=0.905)
    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUTPUT / f"risk_log1p_f_curves_all.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_log1p_f_all_95ci(
    rows: list[dict],
    *,
    cvpr: bool,
    models: tuple[tuple[str, str], ...] = MODELS,
    output_tag: str = "",
) -> None:
    """Plot all samples with normal-approximation 95% confidence intervals."""
    feature_specs = (
        ("log_s_hpre_k032_tau003", "Risk"),
        ("log1p_f_log", r"$\log(1+F_{\log})$"),
    )
    if cvpr:
        rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
        figsize = (7.1, 3.45 if len(models) == 3 else 3.25)
        layout = dict(
            left=0.075,
            right=0.995,
            bottom=0.145,
            top=0.78,
            wspace=0.25 if len(models) == 3 else 0.31,
            hspace=0.19,
        )
        title_size = 8.2 if len(models) == 3 else 7.0
        label_size, tick_size, legend_size = 7.8, 6.8 if len(models) == 3 else 6.1, 7.5
        line_width, band_alpha = 1.35, 0.15
    else:
        rc = {}
        figsize = (13.2 if len(models) == 3 else 17.2, 6.3)
        layout = dict(left=0.075, right=0.99, bottom=0.10, top=0.82, wspace=0.19, hspace=0.18)
        title_size, label_size, tick_size, legend_size = 12, 10, 9, 10
        line_width, band_alpha = 2.05, 0.17

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, len(models), figsize=figsize, squeeze=False)
        fig.subplots_adjust(**layout)
        for column, (model, display) in enumerate(models):
            axes[0, column].set_title(display, fontsize=title_size, fontweight="semibold", pad=5 if cvpr else 9)
            for row_index, (feature_key, feature_name) in enumerate(feature_specs):
                axis = axes[row_index, column]
                for _, label_name, color in GROUPS:
                    selected = [
                        row for row in rows
                        if row["model"] == model
                        and row["feature"] == feature_key
                        and row["label"] == label_name
                    ]
                    layers = np.asarray([row["layer"] for row in selected])
                    mean = np.asarray([row["mean"] for row in selected])
                    ci95 = 1.96 * np.asarray([row["sem"] for row in selected])
                    axis.plot(
                        layers,
                        mean,
                        color=color,
                        linewidth=line_width,
                        linestyle="-",
                        label=label_name,
                    )
                    axis.fill_between(
                        layers,
                        mean - ci95,
                        mean + ci95,
                        color=color,
                        alpha=band_alpha,
                        linewidth=0,
                    )
                axis.set_xlim(1, int(layers[-1]))
                axis.margins(y=0.08)
                axis.grid(axis="y", alpha=0.18, linewidth=0.45 if cvpr else 0.7)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.tick_params(labelsize=tick_size, length=2.5 if cvpr else 3.5)
                if column == 0:
                    axis.set_ylabel(feature_name, fontsize=label_size, fontweight="semibold")
                if row_index == 1:
                    axis.set_xlabel("Decoder layer", fontsize=label_size)

        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        if cvpr:
            fig.legend(
                handles,
                legend_labels,
                loc="upper center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.45, 0.985),
                fontsize=legend_size,
                handlelength=2.5,
                columnspacing=1.4,
            )
            fig.text(0.988, 0.968, "Mean ± 95% CI", ha="right", va="top", fontsize=6.8)
            stem = f"risk_log1p_f_curves_all_95ci{output_tag}_cvpr"
        else:
            fig.legend(
                handles,
                legend_labels,
                loc="upper center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 0.995),
                fontsize=legend_size,
            )
            fig.suptitle("All aligned object tokens (mean ± 95% CI)", fontsize=11, y=0.905)
            stem = f"risk_log1p_f_curves_all_95ci{output_tag}"

        fig.savefig(OUTPUT / f"{stem}.png", dpi=600 if cvpr else 300, bbox_inches="tight")
        fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
        if cvpr:
            fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, summaries = load_curves()
    with (OUTPUT / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for scope in ("test", "all"):
        plot(scope, rows)
    log1p_f_rows = load_log1p_f_all_curves()
    with (OUTPUT / "curves_log1p_f_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log1p_f_rows[0]))
        writer.writeheader()
        writer.writerows(log1p_f_rows)
    plot_log1p_f_all(log1p_f_rows)
    ci_rows = [
        {
            **row,
            "ci95_lower": row["mean"] - 1.96 * row["sem"],
            "ci95_upper": row["mean"] + 1.96 * row["sem"],
        }
        for row in log1p_f_rows
    ]
    with (OUTPUT / "curves_log1p_f_all_95ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ci_rows[0]))
        writer.writeheader()
        writer.writerows(ci_rows)
    plot_log1p_f_all_95ci(log1p_f_rows, cvpr=False)
    plot_log1p_f_all_95ci(log1p_f_rows, cvpr=True)
    four_model_rows = load_log1p_f_all_curves(FOUR_MODELS)
    four_model_ci_rows = [
        {
            **row,
            "ci95_lower": row["mean"] - 1.96 * row["sem"],
            "ci95_upper": row["mean"] + 1.96 * row["sem"],
        }
        for row in four_model_rows
    ]
    with (OUTPUT / "curves_log1p_f_all_95ci_four_models.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(four_model_ci_rows[0]))
        writer.writeheader()
        writer.writerows(four_model_ci_rows)
    plot_log1p_f_all_95ci(four_model_rows, cvpr=False, models=FOUR_MODELS, output_tag="_four_models")
    plot_log1p_f_all_95ci(four_model_rows, cvpr=True, models=FOUR_MODELS, output_tag="_four_models")

    lines = [
        "# Risk 与 F 的幻觉/非幻觉逐层曲线",
        "",
        "- Risk：`log_s_hpre_k032_tau003`（K=32，tau=0.03）",
        "- F：`f_log`",
        "- 标签：0=Hallucination，1=Non-hallucination",
        "- 曲线：目标 token 均值；阴影：±1 SEM",
        "",
        "## 图",
        "",
        "- 固定 8:2 测试集：[PNG](risk_f_curves_test.png) · [PDF](risk_f_curves_test.pdf)",
        "- 全部对齐目标：[PNG](risk_f_curves_all.png) · [PDF](risk_f_curves_all.pdf)",
        "- 全部对齐目标，F 先取 `log(1+F)`：[PNG](risk_log1p_f_curves_all.png) · [PDF](risk_log1p_f_curves_all.pdf)",
        "- `log(1+F)` 逐层数据：[CSV](curves_log1p_f_all.csv)",
        "- Mean ± 95% CI：[PNG](risk_log1p_f_curves_all_95ci.png) · [PDF](risk_log1p_f_curves_all_95ci.pdf)",
        "- Mean ± 95% CI，CVPR 风格：[PNG](risk_log1p_f_curves_all_95ci_cvpr.png) · [PDF](risk_log1p_f_curves_all_95ci_cvpr.pdf)",
        "- 95% CI 逐层数据：[CSV](curves_log1p_f_all_95ci.csv)",
        "- 四模型（含 Qwen2.5）Mean ± 95% CI：[PNG](risk_log1p_f_curves_all_95ci_four_models.png) · [PDF](risk_log1p_f_curves_all_95ci_four_models.pdf)",
        "- 四模型（含 Qwen2.5）CVPR 风格：[PNG](risk_log1p_f_curves_all_95ci_four_models_cvpr.png) · [PDF](risk_log1p_f_curves_all_95ci_four_models_cvpr.pdf) · [SVG](risk_log1p_f_curves_all_95ci_four_models_cvpr.svg)",
        "- 四模型 95% CI 数据：[CSV](curves_log1p_f_all_95ci_four_models.csv)",
        "- 数据：[逐层 CSV](curves.csv) · [汇总 CSV](summary.csv)",
        "",
        "## 样本数",
        "",
        "| 范围 | 模型 | Hall | Real |",
        "|---|---|---:|---:|",
    ]
    for scope in ("test", "all"):
        for model, display in MODELS:
            counts = {
                row["label"]: row["count"]
                for row in summaries
                if row["scope"] == scope
                and row["model"] == model
                and row["feature"] == FEATURES[0][0]
            }
            lines.append(
                f"| {scope} | {display} | {counts['Hallucination']} | {counts['Non-hallucination']} |"
            )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
