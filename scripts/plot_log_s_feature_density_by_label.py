#!/usr/bin/env python3
"""Plot held-out object-token densities for Risk/F_log and EMS/SAR."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs/log_s_hpre_k_tau_ablation"
SPLIT = ROOT.parent / "gfchair/outputs/coco4000_512_endac_811/image_splits.json"
OUTPUT = ROOT / "outputs/log_s_hpre_feature_density_by_label"
MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
)
GROUPS = ((1, "Real", "#248FD2"), (0, "Hallucination", "#E45963"))


def load_scores(variant: str, scope: str) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], list[dict]]:
    if scope not in ("test", "all"):
        raise ValueError(f"Unknown scope: {scope}")
    test_ids = np.asarray(json.loads(SPLIT.read_text())["test"], dtype=np.int64) if scope == "test" else None
    if test_ids is not None and len(test_ids) != 400:
        raise ValueError("Expected 400 held-out images in the 8:1:1 split")
    scores = {}
    summary = []
    for model, display in MODELS:
        with np.load(FEATURES / model / "compact_features.npz", allow_pickle=False) as saved:
            mask = np.isin(saved["image_id"], test_ids) if test_ids is not None else np.ones(len(saved["label"]), dtype=bool)
            labels = np.asarray(saved["label"][mask], dtype=np.int8)
            risk = np.asarray(saved["log_s_hpre_k032_tau003"][mask], dtype=np.float64)
            f_log = np.asarray(saved["f_log"][mask], dtype=np.float64)
            if variant == "ems_log_sar":
                sar = f_log.mean(axis=1)
                if np.any(sar <= 0):
                    raise ValueError(f"{model}: SAR must be positive before taking log")
                values = {"ems": risk.mean(axis=1), "sar": np.log(sar)}
            elif variant == "risk_f_log":
                # Original visualization: log1p per layer, then average.
                values = {"risk": risk.mean(axis=1), "f_log": np.log1p(f_log).mean(axis=1)}
            else:
                raise ValueError(f"Unknown variant: {variant}")
        if len(labels) < 50 or not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{model}: invalid test labels")
        scores[model] = {}
        for feature, scalar in values.items():
            if not np.isfinite(scalar).all():
                raise ValueError(f"{model}/{feature}: nonfinite scalar score")
            scores[model][feature] = (labels, scalar)
            for label_value, label_name, _ in GROUPS:
                subset = scalar[labels == label_value]
                if len(subset) < 30:
                    raise ValueError(f"{model}/{feature}/{label_name}: too few samples")
                summary.append({
                    "model": model, "model_display": display, "feature": feature,
                    "label": label_name, "count": len(subset),
                    "mean": float(subset.mean()), "median": float(np.median(subset)),
                    "std": float(subset.std(ddof=1)),
                })
    return scores, summary


def common_bandwidth(values: np.ndarray) -> float:
    n = len(values)
    sd = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, (75, 25))))
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd
    return max(0.9 * scale * n ** (-0.2), np.finfo(float).eps)


def plot(scores: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], variant: str, scope: str) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 5.2))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.105, top=0.84, wspace=0.26, hspace=0.42)
    feature_keys = ("ems", "sar") if variant == "ems_log_sar" else ("risk", "f_log")
    for col, (model, display) in enumerate(MODELS):
        axes[0, col].set_title(display, fontsize=11, fontweight="bold", pad=10)
        for row, feature in enumerate(feature_keys):
            ax = axes[row, col]
            labels, scalar = scores[model][feature]
            bw = common_bandwidth(scalar)
            xx = np.linspace(float(scalar.min() - 3 * bw), float(scalar.max() + 3 * bw), 500)
            for label_value, _, color in GROUPS:
                subset = scalar[labels == label_value]
                sd = float(np.std(subset, ddof=1))
                kde = gaussian_kde(subset, bw_method=bw / sd)
                density = kde(xx)
                ax.fill_between(xx, density, color=color, alpha=0.43, linewidth=0)
                ax.plot(xx, density, color=color, linewidth=1.25)
            ax.set_xlim(xx[0], xx[-1])
            ax.set_ylim(bottom=0)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=8, length=3)
            if variant == "ems_log_sar":
                ax.set_xlabel("EMS" if feature == "ems" else r"$\log(\mathrm{SAR})$", fontsize=9)
            else:
                ax.set_xlabel("Mean layer Risk" if feature == "risk" else r"Mean layer $\log(1+F_{\log})$", fontsize=9)
            if col == 0:
                ax.set_ylabel("Density", fontsize=9)

    if variant == "ems_log_sar":
        fig.text(0.014, 0.62, "(a)  EMS", rotation=90, va="center", fontsize=11, fontweight="bold")
        fig.text(0.014, 0.235, "(b)  SAR", rotation=90, va="center", fontsize=11, fontweight="bold")
    else:
        fig.text(0.014, 0.62, "(a)  Risk", rotation=90, va="center", fontsize=11, fontweight="bold")
        fig.text(0.014, 0.235, r"(b)  $F_{\log}$", rotation=90, va="center", fontsize=11, fontweight="bold")
    handles = [Patch(facecolor=color, edgecolor=color, alpha=0.55, label=name) for _, name, color in GROUPS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.97), ncol=2, frameon=False, fontsize=10)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        tag = "811_test" if scope == "test" else "all"
        fig.savefig(OUTPUT / f"{variant}_density_{tag}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for variant, scope in (("risk_f_log", "test"), ("ems_log_sar", "test"), ("ems_log_sar", "all")):
        scores, summary = load_scores(variant, scope)
        plot(scores, variant, scope)
        summary_name = (
            "summary.csv" if variant == "risk_f_log" else
            "summary_ems_log_sar.csv" if scope == "test" else
            "summary_ems_log_sar_all.csv"
        )
        with (OUTPUT / summary_name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        if variant == "risk_f_log":
            lines = [
                "# Risk 与 F_log 的 Real/Hall 密度图", "",
                "按 COCO 图片级 8:1:1 划分，只用 400 张测试图片中的已对齐 object token；标签 1 为 Real，0 为 Hallucination。每条类别曲线各自归一化为面积 1，展示类别内的分数分布，不表示类别比例。", "",
                "- 第一排：每个 token 的 `log_s_hpre_k032_tau003` 逐层均值（K=32，τ=0.03）。Risk 大通常更倾向 Hall。",
                "- 第二排：每个 token 先逐层计算 `log(1 + f_log)`，再取层均值。分数大通常更倾向 Real。此对数仅用于图示，没有改动训练特征。",
                "- 曲线使用高斯核密度估计；同一面板两类使用相同的绝对带宽。各模型分别设置横轴范围，不能直接比较不同模型的横轴尺度。",
                "- 这张图展示预先指定的单变量摘要，不能代替逐层特征拼接后的 MLP 检测结果。", "",
                "图：[PNG](risk_f_log_density_811_test.png) · [PDF](risk_f_log_density_811_test.pdf) · [SVG](risk_f_log_density_811_test.svg)。[分组统计](summary.csv)。", "",
            ]
            (OUTPUT / "README.md").write_text("\n".join(lines))
        elif scope == "test":
            lines = [
                "# EMS 与 log(SAR) 的 Real/Hall 密度图", "",
                "按 COCO 图片级 8:1:1 划分，只用 400 张测试图片中的已对齐 object token；标签 1 为 Real，0 为 Hallucination。每条类别曲线各自归一化为面积 1。", "",
                "- EMS：原 `log_s_hpre_k032_tau003`，每个 token 的逐层均值（K=32，τ=0.03）。这是图示中的标量摘要。",
                "- SAR：原 `f_log`，每个 token 先取逐层均值 `SAR = mean_l f_log,l`，再在横轴使用 `log(SAR)`。所有 SAR 均大于 0；此对数仅用于图示，没有改动训练特征。",
                "- 两类使用相同的核密度估计带宽；各模型横轴独立，不应直接比较分数尺度。图中分布不能代替逐层特征拼接后的 MLP 检测结果。", "",
                "图：[PNG](ems_log_sar_density_811_test.png) · [PDF](ems_log_sar_density_811_test.pdf) · [SVG](ems_log_sar_density_811_test.svg)。[分组统计](summary_ems_log_sar.csv)。", "",
            ]
            (OUTPUT / "README_ems_sar.md").write_text("\n".join(lines))
        else:
            lines = [
                "# EMS 与 log(SAR) 的 Real/Hall 密度图：全部对齐 object token", "",
                "使用 COCO4000 中各模型所有已对齐且保存了特征的 object token，包含原 8:1:1 划分的 train、validation 和 test。各模型可能有不同数量的可用图片和 token。标签 1 为 Real，0 为 Hallucination。每条类别曲线各自归一化为面积 1。", "",
                "- EMS：原 `log_s_hpre_k032_tau003`，每个 token 的逐层均值（K=32，τ=0.03）。",
                "- SAR：原 `f_log`，每个 token 先取逐层均值 `SAR = mean_l f_log,l`，横轴使用自然对数 `log(SAR)`。所有 SAR 均大于 0；此对数仅用于图示，没有改动训练特征。",
                "- 两类使用相同的核密度估计带宽；各模型横轴独立，不应直接比较分数尺度。此图是全量描述性分布，不是独立测试集性能评估。", "",
                "图：[PNG](ems_log_sar_density_all.png) · [PDF](ems_log_sar_density_all.pdf) · [SVG](ems_log_sar_density_all.svg)。[分组统计](summary_ems_log_sar_all.csv)。", "",
            ]
            (OUTPUT / "README_ems_sar_all.md").write_text("\n".join(lines))
        tag = "811_test" if scope == "test" else "all"
        print(OUTPUT / f"{variant}_density_{tag}.png")


if __name__ == "__main__":
    main()
