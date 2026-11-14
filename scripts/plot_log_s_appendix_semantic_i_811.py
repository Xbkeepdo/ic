#!/usr/bin/env python3
"""Plot final-norm semantic-only and visual write I across all 4,000 COCO images."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GF = ROOT.parent / "gfchair" / "outputs"
OUT = ROOT / "outputs" / "log_s_appendix"
MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
)
COLORS = {"Hall": "#D55E00", "Real": "#0072B2"}


def read(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def summarize(model: str) -> tuple[dict, list[dict]]:
    semantic_path = GF / "coco4000_512_endac_semantic_attention_all_visual_811" / model / "features.pt"
    write_path = GF / "coco4000_512_endac_attention_i_fusion_811" / model / "features.pt"
    semantic = read(semantic_path)
    write = read(write_path)
    if list(semantic["sample_id"]) != list(write["sample_id"]):
        raise ValueError(f"{model}: sample IDs differ")
    labels = np.asarray(semantic["labels"], dtype=np.int8)
    images = np.asarray(semantic["image_id"], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(write["labels"])):
        raise ValueError(f"{model}: labels differ")
    if not np.array_equal(images, np.asarray(write["image_id"])):
        raise ValueError(f"{model}: image IDs differ")
    if any(not np.array_equal(np.asarray(semantic["masks"][name], dtype=bool),
                                  np.asarray(write["masks"][name], dtype=bool))
           for name in ("train", "val", "test")):
        raise ValueError(f"{model}: split masks differ")
    # The all-visual semantic_only reduction is the mean over visual-token p.
    p = np.asarray(semantic["cubes"]["norm"], dtype=np.float64)[:, :, 0]
    i = np.asarray(write["I"], dtype=np.float64)
    if p.shape != i.shape or not np.isfinite(p).all() or not np.isfinite(i).all():
        raise ValueError(f"{model}: invalid feature shapes or non-finite values")
    unique_images, inv = np.unique(images, return_inverse=True)
    data: dict = {"image_count": len(unique_images), "n_tokens": len(labels), "features": {}}
    rows: list[dict] = []
    for name, feature in (("semantic_only", p), ("I", i)):
        data["features"][name] = {}
        for value, group in ((0, "Hall"), (1, "Real")):
            selected = labels == value
            sums = np.zeros((len(unique_images), feature.shape[1]), dtype=np.float64)
            counts = np.bincount(inv[selected], minlength=len(unique_images)).astype(np.float64)
            np.add.at(sums, inv[selected], feature[selected])
            mean = sums.sum(axis=0) / counts.sum()
            # Cluster-robust delta-method CI for the token-pooled mean,
            # treating each image (including images without this class) as a cluster.
            influence = sums - counts[:, None] * mean[None, :]
            se = np.sqrt(len(unique_images) / (len(unique_images) - 1) *
                         np.sum(influence ** 2, axis=0)) / counts.sum()
            low, high = np.maximum(0, mean - 1.96 * se), mean + 1.96 * se
            data["features"][name][group] = (mean, low, high, int(selected.sum()))
            for layer, (center, lower, upper) in enumerate(zip(mean, low, high), 1):
                rows.append({
                    "model": model, "feature": name, "label": group,
                    "layer": layer, "mean": center, "ci95_low": lower,
                    "ci95_high": upper, "tokens": int(selected.sum()),
                    "images_with_object_tokens": len(unique_images),
                })
    return data, rows


def panel(ax, model: str, data: dict, feature: str) -> None:
    values = data["features"][feature]
    layers = np.arange(1, len(values["Hall"][0]) + 1)
    for group in ("Hall", "Real"):
        mean, low, high, _ = values[group]
        ax.plot(layers, mean, color=COLORS[group], lw=1.6, label=group)
        ax.fill_between(layers, low, high, color=COLORS[group], alpha=0.18, linewidth=0)
    ax.set_xlim(1, layers[-1])
    ax.set_xticks([1, (layers[-1] + 1) // 2, layers[-1]])
    ax.set_title(model, fontsize=9)
    ax.grid(color="0.84", linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Decoder layer", fontsize=8)
    if feature == "semantic_only":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_ylabel(r"Mean visual-token $p(y_t)$", fontsize=8)
    else:
        ax.set_ylabel(r"$I=\sum_m\Vert a_m\Vert$", fontsize=8)


def draw(results: dict, feature: str, stem: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, (model, title) in zip(axes.flat, MODELS):
        panel(ax, title, results[model], feature)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=8)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.09, top=0.89,
                        wspace=0.34, hspace=0.46)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=240)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    all_rows = []
    for model, _ in MODELS:
        results[model], rows = summarize(model)
        all_rows.extend(rows)
        print(model, "images", results[model]["image_count"],
              "Hall/Real", *(results[model]["features"]["I"][group][3]
                            for group in ("Hall", "Real")))
    with (OUT / "semantic_i_all4000_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    draw(results, "semantic_only", "semantic_only_all4000")
    draw(results, "I", "visual_write_i_all4000")


if __name__ == "__main__":
    main()
