#!/usr/bin/env python3
"""Build and plot sqrt(C*p) semantic features, alone and with write norms."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from scripts.train_iclr_endac_attention_i_fusion import MODELS  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


PROBABILITY_ROOT = GF_ROOT / "outputs/coco4000_512_endac_semantic_attention_topk"
PROB_FIELD = "object_probability_raw"
VOCAB_CONFIGS = {
    "llava_1_5_7b": Path("/home/apulis-dev/userdata/models/llava/llava-hf/llava-1___5-7b-hf/config.json"),
    "internvl_2_5_8b": Path("/home/apulis-dev/userdata/models/InternVL2_5-8B/config.json"),
    "qwen2_5_vl_7b": Path("/home/apulis-dev/userdata/models/Qwen2.5-VL-7B-Instruct/config.json"),
    "qwen3_vl_8b": Path("/home/apulis-dev/userdata/models/Qwen3-VL-8B-Instruct/config.json"),
}
OUT = ROOT / "outputs/sqrt_semantic_probability_curves_by_label"
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}
FEATURES = {
    "sqrt_semantic_sum": r"$S_l=\sum_m\sqrt{C p_{l,m}}$",
    "I_sum": r"$I_l=\sum_m I_{l,m}$",
    "residual_fusion": r"$F_l=I_l+\sum_m I_{l,m}\sqrt{C p_{l,m}}$",
}


def target_key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["image_id"]),
        int(row["response_token_idx"]),
        int(row["target_token_id"]),
    )


def vocabulary_size(model: str) -> int:
    config = json.loads(VOCAB_CONFIGS[model].read_text())
    if model == "internvl_2_5_8b":
        return int(config["llm_config"]["vocab_size"])
    if model == "qwen3_vl_8b":
        return int(config["text_config"]["vocab_size"])
    return int(config["vocab_size"])


def build_compact(model: str, output_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    model_out = output_dir / model
    model_out.mkdir(parents=True, exist_ok=True)
    cache = model_out / "compact_features.npz"
    audit_path = model_out / "audit.json"
    if cache.exists() and audit_path.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return ({key: np.asarray(saved[key]) for key in saved.files},
                    json.loads(audit_path.read_text()))

    probability_path = PROBABILITY_ROOT / model / "semantic_attention.pkl"
    am_path = GF_ROOT / "outputs/coco4000_512_endac_811" / model / "am_norms/features.pkl"
    rows = load_pkl(probability_path)
    am_rows = load_pkl(am_path)
    if not rows or "matrices" not in rows[0] or PROB_FIELD not in rows[0]["matrices"]:
        raise KeyError(f"{model}: direct semantic probability matrix is missing")
    am_by_key = {target_key(row): row for row in am_rows}
    if len(am_by_key) != len(am_rows):
        raise ValueError(f"{model}: duplicate a_m target keys")
    row_keys = [target_key(row) for row in rows]
    if len(set(row_keys)) != len(rows) or set(row_keys) != set(am_by_key):
        raise ValueError(f"{model}: semantic probability and a_m targets differ")

    layers = int(np.asarray(rows[0]["matrices"][PROB_FIELD]).shape[0])
    count = len(rows)
    sqrt_sum = np.empty((count, layers), dtype=np.float32)
    i_sum = np.empty((count, layers), dtype=np.float32)
    residual = np.empty((count, layers), dtype=np.float32)
    sqrt_mean = np.empty((count, layers), dtype=np.float32)
    residual_mean = np.empty((count, layers), dtype=np.float32)
    labels = np.empty(count, dtype=np.int8)
    image_ids = np.empty(count, dtype=np.int64)
    visual_tokens = np.empty(count, dtype=np.int32)
    vocab_size = vocabulary_size(model)
    probability_min, probability_max = np.inf, -np.inf

    for index, row in enumerate(rows):
        key = row_keys[index]
        am_row = am_by_key[key]
        probability = np.asarray(row["matrices"][PROB_FIELD], dtype=np.float64)
        am_norm = np.asarray(am_row["am_norm"], dtype=np.float64)
        if probability.shape != am_norm.shape or probability.shape[0] != layers:
            raise ValueError(
                f"{model}/{key}: probability {probability.shape} != a_m {am_norm.shape}"
            )
        if (not np.isfinite(probability).all() or not np.isfinite(am_norm).all()
                or (probability < 0).any() or (probability > 1).any()
                or (am_norm < 0).any() or vocab_size <= 0):
            raise ValueError(f"{model}/{key}: invalid source values")
        if int(row["label"]) != int(am_row["label"]):
            raise ValueError(f"{model}/{key}: label mismatch")
        transformed = np.sqrt(vocab_size * probability)
        semantic_increment = am_norm * transformed
        sqrt_sum[index] = transformed.sum(axis=1).astype(np.float32)
        i_sum[index] = am_norm.sum(axis=1).astype(np.float32)
        residual[index] = (am_norm + semantic_increment).sum(axis=1).astype(np.float32)
        sqrt_mean[index] = transformed.mean(axis=1).astype(np.float32)
        residual_mean[index] = (am_norm + semantic_increment).mean(axis=1).astype(np.float32)
        labels[index] = int(row["label"])
        image_ids[index] = int(row["image_id"])
        visual_tokens[index] = int(probability.shape[1])
        probability_min = min(probability_min, float(probability.min()))
        probability_max = max(probability_max, float(probability.max()))

    split = json.loads((
        ROOT / "outputs" / model / "COCO4000-512-WCOST-ALL" / "image_splits.json"
    ).read_text())
    train_mask = np.isin(image_ids, np.asarray(split["train"], dtype=np.int64))
    test_mask = np.isin(image_ids, np.asarray(split["test"], dtype=np.int64))
    if not np.all(train_mask | test_mask) or np.any(train_mask & test_mask):
        raise ValueError(f"{model}: invalid image split masks")
    data = {
        "sqrt_semantic_sum": sqrt_sum,
        "I_sum": i_sum,
        "residual_fusion": residual,
        "sqrt_semantic_mean": sqrt_mean,
        "residual_fusion_mean": residual_mean,
        "label": labels,
        "image_id": image_ids,
        "visual_tokens": visual_tokens,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }
    np.savez_compressed(cache, **data)
    audit = {
        "model": model,
        "definition": {
            "sqrt_semantic_sum": "sum_m sqrt(C * p_lm)",
            "I_sum": "sum_m ||a_lm||_2",
            "residual_fusion": "sum_m ||a_lm||_2 * (1 + sqrt(C * p_lm))",
            "sqrt_semantic_mean": "mean_m sqrt(C * p_lm), token-count diagnostic",
            "residual_fusion_mean": "mean_m ||a_lm||_2 * (1 + sqrt(C * p_lm)), token-count diagnostic",
        },
        "probability_source": str(probability_path),
        "am_norm_source": str(am_path),
        "probability_field": f"matrices.{PROB_FIELD}",
        "vocabulary_size": vocab_size,
        "vocabulary_config": str(VOCAB_CONFIGS[model]),
        "targets": count,
        "train_targets": int(train_mask.sum()),
        "test_targets": int(test_mask.sum()),
        "layers": layers,
        "visual_token_count_min": int(visual_tokens.min()),
        "visual_token_count_max": int(visual_tokens.max()),
        "probability_min": probability_min,
        "probability_max": probability_max,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    del rows, am_rows, am_by_key
    gc.collect()
    return data, audit


def class_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def summarize_feature(
    model: str, feature: str, split: str, data: dict[str, np.ndarray]
) -> tuple[list[dict], dict]:
    selected = np.ones(len(data["label"]), dtype=bool) if split == "all" else data["test_mask"]
    labels = data["label"][selected]
    values = data[feature][selected]
    hall, real = values[labels == 0], values[labels == 1]
    hall_mean, hall_sem = class_stats(hall)
    real_mean, real_sem = class_stats(real)
    gap = hall_mean - real_mean
    peak = int(np.argmax(np.abs(gap)))
    curves = [{
        "model": model,
        "feature": feature,
        "split": split,
        "layer": layer + 1,
        "hall_count": len(hall),
        "real_count": len(real),
        "hall_mean": float(hall_mean[layer]),
        "hall_sem": float(hall_sem[layer]),
        "real_mean": float(real_mean[layer]),
        "real_sem": float(real_sem[layer]),
        "hall_minus_real": float(gap[layer]),
    } for layer in range(values.shape[1])]
    summary = {
        "model": model,
        "feature": feature,
        "split": split,
        "hall_count": len(hall),
        "real_count": len(real),
        "hall_layer_average": float(hall_mean.mean()),
        "real_layer_average": float(real_mean.mean()),
        "layer_average_gap": float(gap.mean()),
        "peak_abs_gap_layer": peak + 1,
        "peak_gap": float(gap[peak]),
    }
    return curves, summary


def plot_feature(feature: str, split: str, rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS):
        selected = [row for row in rows if row["model"] == model
                    and row["feature"] == feature and row["split"] == split]
        layers = np.asarray([row["layer"] for row in selected])
        for prefix, label, color in (
            ("hall", "Hallucination", "#d55e00"),
            ("real", "Non-hallucination", "#0072b2"),
        ):
            mean = np.asarray([row[f"{prefix}_mean"] for row in selected])
            sem = np.asarray([row[f"{prefix}_sem"] for row in selected])
            axis.plot(layers, mean, color=color, linewidth=2.1, label=label)
            axis.fill_between(layers, mean - sem, mean + sem, color=color, alpha=0.18)
        axis.set_title(DISPLAY[model])
        axis.set_xlabel("Decoder layer (1-based)")
        axis.set_ylabel("Mean feature value")
        axis.set_xlim(1, int(layers[-1]))
        axis.grid(alpha=0.23)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.025))
    scope = "all aligned targets" if split == "all" else "held-out 800-image test split"
    fig.suptitle(f"{FEATURES[feature]}\n{scope}; shaded band = ±1 SEM", y=1.075)
    fig.savefig(output_dir / f"{feature}_{split}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{feature}_{split}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curves, summaries, audits = [], [], []
    for model in MODELS:
        data, audit = build_compact(model, output_dir)
        audits.append(audit)
        for feature in FEATURES:
            for split in ("all", "test"):
                feature_rows, summary = summarize_feature(model, feature, split, data)
                curves.extend(feature_rows)
                summaries.append(summary)
        del data
        gc.collect()
        print("DONE", model, flush=True)

    with (output_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader(); writer.writerows(curves)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (output_dir / "audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    for feature in FEATURES:
        for split in ("all", "test"):
            plot_feature(feature, split, curves, output_dir)

    lines = [
        "# Square-root semantic probability curves", "",
        "`p_lm` is the absolute target-token vocabulary-softmax probability and `C` is the model's actual output vocabulary size. Existing MAD experiments remain unchanged as the baseline.", "",
        "| Model | Feature | Split | HALL / REAL | HALL layer avg | REAL layer avg | HALL−REAL | Peak layer | Peak gap |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {DISPLAY[row['model']]} | `{row['feature']}` | {row['split']} | "
            f"{row['hall_count']} / {row['real_count']} | {row['hall_layer_average']:.5f} | "
            f"{row['real_layer_average']:.5f} | {row['layer_average_gap']:+.5f} | "
            f"{row['peak_abs_gap_layer']} | {row['peak_gap']:+.5f} |"
        )
    lines += ["", "## Artifacts", ""]
    for feature in FEATURES:
        lines.append(
            f"- `{feature}`: [test PNG]({feature}_test.png) · [all PNG]({feature}_all.png)"
        )
    lines += ["- [Layerwise CSV](curves.csv) · [Summary CSV](summary.csv) · [Audits](audits.json)"]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(output_dir / "summary.md", flush=True)


if __name__ == "__main__":
    main()
