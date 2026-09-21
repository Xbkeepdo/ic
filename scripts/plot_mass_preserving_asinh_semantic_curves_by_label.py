#!/usr/bin/env python3
"""Build mass-preserving/asinh semantic features and label-wise layer curves."""

from __future__ import annotations

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

from scripts.plot_sqrt_semantic_probability_curves_by_label import (  # noqa: E402
    DISPLAY,
    MODELS,
    PROBABILITY_ROOT,
    PROB_FIELD,
    VOCAB_CONFIGS,
    target_key,
    vocabulary_size,
)
from utils.io_utils import load_pkl  # noqa: E402


OUT = ROOT / "outputs/mass_preserving_asinh_semantic_curves_by_label"
FEATURES = {
    "semantic_mass": r"$E_l=\sum_m p_{l,m}$",
    "sqrt_diffuseness": r"$D_l^{sqrt}=\sum_m\sqrt{p_{l,m}}/\sqrt{M E_l}$",
    "log_diffuseness": r"$D_l^{log}=\sum_m\log(1+C p_{l,m})/[M\log(1+C E_l/M)]$",
    "mass_preserving_fusion": r"$F_l^{MP}=I_l+C\sum_m I_{l,m}\tilde p_{l,m}$",
    "asinh_sqrt_fusion": r"$F_l^{AS}=\sum_m I_{l,m}[1+asinh(\sqrt{C p_{l,m}})]$",
}


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.ones_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def build_compact(model: str, output_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    model_out = output_dir / model
    model_out.mkdir(parents=True, exist_ok=True)
    cache = model_out / "compact_features.npz"
    audit_path = model_out / "audit.json"
    if cache.exists() and audit_path.exists():
        with np.load(cache, allow_pickle=False) as saved:
            data = {key: np.asarray(saved[key]) for key in saved.files}
        return data, json.loads(audit_path.read_text())

    probability_path = PROBABILITY_ROOT / model / "semantic_attention.pkl"
    am_path = GF_ROOT / "outputs/coco4000_512_endac_811" / model / "am_norms/features.pkl"
    rows = load_pkl(probability_path)
    am_rows = load_pkl(am_path)
    am_by_key = {target_key(row): row for row in am_rows}
    row_keys = [target_key(row) for row in rows]
    if len(am_by_key) != len(am_rows) or len(set(row_keys)) != len(rows):
        raise ValueError(f"{model}: duplicate target keys")
    if set(row_keys) != set(am_by_key):
        raise ValueError(f"{model}: semantic probability and a_m targets differ")
    if not rows or PROB_FIELD not in rows[0].get("matrices", {}):
        raise KeyError(f"{model}: {PROB_FIELD} is missing")

    first = np.asarray(rows[0]["matrices"][PROB_FIELD])
    count, layers = len(rows), int(first.shape[0])
    names = tuple(FEATURES)
    arrays = {name: np.empty((count, layers), dtype=np.float32) for name in names}
    labels = np.empty(count, dtype=np.int8)
    image_ids = np.empty(count, dtype=np.int64)
    visual_tokens = np.empty(count, dtype=np.int32)
    vocab_size = vocabulary_size(model)
    probability_min, probability_max = np.inf, -np.inf
    zero_mass_layers = 0

    for index, row in enumerate(rows):
        key = row_keys[index]
        am_row = am_by_key[key]
        probability = np.asarray(row["matrices"][PROB_FIELD], dtype=np.float64)
        am_norm = np.asarray(am_row["am_norm"], dtype=np.float64)
        if probability.shape != am_norm.shape or probability.shape[0] != layers:
            raise ValueError(f"{model}/{key}: shape mismatch {probability.shape} vs {am_norm.shape}")
        if (not np.isfinite(probability).all() or not np.isfinite(am_norm).all()
                or (probability < 0).any() or (probability > 1).any() or (am_norm < 0).any()):
            raise ValueError(f"{model}/{key}: invalid source values")
        if int(row["label"]) != int(am_row["label"]):
            raise ValueError(f"{model}/{key}: label mismatch")

        token_count = probability.shape[1]
        mass = probability.sum(axis=1)
        root_probability = np.sqrt(probability)
        root_sum = root_probability.sum(axis=1)
        q = np.full_like(probability, 1.0 / token_count)
        np.divide(root_probability, root_sum[:, None], out=q, where=root_sum[:, None] > 0)
        reshaped_probability = mass[:, None] * q
        i_sum = am_norm.sum(axis=1)
        mass_preserving = i_sum + vocab_size * (am_norm * reshaped_probability).sum(axis=1)
        asinh_sqrt = (am_norm * (1.0 + np.arcsinh(np.sqrt(vocab_size * probability)))).sum(axis=1)
        sqrt_diffuseness = safe_divide(root_sum, np.sqrt(token_count * mass))
        log_numerator = np.log1p(vocab_size * probability).sum(axis=1)
        log_denominator = token_count * np.log1p(vocab_size * mass / token_count)
        log_diffuseness = safe_divide(log_numerator, log_denominator)

        if ((sqrt_diffuseness < -1e-8).any() or (sqrt_diffuseness > 1 + 2e-6).any()
                or (log_diffuseness < -1e-8).any() or (log_diffuseness > 1 + 2e-6).any()
                or not np.allclose(reshaped_probability.sum(axis=1), mass, rtol=2e-6, atol=1e-12)):
            raise ValueError(f"{model}/{key}: mass preservation or diffuseness bounds failed")

        values = {
            "semantic_mass": mass,
            "sqrt_diffuseness": sqrt_diffuseness,
            "log_diffuseness": log_diffuseness,
            "mass_preserving_fusion": mass_preserving,
            "asinh_sqrt_fusion": asinh_sqrt,
        }
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{model}/{key}: non-finite {name}")
            arrays[name][index] = value.astype(np.float32)
        labels[index] = int(row["label"])
        image_ids[index] = int(row["image_id"])
        visual_tokens[index] = token_count
        zero_mass_layers += int((mass == 0).sum())
        probability_min = min(probability_min, float(probability.min()))
        probability_max = max(probability_max, float(probability.max()))

    split = json.loads((
        ROOT / "outputs" / model / "COCO4000-512-WCOST-ALL" / "image_splits.json"
    ).read_text())
    train_mask = np.isin(image_ids, np.asarray(split["train"], dtype=np.int64))
    test_mask = np.isin(image_ids, np.asarray(split["test"], dtype=np.int64))
    if not np.all(train_mask | test_mask) or np.any(train_mask & test_mask):
        raise ValueError(f"{model}: invalid split masks")
    data = {
        **arrays,
        "label": labels,
        "image_id": image_ids,
        "visual_tokens": visual_tokens,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }
    np.savez_compressed(cache, **data)
    audit = {
        "model": model,
        "definitions": {
            "semantic_mass": "E_l = sum_m p_lm",
            "sqrt_diffuseness": "sum_m sqrt(p_lm) / sqrt(M * E_l)",
            "log_diffuseness": "sum_m log(1+C*p_lm) / (M*log(1+C*E_l/M))",
            "mass_preserving_fusion": "I_l + C*sum_m I_lm*(E_l*sqrt(p_lm)/sum_j sqrt(p_lj))",
            "asinh_sqrt_fusion": "sum_m I_lm*(1+asinh(sqrt(C*p_lm)))",
        },
        "zero_mass_convention": "q=uniform and diffuseness=1; transformed semantic mass remains zero",
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
        "zero_mass_layers": zero_mass_layers,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    del rows, am_rows, am_by_key
    gc.collect()
    return data, audit


def class_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def summarize_feature(model: str, feature: str, split: str, data: dict[str, np.ndarray]):
    selected = np.ones(len(data["label"]), dtype=bool) if split == "all" else data["test_mask"]
    labels, values = data["label"][selected], data[feature][selected]
    hall, real = values[labels == 0], values[labels == 1]
    hall_mean, hall_sem = class_stats(hall)
    real_mean, real_sem = class_stats(real)
    gap = hall_mean - real_mean
    peak = int(np.argmax(np.abs(gap)))
    curves = [{
        "model": model, "feature": feature, "split": split, "layer": layer + 1,
        "hall_count": len(hall), "real_count": len(real),
        "hall_mean": float(hall_mean[layer]), "hall_sem": float(hall_sem[layer]),
        "real_mean": float(real_mean[layer]), "real_sem": float(real_sem[layer]),
        "hall_minus_real": float(gap[layer]),
    } for layer in range(values.shape[1])]
    summary = {
        "model": model, "feature": feature, "split": split,
        "hall_count": len(hall), "real_count": len(real),
        "hall_layer_average": float(hall_mean.mean()),
        "real_layer_average": float(real_mean.mean()),
        "layer_average_gap": float(gap.mean()),
        "peak_abs_gap_layer": peak + 1, "peak_gap": float(gap[peak]),
    }
    return curves, summary


def plot_feature(feature: str, split: str, rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS):
        selected = [row for row in rows if row["model"] == model
                    and row["feature"] == feature and row["split"] == split]
        layers = np.asarray([row["layer"] for row in selected])
        for prefix, label, color in (("hall", "Hallucination", "#d55e00"),
                                     ("real", "Non-hallucination", "#0072b2")):
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
    OUT.mkdir(parents=True, exist_ok=True)
    curves, summaries, audits = [], [], []
    for model in MODELS:
        data, audit = build_compact(model, OUT)
        audits.append(audit)
        for feature in FEATURES:
            for split in ("all", "test"):
                feature_rows, summary = summarize_feature(model, feature, split, data)
                curves.extend(feature_rows)
                summaries.append(summary)
        del data
        gc.collect()
        print("DONE", model, flush=True)
    with (OUT / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    (OUT / "audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    for feature in FEATURES:
        for split in ("all", "test"):
            plot_feature(feature, split, curves, OUT)
    lines = [
        "# Mass-preserving and asinh-sqrt semantic curves", "",
        "Raw probabilities are GF `object_probability_raw`; `C` is the actual model vocabulary size. Values are mean±SEM by label.", "",
        "| Model | Feature | Split | HALL / REAL | HALL layer avg | REAL layer avg | HALL−REAL | Peak layer | Peak gap |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {DISPLAY[row['model']]} | `{row['feature']}` | {row['split']} | "
            f"{row['hall_count']} / {row['real_count']} | {row['hall_layer_average']:.6g} | "
            f"{row['real_layer_average']:.6g} | {row['layer_average_gap']:+.6g} | "
            f"{row['peak_abs_gap_layer']} | {row['peak_gap']:+.6g} |"
        )
    lines += ["", "## Artifacts", ""]
    for feature in FEATURES:
        lines.append(f"- `{feature}`: [test PNG]({feature}_test.png) · [all PNG]({feature}_all.png)")
    lines.append("- [Layerwise CSV](curves.csv) · [Summary CSV](summary.csv) · [Audits](audits.json)")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(OUT / "summary.md", flush=True)


if __name__ == "__main__":
    main()
