#!/usr/bin/env python3
"""Analyze image 831 with LLaVA log_s_hpre spatial maps and regional F_log."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    ROOT.parent
    / "gfchair/outputs/coco4000_512_linear_cost_log_s_risk_811"
    / "llava_1_5_7b/features.pkl"
)
DEFAULT_GENERATIONS = (
    ROOT.parent
    / "gfchair/outputs/coco4000_512_endac_811"
    / "llava_1_5_7b/generations.json"
)
DEFAULT_IMAGE = Path(
    "/home/apulis-dev/userdata/val2014/COCO_val2014_000000000831.jpg"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/llava_1_5_7b/COCO831-LOGS-PRE-TOPK-ANALYSIS"
)

IMAGE_ID = 831
TOKENS = ("skateboard", "handbag")
TOKEN_RESPONSE_INDICES = {"skateboard": 37, "handbag": 123}
LAYERS = (18, 21, 24, 27, 30, 32)
SIGNALS = (
    ("raw_attention", "attention", "Raw attention"),
    ("target_distribution", "target_log_s", r"$T$: target distribution"),
    ("source_distribution", "source_hpre", r"$P$: source distribution"),
)
GRID = (24, 24)
TOP32 = 32
TOP16 = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_rows(path: Path) -> dict[str, dict]:
    """Load the consolidated artifact once and retain only the two exact targets."""
    with path.open("rb") as handle:
        records = pickle.load(handle)
    selected: dict[str, dict] = {}
    for row in records:
        if int(row.get("image_id", -1)) != IMAGE_ID:
            continue
        for token in TOKENS:
            if (
                str(row.get("token")) == token
                and int(row.get("response_token_idx", -1))
                == TOKEN_RESPONSE_INDICES[token]
            ):
                selected[token] = row
    del records
    gc.collect()
    missing = set(TOKENS) - set(selected)
    if missing:
        raise RuntimeError(f"Missing exact image-831 targets: {sorted(missing)}")
    return selected


def validate_rows(rows: dict[str, dict]) -> None:
    for token, row in rows.items():
        matrices = row["matrices"]
        required = {
            "attention",
            "target_log_s",
            "source_hpre",
            "object_probability_norm_hpre",
            "am_norm",
        }
        missing = required - set(matrices)
        if missing:
            raise ValueError(f"{token}: missing matrices {sorted(missing)}")
        for name in required:
            value = np.asarray(matrices[name], dtype=np.float64)
            if value.shape != (32, 576):
                raise ValueError(f"{token}/{name}: unexpected shape {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{token}/{name}: contains non-finite values")
        for name in ("attention", "target_log_s", "source_hpre"):
            totals = np.asarray(matrices[name], dtype=np.float64).sum(axis=1)
            if not np.allclose(totals, 1.0, rtol=2e-5, atol=2e-6):
                raise ValueError(f"{token}/{name}: rows are not normalized")
        if list(row["visual_grid"]) != list(GRID):
            raise ValueError(f"{token}: expected visual grid {GRID}, got {row['visual_grid']}")


def center_square_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    box = (left, top, left + side, top + side)
    return image.crop(box), box


def topk(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    count = min(int(k), flat.size)
    indices = np.argpartition(flat, -count)[-count:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    retained = np.zeros_like(flat)
    retained[indices] = flat[indices]
    return retained, indices.astype(np.int32), float(flat[indices].sum())


def blue_base(image: Image.Image) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32) / 255.0
    pale_blue = np.asarray([0.76, 0.88, 0.98], dtype=np.float32)
    return np.clip(0.64 * base + 0.36 * pale_blue, 0.0, 1.0)


def resize_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    source = Image.fromarray(np.asarray(values, dtype=np.float32))
    return np.asarray(
        source.resize(size, resample=Image.Resampling.BICUBIC), dtype=np.float32
    )


def overlay(image: Image.Image, retained: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = retained.reshape(GRID)
    maximum = float(grid.max())
    relative = grid / maximum if maximum > 0.0 else grid
    smooth = np.clip(resize_map(relative, image.size), 0.0, 1.0)
    color = plt.get_cmap("turbo")(smooth)[..., :3]
    alpha = (0.82 * np.power(smooth, 0.72))[..., None]
    base = blue_base(image)
    blended = np.clip(base * (1.0 - alpha) + color * alpha, 0.0, 1.0)
    return blended, smooth


def build_raw_arrays(rows: dict[str, dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for token, row in rows.items():
        matrices = row["matrices"]
        p_norm = np.asarray(matrices["object_probability_norm_hpre"], dtype=np.float64)
        am_norm = np.asarray(matrices["am_norm"], dtype=np.float64)
        vocabulary_size = int(row["vocabulary_size"])
        f_component = am_norm * (1.0 + np.log1p(vocabulary_size * p_norm))
        arrays[f"{token}__raw_attention"] = np.asarray(
            matrices["attention"], dtype=np.float32
        )
        arrays[f"{token}__target_distribution"] = np.asarray(
            matrices["target_log_s"], dtype=np.float32
        )
        arrays[f"{token}__source_distribution"] = np.asarray(
            matrices["source_hpre"], dtype=np.float32
        )
        arrays[f"{token}__object_probability_norm_hpre"] = p_norm.astype(np.float32)
        arrays[f"{token}__am_norm"] = am_norm.astype(np.float32)
        arrays[f"{token}__f_log_component"] = f_component.astype(np.float32)
        arrays[f"{token}__f_log_full_visual"] = f_component.sum(axis=1).astype(
            np.float32
        )
    return arrays


def build_topk_artifacts(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    saved: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for token in TOKENS:
        for signal, _matrix, _title in SIGNALS:
            values = arrays[f"{token}__{signal}"]
            for layer in LAYERS:
                layer_values = values[layer - 1]
                retained32, indices32, mass32 = topk(layer_values, TOP32)
                _retained16, indices16, mass16 = topk(layer_values, TOP16)
                prefix = f"{token}__{signal}__layer{layer:02d}"
                saved[f"{prefix}__top32_values"] = retained32.astype(np.float32)
                saved[f"{prefix}__top32_indices"] = indices32
                saved[f"{prefix}__top16_indices"] = indices16
                rows.append(
                    {
                        "token": token,
                        "signal": signal,
                        "layer": layer,
                        "top32_mass": mass32,
                        "top16_mass": mass16,
                        "top32_indices": " ".join(map(str, indices32.tolist())),
                        "top16_indices": " ".join(map(str, indices16.tolist())),
                    }
                )
    return saved, rows


def f_log_cross_scores(
    arrays: dict[str, np.ndarray],
) -> list[dict]:
    results: list[dict] = []
    for signal, _matrix, _title in SIGNALS:
        for layer in LAYERS:
            for region_token in TOKENS:
                region_values = arrays[f"{region_token}__{signal}"][layer - 1]
                _retained, indices, region_mass = topk(region_values, TOP16)
                for score_token in TOKENS:
                    components = arrays[f"{score_token}__f_log_component"][layer - 1]
                    restricted = float(components[indices].sum())
                    full = float(components.sum())
                    results.append(
                        {
                            "signal": signal,
                            "layer": layer,
                            "region_token": region_token,
                            "score_token": score_token,
                            "top_k": TOP16,
                            "region_signal_mass": region_mass,
                            "f_log_top16": restricted,
                            "f_log_full_visual": full,
                            "f_log_top16_fraction": restricted / full if full > 0 else 0.0,
                            "region_indices": " ".join(map(str, indices.tolist())),
                        }
                    )
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_token(
    *,
    token: str,
    arrays: dict[str, np.ndarray],
    image: Image.Image,
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        nrows=len(SIGNALS),
        ncols=len(LAYERS),
        figsize=(16.2, 8.4),
        constrained_layout=True,
    )
    last_mappable = None
    for row_index, (signal, _matrix, title) in enumerate(SIGNALS):
        for column_index, layer in enumerate(LAYERS):
            axis = axes[row_index, column_index]
            values = arrays[f"{token}__{signal}"][layer - 1]
            retained, _indices, mass = topk(values, TOP32)
            rendered, smooth = overlay(image, retained)
            axis.imshow(rendered)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"Layer {layer}\nTop-32 mass={mass:.3f}", fontsize=9)
            if column_index == 0:
                axis.set_ylabel(title, fontsize=11)
            last_mappable = axis.imshow(
                smooth,
                cmap="turbo",
                vmin=0.0,
                vmax=1.0,
                alpha=0.0,
            )
    figure.suptitle(
        f"LLaVA-1.5-7B · COCO 831 · {token} · log_s_hpre · independent Top-32",
        fontsize=14,
    )
    if last_mappable is not None:
        colorbar = figure.colorbar(
            last_mappable,
            ax=axes,
            location="right",
            fraction=0.018,
            pad=0.012,
        )
        colorbar.set_label("Within-panel relative intensity")
    figure.savefig(output / f"{token}_top32_heatmaps.png", dpi=dpi)
    figure.savefig(output / f"{token}_top32_heatmaps.pdf")
    plt.close(figure)


def plot_top16_regions(
    *,
    region_token: str,
    arrays: dict[str, np.ndarray],
    image: Image.Image,
    output: Path,
    dpi: int,
) -> None:
    other_token = next(token for token in TOKENS if token != region_token)
    figure, axes = plt.subplots(
        nrows=len(SIGNALS),
        ncols=len(LAYERS),
        figsize=(17.4, 8.8),
        constrained_layout=True,
    )
    last_mappable = None
    for row_index, (signal, _matrix, title) in enumerate(SIGNALS):
        for column_index, layer in enumerate(LAYERS):
            axis = axes[row_index, column_index]
            values = arrays[f"{region_token}__{signal}"][layer - 1]
            retained, indices, mass = topk(values, TOP16)
            rendered, smooth = overlay(image, retained)
            self_score = float(
                arrays[f"{region_token}__f_log_component"][layer - 1][indices].sum()
            )
            cross_score = float(
                arrays[f"{other_token}__f_log_component"][layer - 1][indices].sum()
            )
            axis.imshow(rendered)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"Layer {layer} · mass={mass:.3f}\n"
                f"F({region_token})={self_score:.3f} · F({other_token})={cross_score:.3f}",
                fontsize=8.2,
            )
            if column_index == 0:
                axis.set_ylabel(title, fontsize=11)
            last_mappable = axis.imshow(
                smooth,
                cmap="turbo",
                vmin=0.0,
                vmax=1.0,
                alpha=0.0,
            )
    figure.suptitle(
        f"LLaVA-1.5-7B · COCO 831 · {region_token} independent Top-16 regions · regional F_log",
        fontsize=14,
    )
    if last_mappable is not None:
        colorbar = figure.colorbar(
            last_mappable,
            ax=axes,
            location="right",
            fraction=0.018,
            pad=0.012,
        )
        colorbar.set_label("Within-panel relative intensity")
    figure.savefig(output / f"{region_token}_top16_regions_f_log.png", dpi=dpi)
    figure.savefig(output / f"{region_token}_top16_regions_f_log.pdf")
    plt.close(figure)


def mean_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["signal"], row["region_token"], row["score_token"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (signal, region_token, score_token), values in grouped.items():
        summary.append(
            {
                "signal": signal,
                "region_token": region_token,
                "score_token": score_token,
                "layers": "18 21 24 27 30 32",
                "mean_f_log_top16": float(
                    np.mean([row["f_log_top16"] for row in values])
                ),
                "mean_f_log_full_visual": float(
                    np.mean([row["f_log_full_visual"] for row in values])
                ),
                "mean_f_log_top16_fraction": float(
                    np.mean([row["f_log_top16_fraction"] for row in values])
                ),
                "mean_region_signal_mass": float(
                    np.mean([row["region_signal_mass"] for row in values])
                ),
            }
        )
    return summary


def write_report(
    path: Path,
    *,
    generation: str,
    source_hash: str,
    summary: list[dict],
) -> None:
    lookup = {
        (row["signal"], row["region_token"], row["score_token"]): row
        for row in summary
    }
    lines = [
        "# COCO 831 · LLaVA log_s_hpre Top-K analysis",
        "",
        "- Image: `COCO_val2014_000000000831.jpg`",
        "- Model: `LLaVA-1.5-7B`",
        "- Layers: `18, 21, 24, 27, 30, 32` (1-based)",
        "- Visual grid: `24×24` (LLaVA center-square model view)",
        "- `T = renorm(raw_attention × log(1 + |V| × p_hpre))`",
        "- `P = softmax(cos(o_ffn, hpre_visual) / 0.07)`",
        "- Raw attention, T and P select Top-K independently.",
        "- F_log regional score uses the selected token/signal Top-16 mask.",
        f"- Source SHA-256: `{source_hash}`",
        "",
        "## Historical mainline generation",
        "",
        generation,
        "",
    ]
    for signal, _matrix, title in SIGNALS:
        lines += [f"## Six-layer mean F_log · {title}", ""]
        lines += [
            "| Top-16 region | Score token | F_log Top-16 | Full visual F_log | Fraction | Region signal mass |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for region_token in TOKENS:
            for score_token in TOKENS:
                row = lookup[(signal, region_token, score_token)]
                lines.append(
                    f"| {region_token} | {score_token} | "
                    f"{row['mean_f_log_top16']:.6f} | "
                    f"{row['mean_f_log_full_visual']:.6f} | "
                    f"{row['mean_f_log_top16_fraction']:.4f} | "
                    f"{row['mean_region_signal_mass']:.4f} |"
                )
        lines.append("")
    lines += [
        "## Files",
        "",
        "- `raw_matrices.npz`: complete 32×576 raw attention/T/P, hpre probability, a_m and F_log components.",
        "- `topk_masks_and_values.npz`: selected-layer Top-32 retained maps and Top-16/Top-32 indices.",
        "- `topk_regions.csv`: Top-K mass and indices.",
        "- `f_log_top16_cross_scores.csv`: all 72 layer/signal/region/score combinations.",
        "- `f_log_top16_six_layer_means.csv`: the 12 six-layer means shown above.",
        "- `*_top16_regions_f_log.{png,pdf}`: explicit Top-16 masks with self/cross F_log in each panel.",
        "- `metadata.json`: exact source, token identities, crop box and formulas.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(args.features)
    rows = load_selected_rows(args.features)
    validate_rows(rows)
    arrays = build_raw_arrays(rows)
    topk_arrays, topk_rows = build_topk_artifacts(arrays)
    f_rows = f_log_cross_scores(arrays)
    f_summary = mean_summary(f_rows)

    np.savez_compressed(args.output_dir / "raw_matrices.npz", **arrays)
    np.savez_compressed(args.output_dir / "topk_masks_and_values.npz", **topk_arrays)
    write_csv(args.output_dir / "topk_regions.csv", topk_rows)
    write_csv(args.output_dir / "f_log_top16_cross_scores.csv", f_rows)
    write_csv(args.output_dir / "f_log_top16_six_layer_means.csv", f_summary)

    with args.image.open("rb") as handle:
        full_image = Image.open(handle).convert("RGB")
    model_image, crop_box = center_square_crop(full_image)
    full_image.save(args.output_dir / "input_full.png")
    model_image.save(args.output_dir / "model_view_center_crop.png")
    for token in TOKENS:
        plot_token(
            token=token,
            arrays=arrays,
            image=model_image,
            output=args.output_dir,
            dpi=args.dpi,
        )
        plot_top16_regions(
            region_token=token,
            arrays=arrays,
            image=model_image,
            output=args.output_dir,
            dpi=args.dpi,
        )

    generations = json.loads(args.generations.read_text(encoding="utf-8"))
    generation_row = generations[str(IMAGE_ID)]
    metadata = {
        "schema": "llava-831-log-s-hpre-topk-analysis-v1",
        "image_id": IMAGE_ID,
        "image_path": str(args.image.resolve()),
        "full_image_size": list(full_image.size),
        "model_view": "center-square crop",
        "model_view_crop_box_left_top_right_bottom": list(crop_box),
        "model_view_size": list(model_image.size),
        "model": "llava_1_5_7b",
        "source_features": str(args.features.resolve()),
        "source_features_sha256": source_hash,
        "source_schema": rows["skateboard"]["schema"],
        "generation": generation_row["generated_text"],
        "layers_1_based": list(LAYERS),
        "visual_grid": list(GRID),
        "top32": TOP32,
        "top16": TOP16,
        "tokens": {
            token: {
                "response_token_idx": int(rows[token]["response_token_idx"]),
                "target_token_id": int(rows[token]["target_token_id"]),
                "label": int(rows[token]["label"]),
                "canonical_object": str(rows[token]["canonical_object"]),
            }
            for token in TOKENS
        },
        "signals": {
            "raw_attention": "mean-head visual attention, normalized over 576 visual tokens",
            "target_distribution": "target_log_s = renorm(attention * log(1 + vocabulary_size * object_probability_norm_hpre))",
            "source_distribution": "source_hpre = softmax(cos(o_ffn_target, hpre_visual) / 0.07)",
        },
        "f_log": "sum_{m in region} am_norm[l,m] * (1 + log(1 + vocabulary_size * object_probability_norm_hpre[l,m]))",
        "topk_policy": "raw attention, target distribution and source distribution select patches independently for each token and layer",
        "heatmap_background": "RGB image blended 36% with pale blue [0.76, 0.88, 0.98] before turbo overlay",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        args.output_dir / "README.md",
        generation=generation_row["generated_text"],
        source_hash=source_hash,
        summary=f_summary,
    )

    expected = {
        "topk_rows": len(TOKENS) * len(SIGNALS) * len(LAYERS),
        "f_rows": len(SIGNALS) * len(LAYERS) * len(TOKENS) * len(TOKENS),
        "summary_rows": len(SIGNALS) * len(TOKENS) * len(TOKENS),
    }
    actual = {
        "topk_rows": len(topk_rows),
        "f_rows": len(f_rows),
        "summary_rows": len(f_summary),
    }
    if actual != expected:
        raise AssertionError(f"Unexpected output cardinality: {actual} != {expected}")
    print(json.dumps({"output": str(args.output_dir), **actual}, indent=2))


if __name__ == "__main__":
    main()
