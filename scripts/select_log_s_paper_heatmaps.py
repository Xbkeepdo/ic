#!/usr/bin/env python3
"""Rank and render COCO Log-S paper heatmap candidates for LLaVA, Qwen3, and InternVL.

The spatial matrices are saved at source tau=.07. Reweighting their positive
softmax probabilities to tau=.03 preserves the original cosine ranking.
Top-16 is selected independently for each signal, token, and decoder layer.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import pickle
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
GF = ROOT.parent / "gfchair/outputs"
SPATIAL = GF / "coco4000_512_linear_cost_log_s_risk_811"
LABELS = GF / "coco4000_512_endac_811"
IMAGES = Path("/home/apulis-dev/userdata/val2014")
INSTANCES = Path("/home/apulis-dev/userdata/annotations/instances_val2014.json")
DEFAULT_OUT = ROOT / "outputs/log_s_paper_heatmap_candidates"
LAYERS = (15, 18, 21, 24, 27)
LAYER_INDEX = np.asarray(LAYERS) - 1
TOP_K = 16
SMALL_CLASSES = {
    "apple", "baseball bat", "baseball glove", "bird", "book", "bottle",
    "bowl", "cell phone", "clock", "cup", "donut", "fork", "frisbee",
    "handbag", "kite", "knife", "laptop", "mouse", "orange", "remote", "scissors",
    "sandwich", "skateboard", "spoon", "sports ball", "tie", "toaster", "toothbrush",
    "traffic light", "umbrella", "vase", "wine glass",
}
PAIR_FALLBACK_CLASSES = {"cat"}
CMAP = LinearSegmentedColormap.from_list(
    "reference_blue_to_warm",
    [(0, "#152878"), (.20, "#233fff"), (.45, "#12bdf4"),
     (.70, "#91ed67"), (.88, "#fff04f"), (1, "#f15b3e")],
)
FONT_PATH = font_manager.findfont("DejaVu Sans")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "enrich", "render", "catalog"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=("llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b"))
    parser.add_argument("--shortlist", type=int, default=30)
    parser.add_argument("--render-list", type=Path, help="JSON list of {model, image_id, response_token_idx} records")
    parser.add_argument("--previews-only", action="store_true")
    return parser.parse_args()


def key(row: dict) -> tuple[int, int, int]:
    return int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"])


def top16(values: np.ndarray) -> np.ndarray:
    return np.argpartition(values, -TOP_K)[-TOP_K:]


def retemperature(source: np.ndarray) -> np.ndarray:
    log_p = np.log(np.maximum(np.asarray(source, dtype=np.float64), 1e-45))
    logits = (log_p - log_p.max(axis=1, keepdims=True)) * (0.07 / 0.03)
    weights = np.exp(logits)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def f_log_components(row: dict) -> np.ndarray:
    """Per-patch mainline F_log contributions at the five selected layers."""
    matrices = row["matrices"]
    p_norm = np.asarray(matrices["object_probability_norm_hpre"], dtype=np.float64)[LAYER_INDEX]
    write_norm = np.asarray(matrices["am_norm"], dtype=np.float64)[LAYER_INDEX]
    return (write_norm * (1 + np.log1p(int(row["vocabulary_size"]) * p_norm))).astype(np.float32)


def load_coco() -> tuple[dict[int, dict], dict[int, list[dict]], dict[str, int]]:
    data = json.loads(INSTANCES.read_text())
    images = {int(row["id"]): row for row in data["images"]}
    boxes: dict[int, list[dict]] = defaultdict(list)
    for row in data["annotations"]:
        boxes[int(row["image_id"])].append(row)
    categories = {row["name"].lower(): int(row["id"]) for row in data["categories"]}
    return images, boxes, categories


def coverage(
    *, image_id: int, category: str, grid: tuple[int, int], model: str,
    image_meta: dict[int, dict], boxes: dict[int, list[dict]], categories: dict[str, int],
) -> tuple[np.ndarray, float, int]:
    h, w = grid
    meta = image_meta[image_id]
    width, height = int(meta["width"]), int(meta["height"])
    if model == "llava_1_5_7b":
        side = min(width, height)
        ox, oy = (width - side) // 2, (height - side) // 2
        width = height = side
    else:
        ox = oy = 0
    x0 = (np.arange(w) * width / w)[None, :]
    x1 = ((np.arange(w) + 1) * width / w)[None, :]
    y0 = (np.arange(h) * height / h)[:, None]
    y1 = ((np.arange(h) + 1) * height / h)[:, None]
    cov = np.zeros((h, w), dtype=np.float32)
    selected = [row for row in boxes.get(image_id, ()) if row["category_id"] == categories.get(category)]
    areas = []
    for row in selected:
        bx, by, bw, bh = map(float, row["bbox"])
        bx -= ox
        by -= oy
        ax, ay = max(0, bx), max(0, by)
        cx, cy = min(width, bx + bw), min(height, by + bh)
        if cx <= ax or cy <= ay:
            continue
        areas.append((cx - ax) * (cy - ay) / (width * height))
        horizontal = np.maximum(0, np.minimum(x1, cx) - np.maximum(x0, ax))
        vertical = np.maximum(0, np.minimum(y1, cy) - np.maximum(y0, ay))
        cov += horizontal * vertical / ((width / w) * (height / h))
    return np.minimum(cov, 1).reshape(-1), (min(areas) if areas else math.nan), len(areas)


def snippet(text: str, span: dict, flank: int = 80) -> str:
    start, end = int(span["char_start"]), int(span["char_end"])
    left, right = max(0, start - flank), min(len(text), end + flank)
    return " ".join((
        ("…" if left else "") + text[left:start].replace("\n", " ").strip(),
        "[" + text[start:end].replace("\n", " ") + "]",
        text[end:right].replace("\n", " ").strip() + ("…" if right < len(text) else ""),
    )).strip()


def map_metrics(row: dict, cov: np.ndarray) -> dict:
    matrices = row["matrices"]
    attention = np.asarray(matrices["attention"], dtype=np.float32)[LAYER_INDEX]
    target = np.asarray(matrices["target_log_s"], dtype=np.float32)[LAYER_INDEX]
    source = retemperature(np.asarray(matrices["source_hpre"], dtype=np.float32)[LAYER_INDEX])
    if not all(np.allclose(arr.sum(axis=1), 1, atol=2e-5) for arr in (attention, target, source)):
        raise ValueError(f"Distributions are not normalized for {key(row)}")
    grid = tuple(map(int, row["visual_grid"]))
    if len(grid) != 2 or math.prod(grid) != attention.shape[1]:
        raise ValueError(f"Invalid grid {grid} for {key(row)}")
    layer_metrics = []
    for a, t, s in zip(attention, target, source):
        ai, ti, si = top16(a), top16(t), top16(s)
        local = {
            "attention_hit": float(np.dot(a[ai], cov[ai]) / a[ai].sum()),
            "target_hit": float(np.dot(t[ti], cov[ti]) / t[ti].sum()),
            "source_hit": float(np.dot(s[si], cov[si]) / s[si].sum()),
            "attention_mass_object": float(np.dot(a, cov)),
            "target_mass_object": float(np.dot(t, cov)),
            "source_mass_object": float(np.dot(s, cov)),
            "source_mass_target_top16": float(s[ti].sum()),
            "source_target_top16_overlap": int(np.intersect1d(si, ti).size),
            "target_top16_mass": float(t[ti].sum()),
            "source_top16_mass": float(s[si].sum()),
        }
        layer_metrics.append(local)
    return {
        "layers": layer_metrics,
        "mean_target_gain": float(np.mean([v["target_hit"] - v["attention_hit"] for v in layer_metrics])),
        "mean_target_hit": float(np.mean([v["target_hit"] for v in layer_metrics])),
        "mean_source_hit": float(np.mean([v["source_hit"] for v in layer_metrics])),
        "mean_source_outside_target": float(np.mean([1 - v["source_mass_target_top16"] for v in layer_metrics])),
        "mean_source_top16_mass": float(np.mean([v["source_top16_mass"] for v in layer_metrics])),
        "mean_target_top16_mass": float(np.mean([v["target_top16_mass"] for v in layer_metrics])),
        "positive_gain_layers": int(sum(v["target_hit"] > v["attention_hit"] + .08 for v in layer_metrics)),
        "overlap_mean": float(np.mean([v["source_target_top16_overlap"] for v in layer_metrics])),
    }


def score(summary: dict) -> float:
    small = summary["canonical_object"] in SMALL_CLASSES
    small_bonus = .18 if small else 0
    if summary["label"] == 1:
        area = summary["smallest_box_fraction"]
        area_bonus = .12 if math.isfinite(area) and area <= .04 else 0
        return (
            .40 * summary["mean_target_gain"]
            + .25 * summary["mean_target_hit"]
            + .30 * summary["mean_source_hit"]
            + .05 * summary["positive_gain_layers"] / 5
            + small_bonus + area_bonus
        )
    return (
        .38 * summary["mean_source_outside_target"]
        + .27 * (1 - summary["overlap_mean"] / TOP_K)
        + .18 * summary["mean_target_top16_mass"]
        + .17 * summary["mean_source_top16_mass"]
        + small_bonus
    )


def diverse(items: list[dict], limit: int) -> list[dict]:
    selected = []
    seen_images = set()
    for item in items:
        if item["image_id"] in seen_images:
            continue
        selected.append(item)
        seen_images.add(item["image_id"])
        if len(selected) >= limit:
            break
    return selected


def scan_model(model: str, output: Path, coco: tuple) -> None:
    image_meta, boxes, categories = coco
    labels = json.loads((LABELS / model / "labeling.json").read_text())
    with (SPATIAL / model / "features.pkl").open("rb") as handle:
        records = pickle.load(handle)
    summaries = []
    record_by_key = {}
    coverage_cache = {}
    for index, row in enumerate(records):
        sample_key = key(row)
        image_id, token_idx, _token_id = sample_key
        grid = tuple(map(int, row["visual_grid"]))
        label_row = labels[str(image_id)]
        span = next((s for s in label_row["object_token_spans"] if token_idx == int(s["token_indices"][0])), None)
        if span is None:
            raise ValueError(f"Missing exact label for {model} {sample_key}")
        category = str(row["canonical_object"]).lower()
        cov_key = (image_id, category, grid)
        if cov_key not in coverage_cache:
            coverage_cache[cov_key] = coverage(
                image_id=image_id, category=category, grid=grid, model=model,
                image_meta=image_meta, boxes=boxes, categories=categories,
            )
        cov, area, box_count = coverage_cache[cov_key]
        if int(row["label"]) == 1 and box_count == 0:
            continue
        metrics = map_metrics(row, cov)
        summary = {
            "model": model, "image_id": image_id, "response_token_idx": token_idx,
            "target_token_id": sample_key[2], "token": str(row["token"]),
            "canonical_object": category, "label": int(row["label"]),
            "split": row["split"], "visual_grid": list(grid),
            "smallest_box_fraction": area if math.isfinite(area) else None,
            "box_count": box_count,
            "caption_snippet": snippet(label_row["generated_text"], span),
            "caption_full": label_row["generated_text"],
            **metrics,
        }
        summary["score"] = score(summary)
        summaries.append(summary)
        record_by_key[sample_key] = index
    by_image = defaultdict(lambda: {0: [], 1: []})
    for summary in summaries:
        by_image[summary["image_id"]][summary["label"]].append(summary)
    pair_summaries = []
    for image_id, groups in by_image.items():
        if not groups[0] or not groups[1]:
            continue
        for hall in groups[0]:
            for real in groups[1]:
                pair_summaries.append({
                    "model": model, "image_id": image_id,
                    "hall_index": hall["response_token_idx"], "real_index": real["response_token_idx"],
                    "hall_token": hall["token"], "real_token": real["token"],
                    "small_pair": hall["canonical_object"] in SMALL_CLASSES and real["canonical_object"] in SMALL_CLASSES,
                    "split": hall["split"],
                    "score": .5 * (hall["score"] + real["score"]),
                })
    rankings = {
        "hall": sorted((s for s in summaries if s["label"] == 0), key=lambda s: s["score"], reverse=True),
        "real": sorted((s for s in summaries if s["label"] == 1), key=lambda s: s["score"], reverse=True),
        "pair": sorted(pair_summaries, key=lambda s: s["score"], reverse=True),
    }
    model_out = output / model
    model_out.mkdir(parents=True, exist_ok=True)
    for kind, items in rankings.items():
        (model_out / f"ranked_{kind}.json").write_text(json.dumps(items, ensure_ascii=False, indent=2, allow_nan=False))
        with (model_out / f"ranked_{kind}.csv").open("w", newline="") as handle:
            fields = (["model", "image_id", "response_token_idx", "target_token_id", "token", "canonical_object", "split", "score", "mean_target_gain", "mean_target_hit", "mean_source_hit", "mean_source_outside_target", "caption_snippet"] if kind != "pair" else ["model", "image_id", "hall_index", "real_index", "hall_token", "real_token", "split", "score"])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in items:
                writer.writerow({field: item.get(field, "") for field in fields})
    chosen = {}
    for kind, items in rankings.items():
        for item in diverse(items, args_global.shortlist):
            candidates = (item,) if kind != "pair" else (next(s for s in by_image[item["image_id"]][0] if s["response_token_idx"] == item["hall_index"]), next(s for s in by_image[item["image_id"]][1] if s["response_token_idx"] == item["real_index"]))
            for candidate in candidates:
                candidate_key = (candidate["image_id"], candidate["response_token_idx"], candidate["target_token_id"])
                source_row = records[record_by_key[candidate_key]]
                matrices = source_row["matrices"]
                chosen[candidate_key] = {
                    "summary": candidate,
                    "attention": np.asarray(matrices["attention"])[LAYER_INDEX],
                    "target": np.asarray(matrices["target_log_s"])[LAYER_INDEX],
                    "source": retemperature(np.asarray(matrices["source_hpre"])[LAYER_INDEX]),
                    "f_log_component": f_log_components(source_row),
                }
    with (model_out / "shortlist.pkl").open("wb") as handle:
        pickle.dump(chosen, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{model}: {len(summaries)} tokens, {len(pair_summaries)} pairs, {len(chosen)} cached shortlist tokens", flush=True)
    del records, summaries, chosen, coverage_cache
    gc.collect()


def model_view(image: Image.Image, model: str) -> Image.Image:
    image = image.convert("RGB")
    if model != "llava_1_5_7b":
        return image
    w, h = image.size
    side = min(w, h)
    return image.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))


def overlay(image: Image.Image, values: np.ndarray, grid: tuple[int, int]) -> Image.Image:
    indices = top16(values)
    retained = np.zeros_like(values, dtype=np.float32)
    retained[indices] = values[indices]
    scaled = retained.reshape(grid) / max(float(retained.max()), 1e-12)
    expanded = Image.fromarray(np.uint8(np.round(scaled * 255))).resize(
        image.size, Image.Resampling.BICUBIC
    )
    smoothed = expanded.filter(ImageFilter.GaussianBlur(radius=min(image.size) / 40))
    heat = np.clip(np.asarray(smoothed, dtype=np.float32) / 255, 0, 1)
    heat /= max(float(heat.max()), 1e-12)
    base = np.asarray(image, dtype=np.float32) / 255
    base = .44 * base + .56 * np.asarray([.04, .07, .25], dtype=np.float32)
    color = CMAP(heat)[..., :3]
    alpha = (.96 * np.power(heat, .63))[..., None]
    return Image.fromarray(np.uint8(np.round(np.clip(base * (1 - alpha) + color * alpha, 0, 1) * 255)))


def canvas_for_token(model: str, record: dict, full: bool) -> Image.Image:
    summary = record["summary"]
    image_path = IMAGES / f"COCO_val2014_{summary['image_id']:012d}.jpg"
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    view = model_view(original, model)
    grid = tuple(summary["visual_grid"])
    if full:
        panel_w, panel_h = 320, 320
        canvas = Image.new("RGB", (panel_w * 6, panel_h * 3 + 120), "white")
        draw = ImageDraw.Draw(canvas)
        title_font = ImageFont.truetype(FONT_PATH, 21)
        body_font = ImageFont.truetype(FONT_PATH, 16)
        draw.text((10, 6), f"{model} | COCO {summary['image_id']} | {summary['token']} | {'real' if summary['label'] else 'hallucinated'}", fill="black", font=title_font)
        caption = summary["caption_snippet"].replace("\n", " ")
        for line_number, line in enumerate(textwrap.wrap(caption, width=115)[:2]):
            draw.text((10, 39 + line_number * 22), line, fill="black", font=body_font)
        original_panel = original.copy()
        original_panel.thumbnail((panel_w - 8, panel_h - 8))
        canvas.paste(original_panel, ((panel_w - original_panel.width) // 2, 100 + (panel_h - original_panel.height) // 2))
        draw.text((8, 105), "Original image", fill="white", stroke_width=2, stroke_fill="black", font=body_font)
        notes = [
            f"Token: {summary['token']}",
            f"COCO class: {summary['canonical_object']}",
            f"Response index: {summary['response_token_idx']}",
            f"Token ID: {summary['target_token_id']}",
            f"Split: {summary['split']}",
            f"Grid: {grid[0]} x {grid[1]}",
            "Source temperature: 0.03",
            "Each panel: independent Top-16",
        ]
        for line_number, line in enumerate(notes):
            draw.text((10, 440 + line_number * 30), line, fill="#152878", font=body_font)
        footer = [
            "Rows:", "A = support attention", "T = Log-S target distribution",
            "P = FFN source distribution", "Columns: decoder layers", "15, 18, 21, 24, 27",
            "Heatmaps use model image view.",
        ]
        for line_number, line in enumerate(footer):
            draw.text((10, 780 + line_number * 30), line, fill="#152878", font=body_font)
        for r, name in enumerate(("attention", "target", "source")):
            for c, layer in enumerate(LAYERS, start=1):
                panel = overlay(view, record[name][c - 1], grid).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
                canvas.paste(panel, (c * panel_w, 100 + r * panel_h))
                row_label = {"attention": "A: support", "target": "T: target", "source": "P: source"}[name]
                draw.text((c * panel_w + 8, 100 + r * panel_h + 7), f"{row_label} | L{layer}", fill="white", stroke_width=2, stroke_fill="black", font=body_font)
        return canvas
    layer_idx = 3  # Layer 24 is a useful mid-depth preview.
    panel = 230
    canvas = Image.new("RGB", (panel * 4, panel + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"{summary['image_id']} {summary['token']} ({'R' if summary['label'] else 'H'}) score {summary['score']:.3f}", fill="black")
    for col, source in enumerate((original, *(overlay(view, record[name][layer_idx], grid) for name in ("attention", "target", "source")))):
        image = source.copy()
        image.thumbnail((panel, panel))
        canvas.paste(image, (col * panel + (panel - image.width) // 2, 28 + (panel - image.height) // 2))
    return canvas


def figure_stem(summary: dict) -> str:
    return f"{summary['image_id']:06d}_token{summary['response_token_idx']:03d}_{summary['canonical_object'].replace(' ', '_')}"


def enrich_model(model: str, output: Path, requested: list[dict]) -> None:
    model_out = output / model
    with (model_out / "shortlist.pkl").open("rb") as handle:
        cached = pickle.load(handle)
    desired = {
        (int(item["image_id"]), int(item["response_token_idx"]))
        for item in requested if item["model"] == model
    }
    present = {(key[0], key[1]) for key in cached}
    missing = desired - present
    missing_f = {
        (cache_key[0], cache_key[1]) for cache_key, value in cached.items()
        if (cache_key[0], cache_key[1]) in desired and "f_log_component" not in value
    }
    if not missing and not missing_f:
        print(f"{model}: all {len(desired)} requested tokens are cached", flush=True)
        return
    summaries = {}
    for kind in ("hall", "real"):
        for row in json.loads((model_out / f"ranked_{kind}.json").read_text()):
            identity = (int(row["image_id"]), int(row["response_token_idx"]))
            if identity in missing:
                summaries[identity] = row
    if set(summaries) != missing:
        raise ValueError(f"Unknown requested tokens: {missing - set(summaries)}")
    with (SPATIAL / model / "features.pkl").open("rb") as handle:
        records = pickle.load(handle)
    for row in records:
        identity = (int(row["image_id"]), int(row["response_token_idx"]))
        if identity not in missing and identity not in missing_f:
            continue
        matrices = row["matrices"]
        if identity in missing:
            cached[key(row)] = {
                "summary": summaries[identity],
                "attention": np.asarray(matrices["attention"])[LAYER_INDEX],
                "target": np.asarray(matrices["target_log_s"])[LAYER_INDEX],
                "source": retemperature(np.asarray(matrices["source_hpre"])[LAYER_INDEX]),
            }
        cached[key(row)]["f_log_component"] = f_log_components(row)
    if not missing.issubset({(key[0], key[1]) for key in cached}):
        raise ValueError(f"Missing requested records after reading {model} features")
    with (model_out / "shortlist.pkl").open("wb") as handle:
        pickle.dump(cached, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{model}: cached {len(missing)} additional tokens and {len(missing_f)} F_log records", flush=True)


def render_model(model: str, output: Path, render_list: list[dict] | None) -> None:
    model_out = output / model
    with (model_out / "shortlist.pkl").open("rb") as handle:
        cached = pickle.load(handle)
    by_index = {(k[0], k[1]): value for k, value in cached.items()}
    if render_list is not None:
        targets = [(int(row["image_id"]), int(row["response_token_idx"])) for row in render_list if row["model"] == model]
    else:
        targets = list(by_index)
    render_dir = model_out / "figures"
    render_dir.mkdir(exist_ok=True)
    if not args_global.previews_only:
        for image_id, token_idx in targets:
            record = by_index[(image_id, token_idx)]
            stem = figure_stem(record["summary"])
            png_path, pdf_path = render_dir / f"{stem}.png", render_dir / f"{stem}.pdf"
            if not (png_path.exists() and pdf_path.exists()):
                figure = canvas_for_token(model, record, full=True)
                figure.save(png_path)
                figure.save(pdf_path, resolution=160)
            preview_path = render_dir / f"{stem}_preview.jpg"
            if not preview_path.exists():
                canvas_for_token(model, record, full=False).save(preview_path, quality=91)
        by_image = defaultdict(lambda: {0: [], 1: []})
        selected_groups = {
            (int(row["image_id"]), int(row["response_token_idx"])): row.get("group")
            for row in (render_list or []) if row["model"] == model
        }
        for image_id, token_idx in targets:
            group = selected_groups.get((image_id, token_idx))
            if group in ("pair-real", "pair-hall"):
                by_image[image_id][int(group == "pair-real")].append(token_idx)
        pair_dir = model_out / "pairs"
        pair_dir.mkdir(exist_ok=True)
        for image_id, groups in by_image.items():
            if not groups[0] or not groups[1]:
                continue
            real_idx = max(groups[1], key=lambda i: by_index[(image_id, i)]["summary"]["mean_source_hit"])
            hall_idx = max(groups[0], key=lambda i: by_index[(image_id, i)]["summary"]["mean_source_outside_target"])
            stem = f"{image_id:06d}_real{real_idx:03d}_hall{hall_idx:03d}"
            png_path, pdf_path = pair_dir / f"{stem}.png", pair_dir / f"{stem}.pdf"
            if png_path.exists() and pdf_path.exists():
                continue
            real_path = render_dir / f"{figure_stem(by_index[(image_id, real_idx)]['summary'])}.png"
            hall_path = render_dir / f"{figure_stem(by_index[(image_id, hall_idx)]['summary'])}.png"
            with Image.open(real_path) as real_image, Image.open(hall_path) as hall_image:
                pair_image = Image.new("RGB", (real_image.width, real_image.height + hall_image.height + 30), "#d6e5fa")
                pair_image.paste(real_image, (0, 0))
                pair_image.paste(hall_image, (0, real_image.height + 30))
                pair_image.save(png_path)
                pair_image.save(pdf_path, resolution=160)
    if targets:
        requested_thumbs = [canvas_for_token(model, by_index[identity], full=False) for identity in targets]
        prefix = f"preview_{args_global.render_list.stem}" if args_global.render_list else "preview_requested"
        for page in range(0, len(requested_thumbs), 4):
            sheet = Image.new("RGB", (920, 258 * min(4, len(requested_thumbs) - page)), "white")
            for row, thumb in enumerate(requested_thumbs[page:page + 4]):
                sheet.paste(thumb, (0, 258 * row))
            sheet.save(model_out / f"{prefix}_{page // 4 + 1:02d}.jpg", quality=87)
    for kind in ("hall", "real", "pair"):
        ranking = json.loads((model_out / f"ranked_{kind}.json").read_text())
        if kind == "pair":
            identities = [
                (int(item["image_id"]), int(index))
                for item in diverse(ranking, args_global.shortlist)
                for index in (item["real_index"], item["hall_index"])
            ]
        else:
            identities = [(int(item["image_id"]), int(item["response_token_idx"])) for item in diverse(ranking, args_global.shortlist)]
        thumbs = [canvas_for_token(model, by_index[identity], full=False) for identity in identities if identity in by_index]
        for page in range(0, len(thumbs), 4):
            sheet = Image.new("RGB", (920, 258 * min(4, len(thumbs) - page)), "white")
            for row, thumb in enumerate(thumbs[page:page + 4]):
                sheet.paste(thumb, (0, 258 * row))
            sheet.save(model_out / f"preview_{kind}_{page // 4 + 1:02d}.jpg", quality=87)
    print(f"{model}: {'previewed' if args_global.previews_only else 'rendered'} {len(targets)} requested tokens", flush=True)


def build_catalog(output: Path, requested: list[dict]) -> None:
    notes_path = output / "selection_notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    caches = {}
    for model in {item["model"] for item in requested}:
        with (output / model / "shortlist.pkl").open("rb") as handle:
            cached = pickle.load(handle)
        caches[model] = {(k[0], k[1]): value for k, value in cached.items()}
    manifest, layer_rows, patch_rows = [], [], []
    for item in requested:
        model, image_id, token_idx = item["model"], int(item["image_id"]), int(item["response_token_idx"])
        record = caches[model][(image_id, token_idx)]
        summary = record["summary"]
        if "f_log_component" not in record:
            raise ValueError(f"Missing F_log for {model} {image_id} {token_idx}; run enrich")
        identity = f"{model}:{image_id}:{token_idx}"
        stem = figure_stem(summary)
        relative = f"{model}/figures/{stem}"
        note = notes.get(identity, "")
        manifest.append({
            "model": model, "image_id": image_id, "response_token_idx": token_idx,
            "target_token_id": summary["target_token_id"], "token": summary["token"],
            "canonical_object": summary["canonical_object"], "label": summary["label"],
            "group": item["group"], "tier": item["tier"], "split": summary["split"],
            "caption_snippet": summary["caption_snippet"], "caption_full": summary["caption_full"],
            "manual_note": note, "mean_target_gain": summary["mean_target_gain"],
            "mean_target_hit": summary["mean_target_hit"], "mean_source_hit": summary["mean_source_hit"],
            "mean_source_outside_target": summary["mean_source_outside_target"],
            "mean_target_top16_mass": summary["mean_target_top16_mass"],
            "mean_source_top16_mass": summary["mean_source_top16_mass"],
            "figure_png": relative + ".png", "figure_pdf": relative + ".pdf",
            "preview_jpg": relative + "_preview.jpg",
        })
        for layer_idx, layer in enumerate(LAYERS):
            f_component = record["f_log_component"][layer_idx]
            f_total = float(f_component.sum())
            metrics = summary["layers"][layer_idx]
            layer_row = {
                "model": model, "image_id": image_id, "response_token_idx": token_idx,
                "group": item["group"], "layer": layer, "f_log_full_visual": f_total,
                **metrics,
            }
            for signal in ("attention", "target", "source"):
                values = record[signal][layer_idx]
                indices = sorted(top16(values), key=lambda index: -values[index])
                layer_row[f"f_log_{signal}_top16"] = float(f_component[indices].sum())
                grid_width = int(summary["visual_grid"][1])
                for rank, index in enumerate(indices, 1):
                    patch_rows.append({
                        "model": model, "image_id": image_id,
                        "response_token_idx": token_idx, "layer": layer,
                        "signal": signal, "rank": rank, "patch_index": int(index),
                        "grid_row": int(index) // grid_width, "grid_col": int(index) % grid_width,
                        "probability": float(values[index]), "f_log_component": float(f_component[index]),
                    })
            layer_rows.append(layer_row)

    def write_rows(name: str, rows: list[dict]) -> None:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    by_token_layers = defaultdict(list)
    for row in layer_rows:
        by_token_layers[(row["model"], row["image_id"], row["response_token_idx"])].append(row)
    for row in manifest:
        layers = by_token_layers[(row["model"], row["image_id"], row["response_token_idx"])]
        row["mean_f_log_target_top16_fraction"] = float(np.mean([
            layer["f_log_target_top16"] / layer["f_log_full_visual"] for layer in layers
        ]))

    write_rows("selected_manifest.csv", manifest)
    write_rows("selected_layer_metrics.csv", layer_rows)
    write_rows("selected_top16_patches.csv", patch_rows)
    pair_groups = defaultdict(lambda: {0: [], 1: []})
    for row in manifest:
        if row["group"] not in ("pair-real", "pair-hall"):
            continue
        expected_label = int(row["group"] == "pair-real")
        if row["label"] != expected_label or row["canonical_object"] not in SMALL_CLASSES | PAIR_FALLBACK_CLASSES:
            raise ValueError(f"Invalid pair token: {row['model']} {row['image_id']} {row['token']}")
        pair_groups[(row["model"], row["image_id"])][expected_label].append(row)
    pair_manifest = []
    for (model, image_id), groups in pair_groups.items():
        if not groups[0] or not groups[1]:
            continue
        real = max(groups[1], key=lambda row: row["mean_source_hit"])
        hall = max(groups[0], key=lambda row: row["mean_source_outside_target"])
        stem = f"{image_id:06d}_real{real['response_token_idx']:03d}_hall{hall['response_token_idx']:03d}"
        pair_manifest.append({
            "model": model, "image_id": image_id,
            "tier": "priority" if real["tier"] == hall["tier"] == "priority" else "review",
            "real_token": real["token"], "real_response_token_idx": real["response_token_idx"],
            "real_caption_snippet": real["caption_snippet"],
            "hall_token": hall["token"], "hall_response_token_idx": hall["response_token_idx"],
            "hall_caption_snippet": hall["caption_snippet"],
            "pair_png": f"{model}/pairs/{stem}.png", "pair_pdf": f"{model}/pairs/{stem}.pdf",
        })
    pair_manifest.sort(key=lambda row: (row["tier"] != "priority", row["model"] != "llava_1_5_7b", row["image_id"]))
    write_rows("selected_pairs.csv", pair_manifest)
    cards = []
    for row in manifest:
        title = f"{row['model']} · {row['image_id']} · {row['token']} ({row['group']})"
        metrics_text = (
            f'T gain {row["mean_target_gain"]:+.2f} · P hit {row["mean_source_hit"]:.2f}'
            if row["label"] == 1 else
            f'P outside T16 {row["mean_source_outside_target"]:.2f} · T16 mass {row["mean_target_top16_mass"]:.2f}'
        )
        cards.append(
            f'<article data-model="{html.escape(row["model"])}" data-group="{html.escape(row["group"])}" data-tier="{html.escape(row["tier"])}">'
            f'<a href="{html.escape(row["figure_png"])}"><img src="{html.escape(row["preview_jpg"])}" alt="{html.escape(title)}"></a>'
            f'<h3>{html.escape(title)}</h3><p>{html.escape(row["caption_snippet"])}</p>'
            f'<p>{html.escape(row["manual_note"])}</p>'
            f'<small>{row["tier"]} · token #{row["response_token_idx"]} / ID {row["target_token_id"]} · '
            f'{metrics_text} · '
            f'<a href="{html.escape(row["figure_png"])}">PNG</a> · <a href="{html.escape(row["figure_pdf"])}">PDF</a></small></article>'
        )
    pair_links = "".join(
        f'<li>{html.escape(row["tier"])} · {html.escape(row["model"])} · COCO {row["image_id"]}: '
        f'{html.escape(row["real_token"])} / {html.escape(row["hall_token"])} '
        f'<a href="{html.escape(row["pair_png"])}">PNG</a> · '
        f'<a href="{html.escape(row["pair_pdf"])}">PDF</a></li>'
        for row in pair_manifest
    )
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Log-S/F_log paper heatmap candidates</title>
<style>body{font:16px/1.45 system-ui;margin:2rem;background:#f6f8fc;color:#142044}h1{margin-bottom:.2em}
nav{position:sticky;top:0;background:#f6f8fc;padding:.8rem 0}button{margin:.2rem;padding:.5rem 1rem;cursor:pointer}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:1rem}article{background:white;border:1px solid #d8e0ed;padding:1rem;border-radius:10px}
article img{width:100%;height:auto}article h3{margin:.3rem 0}article p{margin:.3rem 0}small{color:#405078}</style>
<h1>Log-S/F_log 热力图候选</h1><p>缩略图顺序：原图、A、T、P（第 24 层）。点击看五层完整图；priority 为优先，review 需再筛选。</p>
<p><a href="appendix_pairs/README.md">已选的八组附录成图（PNG / PDF / SVG）</a></p>
<p><a href="appendix_single/README.md">13 张单 token 附录图（PNG / PDF / SVG）</a></p>
<p><a href="expansion_2026-09-25.md">Qwen3 新配对与 InternVL 新候选：逐例复核记录</a></p>
<details><summary>同图合并图（真实 token / 幻觉 token）</summary><ul>""" + pair_links + """</ul></details>
<nav><button onclick="setFilter('all')">全部</button><button onclick="setFilter('priority')">优先</button>
<button onclick="setFilter('llava_1_5_7b')">LLaVA</button><button onclick="setFilter('qwen3_vl_8b')">Qwen3</button><button onclick="setFilter('internvl_2_5_8b')">InternVL</button>
<button onclick="setFilter('real')">真实</button><button onclick="setFilter('hall')">幻觉</button><button onclick="setFilter('pair')">同图成对</button></nav>
<main>""" + "\n".join(cards) + """</main><script>function setFilter(f){for(const e of document.querySelectorAll('article'))
e.style.display=(f==='all'||e.dataset.tier===f||e.dataset.model===f||e.dataset.group===f||
(f==='pair'&&e.dataset.group.startsWith('pair-')))?'':'none'}</script></html>"""
    (output / "gallery.html").write_text(page)
    md = [
        "# Log-S / F_log 论文热力图候选", "",
        f"共 {len(manifest)} 个 object token、{len({(r['model'], r['image_id']) for r in manifest})} 张图；"
        "[筛选图库](gallery.html)可按模型、类别和优先级查看；"
        "[新增候选复核记录](expansion_2026-09-25.md)说明 Qwen3 与 InternVL 的取舍；"
        "[八组配对附录](appendix_pairs/README.md)和[13 张单 token 附录](appendix_single/README.md)另有 PNG、PDF、SVG。每张完整图提供 [PNG] 和 [PDF]，左侧先给原图，"
        "右侧按 A（support attention）、T（Log-S target）、P（FFN source）三行显示第 15/18/21/24/27 层。", "",
        "候选来自三模型已有的 COCO4000-512 generation、Log-S 空间矩阵、CHAIR token 标注及 COCO instance 框；"
        "不重新生成 caption。先用框覆盖率和分布指标初筛，再核对原图与 caption 语义。优先挑小物体，"
        "没有足够清晰的小物体时保留大物体案例。", "",
        "## 画法与数值", "",
        "- 每个 token、层、分布独立选概率最高的 16 个视觉 patch；只对显示热斑做插值和模糊，不改变 Top-16 区域或分数。"
        "背景是原图叠加冷蓝色，热斑由蓝到青、黄、橙。每格颜色按该格最大值缩放，因此颜色深浅不能跨格当作概率值比较。",
        "- LLaVA 热图映射到模型的中央正方形裁剪，Qwen3 使用记录中的动态视觉网格，InternVL 的 16×16 网格按整图缩放的逆映射回原图；完整原图均单独显示。",
        "- 主线用 Log-S target、hpre source 的 tau=0.03、transport K=32。图上的 Top-16 只用于可视化，"
        "并非 transport K。F_log 是视觉位置的 write norm × [1 + log(1 + V × p_norm)] 之和，不把它误标为 T/P 热图。",
        "- `T gain` 是五层 T 相对 A 的 Top-16 物体框加权命中率差；`P hit` 是 P 的同一命中率。"
        "它们是候选排序线索，COCO 框并不能替代原图目视核对。", "",
        "- 幻觉 token 没有同类物体框，因此报告 `P outside T16`（P 在 T 的 Top-16 patch 之外的概率质量）和 `T16 mass`。"
        "前者不等于 P 在‘物体之外’的概率，尤其当 P 本身很分散时应结合热图看。", "",
        "数据表：[完整 caption 与 token](selected_manifest.csv)、"
        "[逐层 A/T/P 与 F_log 指标](selected_layer_metrics.csv)、"
        "[逐层独立 Top-16 patch](selected_top16_patches.csv)、"
        "[同图配对索引](selected_pairs.csv)。", "",
        "## 同图上下对照", "",
        "每张对照图上半是真实 token，下半是幻觉 token；各部分仍含原图、A/T/P 五层。"
        "配对优先小物体类 token；仅在小物体配对不足时保留猫的备选，不包含人物词。", "",
    ]
    for pair in pair_manifest:
        md += [
            f"- **{pair['tier']} · {pair['model']} · COCO {pair['image_id']}**："
            f"真实 `{pair['real_token']}`（index {pair['real_response_token_idx']}）/ "
            f"幻觉 `{pair['hall_token']}`（index {pair['hall_response_token_idx']}）；"
            f"[PNG]({pair['pair_png']}) / [PDF]({pair['pair_pdf']})。", "",
        ]
    md += [
        "## 可直接改写的英文图注", "",
        "1. **Hallucinated object.** `For the hallucinated object token, support attention and the Log-S target may peak near a visually plausible but incorrect region, while the FFN source distribution assigns most of its mass elsewhere.`",
        "2. **Grounded object.** `For a grounded object token, the Log-S target highlights more patches on the annotated object than support attention, and the FFN source largely follows the object.`",
        "3. **Same image pair.** `Within the same image, the grounded token has overlapping target and source regions on the visible object; the hallucinated token retains a focused target but its source shifts to other regions.`",
        "这些句子须按最终所选图和层数删改，不能笼统声称每一层都满足。", "",
        "## 候选", "",
    ]
    for tier, title in (("priority", "优先看"), ("review", "更多备选与谨慎案例")):
        md += [f"### {title}", ""]
        for model in ("llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b"):
            relevant = [row for row in manifest if row["tier"] == tier and row["model"] == model]
            if not relevant:
                continue
            md += [f"#### {model}", ""]
            for row in relevant:
                caption = (row["caption_snippet"].replace("|", "\\|").replace("`", "'")
                           .replace("*", "\\*").replace("[", "\\[").replace("]", "\\]"))
                note = row["manual_note"] or "需逐层目视复核。"
                metric_description = (
                    f"T gain `{row['mean_target_gain']:+.2f}`，T hit `{row['mean_target_hit']:.2f}`，"
                    f"P hit `{row['mean_source_hit']:.2f}`"
                    if row["label"] == 1 else
                    f"P outside T16 `{row['mean_source_outside_target']:.2f}`，"
                    f"T16 mass `{row['mean_target_top16_mass']:.2f}`，"
                    f"P16 mass `{row['mean_source_top16_mass']:.2f}`"
                )
                md += [
                    f"- **{row['group']} · COCO {row['image_id']} · {row['token']}** "
                    f"([PNG]({row['figure_png']}) / [PDF]({row['figure_pdf']})): "
                    f"token index `{row['response_token_idx']}`, ID `{row['target_token_id']}`, "
                    f"class `{row['canonical_object']}`, split `{row['split']}`；"
                    f"{metric_description}，"
                    f"T Top-16 的 F_log 占比 `{row['mean_f_log_target_top16_fraction']:.2f}`。"
                    f"{note} Caption 片段：「{caption}」", "",
                ]
    md += [
        "## 排除与限制", "",
        "- Qwen3 COCO 81390 的 `bird` 在原图树枝上真实存在；已排除。颜色词 `orange`、"
        "书名 `Kite Runner`、`ball of yarn` 等自动 object-token 映射误例也不作为幻觉图。",
        "- 新增 Qwen3 配对均列为 `review`：暂未发现兼具明确缺失物体、真实 token 五层 T/P 稳定命中、且没有语义歧义的新优先案例。",
        "- `review` 包含目标语义或可见性有歧义、或者 P 仍较分散的案例；优先选 `priority`，"
        "尤其不要把带‘谨慎’说明的案例直接写成确定幻觉。",
        "- 所有图是生成 caption 的单图解释示例。命中率与 F_log 贡献不是检测 AUROC，不能由这些样例推出整体改进。",
        "", "## 复现", "",
        "从项目根目录运行；本机 `/opt/conda/bin/python` 缺少 `matplotlib`，"
        "本次用 `/opt/conda/private/envs/glsim/bin/python` 完成生成：", "",
        "```bash", "PY=/opt/conda/private/envs/glsim/bin/python",
        "$PY scripts/select_log_s_paper_heatmaps.py scan",
        "$PY scripts/select_log_s_paper_heatmaps.py enrich --render-list outputs/log_s_paper_heatmap_candidates/curated_candidates.json",
        "$PY scripts/select_log_s_paper_heatmaps.py render --render-list outputs/log_s_paper_heatmap_candidates/curated_candidates.json",
        "$PY scripts/select_log_s_paper_heatmaps.py catalog --render-list outputs/log_s_paper_heatmap_candidates/curated_candidates.json",
        "```", "",
    ]
    (output / "README.md").write_text("\n".join(md) + "\n")
    print(f"Catalog: {len(manifest)} tokens, {len(layer_rows)} layer rows, {len(patch_rows)} Top-16 patches", flush=True)


if __name__ == "__main__":
    args_global = args()
    args_global.output.mkdir(parents=True, exist_ok=True)
    if args_global.mode == "scan":
        coco_data = load_coco()
        for model_name in args_global.models:
            scan_model(model_name, args_global.output, coco_data)
    elif args_global.mode == "enrich":
        if args_global.render_list is None:
            raise ValueError("enrich requires --render-list")
        requested = json.loads(args_global.render_list.read_text())
        for model_name in args_global.models:
            enrich_model(model_name, args_global.output, requested)
    elif args_global.mode == "render":
        selected = json.loads(args_global.render_list.read_text()) if args_global.render_list else None
        for model_name in args_global.models:
            render_model(model_name, args_global.output, selected)
    else:
        if args_global.render_list is None:
            raise ValueError("catalog requires --render-list")
        build_catalog(args_global.output, json.loads(args_global.render_list.read_text()))
