#!/usr/bin/env python3
"""Render selected Log-S token pairs as appendix-ready PNG/PDF/SVG figures.

The displayed caption sentences are exact excerpts of the cached model output.
Each heatmap reuses the paper-candidate renderer's independent Top-16 overlay.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont, ImageOps

from select_log_s_paper_heatmaps import IMAGES, LAYERS, model_view, overlay


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/log_s_paper_heatmap_candidates"
OUTPUT = SOURCE / "appendix_pairs"
DPI = 300
INK = "#151515"
MUTED = "#555555"
EDGE = "#d8d8d8"
GREEN = "#008043"
RED = "#c3232b"

CASES = (
    {
        "model": "llava_1_5_7b", "image_id": 7320, "real_idx": 26, "hall_idx": 69,
        "real_quote": "The person is holding a Wii remote, likely playing the game.",
        "hall_quote": "In addition to the TV, there is a computer mouse on a surface in the room.",
        "segments": (
            "The image features a person sitting in front of a TV, which is displaying a video game.",
            "The person is holding a Wii remote, likely playing the game.",
            "In addition to the TV, there is a computer mouse on a surface in the room.",
        ),
        "review_note": "The visible Wii remote is grounded; no computer mouse is apparent in the photo.",
    },
    {
        "model": "llava_1_5_7b", "image_id": 287741, "real_idx": 8, "hall_idx": 53,
        "real_quote": "The image features a small green and yellow bird perched on a person's hand.",
        "hall_quote": "The scene takes place in a room with a keyboard and a computer mouse nearby, indicating that the person might be working or using a computer.",
        "segments": (
            "The image features a small green and yellow bird perched on a person's hand.",
            "The person is holding the bird, providing a close-up view of the bird's head and body.",
            "The scene takes place in a room with a keyboard and a computer mouse nearby, indicating that the person might be working or using a computer.",
        ),
        "review_note": "The bird is visible; a keyboard is visible, but a computer mouse is not apparent.",
    },
    {
        "model": "llava_1_5_7b", "image_id": 291245, "real_idx": 6, "hall_idx": 78,
        "real_quote": "The image features a white computer mouse sitting on a brown tiled floor.",
        "hall_quote": "A cell phone can also be seen on the right side of the image, slightly above the keyboard.",
        "segments": (
            "The image features a white computer mouse sitting on a brown tiled floor.",
            "The mouse is positioned towards the center of the scene, with its cord extending to the left side.",
            "In addition to the mouse, there are two keyboards in the scene.",
            "A cell phone can also be seen on the right side of the image, slightly above the keyboard.",
        ),
        "review_note": "The white mouse is visible; no cell phone is apparent beside the keyboards.",
    },
    {
        "model": "qwen3_vl_8b", "image_id": 407717, "real_idx": 30, "hall_idx": 18,
        "real_quote": "The image is dominated by vibrant, glossy oranges and pale yellow bananas.",
        "hall_quote": "This is a close-up, slightly angled photograph of a fruit arrangement, likely in a fruit bowl or bag.",
        "segments": (
            "This is a close-up, slightly angled photograph of a fruit arrangement, likely in a fruit bowl or bag.",
            "The image is dominated by vibrant, glossy oranges and pale yellow bananas.",
            "Several oranges are visible, filling most of the frame.",
            "They have a bright, textured peel with a slight sheen, suggesting they are fresh.",
        ),
        "review_note": "The caption qualifies “bowl” with “likely”; the container is not visible.",
    },
    {
        "model": "qwen3_vl_8b", "image_id": 553698, "real_idx": 107, "hall_idx": 267,
        "real_quote": "The cat is sitting upright, looking regal and composed, as if it owns the entire book pile.",
        "hall_quote": "To the right, more books are stacked on a shelf or table, and a black laptop is also visible among them.",
        "segments": (
            "This is a charming and slightly whimsical street photograph capturing a fluffy, long-haired tabby cat perched atop a stack of books.",
            "The cat is sitting upright, looking regal and composed, as if it owns the entire book pile.",
            "To the right, more books are stacked on a shelf or table, and a black laptop is also visible among them.",
        ),
        "review_note": "The dark object at far right is visually ambiguous; review the laptop label before publication.",
    },
    {
        "model": "internvl_2_5_8b", "image_id": 17409, "real_idx": 27, "hall_idx": 319,
        "real_quote": "The primary focus of the image is a bottle of champagne and a champagne flute.",
        "hall_quote": "Additionally, there is a small dish with a spoon in it, which might contain butter or another spread to be served with the bread.",
        "segments": (
            "The primary focus of the image is a bottle of champagne and a champagne flute.",
            "The bottle is positioned to the right side of the image, and it is labeled \"Champagne\" and \"Brut Rosé.\"",
            "Additionally, there is a small dish with a spoon in it, which might contain butter or another spread to be served with the bread.",
        ),
        "review_note": "The champagne bottle is visible; no spoon is apparent in the small dish. The real-token FFN preference is partly dispersed.",
    },
    {
        "model": "internvl_2_5_8b", "image_id": 155732, "real_idx": 126, "hall_idx": 195,
        "real_quote": "A silver spoon is placed inside the bowl, resting on the surface of the crème brûlée.",
        "hall_quote": "The background also includes a fork and a knife, indicating that the meal is in progress or has recently been served.",
        "segments": (
            "The bowl itself is white with a decorative green and blue pattern around the rim.",
            "A silver spoon is placed inside the bowl, resting on the surface of the crème brûlée.",
            "The background also includes a fork and a knife, indicating that the meal is in progress or has recently been served.",
        ),
        "review_note": "The spoon is visible; no knife is apparent in the blurred background.",
    },
    {
        "model": "qwen3_vl_8b", "image_id": 79070, "real_idx": 246, "hall_idx": 94,
        "real_quote": "A skateboard is propped up against the bench beside him, with its wheels and deck clearly visible.",
        "hall_quote": "He is actively eating a large, round, dark-colored food item—possibly a donut or a burger—holding it with his right hand and bringing it to his mouth.",
        "segments": (
            "He is actively eating a large, round, dark-colored food item—possibly a donut or a burger—holding it with his right hand and bringing it to his mouth.",
            "A skateboard is propped up against the bench beside him, with its wheels and deck clearly visible.",
        ),
        "review_note": "The skateboard is visible. The food at his mouth cannot be identified with certainty; the caption itself says 'possibly a donut or a burger'.",
    },
)


def add_rect(fig, x, y, w, h, *, face="white", edge=EDGE, radius=.10, lw=.8):
    fig_w, fig_h = fig.get_size_inches()
    patch = FancyBboxPatch(
        (x / fig_w, y / fig_h), w / fig_w, h / fig_h,
        boxstyle=f"round,pad=0,rounding_size={radius / fig_h}",
        transform=fig.transFigure, linewidth=lw, edgecolor=edge,
        facecolor=face, zorder=0,
    )
    fig.add_artist(patch)


def put_text(fig, x, y, value, **kwargs):
    fig_w, fig_h = fig.get_size_inches()
    fig.text(x / fig_w, y / fig_h, value, transform=fig.transFigure, **kwargs)


def put_image(fig, image: Image.Image, x, y, w, h, *, fill="white"):
    fig_w, fig_h = fig.get_size_inches()
    box = (round(w * DPI), round(h * DPI))
    framed = ImageOps.pad(image.convert("RGB"), box, method=Image.Resampling.LANCZOS, color=fill)
    ax = fig.add_axes((x / fig_w, y / fig_h, w / fig_w, h / fig_h), zorder=2)
    ax.imshow(framed, interpolation="none")
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def prepare_caption(caption: str, case: dict, real_token: str,
                    hall_token: str, content_width: float) -> list[list[tuple[str, str, float]]]:
    """Keep selected sentences in their original order and wrap the colored words."""
    positions = [(caption.index(segment), segment) for segment in case["segments"]]
    if positions != sorted(positions):
        raise ValueError("Caption segments must follow the original response order")
    pieces = []
    previous_end = None
    for start, segment in positions:
        if previous_end is not None:
            pieces.append(" " if not caption[previous_end:start].strip() else " … ")
        pieces.append(segment)
        previous_end = start + len(segment)
    content = "".join(pieces)
    marks = [
        (match.start(), match.end(), color)
        for token, color in ((real_token, GREEN), (hall_token, RED))
        for match in re.finditer(rf"\b{re.escape(token)}\b", content, flags=re.IGNORECASE)
    ]
    if not any(color == GREEN for _, _, color in marks) or not any(color == RED for _, _, color in marks):
        raise ValueError("Selected caption is missing a colored pair token")
    font_size = 11.5
    measured_font = ImageFont.truetype(findfont("DejaVu Serif"), round(font_size * DPI / 72))
    space_width = measured_font.getlength(" ") / DPI
    lines: list[list[tuple[str, str, float]]] = [[]]
    line_width = 0.0
    phrases = [token for token in (real_token, hall_token) if " " in token]
    word_pattern = ("|".join(rf"\b{re.escape(token)}\b" for token in phrases) + r"|\S+"
                    if phrases else r"\S+")
    for match in re.finditer(word_pattern, content, flags=re.IGNORECASE):
        word = match.group()
        width = measured_font.getlength(word) / DPI
        if lines[-1] and line_width + width > content_width:
            lines.append([])
            line_width = 0.0
        color = next((shade for start, end, shade in marks
                      if start <= match.start() < end), INK)
        lines[-1].append((word, color, width))
        line_width += width + space_width
    return lines


def draw_caption(fig, lines: list[list[tuple[str, str, float]]], x: float, top: float):
    font_size = 11.5
    measured_font = ImageFont.truetype(findfont("DejaVu Serif"), round(font_size * DPI / 72))
    space_width = measured_font.getlength(" ") / DPI
    for row, line in enumerate(lines):
        cursor = x
        for word, color, width in line:
            put_text(fig, cursor, top - row * .235, word, color=color,
                     fontsize=font_size, fontfamily="DejaVu Serif", va="top", zorder=3)
            cursor += width + space_width


def draw_block(fig, top: float, label: str, token: str, color: str,
               record: dict, view: Image.Image, x0: float, tile_w: float,
               tile_h: float, gap: float):
    fig_w, fig_h = fig.get_size_inches()
    fig.add_artist(Rectangle(
        (.40 / fig_w, (top - .012) / fig_h), (fig_w - .80) / fig_w, .012 / fig_h,
        transform=fig.transFigure, facecolor="#c5c5c5", edgecolor="none", zorder=2,
    ))
    put_text(fig, .43, top - .19, f"{label}: {token}", color=color, fontsize=13.0,
             weight="bold", va="center", zorder=3)

    names = (
        ("attention", "Raw\nattention"),
        ("target", "Visual\nevidence"),
        ("source", "FFN\npreference"),
    )
    grid = tuple(record["summary"]["visual_grid"])
    for row, (signal, row_label) in enumerate(names):
        y = top - .35 - tile_h - row * (tile_h + .06)
        put_text(fig, 1.02, y + tile_h / 2, row_label, ha="center", va="center",
                 fontsize=11.0, color=INK, linespacing=1.13, zorder=3)
        for col in range(len(LAYERS)):
            x = x0 + col * (tile_w + gap)
            heat = overlay(view, record[signal][col], grid)
            put_image(fig, heat, x, y, tile_w, tile_h)


def draw_layer_row(fig, y: float, x0: float, tile_w: float, gap: float):
    put_text(fig, 1.02, y, "Layer", fontsize=11.0, color=INK,
             ha="center", va="center")
    for col, layer in enumerate(LAYERS):
        x = x0 + col * (tile_w + gap)
        put_text(fig, x + tile_w / 2, y, str(layer), ha="center", va="center",
                 fontsize=11.0, color=INK)


def render(case: dict, caches: dict, formats: tuple[str, ...], force: bool = False) -> dict:
    model, image_id = case["model"], case["image_id"]
    real = caches[model][(image_id, case["real_idx"])]
    hall = caches[model][(image_id, case["hall_idx"])]
    real_summary, hall_summary = real["summary"], hall["summary"]
    caption = real_summary["caption_full"]
    if caption != hall_summary["caption_full"]:
        raise ValueError(f"Caption mismatch for {model} COCO {image_id}")
    if real_summary["label"] != 1 or hall_summary["label"] != 0:
        raise ValueError(f"Pair label mismatch for {model} COCO {image_id}")
    for quote in (case["real_quote"], case["hall_quote"]):
        if quote not in caption:
            raise ValueError(f"Excerpt is not verbatim for {model} COCO {image_id}: {quote}")

    image_path = IMAGES / f"COCO_val2014_{image_id:012d}.jpg"
    with Image.open(image_path) as image:
        original = image.convert("RGB")
    view = model_view(original, model)

    tile_h = (1.75 if view.width / view.height < .85 else
              1.40 if model == "llava_1_5_7b" else 1.30)
    tile_w = tile_h * view.width / view.height
    x0, gap = 1.58, .08
    fig_w = x0 + 5 * tile_w + 4 * gap + .38
    photo_ratio = original.width / original.height
    photo_h = 1.70
    for _ in range(6):
        photo_w = photo_h * photo_ratio
        caption_x = .42 + photo_w + .16
        caption_w = min(fig_w - caption_x - .42, 8.0)
        caption_rows = prepare_caption(caption, case, real_summary["token"],
                                       hall_summary["token"], caption_w - .22)
        caption_h = max(1.05, len(caption_rows) * .235 + .18)
        required_photo_h = max(1.70, caption_h + .53)
        if abs(required_photo_h - photo_h) < .005:
            break
        photo_h = required_photo_h
    photo_w = photo_h * photo_ratio
    caption_x = .42 + photo_w + .16
    caption_w = min(fig_w - caption_x - .42, 8.0)
    caption_rows = prepare_caption(caption, case, real_summary["token"],
                                   hall_summary["token"], caption_w - .22)
    caption_h = max(1.05, len(caption_rows) * .235 + .18)
    if caption_h + .53 > photo_h + .005:
        raise ValueError(f"Caption box would extend below the image for COCO {image_id}")
    top_h = max(.95 + caption_h, photo_h + .48)
    block_h = .35 + 3 * tile_h + 2 * .06
    fig_h = .20 + top_h + .16 + block_h + .16 + block_h + .34 + .20
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, facecolor="white")
    model_name = {"llava_1_5_7b": "LLaVA-1.5-7B", "qwen3_vl_8b": "Qwen3-VL-8B",
                  "internvl_2_5_8b": "InternVL-2.5-8B"}[model]
    top = fig_h - .20
    top_bottom = top - top_h
    input_label_size = 11.5 if photo_ratio < .85 else 14
    put_text(fig, .42, top - .13, "[Input image]:", color=INK,
             fontsize=input_label_size, fontfamily="DejaVu Serif", va="center")
    put_image(fig, original, .42, top - .38 - photo_h, photo_w, photo_h)
    put_text(fig, caption_x, top - .13, "User:", color=INK,
             fontsize=14, fontfamily="DejaVu Serif", va="center")
    add_rect(fig, caption_x, top - .58, caption_w, .31,
             face="#f3f4f6", edge="none", radius=.025, lw=0)
    put_text(fig, caption_x + .12, top - .425, "Describe this image.", color=INK,
             fontsize=12.5, fontfamily="DejaVu Serif", va="center", zorder=3)
    put_text(fig, caption_x, top - .73, f"{model_name}:", color=INK,
             fontsize=14, fontfamily="DejaVu Serif", va="center")
    add_rect(fig, caption_x, top - .91 - caption_h, caption_w, caption_h,
             face="#e9f3ff", edge="none", radius=.025, lw=0)
    draw_caption(fig, caption_rows, caption_x + .11, top - 1.01)

    real_top = top_bottom - .16
    hall_top = real_top - block_h - .16
    draw_block(fig, real_top, "Real token", real_summary["token"], GREEN,
               real, view, x0, tile_w, tile_h, gap)
    draw_block(fig, hall_top, "Hallucinated token", hall_summary["token"], RED,
               hall, view, x0, tile_w, tile_h, gap)
    draw_layer_row(fig, hall_top - block_h - .18, x0, tile_w, gap)

    stem = f"{model}_{image_id:06d}_pair_appendix"
    for ext in formats:
        path = OUTPUT / f"{stem}.{ext}"
        if force or not path.exists():
            fig.savefig(path, dpi=DPI, facecolor="white",
                        metadata={"Title": f"{model_name} COCO {image_id:06d} paired Log-S heatmaps"})
    plt.close(fig)
    return {
        "model": model, "image_id": image_id, "real_token": real_summary["token"],
        "real_response_token_idx": case["real_idx"], "hall_token": hall_summary["token"],
        "hall_response_token_idx": case["hall_idx"], "real_quote": case["real_quote"],
        "hall_quote": case["hall_quote"], "full_caption": caption,
        "review_note": case["review_note"],
        **{ext: f"{stem}.{ext}" for ext in ("png", "pdf", "svg")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf", "svg"))
    parser.add_argument("--case", action="append", metavar="MODEL:COCO_ID",
                        help="Render only these cases; still refresh the complete index")
    parser.add_argument("--force", action="store_true", help="Replace existing files for selected cases")
    opts = parser.parse_args()
    formats = tuple(opts.formats)
    selected = set(opts.case or ())
    available = {f"{case['model']}:{case['image_id']}" for case in CASES}
    if selected - available:
        parser.error(f"Unknown case(s): {', '.join(sorted(selected - available))}")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.pad_inches": 0,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    caches = {}
    for model in {case["model"] for case in CASES}:
        with (SOURCE / model / "shortlist.pkl").open("rb") as handle:
            data = pickle.load(handle)
        caches[model] = {(key[0], key[1]): value for key, value in data.items()}
    rows = [render(case, caches, formats if not selected or f"{case['model']}:{case['image_id']}" in selected else (), opts.force)
            for case in CASES]
    overview = Image.new("RGB", (1452, 20 + ((len(rows) + 1) // 2) * 950), "#e8edf4")
    overview_draw = ImageDraw.Draw(overview)
    for index, row in enumerate(rows):
        with Image.open(OUTPUT / row["png"]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (696, 900), Image.Resampling.LANCZOS)
        x, y = 20 + (index % 2) * 716, 20 + (index // 2) * 950
        overview_draw.text((x, y), f"{row['model']} · COCO {row['image_id']:06d}", fill=INK)
        overview.paste(thumb, (x + (696 - thumb.width) // 2, y + 20))
    overview.save(OUTPUT / "contact_sheet.jpg", quality=88)
    with (OUTPUT / "figure_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Appendix pair figures", "",
        f"{len(rows)} selected COCO pairs. Each figure shows the input image, the actual user prompt "
        "(`Describe this image.`), selected caption sentences around both tokens, "
        "and layer 15/18/21/24/27 heatmaps. Real tokens are green; hallucinated tokens "
        "are red. The rows are Raw attention, Visual evidence (Log-S target), and "
        "FFN preference (hpre source). Each panel uses its own Top-16 visual patches.", "",
        "The displayed sentences are verbatim; the complete generated captions are in "
        "[figure_manifest.csv](figure_manifest.csv). Omitted sentences are marked with an ellipsis.", "",
        f"[Contact sheet](contact_sheet.jpg) for a quick view of all {len(rows)} figures.", "",
        "| Model | COCO | Real / hallucinated | Files | Review note |",
        "|---|---:|---|---|---|",
    ]
    for row in rows:
        files = " / ".join(f"[{ext.upper()}]({row[ext]})" for ext in ("png", "pdf", "svg"))
        lines.append(
            f"| {row['model']} | {row['image_id']:06d} | "
            f"`{row['real_token']}` / `{row['hall_token']}` | {files} | "
            f"{row['review_note']} |"
        )
    lines += ["", "## Reproduce", "", "Run from the token-detector repository root:", "",
              "```bash", "/opt/conda/private/envs/glsim/bin/python scripts/render_log_s_appendix_pairs.py",
              "```", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    print(f"Rendered {len(selected) if selected else len(rows)} pair figures ({', '.join(formats)}) into {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
