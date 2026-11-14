#!/usr/bin/env python3
"""Render selected single-token Log-S heatmaps in the appendix figure style.

Caption text is selected verbatim from each cached model response. Every panel
uses the existing independent Top-16 overlay for its signal and decoder layer.
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
from PIL import Image, ImageDraw, ImageFont, ImageOps

from render_log_s_appendix_pairs import (
    DPI, GREEN, INK, RED, SOURCE, add_rect, draw_block, draw_layer_row,
    put_image, put_text,
)
from select_log_s_paper_heatmaps import IMAGES, model_view


OUTPUT = SOURCE / "appendix_single"
CASES = (
    {
        "model": "qwen3_vl_8b", "image_id": 550350, "token_idx": 99, "label": 1,
        "segments": (
            "A white paper bag, partially open, sits on the couch cushion.",
            "It contains several rectangular, golden-brown crackers, which appear to be a type of saltine or similar snack.",
            'A clear plastic bottle of "Purified Water" is positioned behind the cracker bag.',
            "The label is green and white, and the bottle is about half-full.",
            "A silver, black, and gray remote control is lying on the couch to the left of the cracker bag.",
        ),
    },
    {
        "model": "qwen3_vl_8b", "image_id": 516856, "token_idx": 266, "label": 0,
        "segments": (
            "On the wooden table in front of the person:",
            "A white plate holds several small, round, toasted bread slices (possibly crostini or small sandwiches).",
            "Two lit candles in decorative, orange-colored glass holders provide a warm glow.",
            "A folded white napkin is placed beside the plate.",
            "A silver spoon rests on the plate.",
        ),
    },
    {
        "model": "qwen3_vl_8b", "image_id": 436470, "token_idx": 214, "label": 0,
        "segments": (
            "A white, ornate fireplace with a mirror above it is a central feature.",
            "The mantel is decorated with small items, including a clock and a vase.",
            "The room is lit by multiple sources, including a large, modern-style lamp with a grey shade on a side table, a chandelier hanging from the ceiling, and natural light coming through the windows.",
        ),
    },
    {
        "model": "llava_1_5_7b", "image_id": 204256, "token_idx": 93, "label": 0,
        "segments": (
            "The image features a couch with a yellow and pink backpack placed on it.",
            "The backpack is filled with various items, including a cell phone, a wallet, and a book.",
            "There are also several pens scattered around the backpack, indicating that it might be used for work or study purposes.",
            "In addition to the backpack, there are multiple books placed on the couch, with some of them located near the backpack.",
            "A clock can also be seen on the couch, possibly indicating the time or serving as a decorative element.",
        ),
    },
    {
        "model": "llava_1_5_7b", "image_id": 217509, "token_idx": 83, "label": 0,
        "segments": (
            "The image features a young girl standing on the side of a road, posing for a picture in front of a stop sign.",
            "She is wearing a purple shirt and appears to be enjoying her time outdoors.",
            "In the background, there are two cars parked along the street, one closer to the left side and the other further to the right.",
            "Additionally, there is a bird perched on a branch in the upper right corner of the image.",
        ),
    },
    {
        "model": "llava_1_5_7b", "image_id": 390435, "token_idx": 17, "label": 1,
        "segments": (
            "The image captures a young man in a soccer uniform running towards a soccer ball on a field.",
            "He is focused on the ball, which is located towards the center of the field.",
            "The soccer player is in motion, likely preparing to kick the ball or make a play.",
            "The scene is set on a grassy field, and the young man appears to be the main subject of the image.",
        ),
    },
    {
        "model": "llava_1_5_7b", "image_id": 343978, "token_idx": 15, "label": 1,
        "segments": (
            "The image features a young child, possibly a little girl, riding a skateboard down a sidewalk.",
            "The child is wearing a cast on her arm, indicating that she might have recently had an injury.",
            "The skateboard is positioned under her feet, and she appears to be enjoying her time on the board.",
            "The scene also includes a potted plant located near the sidewalk, adding a touch of greenery to the environment.",
        ),
    },
    {
        "model": "llava_1_5_7b", "image_id": 113588, "token_idx": 109, "label": 1,
        "segments": (
            "The image depicts a group of people working in an office setting.",
            "There are three people in the scene, with one person sitting at a desk using a laptop, another person standing nearby, and the third person sitting at a separate desk.",
            "The office is equipped with two computer monitors, one on the left side of the room and the other on the right side.",
            "There are also two keyboards, one in front of the person using the laptop and the other in front of the person standing nearby.",
            "A mouse can be seen on the right side of the room.",
        ),
    },
    {
        "model": "internvl_2_5_8b", "image_id": 2295, "token_idx": 52, "label": 0,
        "segments": (
            "The image depicts a black and white cat with its head close to a wooden bowl.",
            "The bowl is intricately designed with a carved pattern around its rim, featuring a series of interconnected loops and swirls.",
            "Inside the bowl, there is a ball of yarn, which appears to be the cat's focus of interest.",
            "The cat's nose is almost touching the yarn, indicating curiosity or interest in the object.",
        ),
    },
    {
        "model": "internvl_2_5_8b", "image_id": 468778, "token_idx": 311, "label": 0,
        "segments": (
            "In the foreground, there are five teddy bears, each uniquely dressed and positioned around a small table.",
            "The table is set with various items, including a teapot, cups, and plates, indicating a tea party setup.",
            "This bear is the focal point of the scene.",
            "It is dressed in a colorful outfit with a red hat and is holding a plate with a piece of cake.",
            "The bear is also holding a small book or menu, suggesting it might be reading or deciding what to order.",
        ),
    },
    {
        "model": "internvl_2_5_8b", "image_id": 353347, "token_idx": 245, "label": 0,
        "segments": (
            "To the right of the dancing pair, there is a desk with a computer setup.",
            "The desk is cluttered with various items, including a monitor, a keyboard, a mouse, and some papers or books.",
            "There is also a lamp on the desk, which is turned on, adding to the room's illumination.",
            "Behind the desk, a staircase is visible, leading to an upper level of the house.",
        ),
    },
    {
        "model": "internvl_2_5_8b", "image_id": 30793, "token_idx": 51, "label": 1,
        "segments": (
            "The image depicts a cluttered desk with various objects.",
            "In the foreground, there is a coaster with the Heineken logo on it, placed on top of a stack of papers.",
            "To the right of the coaster, there is a black remote control.",
            "The background shows a computer monitor and some other indistinct items.",
            "The overall setting suggests a casual, possibly work-related environment.",
        ),
    },
    {
        "model": "internvl_2_5_8b", "image_id": 210032, "token_idx": 214, "label": 1,
        "segments": (
            "To the right of the image, there is a bird, possibly a sparrow, perched on the edge of the table.",
            "The bird is facing the plate of food, and it appears to be pecking at the condiments, particularly the ketchup.",
            "The bird's feathers are brown, and it has a small beak.",
            "The bird's presence adds a dynamic element to the scene, suggesting that the dining area is in a location where wildlife is present.",
        ),
    },
)


def caption_lines(caption: str, case: dict, token: str, content_width: float):
    """Wrap selected exact excerpts and retain omissions with an ellipsis."""
    positions = [(caption.index(segment), segment) for segment in case["segments"]]
    if positions != sorted(positions):
        raise ValueError(f"Caption segments are out of order for {case['image_id']}")
    pieces, previous_end = [], None
    for start, segment in positions:
        if previous_end is not None:
            gap = caption[previous_end:start]
            formatting_only = not re.sub(r"[\s*\-]+", "", gap)
            pieces.append(" " if formatting_only else " … ")
        pieces.append(segment)
        previous_end = start + len(segment)
    content = "".join(pieces)
    if not re.search(rf"\b{re.escape(token)}\b", content, flags=re.IGNORECASE):
        raise ValueError(f"No highlighted token in caption for {case['image_id']}")
    fontsize = 11.5
    font = ImageFont.truetype(findfont("DejaVu Serif"), round(fontsize * DPI / 72))
    space = font.getlength(" ") / DPI
    words = []
    color = GREEN if case["label"] == 1 else RED
    for match in re.finditer(r"\S+", content):
        word = match.group()
        words.append((word, color if re.search(rf"\b{re.escape(token)}\b", word, re.I) else INK,
                      font.getlength(word) / DPI))
    lines, width = [[]], 0.0
    for word, shade, word_width in words:
        if lines[-1] and width + word_width > content_width:
            lines.append([])
            width = 0.0
        lines[-1].append((word, shade, word_width))
        width += word_width + space
    return lines, content


def draw_caption(fig, lines, x, top):
    font = ImageFont.truetype(findfont("DejaVu Serif"), round(11.5 * DPI / 72))
    space = font.getlength(" ") / DPI
    for row, line in enumerate(lines):
        cursor = x
        for word, shade, width in line:
            put_text(fig, cursor, top - row * .235, word, color=shade,
                     fontsize=11.5, fontfamily="DejaVu Serif", va="top", zorder=3)
            cursor += width + space


def render(case: dict, caches: dict, formats: tuple[str, ...], force: bool = False) -> dict:
    model, image_id, token_idx = case["model"], case["image_id"], case["token_idx"]
    record = caches[model][(image_id, token_idx)]
    summary = record["summary"]
    label = int(summary["label"])
    if label != case["label"]:
        raise ValueError(f"Label mismatch for {model} COCO {image_id}")
    caption, token = summary["caption_full"], summary["token"]
    with Image.open(IMAGES / f"COCO_val2014_{image_id:012d}.jpg") as image:
        original = image.convert("RGB")
    view = model_view(original, model)

    tile_h = (1.75 if view.width / view.height < .85 else
              1.70 if model == "llava_1_5_7b" else 1.30)
    tile_w = tile_h * view.width / view.height
    x0, gap = 1.58, .08
    fig_w = x0 + 5 * tile_w + 4 * gap + .38
    photo_ratio = original.width / original.height
    photo_h = 1.70
    for _ in range(12):
        photo_w = photo_h * photo_ratio
        caption_x = .42 + photo_w + .16
        caption_w = min(fig_w - caption_x - .42, 8.0)
        if caption_w < 2.5:
            raise ValueError(f"Caption column too narrow for {model} COCO {image_id}")
        rows, content = caption_lines(caption, case, token, caption_w - .22)
        caption_h = max(.18 + len(rows) * .235, .55)
        required_h = max(1.70, caption_h + .53)
        if abs(required_h - photo_h) < .005:
            break
        photo_h = required_h
    photo_w = photo_h * photo_ratio
    caption_x = .42 + photo_w + .16
    caption_w = min(fig_w - caption_x - .42, 8.0)
    rows, content = caption_lines(caption, case, token, caption_w - .22)
    caption_h = max(.18 + len(rows) * .235, .55)
    if caption_h + .53 > photo_h + .005:
        raise ValueError(f"Caption box exceeds photo height for {model} COCO {image_id}")
    top_h = max(.95 + caption_h, photo_h + .48)
    block_h = .35 + 3 * tile_h + 2 * .06
    fig_h = .20 + top_h + .16 + block_h + .34 + .20
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
             face="#e8f5ed", edge="none", radius=.025, lw=0)
    draw_caption(fig, rows, caption_x + .11, top - 1.01)

    block_top = top_bottom - .16
    draw_block(fig, block_top, "Real token" if label else "Hallucinated token",
               token, GREEN if label else RED, record, view, x0, tile_w, tile_h, gap)
    draw_layer_row(fig, block_top - block_h - .18, x0, tile_w, gap)
    stem = f"{model}_{image_id:06d}_token{token_idx:03d}_appendix"
    for ext in formats:
        path = OUTPUT / f"{stem}.{ext}"
        if force or not path.exists():
            fig.savefig(path, dpi=DPI, facecolor="white",
                        metadata={"Title": f"{model_name} COCO {image_id:06d} {token} Log-S heatmaps"})
    plt.close(fig)
    return {
        "model": model, "image_id": f"{image_id:06d}", "token": token,
        "response_token_idx": token_idx, "label": "real" if label else "hallucinated",
        "selected_caption": content, "full_caption": caption,
        "caption_rows": len(rows), "caption_height_in": round(caption_h, 3),
        "original_image_height_in": round(photo_h, 3),
        **{ext: f"{stem}.{ext}" for ext in ("png", "pdf", "svg")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf", "svg"))
    parser.add_argument("--case", action="append", metavar="MODEL:COCO_ID",
                        help="Render only these cases; still refresh the complete index")
    parser.add_argument("--force", action="store_true")
    opts = parser.parse_args()
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
    rows = [render(case, caches,
                   tuple(opts.formats) if not selected or f"{case['model']}:{case['image_id']}" in selected else (),
                   opts.force) for case in CASES]
    if all((OUTPUT / row["png"]).exists() for row in rows):
        overview = Image.new("RGB", (1452, 20 + ((len(rows) + 1) // 2) * 800), "#e8edf4")
        draw = ImageDraw.Draw(overview)
        for index, row in enumerate(rows):
            with Image.open(OUTPUT / row["png"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (696, 750), Image.Resampling.LANCZOS)
            x, y = 20 + (index % 2) * 716, 20 + (index // 2) * 800
            draw.text((x, y), f"{row['model']} · COCO {row['image_id']}", fill=INK)
            overview.paste(thumb, (x + (696 - thumb.width) // 2, y + 20))
        overview.save(OUTPUT / "contact_sheet.jpg", quality=88)
    with (OUTPUT / "figure_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Appendix single-token figures", "",
        f"{len(rows)} selected COCO examples. Each figure shows the original image, "
        "the actual user prompt, selected verbatim caption text, and independent Top-16 "
        "heatmaps at layers 15, 18, 21, 24, and 27. The rows show raw attention, "
        "Log-S visual evidence, and FFN preference. Real tokens are green and "
        "hallucinated tokens are red. Omitted text is marked by an ellipsis.", "",
        "The [manifest](figure_manifest.csv) contains the full cached caption and "
        "the selected excerpt for every figure.", "",
        "The real/hallucinated labels follow the existing CHAIR token annotations. "
        "Before publication, visually recheck Qwen3 516856 `spoon` and 436470 `clock`, "
        "and InternVL 468778 `book` and 353347 `computer`: the object identity or "
        "visibility in these images is less certain than the label suggests.", "",
        "[Contact sheet](contact_sheet.jpg)", "",
        "| Model | COCO | Token | Label | Files |",
        "|---|---:|---|---|---|",
    ]
    for row in rows:
        files = " / ".join(f"[{ext.upper()}]({row[ext]})" for ext in ("png", "pdf", "svg"))
        lines.append(f"| {row['model']} | {row['image_id']} | `{row['token']}` | {row['label']} | {files} |")
    lines += ["", "## Reproduce", "", "```bash",
              "/opt/conda/private/envs/glsim/bin/python scripts/render_log_s_appendix_single.py",
              "```", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    print(f"Rendered {len(selected) if selected else len(rows)} single-token figures "
          f"({', '.join(opts.formats)}) into {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
