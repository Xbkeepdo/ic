#!/usr/bin/env python3
"""Export six standalone layer-27 Top-16 maps for COCO image 831."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analyze_llava_831_log_s_pre_topk import overlay, topk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "outputs/llava_1_5_7b/COCO831-LOGS-PRE-K32-TAU003-REFERENCE-STYLE"
)
LAYER = 27
SIGNALS = ("raw_attention", "target_distribution", "source_distribution")
SIGNAL_SLUGS = {
    "raw_attention": "raw_attention",
    "target_distribution": "T",
    "source_distribution": "P",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir / "layer27_top16"
    output.mkdir(parents=True, exist_ok=True)

    with Image.open(args.input_dir / "model_view_center_crop.png") as source:
        image = source.convert("RGB")

    manifest = []
    with np.load(args.input_dir / "raw_matrices.npz", allow_pickle=False) as arrays:
        for token in ("skateboard", "handbag"):
            risk = float(arrays[f"{token}__risk_log_s_hpre"][LAYER - 1])
            for signal in SIGNALS:
                values = arrays[f"{token}__{signal}"][LAYER - 1]
                retained, indices, mass = topk(values, 16)
                rendered, _smooth = overlay(image, retained)
                rgb = np.uint8(np.round(rendered * 255.0))
                name = f"{token}_{SIGNAL_SLUGS[signal]}_layer27_top16"
                Image.fromarray(rgb).resize(
                    (1200, 1200), resample=Image.Resampling.LANCZOS
                ).save(output / f"{name}.png")

                figure = plt.figure(figsize=(4, 4))
                axis = figure.add_axes((0, 0, 1, 1))
                axis.imshow(rgb)
                axis.set_axis_off()
                figure.savefig(output / f"{name}.pdf", dpi=300)
                plt.close(figure)

                manifest.append(
                    {
                        "image_id": 831,
                        "token": token,
                        "signal": signal,
                        "layer_1_based": LAYER,
                        "top_k": 16,
                        "top16_signal_mass": mass,
                        "risk_log_s_hpre_k032_tau003": risk,
                        "indices_row_major_24x24": " ".join(
                            map(str, indices.tolist())
                        ),
                        "png": f"{name}.png",
                        "pdf": f"{name}.pdf",
                    }
                )

    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Saved {len(manifest)} separate layer-27 Top-16 maps to {output}")


if __name__ == "__main__":
    main()
