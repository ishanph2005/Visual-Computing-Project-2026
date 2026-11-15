#!/usr/bin/env python3
"""Zoomed error maps highlighting performance differences between U-Net variants.

Colours (rock class only):
  green   = rock found correctly
  yellow  = rock MISSED (in ground truth, not predicted)
  magenta = FALSE rock (predicted, not in ground truth)

Usage:
    python3 scripts/unet/make_zoom.py [--frame 00_0919] [--out zoom_errors.png]
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_sample_figure import RUNS, ROOT, ROCK, load_run, dataset_for, find_split, find_file, find_checkpoint, DEVICE

DEFAULT_FRAME = "00_0919"
CROP = (240, 640, 190, 300)
ZOOM_RGB_GAIN = 2.5


def error_map(rgb, pred, gt):
    out = np.clip(rgb.astype(np.float32) * ZOOM_RGB_GAIN, 0, 255) * 0.5
    p, g = pred == ROCK, gt == ROCK
    out[p & g] = (0, 210, 0)
    out[~p & g] = (255, 215, 0)
    out[p & ~g] = (230, 0, 230)
    return out.astype(np.uint8)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate zoomed error comparison map.")
    parser.add_argument("--frame", default=DEFAULT_FRAME, help="Sample stem to inspect (default: 00_0919)")
    parser.add_argument("--out", default="zoom_errors.png", help="Output PNG path")
    args = parser.parse_args()

    split = find_split(args.frame)
    img_p = find_file("image", args.frame, split)
    if not img_p:
        print(f"Error: Frame '{args.frame}' image not found.")
        return

    rgb = np.asarray(Image.open(img_p).convert("RGB").resize((640, 480)))
    preds, gt = {}, None

    available_runs = {}
    for title, folder in RUNS.items():
        ckpt_p = find_checkpoint(folder)
        if (ckpt_p / "best.pth").exists():
            available_runs[title] = folder

    if not available_runs:
        print("No UNet checkpoints found in checkpoints/ or runs/unet/.")
        return

    for title, folder in available_runs.items():
        model, cfg = load_run(folder)
        ds = dataset_for(cfg, split)
        try:
            idx = ds.names.index(args.frame)
        except ValueError:
            idx = 0
        x, y, _ = ds[idx]
        gt = y.numpy()
        preds[title] = model(x[None].to(DEVICE)).argmax(1)[0].cpu().numpy()

    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = (preds[names[i]] != preds[names[j]]).sum()
            print(f"{names[i]} vs {names[j]}: {diff} pixels differ")
    for n in names:
        p, g = preds[n] == ROCK, gt == ROCK
        print(f"{n}: rock missed {(~p & g).sum()} px, false rock {(p & ~g).sum()} px")

    x0, x1, y0, y1 = CROP
    n_cols = len(names)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6 * (y1 - y0) / (x1 - x0) + 1.2), squeeze=False)
    for i, n in enumerate(names):
        ax = axes[0, i]
        p, g = preds[n] == ROCK, gt == ROCK
        ax.imshow(error_map(rgb, preds[n], gt)[y0:y1, x0:x1], interpolation="nearest")
        ax.set_title(f"{n}\nmissed {(~p & g).sum()} px · false {(p & ~g).sum()} px", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(0.5, 0.01, "green = rock found   yellow = rock missed   magenta = false rock",
             ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
