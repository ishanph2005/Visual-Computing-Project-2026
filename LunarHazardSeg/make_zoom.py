"""
Zoomed error maps that make the small differences between the three U-Nets visible.
Put next to make_sample_figure.py in ~/LunarHazardSeg and run:

    python make_zoom_figure.py

Colours (rock class only):
  green   = rock found correctly
  yellow  = rock MISSED (in ground truth, not predicted)
  magenta = FALSE rock (predicted, not in ground truth)
Also prints how many pixels each pair of models disagree on, to confirm they differ.
Output: zoom_errors.png (300 dpi)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

from make_sample_figure import RUNS, ROOT, ROCK, load_run, dataset_for, find_split, DEVICE

FRAME = "00_0919"
CROP = (240, 640, 190, 300)   # x0, x1, y0, y1 in the 640x480 image: the horizon strip. Adjust if needed.
ZOOM_RGB_GAIN = 2.5           # brighten the dark frame so the terrain is visible


def error_map(rgb, pred, gt):
    out = np.clip(rgb.astype(np.float32) * ZOOM_RGB_GAIN, 0, 255) * 0.5
    p, g = pred == ROCK, gt == ROCK
    out[p & g] = (0, 210, 0)       # correct
    out[~p & g] = (255, 215, 0)    # missed
    out[p & ~g] = (230, 0, 230)    # false alarm
    return out.astype(np.uint8)


@torch.no_grad()
def main():
    split = find_split(FRAME)
    rgb = np.asarray(Image.open(os.path.join(ROOT, "image", f"{FRAME}.png")).convert("RGB").resize((640, 480)))
    preds, gt = {}, None
    for title, folder in RUNS.items():
        model, cfg = load_run(folder)
        ds = dataset_for(cfg, split)
        x, y, _ = ds[ds.names.index(FRAME)]
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 18 * (y1 - y0) / (x1 - x0) / 3 + 1.2))
    for ax, n in zip(axes, names):
        p, g = preds[n] == ROCK, gt == ROCK
        ax.imshow(error_map(rgb, preds[n], gt)[y0:y1, x0:x1], interpolation="nearest")
        ax.set_title(f"{n}\nmissed {(~p & g).sum()} px · false {(p & ~g).sum()} px", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.01, "green = rock found   yellow = rock missed   magenta = false rock",
             ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("zoom_errors.png", dpi=300)
    print("saved zoom_errors.png")


if __name__ == "__main__":
    main()