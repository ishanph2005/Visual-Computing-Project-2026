"""
5-panel sample figure for the presentation. Put this file in ~/LunarHazardSeg
(next to visualize_sample.py) and run:

    python make_sample_figure.py

Panels (one row per frame):
  1 Original RGB image
  2 Ground-truth segmentation mask
  3 U-Net prediction (RGB)                -> checkpoints/unet_r34_rgb
  4 U-Net prediction (RGB + GT depth)     -> checkpoints/unet_r34_rgbd_lin   (plain depth, gt_linear)
  5 U-Net prediction (RGB + DA V2-FT)     -> checkpoints/unet_r34_rgbpd_ft

Inputs are built with src/dataset.py (LunarSeg), so normalisation and depth
handling are identical to training. Colours: green = rock, red = crater.
Output: sample_predictions.png (300 dpi)
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from dataset import LunarSeg  # noqa: E402

FRAMES = ["03_0634", "00_0919"]            # change to any frame names you like
RUNS = {                                    # panel title -> checkpoint folder
    "U-Net (RGB)": "unet_r34_rgb",
    "U-Net (RGB + GT depth)": "unet_r34_rgbd_lin",
    "U-Net (RGB + DA V2-FT)": "unet_r34_rgbpd_ft",
}
ROOT = "data"
CKPT_DIR = "checkpoints"
BG, CRATER, ROCK = 0, 1, 2                  # from src/dataset.py
SCENARIO = {"00": "low light, rough", "01": "low light, flat",
            "02": "high light, rough", "03": "high light, flat"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_split(name):
    for split in ("test", "validation", "train"):
        with open(os.path.join(ROOT, f"{split}.txt")) as f:
            if name in {l.strip() for l in f}:
                return split
    raise SystemExit(f"{name} is not listed in test/validation/train.txt")


def load_run(folder):
    cfg = json.load(open(os.path.join(CKPT_DIR, folder, "config.json")))
    in_ch = 3 if cfg["mode"] == "rgb" else 4
    model = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=in_ch, classes=3)
    state = torch.load(os.path.join(CKPT_DIR, folder, "best.pth"), map_location="cpu")
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(DEVICE).eval(), cfg


def dataset_for(cfg, split):
    return LunarSeg(cfg.get("root", ROOT), split, mode=cfg["mode"],
                    pred_depth_dir=cfg.get("pred_depth_dir"),
                    pred_invert=cfg.get("pred_invert", False),
                    gt_linear=cfg.get("gt_linear", False), augment=False)


def overlay(rgb, cls):
    out = rgb.astype(np.float32).copy()
    for c, col in [(ROCK, (0, 220, 0)), (CRATER, (220, 0, 0))]:
        m = cls == c
        out[m] = 0.4 * out[m] + 0.6 * np.array(col)
    return out.astype(np.uint8)


def rock_iou(pred, gt):
    p, g = pred == ROCK, gt == ROCK
    union = (p | g).sum()
    return (p & g).sum() / union if union else float("nan")


@torch.no_grad()
def main():
    runs = {title: load_run(folder) for title, folder in RUNS.items()}
    titles = ["Original RGB image", "Ground-truth mask"] + list(RUNS)

    fig, axes = plt.subplots(len(FRAMES), 5, figsize=(20, 3.4 * len(FRAMES)), squeeze=False)
    for r, name in enumerate(FRAMES):
        split = find_split(name)
        rgb = np.asarray(Image.open(os.path.join(ROOT, "image", f"{name}.png")).convert("RGB").resize((640, 480)))
        panels, ious, gt = [rgb], [], None
        for title, (model, cfg) in runs.items():
            ds = dataset_for(cfg, split)
            x, y, _ = ds[ds.names.index(name)]
            gt = y.numpy()
            pred = model(x[None].to(DEVICE)).argmax(1)[0].cpu().numpy()
            panels.append(overlay(rgb, pred))
            ious.append(rock_iou(pred, gt))
        panels.insert(1, overlay(rgb, gt))

        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            t = titles[c] if r == 0 else ""
            if c >= 2:
                t += ("\n" if t else "") + f"rock IoU {ious[c - 2]:.2f}"
            ax.set_title(t, fontsize=13)
        axes[r, 0].set_ylabel(f"{name}\n{SCENARIO.get(name[:2], '')}", fontsize=12)
        print(f"{name} ({split}): rock IoU RGB {ious[0]:.3f} | +GT {ious[1]:.3f} | +DA V2-FT {ious[2]:.3f}")

    fig.text(0.5, 0.005, "green = rock, red = crater", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig("sample_predictions.png", dpi=300)
    print("saved sample_predictions.png")


if __name__ == "__main__":
    main()