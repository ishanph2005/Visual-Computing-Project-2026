#!/usr/bin/env python3
"""5-panel sample figure generation.

Usage:
    python3 scripts/unet/make_sample_figure.py [--out sample_predictions.png]

Panels:
  1 Original RGB image
  2 Ground-truth segmentation mask
  3 U-Net (RGB)                -> unet_r34_rgb
  4 U-Net (RGB + GT depth)     -> unet_r34_rgbd_lin
  5 U-Net (RGB + DA V2-FT)     -> unet_r34_rgbpd_ft

Colours: green = rock, red = crater.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LunarSeg, resolve_dataset_root  # noqa: E402

FRAMES = ["03_0634", "00_0919"]
RUNS = {
    "U-Net (RGB)": "unet_r34_rgb",
    "U-Net (RGB + GT depth)": "unet_r34_rgbd_lin",
    "U-Net (RGB + DA V2-FT)": "unet_r34_rgbpd_ft",
}

ROOT = resolve_dataset_root()
BG, CRATER, ROCK = 0, 1, 2
SCENARIO = {
    "00": "low light, rough",
    "01": "low light, flat",
    "02": "high light, rough",
    "03": "high light, flat",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_file(folder, name, split="test"):
    for cand in [
        ROOT / folder / f"{name}.png",
        ROOT / folder / split / f"{name}.png",
    ]:
        if cand.exists():
            return cand
    return None


def find_split(name):
    for split in ("test", "validation", "train"):
        txt = ROOT / f"{split}.txt"
        if txt.exists():
            with open(txt) as f:
                if name in {l.strip() for l in f}:
                    return split
    return "test"


def find_checkpoint(folder):
    for cand in [
        Path(folder),
        Path("checkpoints") / folder,
        Path("runs/unet") / folder,
    ]:
        if (cand / "config.json").exists() and (cand / "best.pth").exists():
            return cand
    return Path("checkpoints") / folder


def load_run(folder):
    ckpt_path = find_checkpoint(folder)
    cfg = json.load(open(ckpt_path / "config.json"))
    in_ch = 3 if cfg["mode"] == "rgb" else 4
    model = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=in_ch, classes=3)
    state = torch.load(ckpt_path / "best.pth", map_location="cpu")
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
    parser = argparse.ArgumentParser(description="Generate 5-panel sample comparison figure.")
    parser.add_argument("--out", default="sample_predictions.png", help="Output PNG path")
    parser.add_argument("--frames", nargs="+", default=FRAMES, help="List of sample frame IDs")
    args = parser.parse_args()

    available_runs = {}
    for title, folder in RUNS.items():
        ckpt_p = find_checkpoint(folder)
        if (ckpt_p / "best.pth").exists():
            available_runs[title] = load_run(folder)

    if not available_runs:
        print("No UNet checkpoints found in checkpoints/ or runs/unet/.")
        return

    titles = ["Original RGB image", "Ground-truth mask"] + list(available_runs)
    n_frames = len(args.frames)
    fig, axes = plt.subplots(n_frames, len(titles), figsize=(4 * len(titles), 3.4 * n_frames), squeeze=False)

    for r, name in enumerate(args.frames):
        split = find_split(name)
        img_p = find_file("image", name, split)
        if not img_p:
            continue
        rgb = np.asarray(Image.open(img_p).convert("RGB").resize((640, 480)))
        panels, ious, gt = [rgb], [], None

        for title, (model, cfg) in available_runs.items():
            ds = dataset_for(cfg, split)
            try:
                idx = ds.names.index(name)
            except ValueError:
                idx = 0
            x, y, _ = ds[idx]
            gt = y.numpy()
            pred = model(x[None].to(DEVICE)).argmax(1)[0].cpu().numpy()
            panels.append(overlay(rgb, pred))
            ious.append(rock_iou(pred, gt))

        if gt is not None:
            panels.insert(1, overlay(rgb, gt))
        else:
            panels.insert(1, rgb)

        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            t = titles[c] if r == 0 else ""
            if c >= 2 and (c - 2) < len(ious):
                t += ("\n" if t else "") + f"rock IoU {ious[c - 2]:.2f}"
            ax.set_title(t, fontsize=12)
        axes[r, 0].set_ylabel(f"{name}\n{SCENARIO.get(name[:2], '')}", fontsize=11)

    fig.text(0.5, 0.005, "green = rock, red = crater", ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
