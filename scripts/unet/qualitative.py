#!/usr/bin/env python3
"""Qualitative figure for evaluation: test frames with predictions side by side.

Usage:
    python3 scripts/unet/qualitative.py
    python3 scripts/unet/qualitative.py --segformer_dir runs/segformer_b2_rgb/preds/test
    python3 scripts/unet/qualitative.py --names 00_0123 02_0456 03_0789

Columns: RGB | DA V2 Depth | Ground Truth | U-Net (RGB) | U-Net (RGB + GT Depth) [| SegFormer (RGB)]
Masks are drawn over a dimmed copy of the image: red = crater, green = rock.

Outputs: results/qualitative.pdf and results/qualitative.png
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LunarSeg, read_split, load_pred, pred_depth_to_disparity, resolve_dataset_root

p = argparse.ArgumentParser(description="Generate qualitative side-by-side comparison figure.")
p.add_argument("--root", default=None, help="LunarSeg dataset root (default: auto-detect)")
p.add_argument("--rgb_run", default="unet_r34_rgb", help="Checkpoint name/dir for RGB UNet")
p.add_argument("--depth_run", default="unet_r34_rgbd_lin", help="Checkpoint name/dir for Depth UNet")
p.add_argument("--pred_depth_dir", default="outputs/dav2_lunar_preds", help="DA V2 predicted depth dir")
p.add_argument("--segformer_dir", default=None, help="Optional directory holding SegFormer prediction masks")
p.add_argument("--names", nargs="+", default=None, help="Explicit list of image stems to plot")
p.add_argument("--n", type=int, default=3, help="Number of frames to pick if --names not given")
p.add_argument("--crop", dest="crop", action="store_true", default=True, help="Crop horizon band containing hazards")
p.add_argument("--no-crop", dest="crop", action="store_false", help="Keep full uncropped frames")
p.add_argument("--out", default="results/qualitative", help="Output path prefix (without extension)")
a = p.parse_args()

root = resolve_dataset_root(a.root)
dev = "cuda" if torch.cuda.is_available() else "cpu"
PAL = np.array([[0, 0, 0], [230, 40, 40], [40, 200, 70]], np.float32)


def find_checkpoint(run):
    candidates = [
        Path(run),
        Path("checkpoints") / run,
        Path("runs/unet") / run,
    ]
    for c in candidates:
        if (c / "config.json").exists() and (c / "best.pth").exists():
            return c
    return Path("checkpoints") / run


def load_run(run):
    ckpt_dir = find_checkpoint(run)
    cfg = json.load(open(ckpt_dir / "config.json"))
    in_ch = 3 if cfg["mode"] == "rgb" else 4
    m = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=in_ch, classes=3)
    state = torch.load(ckpt_dir / "best.pth", map_location=dev)
    if isinstance(state, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    m.load_state_dict(state)
    ds = LunarSeg(root, "test", mode=cfg["mode"], pred_depth_dir=cfg.get("pred_depth_dir"),
                  pred_invert=cfg.get("pred_invert", False), gt_linear=cfg.get("gt_linear", False))
    return m.to(dev).eval(), ds


def find_file(folder, name, split="test"):
    for cand in [
        root / folder / f"{name}.png",
        root / folder / split / f"{name}.png",
    ]:
        if cand.exists():
            return cand
    return None


@torch.no_grad()
def predict(model, ds, idx):
    x, y, _ = ds[idx]
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
        o = model(x[None].to(dev))
    return o.argmax(1)[0].cpu().numpy(), y.numpy()


def overlay(rgb, mask):
    base = rgb.astype(np.float32) * 0.55
    out = base.copy()
    for c in (1, 2):
        m = mask == c
        out[m] = 0.35 * base[m] + 0.65 * PAL[c]
    return out.clip(0, 255).astype(np.uint8)


def crop_rows(gt):
    rows = np.where((gt > 0).any(1))[0]
    if not a.crop or len(rows) == 0:
        return 0, gt.shape[0]
    lo, hi = max(rows.min() - 25, 0), min(rows.max() + 25, gt.shape[0])
    if hi - lo < 220:
        mid = (lo + hi) // 2
        lo, hi = max(mid - 110, 0), min(mid + 110, gt.shape[0])
    return lo, hi


def main():
    m_rgb, ds_rgb = load_run(a.rgb_run)
    m_d, ds_d = load_run(a.depth_run)
    names = ds_rgb.names
    index = {n: i for i, n in enumerate(names)}

    if a.names:
        chosen = a.names
    else:
        score = {}
        for i, n in enumerate(names):
            pr, gt = predict(m_rgb, ds_rgb, i)
            pd, _ = predict(m_d, ds_d, i)
            rock = gt == 2
            if rock.sum() < 400:
                continue
            score[n] = int(((pd == 2) & (pr != 2) & rock).sum() - ((pr == 2) & (pd != 2) & rock).sum())
            if (i + 1) % 200 == 0:
                print(f"scanning {i + 1}/{len(names)}")
        chosen = []
        for group in (["00"], ["02"], ["01", "03"]):
            cand = sorted([n for n in score if n.split("_")[0] in group], key=lambda n: -score[n])
            if cand:
                chosen.append(cand[0])
        chosen = chosen[:a.n]
    print("Selected frames:", " ".join(chosen))

    cols = ["RGB", "DA V2 Depth", "Ground Truth", "U-Net (RGB)", "U-Net (RGB + Depth)"]
    if a.segformer_dir:
        cols.append("SegFormer (RGB)")
    rows = []
    for n in chosen:
        i = index[n]
        img_path = find_file("image", n, "test")
        if not img_path:
            continue
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        pr, gt = predict(m_rgb, ds_rgb, i)
        pd, _ = predict(m_d, ds_d, i)
        try:
            dep = load_pred(a.pred_depth_dir, n, split="test").astype(np.float32)
            dep = pred_depth_to_disparity(dep) if dep.max() > 0 else dep
            dep = (dep - dep.min()) / (np.ptp(dep) + 1e-8)
        except Exception:
            dep = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

        panels = [rgb, dep, overlay(rgb, gt), overlay(rgb, pr), overlay(rgb, pd)]
        if a.segformer_dir:
            sf_path = Path(a.segformer_dir) / f"{n}.png"
            if sf_path.exists():
                sf = np.asarray(Image.open(sf_path))
                sf = sf[..., 0] if sf.ndim == 3 else sf
                panels.append(overlay(rgb, sf))
            else:
                panels.append(np.zeros_like(rgb))

        lo, hi = crop_rows(gt)
        rows.append((n, [pnl[lo:hi] for pnl in panels]))

    if not rows:
        print("No valid rows generated.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})
    nc, nr = len(cols), len(rows)
    h_in = [r[1][0].shape[0] / r[1][0].shape[1] for r in rows]
    fig_w = 7.16
    cell_w = fig_w / nc
    fig, axs = plt.subplots(nr, nc, figsize=(fig_w, sum(h_in) * cell_w * 1.06 + 0.25),
                            gridspec_kw={"height_ratios": h_in}, squeeze=False)
    for r, (n, panels) in enumerate(rows):
        for c, pnl in enumerate(panels):
            ax = axs[r, c]
            ax.imshow(pnl, cmap="magma" if pnl.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
            if r == 0:
                ax.set_title(cols[c], fontsize=7)
            if c == 0:
                scen = {"00": "low light, rough", "01": "low light, flat", "02": "high light, rough", "03": "high light, flat"}.get(n[:2], "")
                ax.text(0.02, 0.96, f"{n} ({scen})", transform=ax.transAxes, fontsize=5.5, color="white",
                        va="top", ha="left", bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
    top = 1 - 0.22 / fig.get_figheight()
    plt.subplots_adjust(left=0.005, right=0.995, top=top, bottom=0.005, wspace=0.03, hspace=0.05)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=300)
    print(f"Saved {a.out}.png")


if __name__ == "__main__":
    main()
