"""Evaluate a trained UNet model on the TEST (or validation) split.

Usage:
    python -m src.evaluate --name unet_r34_rgb [--split test] [--n_vis 8]

Writes:
    - checkpoints/<name>/<split>_results.json with per-class IoU, recall, precision, and latency
    - Side-by-side ground truth vs prediction masks to results/<name>/
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.dataset import LunarSeg
from src.metrics import ConfusionMeter


def find_checkpoint_dir(name):
    """Resolves checkpoint directory whether running from repo root or inside LunarHazardSeg."""
    candidates = [
        name,
        os.path.join("checkpoints", name),
        os.path.join("LunarHazardSeg", "checkpoints", name),
        os.path.join("runs", "unet", name),
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", name),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            return c
    return os.path.join("checkpoints", name)


p = argparse.ArgumentParser(description="Evaluate a trained UNet checkpoint.")
p.add_argument("--name", required=True, help="Checkpoint folder name (e.g. unet_r34_rgb)")
p.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split to evaluate")
p.add_argument("--n_vis", type=int, default=8, help="Number of qualitative prediction masks to export")
a = p.parse_args()

ckpt_dir = find_checkpoint_dir(a.name)
cfg_path = os.path.join(ckpt_dir, "config.json")
if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"Configuration file 'config.json' not found in '{ckpt_dir}'.")

cfg = json.load(open(cfg_path))
dev = "cuda" if torch.cuda.is_available() else "cpu"
in_ch = 3 if cfg["mode"] == "rgb" else 4
model = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=in_ch, classes=3)

weights_path = os.path.join(ckpt_dir, "best.pth")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file 'best.pth' not found in '{ckpt_dir}'.")
model.load_state_dict(torch.load(weights_path, map_location=dev))
model.to(dev).eval()

ds = LunarSeg(
    cfg.get("root", "LunarSeg"),
    a.split,
    mode=cfg["mode"],
    pred_depth_dir=cfg.get("pred_depth_dir"),
    pred_invert=cfg.get("pred_invert", False),
    gt_linear=cfg.get("gt_linear", False),
)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
cm = ConfusionMeter(3)
times = []
vis_dir = os.path.join(ckpt_dir, "vis") if "results" not in os.path.abspath(".") else os.path.join("results", a.name)
os.makedirs(vis_dir, exist_ok=True)
palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)

with torch.no_grad():
    for i, (x, y, name) in enumerate(dl):
        x, y = x.to(dev), y.to(dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= 10:
            times.append(time.perf_counter() - t0)  # skip warm-up passes
        cm.update(o, y)
        if i < a.n_vis:
            from PIL import Image
            pred = o.argmax(1)[0].cpu().numpy()
            gt = y[0].cpu().numpy()
            Image.fromarray(np.concatenate([palette[gt], palette[pred]], 1)).save(
                os.path.join(vis_dir, f"{name[0]}_gt_vs_pred.png")
            )

r = cm.results()
r["latency_ms"] = 1000 * float(np.mean(times)) if times else None
r["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
results_file = os.path.join(ckpt_dir, f"{a.split}_results.json")
json.dump(r, open(results_file, "w"), indent=2)

print(f"\n--- Evaluation Results ({a.split} split) ---")
for k, v in r.items():
    print(f"{k:22s} {v:.4f}" if isinstance(v, float) else f"{k:22s} {v}")
print(f"Saved evaluation metrics to: {results_file}")
if a.n_vis > 0:
    print(f"Saved {min(a.n_vis, len(ds))} visualizations to: {vis_dir}")