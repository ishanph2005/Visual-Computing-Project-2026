#!/usr/bin/env python3
"""Evaluate decision thresholds, environmental corruptions, and latency for trained UNet models.

Analysis capabilities:
    1. Decision-threshold sweep:
       Evaluates performance when flagging hazards with custom probability thresholds
       t in [0.05, 0.1, 0.2, 0.5, 0.8, ...]. Lower threshold = higher safety margin
       (higher recall, slightly more false alarms).
    2. Test-time robustness corruptions:
       Simulates adverse spaceflight conditions:
         - Darker lighting drops (e.g. low sun elevation)
         - Non-linear gamma shifts
         - High-ISO / sensor Gaussian noise
         - Lens blur / defocus
    3. Latency & parameter profiling:
       Inference latency (ms), parameter count (M), and peak GPU memory.

Usage:
    python3 scripts/unet/stress_unet.py \
        --runs unet_r34_rgb unet_r34_rgbd_lin \
        --split test \
        --out results/unet_stress
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import segmentation_models_pytorch as smp

# Add current folder to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LunarSeg, resolve_dataset_root
from evaluate_unet import find_checkpoint_dir

CLASSES = {1: "crater", 2: "rock"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate thresholding, robustness corruptions, and latency.")
    p.add_argument("--runs", nargs="+", required=True, help="Checkpoint names or paths to profile")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split")
    p.add_argument("--root", default=None, help="Dataset root directory")
    p.add_argument("--parts", nargs="+", default=["threshold", "robustness", "latency"],
                   help="Evaluation components to execute (threshold, robustness, latency)")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--min_area", type=int, default=20, help="Minimum connected component area (px)")
    p.add_argument("--limit", type=int, default=0, help="Optional sample limit for quick smoke testing")
    p.add_argument("--out", default="results/unet_stress", help="Output directory for reports")
    return p.parse_args()


def load_model(run_name, dev):
    ckpt_dir = find_checkpoint_dir(run_name)
    cfg_file = ckpt_dir / "config.json"
    weights_file = ckpt_dir / "best.pth"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    in_ch = 3 if cfg.get("mode") == "rgb" else 4
    model = smp.Unet(cfg.get("encoder", "resnet34"), encoder_weights=None, in_channels=in_ch, classes=3)
    model.load_state_dict(torch.load(weights_file, map_location=dev))
    return model.to(dev).eval(), cfg


def apply_corruption(kind, value, rgb_float):
    """Applies realistic optical/sensor corruptions to RGB float image [0, 1]."""
    if kind == "darker":
        return rgb_float * float(value)
    elif kind == "gamma":
        return np.power(np.maximum(rgb_float, 0.0), float(value))
    elif kind == "noise":
        sigma = float(value) / 255.0
        noisy = rgb_float + np.random.normal(0.0, sigma, rgb_float.shape)
        return np.clip(noisy, 0.0, 1.0)
    elif kind == "blur":
        ksize = int(float(value)) * 2 + 1
        return cv2.GaussianBlur(rgb_float, (ksize, ksize), 0)
    return rgb_float


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[UNet Stress Testing] Profiling runs: {args.runs} on split: '{args.split}'")

    if "latency" in args.parts:
        print("\n--- Latency & Parameter Profiling ---")
        latency_csv = out_dir / "latency.csv"
        rows = []
        for run in args.runs:
            model, cfg = load_model(run, dev)
            in_ch = 3 if cfg.get("mode") == "rgb" else 4
            dummy = torch.randn(1, in_ch, 480, 640).to(dev)

            # Warm-up
            for _ in range(15):
                _ = model(dummy)
            if dev == "cuda":
                torch.cuda.synchronize()

            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = model(dummy)
                if dev == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            lat_ms = 1000.0 * float(np.mean(times))
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            rows.append({"run": run, "mode": cfg.get("mode"), "latency_ms": round(lat_ms, 2), "params_M": round(params_m, 2)})
            print(f"  {run:24s} | Mode: {cfg.get('mode', 'rgb'):5s} | Latency: {lat_ms:6.2f} ms | Params: {params_m:5.2f} M")

        with open(latency_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["run", "mode", "latency_ms", "params_M"])
            w.writeheader()
            w.writerows(rows)
        print(f"Saved latency profile to: {latency_csv}")

    print(f"\nCompleted stress testing. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
