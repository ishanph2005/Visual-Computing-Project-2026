#!/usr/bin/env python3
"""Evaluate a trained UNet model checkpoint on the LunarSeg benchmark split.

Computes:
    - Overall mIoU and hazard-only mIoU
    - Per-class IoU, precision, and recall (background, crater, rock)
    - Average per-frame inference latency (ms)
    - Model parameter count (M)
    - Qualitative side-by-side visual comparisons (Ground Truth vs Prediction)

Usage:
    # Evaluate by checkpoint folder name:
    python3 scripts/unet/evaluate_unet.py --name unet_r34_rgb --split test

    # Or specify direct path to checkpoint folder:
    python3 scripts/unet/evaluate_unet.py --ckpt checkpoints/unet_r34_rgb --split test
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import segmentation_models_pytorch as smp

# Add current folder to sys.path so dataset and metrics import cleanly
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LunarSeg, resolve_dataset_root
from metrics import ConfusionMeter


def find_checkpoint_dir(target_name):
    """Finds checkpoint directory across common project output locations."""
    target_path = Path(target_name)
    if target_path.is_dir() and (target_path / "config.json").exists():
        return target_path

    candidates = [
        Path("checkpoints") / target_name,
        Path("runs/unet") / target_name,
        SCRIPT_DIR.parent.parent / "checkpoints" / target_name,
        SCRIPT_DIR.parent.parent / "runs" / "unet" / target_name,
    ]
    for c in candidates:
        if c.is_dir() and (c / "config.json").exists():
            return c
    return target_path


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained UNet checkpoint.")
    p.add_argument("--name", default=None, help="Checkpoint run name (e.g. unet_r34_rgb)")
    p.add_argument("--ckpt", default=None, help="Explicit path to checkpoint directory")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split to evaluate")
    p.add_argument("--root", default=None, help="Dataset root directory (default: auto-detect LunarSeg)")
    p.add_argument("--n_vis", type=int, default=8, help="Number of qualitative prediction images to export (0 to skip)")
    p.add_argument("--vis_dir", default=None, help="Directory to save visual predictions (default: <ckpt>/vis)")
    p.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size (default 1 for latency benchmark)")
    p.add_argument("--workers", type=int, default=4, help="DataLoader background workers")
    return p.parse_args()


def main():
    args = parse_args()
    run_identifier = args.ckpt or args.name
    if not run_identifier:
        raise ValueError("Please provide either --name (e.g. unet_r34_rgb) or --ckpt (path to checkpoint).")

    ckpt_dir = find_checkpoint_dir(run_identifier)
    cfg_file = ckpt_dir / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file 'config.json' not found in '{ckpt_dir}'.")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    weights_file = ckpt_dir / "best.pth"
    if not weights_file.exists():
        raise FileNotFoundError(f"Trained weights 'best.pth' not found in '{ckpt_dir}'.")

    dataset_root = resolve_dataset_root(args.root or cfg.get("root"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 3 if cfg.get("mode") == "rgb" else 4

    print(f"\n[UNet Evaluation] Checkpoint: {ckpt_dir} | Split: {args.split}")
    print(f"Modality: {cfg.get('mode', 'rgb')} ({in_ch} channels) | Encoder: {cfg.get('encoder', 'resnet34')} | Device: {dev}")

    # Build model and load trained state
    model = smp.Unet(cfg.get("encoder", "resnet34"), encoder_weights=None, in_channels=in_ch, classes=3)
    model.load_state_dict(torch.load(weights_file, map_location=dev))
    model.to(dev).eval()

    dataset = LunarSeg(
        root=dataset_root,
        split=args.split,
        mode=cfg.get("mode", "rgb"),
        pred_depth_dir=cfg.get("pred_depth_dir"),
        pred_invert=cfg.get("pred_invert", False),
        gt_linear=cfg.get("gt_linear", False),
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    meter = ConfusionMeter(3)
    frame_times = []
    vis_dir = Path(args.vis_dir) if args.vis_dir else (ckpt_dir / "vis")
    if args.n_vis > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Visualization color palette: Black = Background, Red = Crater, Green = Rock
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)

    with torch.no_grad():
        for i, (images, targets, sample_names) in enumerate(data_loader):
            images = images.to(dev)
            targets = targets.to(dev)

            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(images)
            if dev == "cuda":
                torch.cuda.synchronize()

            if i >= 10:  # Skip warm-up iterations for accurate latency measurement
                frame_times.append(time.perf_counter() - t0)

            meter.update(logits, targets)

            # Export qualitative overlays
            if i < args.n_vis:
                pred_mask = logits.argmax(1)[0].cpu().numpy()
                gt_mask = targets[0].cpu().numpy()
                side_by_side = np.concatenate([palette[gt_mask], palette[pred_mask]], axis=1)
                Image.fromarray(side_by_side).save(vis_dir / f"{sample_names[0]}_gt_vs_pred.png")

    results = meter.results()
    results["latency_ms"] = (1000.0 * float(np.mean(frame_times))) if frame_times else None
    results["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6

    # Save results to JSON
    output_results_file = ckpt_dir / f"{args.split}_results.json"
    with open(output_results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- Evaluation Summary ---")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:22s}: {val:.4f}")
        else:
            print(f"  {key:22s}: {val}")

    print(f"\nSaved results to: {output_results_file}")
    if args.n_vis > 0:
        print(f"Exported {min(args.n_vis, len(dataset))} visual comparisons to: {vis_dir}")


if __name__ == "__main__":
    main()
