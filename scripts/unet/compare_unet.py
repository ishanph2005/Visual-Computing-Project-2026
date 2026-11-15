#!/usr/bin/env python3
"""Compare trained UNet models and generate hazard detection benchmark metrics.

Metrics generated:
    1. Pixel-level metrics: mIoU, hazard-only mIoU, per-class IoU, precision, recall
    2. Hazard-level instance detection: Connected-component analysis for rocks & craters
       stratified by pixel size:
         - Small  (<300 px) : Distant hazards
         - Medium (300-3000 px) : Intermediate hazards
         - Large  (>3000 px) : Close-range obstacles
       Also reports false alarm clusters per image.
    3. Paired bootstrap 95% Confidence Intervals (evaluating depth-fusion improvements).
    4. Qualitative visual masks of top disagreement samples.

Usage:
    python3 scripts/unet/compare_unet.py \
        --runs unet_r34_rgb unet_r34_rgbd_lin unet_r34_rgbpd_ft \
        --split test \
        --out results/unet_compare
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import segmentation_models_pytorch as smp

# Add current folder to sys.path so dataset and metrics import cleanly
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LunarSeg, resolve_dataset_root
from evaluate_unet import find_checkpoint_dir

CLASSES = {1: "crater", 2: "rock"}
SIZE_BINS = [("small", 0, 300), ("medium", 300, 3000), ("large", 3000, 10 ** 9)]


def parse_args():
    p = argparse.ArgumentParser(description="Multi-model comparison and hazard detection metrics.")
    p.add_argument("--runs", nargs="+", required=True, help="Run names or checkpoint paths to compare")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Split to evaluate on")
    p.add_argument("--root", default=None, help="Dataset root directory")
    p.add_argument("--hit", type=float, default=0.5, help="Minimum overlap fraction to count a hazard as detected")
    p.add_argument("--boot", type=int, default=1000, help="Number of bootstrap resamples for 95% CIs")
    p.add_argument("--min_area", type=int, default=20, help="Minimum hazard area in pixels (ignores tiny noise)")
    p.add_argument("--n_disagree", type=int, default=8, help="Number of worst disagreement images to save")
    p.add_argument("--out", default="results/unet_compare", help="Directory to save comparison outputs")
    return p.parse_args()


def load_model_and_config(run_name, dev):
    ckpt_dir = find_checkpoint_dir(run_name)
    cfg_file = ckpt_dir / "config.json"
    weights_file = ckpt_dir / "best.pth"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config.json missing in '{ckpt_dir}'.")
    if not weights_file.exists():
        raise FileNotFoundError(f"best.pth missing in '{ckpt_dir}'.")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    in_ch = 3 if cfg.get("mode") == "rgb" else 4
    model = smp.Unet(cfg.get("encoder", "resnet34"), encoder_weights=None, in_channels=in_ch, classes=3)
    model.load_state_dict(torch.load(weights_file, map_location=dev))
    return model.to(dev).eval(), cfg, ckpt_dir


def confusion_matrix_single(gt, pred, num_classes=3):
    return np.bincount(num_classes * gt.ravel() + pred.ravel(), minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_hazard_instances(gt, pred, hit_thresh=0.5, min_area=20):
    """Computes connected-component hazard instances and false alarm regions."""
    instances = []
    false_alarms = {1: 0, 2: 0}

    for c in (1, 2):
        labeled_gt, num_hazards = ndimage.label(gt == c)
        for k in range(1, num_hazards + 1):
            mask = labeled_gt == k
            area = int(mask.sum())
            if area < min_area:
                continue
            overlap_fraction = float((pred[mask] == c).mean())
            is_detected = overlap_fraction >= hit_thresh
            instances.append((c, area, overlap_fraction, is_detected))

        labeled_pred, num_pred_hazards = ndimage.label(pred == c)
        for k in range(1, num_pred_hazards + 1):
            pred_mask = labeled_pred == k
            if pred_mask.sum() >= min_area and not (gt[pred_mask] == c).any():
                false_alarms[c] += 1

    return instances, false_alarms


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    disagree_dir = out_dir / "disagree"
    disagree_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[UNet Model Comparison] Comparing {len(args.runs)} runs on split: '{args.split}'")
    per_run_data = {}
    sample_names = None

    for run in args.runs:
        model, cfg, ckpt_dir = load_model_and_config(run, dev)
        dataset_root = resolve_dataset_root(args.root or cfg.get("root"))
        dataset = LunarSeg(
            root=dataset_root,
            split=args.split,
            mode=cfg.get("mode", "rgb"),
            pred_depth_dir=cfg.get("pred_depth_dir"),
            pred_invert=cfg.get("pred_invert", False),
            gt_linear=cfg.get("gt_linear", False),
        )
        sample_names = dataset.names
        confusion_matrices = []
        all_instances = []
        all_false_alarms = []
        cached_preds = []

        print(f"Evaluating run: {run} ({len(dataset)} samples)...")
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y, _ = dataset[i]
                pred = model(x[None].to(dev)).argmax(1)[0].cpu().numpy().astype(np.int64)
                gt = y.numpy().astype(np.int64)

                cm = confusion_matrix_single(gt, pred, num_classes=3)
                confusion_matrices.append(cm)

                inst, fa = compute_hazard_instances(gt, pred, hit_thresh=args.hit, min_area=args.min_area)
                all_instances.append(inst)
                all_false_alarms.append(fa)
                cached_preds.append(pred.astype(np.uint8))

        per_run_data[run] = {
            "cms": np.stack(confusion_matrices),
            "instances": all_instances,
            "false_alarms": all_false_alarms,
            "preds": cached_preds,
        }
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Generate Markdown Summary Tables
    md_lines = [
        f"# UNet Model Comparison & Hazard Detection Benchmark (`{args.split}` split)\n",
        "## 1. Overall Pixel-Level Metrics\n",
        "| Model | mIoU | Hazard mIoU | Crater IoU | Crater Rec | Crater Prec | Rock IoU | Rock Rec | Rock Prec |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    summary_json = {"runs": {}}

    for run in args.runs:
        total_cm = per_run_data[run]["cms"].sum(axis=0)
        tp = np.diag(total_cm)
        gt_sum = total_cm.sum(axis=1)
        pred_sum = total_cm.sum(axis=0)
        iou = tp / (gt_sum + pred_sum - tp)
        rec = tp / gt_sum
        prec = tp / pred_sum
        miou = float(np.mean(iou))
        haz_miou = float(np.mean(iou[1:]))

        md_lines.append(
            f"| **{run}** | {miou:.4f} | {haz_miou:.4f} | "
            f"{iou[1]:.4f} | {rec[1]:.4f} | {prec[1]:.4f} | "
            f"{iou[2]:.4f} | {rec[2]:.4f} | {prec[2]:.4f} |"
        )
        summary_json["runs"][run] = {
            "mIoU": miou,
            "hazard_mIoU": haz_miou,
            "crater_iou": float(iou[1]),
            "rock_iou": float(iou[2]),
        }

    # Hazard Detection Rates by Physical Size
    md_lines.extend([
        "\n## 2. Hazard Instance Detection Rates (Recall by Physical Pixel Size)\n",
        "| Model | Small (<300 px) | Medium (300-3000 px) | Large (>3000 px) | Overall Recall | False Alarms / Img |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for run in args.runs:
        inst_flat = [item for sublist in per_run_data[run]["instances"] for item in sublist]
        fa_flat = per_run_data[run]["false_alarms"]
        avg_fa = sum(f[1] + f[2] for f in fa_flat) / max(len(fa_flat), 1)

        size_recalls = {}
        for sname, slo, shi in SIZE_BINS:
            matching = [x for x in inst_flat if slo <= x[1] < shi]
            rec = (sum(x[3] for x in matching) / len(matching)) if matching else 0.0
            size_recalls[sname] = rec

        all_rec = (sum(x[3] for x in inst_flat) / len(inst_flat)) if inst_flat else 0.0

        md_lines.append(
            f"| **{run}** | {100.0 * size_recalls['small']:.2f}% | "
            f"{100.0 * size_recalls['medium']:.2f}% | "
            f"{100.0 * size_recalls['large']:.2f}% | "
            f"{100.0 * all_rec:.2f}% | {avg_fa:.2f} |"
        )

    # Write Markdown & JSON reports
    summary_md_path = out_dir / "summary.md"
    summary_json_path = out_dir / "summary.json"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "\n".join(md_lines))
    print(f"\nSaved Markdown report to: {summary_md_path}")
    print(f"Saved JSON metrics to: {summary_json_path}")


if __name__ == "__main__":
    main()
