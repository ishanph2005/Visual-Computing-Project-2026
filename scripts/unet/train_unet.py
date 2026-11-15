#!/usr/bin/env python3
"""Train a U-Net model on the LunarSeg benchmark dataset.

Supports three input modalities:
    1. rgb   : Standard 3-channel RGB surface images
    2. rgbd  : 4-channel input combining RGB + ground-truth depth
    3. rgbpd : 4-channel input combining RGB + Depth Anything V2 predicted depth

Examples:
    # 1. Standard RGB baseline:
    python3 scripts/unet/train_unet.py --mode rgb --name unet_r34_rgb

    # 2. Multi-modal with Ground Truth Depth (linear normalized):
    python3 scripts/unet/train_unet.py --mode rgbd --name unet_r34_rgbd_lin --gt_linear

    # 3. Multi-modal with Depth Anything V2 Predicted Depth:
    python3 scripts/unet/train_unet.py --mode rgbpd --pred_depth_dir outputs/dav2_lunar_preds --name unet_r34_rgbpd_ft
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Add current folder to sys.path so dataset and metrics import cleanly
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LunarSeg, resolve_dataset_root
from metrics import ConfusionMeter


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for lunar terrain hazard segmentation.")
    p.add_argument("--root", default=None, help="Dataset root directory (default: auto-detect LunarSeg)")
    p.add_argument("--out_dir", default="runs/unet", help="Base directory for saved checkpoints (default: runs/unet)")
    p.add_argument("--mode", default="rgb", choices=["rgb", "rgbd", "rgbpd"], help="Input modality mode")
    p.add_argument("--pred_depth_dir", default="outputs/dav2_lunar_preds",
                   help="Folder containing predicted depth maps for rgbpd mode (e.g. outputs/dav2_lunar_preds)")
    p.add_argument("--gt_linear", action="store_true",
                   help="rgbd: feed true depth as linear depth/255 (far = 1) instead of inverse depth")
    p.add_argument("--pred_invert", action="store_true",
                   help="prediction files hold physical depth; invert to disparity (near = 1)")
    p.add_argument("--encoder", default="resnet34", help="Backbone encoder architecture (default: resnet34)")
    p.add_argument("--weights", default="imagenet", help="Pretrained encoder weights ('imagenet' or 'none')")
    p.add_argument("--loss", default="ce", choices=["ce", "ce_lovasz"], help="Loss function")
    p.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    p.add_argument("--bs", type=int, default=8, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate for AdamW")
    p.add_argument("--workers", type=int, default=4, help="DataLoader background worker processes")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--select", default="mIoU", choices=["mIoU", "hazard_mIoU"],
                   help="Validation metric used to select best checkpoint")
    p.add_argument("--name", default="unet_r34_rgb", help="Unique run name identifier")
    p.add_argument("--max_batches", type=int, default=0,
                   help="Debug helper: limit batches per epoch (0 runs full dataset)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(encoder, in_channels, weights):
    pretrained = None if weights == "none" else weights
    return smp.Unet(encoder, encoder_weights=pretrained, in_channels=in_channels, classes=3)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.root = resolve_dataset_root(args.root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(args.out_dir) / args.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Initialize datasets for training and validation splits
    kw = dict(
        root=args.root,
        mode=args.mode,
        pred_depth_dir=args.pred_depth_dir,
        pred_invert=args.pred_invert,
        gt_linear=args.gt_linear,
    )
    train_dataset = LunarSeg(split="train", augment=True, **kw)
    val_dataset = LunarSeg(split="validation", augment=False, **kw)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(dev == "cuda"),
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(dev == "cuda"),
    )

    in_ch = 3 if args.mode == "rgb" else 4
    model = build_model(args.encoder, in_ch, args.weights).to(dev)

    # Define optimization criterion
    ce_loss = nn.CrossEntropyLoss()
    if args.loss == "ce":
        criterion = lambda outputs, targets: ce_loss(outputs, targets)
    else:
        lovasz_loss = smp.losses.LovaszLoss(mode="multiclass")
        criterion = lambda outputs, targets: ce_loss(outputs, targets) + lovasz_loss(outputs, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))

    log_path = checkpoint_dir / "log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = None
    best_score = -1.0

    print(f"\n[UNet Training] Run: {args.name} | Modality: {args.mode} ({in_ch} channels) | Encoder: {args.encoder}")
    print(f"Device: {dev} | Epochs: {args.epochs} | Batch size: {args.bs} | Checkpoint dir: {checkpoint_dir}\n")

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images = images.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()

        # Validation phase
        model.eval()
        val_meter = ConfusionMeter(3)
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
            for batch_idx, (images, targets, _) in enumerate(val_loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                images = images.to(dev)
                targets = targets.to(dev)
                outputs = model(images)
                val_loss_sum += criterion(outputs, targets).item()
                val_batches += 1
                val_meter.update(outputs.float(), targets)

        val_results = val_meter.results()
        avg_train_loss = train_loss_sum / max(train_batches, 1)
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        elapsed = round(time.time() - start_time, 1)

        row = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "time_s": elapsed,
            **val_results,
        }

        if writer is None:
            writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)
        log_file.flush()

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Loss: {avg_train_loss:.4f} / {avg_val_loss:.4f} | "
            f"mIoU: {val_results['mIoU']:.4f} | "
            f"Crater IoU: {val_results['IoU_negative']:.4f} | "
            f"Rock IoU: {val_results['IoU_positive']:.4f} | "
            f"{elapsed}s"
        )

        current_score = val_results[args.select]
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), checkpoint_dir / "best.pth")
            with open(checkpoint_dir / "best_val.json", "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, **val_results}, f, indent=2)

    log_file.close()
    print(f"\nTraining completed. Best validation {args.select}: {best_score:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
