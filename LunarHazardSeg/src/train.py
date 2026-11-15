"""Train U-Net for lunar terrain hazard segmentation.

Supported Modes:
    --mode rgb   : 3-channel RGB imagery
    --mode rgbd  : 4-channel multi-modal (RGB + ground truth depth)
    --mode rgbpd : 4-channel multi-modal (RGB + predicted depth from Depth Anything V2)

Examples:
    python -m src.train --mode rgb   --name unet_r34_rgb
    python -m src.train --mode rgbd  --name unet_r34_rgbd
    python -m src.train --mode rgbpd --pred_depth_dir ../outputs/dav2_lunar_preds --name unet_r34_rgbpd
"""
import argparse
import csv
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.dataset import LunarSeg, resolve_dataset_root
from src.metrics import ConfusionMeter

p = argparse.ArgumentParser(description="Train UNet for lunar terrain hazard segmentation.")
p.add_argument("--root", default=None, help="Path to LunarSeg dataset (default: auto-detects LunarSeg or data)")
p.add_argument("--out_dir", default="checkpoints", help="Output directory for checkpoints (default: checkpoints)")
p.add_argument("--mode", default="rgb", choices=["rgb", "rgbd", "rgbpd"], help="Input modality mode")
p.add_argument("--pred_depth_dir", default=None, help="Directory containing predicted depth files for rgbpd mode")
p.add_argument("--gt_linear", action="store_true", help="rgbd: feed true depth as depth/255 (far = large) instead of inverse depth")
p.add_argument("--pred_invert", action="store_true", help="prediction files hold depth (far = large); invert to disparity")
p.add_argument("--encoder", default="resnet34", help="Backbone encoder architecture (default: resnet34)")
p.add_argument("--weights", default="imagenet", help="Pretrained encoder weights ('imagenet' or 'none')")
p.add_argument("--loss", default="ce", choices=["ce", "ce_lovasz"], help="Loss function")
p.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
p.add_argument("--bs", type=int, default=8, help="Batch size")
p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
p.add_argument("--seed", type=int, default=0, help="Random seed")
p.add_argument("--select", default="mIoU", choices=["mIoU", "hazard_mIoU"], help="Metric to select best checkpoint")
p.add_argument("--name", required=True, help="Run name identifier")
p.add_argument("--max_batches", type=int, default=0, help="debug: limit batches per epoch (0 for full epoch)")
args = p.parse_args()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model(encoder, in_ch, weights):
    w = None if weights == "none" else weights
    return smp.Unet(encoder, encoder_weights=w, in_channels=in_ch, classes=3)


def main():
    seed_all(args.seed)
    args.root = resolve_dataset_root(args.root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)

    kw = dict(
        root=args.root,
        mode=args.mode,
        pred_depth_dir=args.pred_depth_dir,
        pred_invert=args.pred_invert,
        gt_linear=args.gt_linear,
    )
    tr = LunarSeg(split="train", augment=True, **kw)
    va = LunarSeg(split="validation", **kw)
    g = torch.Generator(); g.manual_seed(args.seed)
    tl = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, pin_memory=True,
                    drop_last=True, generator=g)
    vl = DataLoader(va, args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    in_ch = 3 if args.mode == "rgb" else 4
    model = build_model(args.encoder, in_ch, args.weights).to(dev)
    ce = nn.CrossEntropyLoss()
    lovasz = smp.losses.LovaszLoss(mode="multiclass")
    crit = (lambda o, t: ce(o, t)) if args.loss == "ce" else (lambda o, t: ce(o, t) + lovasz(o, t))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))

    log = open(os.path.join(out, "log.csv"), "w", newline="")
    writer = None
    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for i, (x, y, _) in enumerate(tl):
            if args.max_batches and i >= args.max_batches:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()

        model.eval(); cm = ConfusionMeter(3); vloss = 0.0; vb = 0
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
            for i, (x, y, _) in enumerate(vl):
                if args.max_batches and i >= args.max_batches:
                    break
                x, y = x.to(dev), y.to(dev)
                o = model(x); vloss += crit(o, y).item(); vb += 1
                cm.update(o.float(), y)
        r = cm.results()
        row = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vloss / max(vb, 1),
               "time_s": round(time.time() - t0, 1), **r}
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row)); writer.writeheader()
        writer.writerow(row); log.flush()
        print(f"ep {ep:3d} | loss {row['train_loss']:.4f}/{row['val_loss']:.4f} | "
              f"mIoU {r['mIoU']:.4f} | crater IoU {r['IoU_negative']:.4f} rec {r['recall_negative']:.4f} | "
              f"rock IoU {r['IoU_positive']:.4f} | {row['time_s']}s")
        if r[args.select] > best:
            best = r[args.select]
            torch.save(model.state_dict(), os.path.join(out, "best.pth"))
            json.dump({"epoch": ep, **r}, open(os.path.join(out, "best_val.json"), "w"), indent=2)
    print(f"done. best val {args.select} = {best:.4f}  -> {out}/best.pth")


if __name__ == "__main__":
    main()