#!/usr/bin/env python3
"""Fine-tune Depth Anything V2 (metric head) on LunarSeg so it predicts the GT depth directly.

Target = GT depth PNG / dtype max  (fixed encoding, 0-1, the same constant for every image).
The network ends in a sigmoid * max_depth with max_depth=1.0, so the output is already in the GT's units:
no per-image scale/shift fitting, and the output is depth (not disparity).

Subcommands
  train    fine-tune on train.txt, validate on validation.txt after every epoch (saves best.pth and last.pth)
  eval     metrics (no alignment) of a checkpoint on a split (default: test; also --split train / validation)
  predict  run a checkpoint on a folder of images and write .npy (float32, 0-1, original image size)

Dataset layout (LunarSeg with one folder per split)
  <root>/image/{train,validation,test}/<name>.png     RGB
  <root>/depth/{train,validation,test}/<name>.png     8-bit depth
  <root>/train.txt, validation.txt, test.txt          one <name> per line (give the folder with --split-dir)

Examples (run from repository root):
  python3 scripts/finetune_lunar.py train --encoder vits --pretrained Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth --out-dir runs/vits --epochs 20 --batch-size 2 --accum 8
  python3 scripts/finetune_lunar.py eval --ckpt runs/vits/best.pth --split test
  python3 scripts/finetune_lunar.py predict --ckpt runs/vits/best.pth --img-dir LunarSeg/image/test --out-dir preds_npy --save-png
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DAV2_REPO = ROOT_DIR / "Depth-Anything-V2"
DEFAULT_LUNAR_DIR = ROOT_DIR / "LunarSeg"


# Data loading and depth conversion helpers
def read_depth(path: Path) -> np.ndarray:
    """Depth PNG -> float32 in [0, 1] (divide by dtype max: 255 / 65535)."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read {path}")
    if arr.ndim == 3:
        arr = arr[..., 0] if (arr[..., 0] == arr[..., 1]).all() else cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    max_val = 255.0 if arr.dtype == np.uint8 else 65535.0 if arr.dtype == np.uint16 else float(max(arr.max(), 1))
    return arr.astype(np.float32) / max_val


def prep_image(bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return rgb


def to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((rgb01 - MEAN) / STD).transpose(2, 0, 1).copy())


def make_valid(depth, sky_thresh):
    valid = depth > 0
    if sky_thresh is not None:
        valid = valid & (depth < sky_thresh)
    return valid


def resolve_dirs(args):
    """-> (image_dir, depth_dir, split_dir). --image-dir/--depth-dir may be absolute, or relative to --data-dir."""
    dd = Path(args.data_dir).expanduser() if args.data_dir else None

    def pick(value, default):
        if value:
            p = Path(value).expanduser()
            return p if (p.is_absolute() or dd is None) else dd / p
        if dd is None:
            sys.exit("Give --data-dir, or give --image-dir, --depth-dir and --split-dir explicitly.")
        return dd / default

    image_dir = pick(args.image_dir, "image")
    depth_dir = pick(args.depth_dir, "depth")
    split_dir = Path(args.split_dir).expanduser() if args.split_dir else dd
    if split_dir is None:
        sys.exit("Give --split-dir (folder with train.txt / validation.txt / test.txt) or --data-dir.")
    return image_dir, depth_dir, split_dir


class LunarDepth(Dataset):
    def __init__(self, image_dir, depth_dir, split_dir, split, h, w, train, sky_thresh=None):
        self.image_dir = Path(str(image_dir).replace("{split}", split)).expanduser()
        self.depth_dir = Path(str(depth_dir).replace("{split}", split)).expanduser()
        self.split_dir = Path(split_dir).expanduser()
        txt_path = self.split_dir / f"{split}.txt"
        if not txt_path.exists():
            sys.exit(f"Split file not found: {txt_path}")
        self.names = [Path(l.strip()).stem for l in open(txt_path) if l.strip()]
        self.h, self.w, self.train, self.sky_thresh = h, w, train, sky_thresh
        missing = [n for n in self.names
                   if not (self.image_dir / f"{n}.png").exists() or not (self.depth_dir / f"{n}.png").exists()]
        if missing:
            sys.exit(f"[{split}] {len(missing)}/{len(self.names)} listed files are missing, e.g. {missing[:3]}\n"
                     f"  looked in image dir: {self.image_dir}\n  looked in depth dir: {self.depth_dir}\n"
                     f"  If each split has its own folder, pass e.g. --image-dir LunarSeg/image/{{split}} "
                     f"--depth-dir LunarSeg/depth/{{split}}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        bgr = cv2.imread(str(self.image_dir / f"{name}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Missing image for {name}")
        depth = read_depth(self.depth_dir / f"{name}.png")
        rgb = prep_image(bgr, self.h, self.w)
        if not self.train:
            return to_tensor(rgb), torch.from_numpy(depth), name
        d = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if random.random() < 0.5:
            rgb, d = rgb[:, ::-1], d[:, ::-1]
        rgb = np.clip(rgb * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)
        d = np.ascontiguousarray(d)
        return to_tensor(np.ascontiguousarray(rgb)), torch.from_numpy(d), torch.from_numpy(make_valid(d, self.sky_thresh))


# Model definition and dimension validation
def build_model(args):
    repo_path = Path(args.repo).expanduser()
    if not (repo_path / "metric_depth").exists():
        if (DEFAULT_DAV2_REPO / "metric_depth").exists():
            repo_path = DEFAULT_DAV2_REPO
        else:
            sys.exit(f"Could not locate metric_depth in repo path: {args.repo}")
    sys.path.insert(0, str(repo_path / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2
    return DepthAnythingV2(**CONFIGS[args.encoder], max_depth=1.0)


def check_size(h, w):
    if h % 14 or w % 14:
        sys.exit(f"--height/--width must be multiples of 14 (got {h}x{w}); e.g. 476x644 or 336x448")


# Loss functions and performance metrics
def l1_loss(pred, gt, valid):
    return (pred - gt).abs()[valid].mean()


def silog_loss(pred, gt, valid, lambd=0.5):
    d = torch.log(gt[valid].clamp_min(1e-4)) - torch.log(pred[valid].clamp_min(1e-4))
    return torch.sqrt((d ** 2).mean() - lambd * d.mean() ** 2 + 1e-12)


def metrics(pred, gt, mask):
    if mask.sum() < 10:
        return None
    p, g = pred[mask].clamp_min(1e-4), gt[mask]
    ratio = torch.maximum(p / g, g / p)
    return {"absrel": ((p - g).abs() / g).mean().item(),
            "rmse": ((p - g) ** 2).mean().sqrt().item(),
            "delta1": (ratio < 1.25).float().mean().item()}


@torch.no_grad()
def evaluate(model, loader, device, use_amp, terrain_thresh=0.99):
    model.eval()
    acc = {"all": [], "terrain": []}
    for img, gt, _ in loader:
        img, gt = img.to(device), gt.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(img)
        pred = F.interpolate(pred.float()[:, None], gt.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        for key, mask in (("all", gt > 0), ("terrain", (gt > 0) & (gt < terrain_thresh))):
            m = metrics(pred, gt, mask)
            if m:
                acc[key].append(m)
    return {k: {n: float(np.mean([m[n] for m in v])) for n in ("absrel", "rmse", "delta1")} if v else None
            for k, v in acc.items()}


def fmt(res):
    return " | ".join(f"{k}: AbsRel={v['absrel']:.4f} RMSE={v['rmse']:.4f} d1={v['delta1']:.4f}"
                      for k, v in res.items() if v)


# Training and evaluation command handlers
def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_train(args):
    check_size(args.height, args.width)
    device = pick_device()
    use_amp = device.type == "cuda" and not args.no_amp
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    model = build_model(args)
    if args.pretrained:
        sd = torch.load(args.pretrained, map_location="cpu")
        if not args.load_all:
            sd = {k: v for k, v in sd.items() if "pretrained" in k}
        res = model.load_state_dict(sd, strict=False)
        print(f"Loaded {len(sd)} tensors from {args.pretrained} (missing {len(res.missing_keys)}, "
              f"unexpected {len(res.unexpected_keys)})")
    else:
        print("WARNING: no --pretrained checkpoint given: training from random initialisation.")
    model.to(device)

    enc = [p for n, p in model.named_parameters() if "pretrained" in n]
    dec = [p for n, p in model.named_parameters() if "pretrained" not in n]
    if args.freeze_encoder:
        for p in enc:
            p.requires_grad = False
        groups = [{"params": dec, "lr": args.lr * 10}]
    else:
        groups = [{"params": enc, "lr": args.lr}, {"params": dec, "lr": args.lr * 10}]
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = l1_loss if args.loss == "l1" else silog_loss

    image_dir, depth_dir, split_dir = resolve_dirs(args)
    train_ds = LunarDepth(image_dir, depth_dir, split_dir, "train", args.height, args.width, True, args.sky_thresh)
    val_ds = LunarDepth(image_dir, depth_dir, split_dir, "validation", args.height, args.width, False, None)
    if args.max_samples:
        train_ds.names = train_ds.names[: args.max_samples]
        val_ds.names = val_ds.names[: args.max_samples]
    tl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=device.type == "cuda")
    vl = DataLoader(val_ds, 1, shuffle=False, num_workers=args.workers)

    steps_per_epoch = max(1, len(tl) // args.accum)
    total_steps = args.epochs * steps_per_epoch
    (out / "config.json").write_text(json.dumps({"encoder": args.encoder, "height": args.height, "width": args.width,
                                                  "max_depth": 1.0, "target": "GT PNG / dtype max"}, indent=2))
    print(f"train {len(train_ds)} imgs, val {len(val_ds)} imgs, {steps_per_epoch} optimizer steps/epoch, amp={use_amp}")

    step, best = 0, float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for i, (img, depth, valid) in enumerate(tl):
            img, depth, valid = img.to(device), depth.to(device), valid.to(device)
            if valid.sum() == 0:
                continue
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(img)
            loss = loss_fn(pred.float(), depth, valid)
            scaler.scale(loss / args.accum).backward()
            run += loss.item()
            if (i + 1) % args.accum == 0:
                factor = (1 - step / total_steps) ** 0.9
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * factor
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            if (i + 1) % args.log_every == 0:
                print(f"epoch {epoch + 1}/{args.epochs} iter {i + 1}/{len(tl)} loss {run / args.log_every:.4f}")
                run = 0.0
        res = evaluate(model, vl, device, use_amp)
        score = res[args.select]["absrel"] if res[args.select] else float("inf")
        print(f"== epoch {epoch + 1} ({time.time() - t0:.0f}s) val {fmt(res)}")
        if device.type == "cuda":
            print(f"   peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        torch.save(model.state_dict(), out / "last.pth")
        if score < best:
            best = score
            torch.save(model.state_dict(), out / "best.pth")
            print(f"   new best (val {args.select} AbsRel {best:.4f}) -> {out / 'best.pth'}")


def load_for_inference(args):
    cfg_path = Path(args.ckpt).parent / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        args.encoder = cfg.get("encoder", args.encoder)
        args.height = cfg.get("height", args.height)
        args.width = cfg.get("width", args.width)
    check_size(args.height, args.width)
    device = pick_device()
    model = build_model(args)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    return model.to(device).eval(), device


def cmd_eval(args):
    model, device = load_for_inference(args)
    image_dir, depth_dir, split_dir = resolve_dirs(args)
    ds = LunarDepth(image_dir, depth_dir, split_dir, args.split, args.height, args.width, False, None)
    if args.max_samples:
        ds.names = ds.names[: args.max_samples]
    res = evaluate(model, DataLoader(ds, 1, num_workers=args.workers), device, device.type == "cuda" and not args.no_amp)
    print(f"{args.split} ({len(ds)} imgs), no alignment: {fmt(res)}")


def load_split_map(split_dir: Path | None = None) -> dict[str, str]:
    candidates = []
    if split_dir and Path(split_dir).is_dir():
        candidates.append(Path(split_dir))
    candidates.extend([DEFAULT_LUNAR_DIR, Path("LunarSeg"), Path(".")])
    target_dir = next((c for c in candidates if c.exists() and (c / "train.txt").exists()), None)
    if not target_dir:
        return {}
    mapping = {}
    for s in ("train", "validation", "test"):
        txt = target_dir / f"{s}.txt"
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    mapping[Path(line).name] = s
                    mapping[Path(line).stem] = s
    return mapping


def resolve_split(file_path: Path, split_map: dict[str, str]) -> str | None:
    for p in reversed(file_path.parent.parts):
        low = p.lower()
        if low in ("train", "test", "validation"):
            return low
        if low in ("val", "valid"):
            return "validation"
    if split_map:
        if file_path.name in split_map:
            return split_map[file_path.name]
        if file_path.stem in split_map:
            return split_map[file_path.stem]
        m = re.search(r"\d{2}_\d{4}", file_path.stem)
        if m and m.group(0) in split_map:
            return split_map[m.group(0)]
        for key, s in split_map.items():
            if key in file_path.stem:
                return s
    return None


@torch.no_grad()
def cmd_predict(args):
    model, device = load_for_inference(args)
    use_amp = device.type == "cuda" and not args.no_amp
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)
    files = sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit(f"No images in {args.img_dir}")

    split_map = {} if getattr(args, "no_auto_split", False) else load_split_map(getattr(args, "split_dir", None))
    counts = {"train": 0, "validation": 0, "test": 0, "root": 0}

    for k, f in enumerate(files):
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[skip] {f.name}")
            continue
        h0, w0 = bgr.shape[:2]
        x = to_tensor(prep_image(bgr, args.height, args.width))[None].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(x)
        pred = F.interpolate(pred.float()[:, None], (h0, w0), mode="bilinear", align_corners=False)[0, 0]
        pred = pred.clamp(0, 1).cpu().numpy().astype(np.float32)

        split = None if getattr(args, "no_auto_split", False) else resolve_split(f, split_map)
        if split:
            target_dir = out / split
            counts[split] = counts.get(split, 0) + 1
        else:
            target_dir = out
            counts["root"] = counts.get("root", 0) + 1

        target_dir.mkdir(parents=True, exist_ok=True)
        np.save(target_dir / f"{f.stem}.npy", pred)
        if args.save_png:
            cv2.imwrite(str(target_dir / f"{f.stem}.png"), np.round(pred * 255).astype(np.uint8))
        if (k + 1) % 50 == 0 or k + 1 == len(files):
            print(f"{k + 1}/{len(files)}")

    summary = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0)
    print(f"Done: {len(files)} image(s) -> {out} ({summary})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, data=True):
        default_repo = "Depth-Anything-V2" if DEFAULT_DAV2_REPO.exists() else "."
        default_split = "LunarSeg" if DEFAULT_LUNAR_DIR.exists() else "."
        p.add_argument("--repo", default=default_repo, help="Path to Depth-Anything-V2 repo")
        p.add_argument("--encoder", choices=list(CONFIGS), default="vits")
        p.add_argument("--height", type=int, default=476, help="network input height (multiple of 14)")
        p.add_argument("--width", type=int, default=644, help="network input width (multiple of 14)")
        p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
        p.add_argument("--workers", type=int, default=4)
        if data:
            p.add_argument("--data-dir", type=Path, default=None,
                           help="Optional root holding image/, depth/ and the split .txt files")
            p.add_argument("--image-dir", default="LunarSeg/image/{split}",
                           help="RGB folder: absolute, or relative to --data-dir (default: LunarSeg/image/{split}).")
            p.add_argument("--depth-dir", default="LunarSeg/depth/{split}",
                           help="Depth folder (default: LunarSeg/depth/{split})")
            p.add_argument("--split-dir", default=default_split,
                           help="Folder with train.txt/validation.txt/test.txt (default: LunarSeg)")
            p.add_argument("--max-samples", type=int, default=0, help="debug: limit images per split")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--pretrained", help="Relative-depth checkpoint, e.g. Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth")
    t.add_argument("--load-all", action="store_true", help="load decoder weights too (default: encoder only)")
    t.add_argument("--out-dir", type=Path, default=Path("runs/lunar"))
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--accum", type=int, default=8, help="gradient accumulation steps (effective batch = batch*accum)")
    t.add_argument("--lr", type=float, default=5e-6, help="encoder lr (decoder uses 10x)")
    t.add_argument("--loss", choices=["l1", "silog"], default="l1")
    t.add_argument("--sky-thresh", type=float, default=None,
                   help="exclude GT >= this from the TRAINING loss (default: sky included)")
    t.add_argument("--select", choices=["all", "terrain"], default="all", help="metric used to pick best.pth")
    t.add_argument("--freeze-encoder", action="store_true", help="train only the decoder (saves memory)")
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval")
    common(e)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default="test")
    e.set_defaults(fn=cmd_eval)

    p = sub.add_parser("predict")
    common(p, data=False)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--save-png", action="store_true", help="also write 8-bit PNG (value*255) like the GT")
    p.add_argument("--split-dir", type=Path, default=None, help="Directory containing split txt files")
    p.add_argument("--no-auto-split", action="store_true", help="Disable automatic split folder organization")
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
