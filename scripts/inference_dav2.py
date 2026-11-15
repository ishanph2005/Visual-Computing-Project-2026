#!/usr/bin/env python3
"""Run a fine-tuned Depth Anything V2 (metric head) checkpoint on a folder of images.

For every image the script writes:
  <out_dir>/[<split>/]<name>.npy   float32 depth map, same size as the input image, values in [0, 1]
  <out_dir>/[<split>/]<name>.png   8-bit grayscale preview of the same map (value * 255)

Outputs are automatically organized into train / validation / test subdirectories
based on directory structure or LunarSeg split definition files.

Usage:
    python3 scripts/inference_dav2.py --ckpt runs/vits/best.pth \
        --img-dir LunarSeg/image/test --out-dir outputs/preds [--minmax]
"""
import argparse
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DAV2_REPO = ROOT_DIR / "Depth-Anything-V2"
DEFAULT_LUNAR_DIR = ROOT_DIR / "LunarSeg" if (ROOT_DIR / "LunarSeg").exists() else Path("LunarSeg")


def load_split_map(split_dir: Path | None = None) -> dict[str, str]:
    """Build a mapping of filename/stem -> split name from train.txt, validation.txt, test.txt."""
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
    """Determine the split subdirectory for file_path."""
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


def preprocess(bgr, h, w):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def main():
    default_repo = "Depth-Anything-V2" if DEFAULT_DAV2_REPO.exists() else "."
    ap = argparse.ArgumentParser(
        description="Fine-tuned Depth Anything V2 inference with automatic split organization."
    )
    ap.add_argument("--repo", default=default_repo, help="Path to Depth-Anything-V2 repo")
    ap.add_argument("--ckpt", required=True, type=Path, help="Fine-tuned checkpoint (.pth)")
    ap.add_argument("--img-dir", required=True, type=Path, help="Folder with input images")
    ap.add_argument("--out-dir", required=True, type=Path, help="Folder for .npy and .png outputs")
    ap.add_argument("--encoder", choices=list(CONFIGS), default=None, help="Default: from config.json, else vits")
    ap.add_argument("--height", type=int, default=None, help="Network input height (multiple of 14)")
    ap.add_argument("--width", type=int, default=None, help="Network input width (multiple of 14)")
    ap.add_argument("--max-depth", type=float, default=None, help="Head scale (default: 1.0)")
    ap.add_argument("--minmax", action="store_true", help="Per-image min-max stretch to exactly 0-1")
    ap.add_argument("--split-dir", type=Path, default=None, help="Folder with train.txt/validation.txt/test.txt")
    ap.add_argument("--no-auto-split", action="store_true", help="Disable automatic split folder organization")
    ap.add_argument("--recursive", action="store_true", help="Search sub-folders recursively")
    ap.add_argument("--no-png", action="store_true", help="Only write .npy files")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true", help="Disable mixed precision on GPU")
    args = ap.parse_args()

    cfg = {}
    cfg_path = args.ckpt.parent / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    encoder = args.encoder or cfg.get("encoder", "vits")
    h = args.height or cfg.get("height", 476)
    w = args.width or cfg.get("width", 644)
    max_depth = args.max_depth if args.max_depth is not None else cfg.get("max_depth", 1.0)
    if h % 14 or w % 14:
        sys.exit(f"--height/--width must be multiples of 14 (got {h}x{w})")
    if not args.img_dir.is_dir():
        sys.exit(f"Input directory not found: {args.img_dir}")

    pattern = args.img_dir.rglob("*") if args.recursive else args.img_dir.iterdir()
    files = sorted(p for p in pattern if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        files = sorted(p for p in args.img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit(f"No images found in {args.img_dir}")

    repo_path = Path(args.repo).expanduser()
    if not (repo_path / "metric_depth").exists() and (DEFAULT_DAV2_REPO / "metric_depth").exists():
        repo_path = DEFAULT_DAV2_REPO
    sys.path.insert(0, str(repo_path / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model = DepthAnythingV2(**CONFIGS[encoder], max_depth=max_depth)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(device).eval()

    split_map = {} if args.no_auto_split else load_split_map(args.split_dir)
    counts = {"train": 0, "validation": 0, "test": 0, "root": 0}

    print(f"{len(files)} images | {encoder}, input {w}x{h}, max_depth {max_depth} | device {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = 0

    for start in range(0, len(files), args.batch_size):
        batch_files, tensors, sizes = [], [], []
        for f in files[start:start + args.batch_size]:
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[skip] could not read {f}")
                continue
            batch_files.append(f)
            sizes.append(bgr.shape[:2])
            tensors.append(preprocess(bgr, h, w))
        if not tensors:
            continue

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(torch.stack(tensors).to(device))

        for f, (h0, w0), p in zip(batch_files, sizes, pred.float()):
            p = F.interpolate(p[None, None], (h0, w0), mode="bilinear", align_corners=False)[0, 0]
            p = p.clamp(0, 1).cpu().numpy().astype(np.float32)
            if args.minmax:
                lo, hi = float(p.min()), float(p.max())
                p = (p - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(p)

            split = None if args.no_auto_split else resolve_split(f, split_map)
            if split:
                target_dir = args.out_dir / split
                counts[split] = counts.get(split, 0) + 1
            elif args.recursive:
                target_dir = args.out_dir / f.relative_to(args.img_dir).parent
                counts["root"] = counts.get("root", 0) + 1
            else:
                target_dir = args.out_dir
                counts["root"] = counts.get("root", 0) + 1

            target_dir.mkdir(parents=True, exist_ok=True)
            np.save(target_dir / f"{f.stem}.npy", p)
            if not args.no_png:
                cv2.imwrite(str(target_dir / f"{f.stem}.png"), np.round(p * 255).astype(np.uint8))
            done += 1

        if done % 50 < args.batch_size or done == len(files):
            print(f"Progress {done}/{len(files)}")

    summary = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0)
    print(f"Done: {done} image(s) -> {args.out_dir} ({summary})")


if __name__ == "__main__":
    main()
