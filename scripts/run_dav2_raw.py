#!/usr/bin/env python3
"""Run Depth Anything V2 relative depth inference on an image or directory.

Exports:
  <outdir>/[<split>/]<base_name>.png        Visualized depth map (side-by-side or depth-only)
  <outdir>/[<split>/]<base_name>_raw.npz    Compressed raw floating-point depth array ('depth' key)

Outputs are automatically organized into train / validation / test subdirectories
when processing split folders or datasets matching LunarSeg split definitions.

Usage:
    python3 scripts/run_dav2_raw.py --encoder vitl --img-path LunarSeg/image/test --outdir outputs/depth --pred-only
"""
import argparse
import glob
import os
from pathlib import Path
import re
import sys

import cv2
import matplotlib
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DAV2_REPO = ROOT_DIR / "Depth-Anything-V2"
DEFAULT_LUNAR_DIR = ROOT_DIR / "LunarSeg" if (ROOT_DIR / "LunarSeg").exists() else Path("LunarSeg")

if DEFAULT_DAV2_REPO.exists():
    sys.path.insert(0, str(DEFAULT_DAV2_REPO))
from depth_anything_v2.dpt import DepthAnythingV2


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Relative Depth + Raw NPZ Exporter')
    parser.add_argument('--repo', type=str, default=str(DEFAULT_DAV2_REPO), help='Path to Depth-Anything-V2 repo')
    parser.add_argument('--img-path', type=str, required=True, help='Path to an image, text file of paths, or directory')
    parser.add_argument('--input-size', type=int, default=518, help='Shorter side input size (multiple of 14)')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Output directory')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Explicit checkpoint path (defaults to checkpoints/)')
    parser.add_argument('--split-dir', type=Path, default=None, help='Folder with train.txt/validation.txt/test.txt')
    parser.add_argument('--no-auto-split', action='store_true', help='Disable automatic split folder organization')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='Only save the depth prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='Render grayscale instead of colormap')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        cand1 = Path(args.repo) / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth"
        cand2 = ROOT_DIR / "Depth-Anything-V2" / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth"
        ckpt_path = str(cand1 if cand1.exists() else cand2)

    if not Path(ckpt_path).exists():
        sys.exit(f"Checkpoint not found at: {ckpt_path}. Please download it or specify via --checkpoint.")

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    filenames = [f for f in filenames if os.path.isfile(f) and f.lower().endswith(valid_extensions)]
    if not filenames:
        sys.exit(f"No valid images found in: {args.img_path}")

    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)

    split_map = {} if args.no_auto_split else load_split_map(args.split_dir)
    counts = {"train": 0, "validation": 0, "test": 0, "root": 0}

    cmap = matplotlib.colormaps['Spectral_r']

    for k, filename in enumerate(filenames):
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Warning: Could not read {filename}. Skipping.")
            continue

        with torch.no_grad():
            raw_depth = depth_anything.infer_image(raw_image, args.input_size)

        fpath = Path(filename)
        base_name = fpath.stem
        split = None if args.no_auto_split else resolve_split(fpath, split_map)

        if split:
            target_dir = out_base / split
            counts[split] = counts.get(split, 0) + 1
        else:
            target_dir = out_base
            counts["root"] = counts.get("root", 0) + 1

        target_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(target_dir / f"{base_name}_raw.npz", depth=raw_depth)

        depth = (raw_depth - raw_depth.min()) / max(raw_depth.max() - raw_depth.min(), 1e-12) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            cv2.imwrite(str(target_dir / f"{base_name}.png"), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            cv2.imwrite(str(target_dir / f"{base_name}.png"), combined_result)

        if (k + 1) % 50 == 0 or (k + 1) == len(filenames):
            print(f"Progress {k+1}/{len(filenames)}")

    summary = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0)
    print(f"Done. Processed {len(filenames)} image(s) -> {out_base} ({summary})")
