#!/usr/bin/env python3
"""Run Depth Anything V2 relative depth inference on an image or directory.

Exports:
  <outdir>/<base_name>.png        Visualized depth map (side-by-side or depth-only)
  <outdir>/<base_name>_raw.npz    Compressed raw floating-point depth array ('depth' key)

Usage:
    python3 scripts/run_dav2_raw.py --encoder vitl --img-path LunarSeg/image/test --outdir outputs/depth --pred-only
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DAV2_REPO = ROOT_DIR / "Depth-Anything-V2"

if DEFAULT_DAV2_REPO.exists():
    sys.path.insert(0, str(DEFAULT_DAV2_REPO))
from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Relative Depth + Raw NPZ Exporter')
    parser.add_argument('--repo', type=str, default=str(DEFAULT_DAV2_REPO), help='Path to Depth-Anything-V2 repo')
    parser.add_argument('--img-path', type=str, required=True, help='Path to an image, text file of paths, or directory')
    parser.add_argument('--input-size', type=int, default=518, help='Shorter side input size (multiple of 14)')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Output directory')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Explicit checkpoint path (defaults to checkpoints/)')
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
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    filenames = [f for f in filenames if os.path.isfile(f) and f.lower().endswith(valid_extensions)]
    if not filenames:
        sys.exit(f"No valid images found in: {args.img_path}")

    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps['Spectral_r']

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Warning: Could not read {filename}. Skipping.")
            continue

        with torch.no_grad():
            raw_depth = depth_anything.infer_image(raw_image, args.input_size)

        base_name = os.path.splitext(os.path.basename(filename))[0]
        np.savez_compressed(os.path.join(args.outdir, base_name + '_raw.npz'), depth=raw_depth)

        depth = (raw_depth - raw_depth.min()) / max(raw_depth.max() - raw_depth.min(), 1e-12) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, base_name + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            cv2.imwrite(os.path.join(args.outdir, base_name + '.png'), combined_result)

    print(f"Done. Processed {len(filenames)} image(s) -> {args.outdir}")
