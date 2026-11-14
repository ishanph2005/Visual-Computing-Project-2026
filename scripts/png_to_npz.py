#!/usr/bin/env python3
"""Convert depth PNG images in an input directory to .npz files with float32 values in [0, 1].

Usage:
    python png_to_npz.py <input_dir> <output_dir> [--key depth] [--compressed]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def load_depth_png(path: Path) -> np.ndarray:
    """Load a PNG with OpenCV as a single-channel float32 array normalized to 0-1."""
    # IMREAD_UNCHANGED keeps 16-bit depth intact (default imread would squash it to 8-bit BGR)
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image: {path}")

    # Reduce to a single channel
    if arr.ndim == 3:
        if arr.shape[2] == 4:  # BGRA -> drop alpha
            arr = arr[..., :3]
        if arr.shape[2] == 3:
            # Depth maps are usually grayscale replicated across channels
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
                arr = arr[..., 0]
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = arr[..., 0]

    # Normalize by the dtype's max value (255 for 8-bit, 65535 for 16-bit)
    if arr.dtype == np.uint8:
        max_val = 255.0
    elif arr.dtype == np.uint16:
        max_val = 65535.0
    else:
        max_val = float(arr.max()) if arr.max() > 1.0 else 1.0

    out = arr.astype(np.float32) / max_val
    return np.clip(out, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Convert depth PNGs to float [0,1] .npz files.")
    parser.add_argument("input_dir", type=Path, help="Directory containing PNG images")
    parser.add_argument("output_dir", type=Path, help="Directory to write .npz files")
    parser.add_argument("--key", default="depth", help="Array name inside the npz (default: depth)")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() == ".png")
    if not pngs:
        sys.exit(f"No PNG files found in {args.input_dir}")

    save = np.savez_compressed if args.compressed else np.savez
    for p in pngs:
        depth = load_depth_png(p)
        out_path = args.output_dir / f"{p.stem}.npz"
        save(out_path, **{args.key: depth})
        print(f"{p.name} -> {out_path.name}  shape={depth.shape} min={depth.min():.3f} max={depth.max():.3f}")

    print(f"Done: {len(pngs)} file(s) converted.")


if __name__ == "__main__":
    main()