#!/usr/bin/env python3
"""Normalize raw Depth Anything V2 .npz outputs to float32 in [0, 1].

Reads *.npz files (as written by the inference script: np.savez_compressed(..., depth=raw_depth)),
min-max normalizes each depth map independently, and writes new .npz files.

Usage:
    python normalize_npz.py <input_dir> <output_dir> [--in-key depth] [--out-key depth]
                            [--invert] [--compressed]
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def normalize(depth: np.ndarray, invert: bool = False) -> np.ndarray:
    depth = depth.astype(np.float32)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-12:  # constant image -> avoid divide-by-zero
        out = np.zeros_like(depth, dtype=np.float32)
    else:
        out = (depth - d_min) / (d_max - d_min)
    if invert:
        out = 1.0 - out
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Normalize raw depth .npz files to [0, 1].")
    parser.add_argument("input_dir", type=Path, help="Directory containing raw .npz files")
    parser.add_argument("output_dir", type=Path, help="Directory to write normalized .npz files")
    parser.add_argument("--in-key", default="depth", help="Array name in the input npz (default: depth)")
    parser.add_argument("--out-key", default="depth", help="Array name in the output npz (default: depth)")
    parser.add_argument("--invert", action="store_true",
                        help="Flip values (1 - x), e.g. to turn disparity (near=1) into near=0")
    parser.add_argument("--keep-suffix", action="store_true",
                        help="Keep the '_raw' suffix in output file names (stripped by default)")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.npz"))
    if not files:
        sys.exit(f"No .npz files found in {args.input_dir}")

    save = np.savez_compressed if args.compressed else np.savez
    for f in files:
        with np.load(f) as data:
            key = args.in_key if args.in_key in data.files else data.files[0]
            raw = data[key]

        out = normalize(raw, invert=args.invert)

        stem = f.stem
        if not args.keep_suffix and stem.endswith("_raw"):
            stem = stem[: -len("_raw")]
        out_path = args.output_dir / f"{stem}.npz"
        save(out_path, **{args.out_key: out})
        print(f"{f.name} -> {out_path.name}  shape={out.shape} "
              f"raw[{raw.min():.4f}, {raw.max():.4f}] -> [{out.min():.3f}, {out.max():.3f}]")

    print(f"Done: {len(files)} file(s) normalized.")


if __name__ == "__main__":
    main()