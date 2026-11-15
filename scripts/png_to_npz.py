#!/usr/bin/env python3
"""Convert depth PNG images to float32 .npz files in [0, 1].

Automatically organizes output into train / validation / test subdirectories
based on input directory layout or LunarSeg split definition files.

Usage:
    python3 scripts/png_to_npz.py <input_dir> <output_dir> [--key depth] [--compressed]
"""

import argparse
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
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
                    name = Path(line).name
                    stem = Path(line).stem
                    mapping[name] = s
                    mapping[stem] = s
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


def load_depth_png(path: Path) -> np.ndarray:
    """Load a PNG with OpenCV as a single-channel float32 array normalized to 0-1."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image: {path}")

    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.shape[2] == 3:
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
                arr = arr[..., 0]
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = arr[..., 0]

    if arr.dtype == np.uint8:
        max_val = 255.0
    elif arr.dtype == np.uint16:
        max_val = 65535.0
    else:
        max_val = float(arr.max()) if arr.max() > 1.0 else 1.0

    out = arr.astype(np.float32) / max_val
    return np.clip(out, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Convert depth PNGs to float [0, 1] .npz files with automatic split organization."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing PNG images")
    parser.add_argument("output_dir", type=Path, help="Directory to write .npz files")
    parser.add_argument("--key", default="depth", help="Array name inside the npz (default: depth)")
    parser.add_argument("--split-dir", type=Path, default=None,
                        help="Directory containing split txt files (default: auto-detect LunarSeg)")
    parser.add_argument("--no-auto-split", action="store_true",
                        help="Disable automatic split folder organization")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")

    pngs = sorted(p for p in args.input_dir.rglob("*.png") if p.is_file())
    if not pngs:
        sys.exit(f"No PNG files found under {args.input_dir}")

    split_map = {} if args.no_auto_split else load_split_map(args.split_dir)
    save = np.savez_compressed if args.compressed else np.savez

    counts = {"train": 0, "validation": 0, "test": 0, "root": 0}

    for p in pngs:
        depth = load_depth_png(p)
        split = None if args.no_auto_split else resolve_split(p, split_map)

        if split:
            target_dir = args.output_dir / split
            counts[split] = counts.get(split, 0) + 1
        else:
            target_dir = args.output_dir
            counts["root"] = counts.get("root", 0) + 1

        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{p.stem}.npz"
        save(out_path, **{args.key: depth})

    summary = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0)
    print(f"Done: {len(pngs)} file(s) converted -> {args.output_dir} ({summary})")


if __name__ == "__main__":
    main()
