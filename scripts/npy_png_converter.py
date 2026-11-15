#!/usr/bin/env python3
"""Convert raw depth .npy arrays to normalized grayscale PNG images using OpenCV.

Automatically organizes outputs into train / validation / test subdirectories
based on input directory layout or LunarSeg split definition files.

Usage:
    python3 scripts/npy_png_converter.py -i <input_dir> -o <output_dir>
"""

import argparse
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


def convert_npy_to_png(input_dir, output_dir, split_dir=None, auto_split=True):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.is_dir():
        sys.exit(f"Input directory not found: {in_dir}")

    npy_files = sorted(p for p in in_dir.rglob("*.npy") if p.is_file())
    if not npy_files:
        print(f"No .npy files found under {in_dir}")
        return

    split_map = load_split_map(split_dir) if auto_split else {}
    counts = {"train": 0, "validation": 0, "test": 0, "root": 0}

    for file_path in npy_files:
        try:
            depth_data = np.load(file_path)
            min_val = float(np.min(depth_data))
            max_val = float(np.max(depth_data))

            if max_val - min_val < 1e-12:
                normalized = np.zeros_like(depth_data, dtype=np.uint8)
            else:
                normalized = ((depth_data - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

            split = resolve_split(file_path, split_map) if auto_split else None
            if split:
                target_dir = out_dir / split
                counts[split] = counts.get(split, 0) + 1
            else:
                target_dir = out_dir
                counts["root"] = counts.get("root", 0) + 1

            target_dir.mkdir(parents=True, exist_ok=True)
            out_file = target_dir / f"{file_path.stem}.png"
            cv2.imwrite(str(out_file), normalized)
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    summary = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0)
    print(f"Done: {len(npy_files)} file(s) converted -> {out_dir} ({summary})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw depth .npy to grayscale PNG with automatic split organization."
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input directory path")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output directory path")
    parser.add_argument("--split-dir", type=Path, default=None, help="Folder with train.txt/validation.txt/test.txt")
    parser.add_argument("--no-auto-split", action="store_true", help="Disable automatic split organization")
    args = parser.parse_args()
    convert_npy_to_png(args.input, args.output, split_dir=args.split_dir, auto_split=not args.no_auto_split)
