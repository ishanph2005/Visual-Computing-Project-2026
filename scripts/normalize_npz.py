#!/usr/bin/env python3
"""Normalize raw Depth Anything V2 .npz outputs to float32 in [0, 1].

Automatically organizes output into train / validation / test subdirectories
based on input directory layout or LunarSeg split definition files.

Usage:
    python3 scripts/normalize_npz.py <input_dir> <output_dir> [--in-key depth] [--out-key depth]
                                     [--invert] [--compressed]
"""

import argparse
import os
from pathlib import Path
import re
import sys

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


def normalize(depth: np.ndarray, invert: bool = False) -> np.ndarray:
    depth = depth.astype(np.float32)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-12:
        out = np.zeros_like(depth, dtype=np.float32)
    else:
        out = (depth - d_min) / (d_max - d_min)
    if invert:
        out = 1.0 - out
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize raw depth .npz files to [0, 1] with automatic split organization."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing raw .npz files")
    parser.add_argument("output_dir", type=Path, help="Directory to write normalized .npz files")
    parser.add_argument("--in-key", default="depth", help="Array name in the input npz (default: depth)")
    parser.add_argument("--out-key", default="depth", help="Array name in the output npz (default: depth)")
    parser.add_argument("--split-dir", type=Path, default=None,
                        help="Directory containing split txt files (default: auto-detect LunarSeg)")
    parser.add_argument("--no-auto-split", action="store_true",
                        help="Disable automatic split folder organization")
    parser.add_argument("--invert", action="store_true",
                        help="Flip values (1 - x), e.g. to turn disparity into depth")
    parser.add_argument("--keep-suffix", action="store_true",
                        help="Keep '_raw' suffix in output filenames (stripped by default)")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")

    files = sorted(p for p in args.input_dir.rglob("*.npz") if p.is_file())
    if not files:
        sys.exit(f"No .npz files found under {args.input_dir}")

    split_map = {} if args.no_auto_split else load_split_map(args.split_dir)
    save = np.savez_compressed if args.compressed else np.savez

    counts = {"train": 0, "validation": 0, "test": 0, "root": 0}

    for f in files:
        with np.load(f) as data:
            key = args.in_key if args.in_key in data.files else data.files[0]
            raw = data[key]

        out = normalize(raw, invert=args.invert)

        stem = f.stem
        if not args.keep_suffix and stem.endswith("_raw"):
            stem = stem[:-len("_raw")]

        split = None if args.no_auto_split else resolve_split(f, split_map)

        if split:
            target_dir = args.output_dir / split
            counts[split] = counts.get(split, 0) + 1
        else:
            target_dir = args.output_dir
            counts["root"] = counts.get("root", 0) + 1

        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{stem}.npz"
        save(out_path, **{args.out_key: out})

    summary = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0)
    print(f"Done: {len(files)} file(s) normalized -> {args.output_dir} ({summary})")


if __name__ == "__main__":
    main()
