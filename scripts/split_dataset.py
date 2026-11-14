#!/usr/bin/env python3
"""Link images from an input dir into train/, test/ and validation/ folders,
according to train.txt, test.txt and validation.txt.

Nothing is copied or moved: each image is exposed through a symbolic link (or a hard link with
--hardlink), so the originals stay where they are and no extra disk space is used.

Usage:
    python3 scripts/split_dataset.py <input_dir> [--split-dir LunarSeg] [--output-dir LunarSeg/image] [--hardlink]
"""
import argparse
import os
import sys
from pathlib import Path

SPLITS = {"train": "train.txt", "test": "test.txt", "validation": "validation.txt"}
DEFAULT_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LUNAR_DIR = ROOT_DIR / "LunarSeg" if (ROOT_DIR / "LunarSeg").exists() else Path("LunarSeg")


def build_index(input_dir: Path, exts):
    by_name, by_stem = {}, {}
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            by_name.setdefault(p.name, p)
            by_stem.setdefault(p.stem, p)
    return by_name, by_stem


def read_names(txt: Path):
    names = []
    for line in txt.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(Path(line).name)
    return names


def main():
    ap = argparse.ArgumentParser(description="Link split images into train/test/validation dirs (no copying or moving).")
    ap.add_argument("input_dir", type=Path, help="Directory containing the images")
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_LUNAR_DIR,
                    help="Directory containing train.txt, test.txt, validation.txt (default: LunarSeg)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Directory to create train/, test/, validation/ in (default: same as split-dir)")
    ap.add_argument("--hardlink", action="store_true",
                    help="Use hard links instead of symlinks (must be on the same filesystem)")
    ap.add_argument("--ext", nargs="+", default=DEFAULT_EXTS,
                    help="Image extensions to match (default: common image formats)")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")
    if not args.split_dir.is_dir():
        sys.exit(f"Split directory not found: {args.split_dir}")

    out_base = args.output_dir if args.output_dir else args.split_dir

    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    by_name, by_stem = build_index(args.input_dir, exts)
    if not by_name:
        sys.exit(f"No image files found under {args.input_dir}")

    for split, txt_name in SPLITS.items():
        txt = args.split_dir / txt_name
        if not txt.exists():
            print(f"[skip] {txt_name} not found in {args.split_dir}")
            continue

        out_dir = out_base / split
        out_dir.mkdir(parents=True, exist_ok=True)

        linked = existing = 0
        missing = []
        for name in read_names(txt):
            matches = [p for fname, p in by_name.items() if name in fname]
            if not matches:
                missing.append(name)
                continue
            for src in matches:
                dst = out_dir / src.name
                if dst.exists() or dst.is_symlink():
                    existing += 1
                    continue
                if args.hardlink:
                    os.link(src.resolve(), dst)
                else:
                    os.symlink(src.resolve(), dst)
                linked += 1

        print(f"{split}: {linked} linked, {existing} already present, {len(missing)} not found in {out_dir}")
        for m in missing[:5]:
            print(f"    missing: {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")


if __name__ == "__main__":
    main()
