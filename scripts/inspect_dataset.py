#!/usr/bin/env python3
"""Dataset diagnostic utility for LunarSeg.

Inspects class ID distribution across labels and validates corresponding
RGB image, class label, color-coded label, and depth files for sample frames.

Usage:
    python3 scripts/inspect_dataset.py [--root LunarSeg] [--sample 00_0001]
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import numpy as np
from PIL import Image


def resolve_dataset_root(candidate=None):
    if candidate and Path(candidate).is_dir():
        return Path(candidate)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    for p in [
        repo_root / "LunarSeg",
        Path("LunarSeg"),
        Path("../LunarSeg"),
        Path.home() / "Documents" / "LunarSeg",
    ]:
        if (p / "train.txt").exists() or (p / "image").exists() or (p / "label").exists():
            return p.resolve()
    return Path("LunarSeg").resolve()


def find_file(root_dir, folder_name, sample_name, ext=".png"):
    stem = sample_name[:-len(ext)] if sample_name.endswith(ext) else sample_name
    candidates = [
        root_dir / folder_name / f"{stem}{ext}",
        root_dir / folder_name / "train" / f"{stem}{ext}",
        root_dir / folder_name / "validation" / f"{stem}{ext}",
        root_dir / folder_name / "test" / f"{stem}{ext}",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect LunarSeg dataset files and class distribution.")
    parser.add_argument("--root", default=None, help="Root path to LunarSeg dataset (default: auto-detect)")
    parser.add_argument("--sample", default="00_0001", help="Sample name to inspect (default: 00_0001)")
    args = parser.parse_args()

    root = resolve_dataset_root(args.root)
    print(f"Dataset root: {root}")

    label_dir = root / "label"
    counts = Counter()

    # Search for labels in label/ directly or label/{train,validation,test}
    label_files = sorted(label_dir.glob("*.png"))
    if not label_files:
        label_files = sorted(label_dir.glob("*/*.png"))

    if label_files:
        for path in label_files:
            label = np.array(Image.open(path))
            values, frequencies = np.unique(label, return_counts=True)
            for value, frequency in zip(values, frequencies):
                counts[int(value)] += int(frequency)

        print("\n=== Pixel counts across all labels ===")
        total = sum(counts.values())
        for class_id in sorted(counts):
            print(
                f"ID {class_id}: {counts[class_id]:,} pixels "
                f"({100 * counts[class_id] / total:.3f}%)"
            )
    else:
        print(f"\n[Notice] No label files found under {label_dir}")

    # Inspect one corresponding RGB, label, color-label, and depth sample
    print(f"\n=== Inspection for sample: '{args.sample}' ===")
    for folder in ["image", "label", "label_color", "depth"]:
        path = find_file(root, folder, args.sample)
        if path and path.exists():
            arr = np.array(Image.open(path))
            print(
                f"{folder:12s}: shape={str(arr.shape):14s} dtype={str(arr.dtype):8s} "
                f"unique_values={len(np.unique(arr)):<4d} ({path.relative_to(root)})"
            )
        else:
            print(f"{folder:12s}: [Not Found under {root / folder}]")

    print("\nDataset diagnostic complete.")


if __name__ == "__main__":
    main()
