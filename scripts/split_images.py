#!/usr/bin/env python3
"""
Segregate images into train / test / validation folders.

Each of train.txt, test.txt and validation.txt (found in --txt_dir) contains one
string per line. An image is assigned to a split if its file name CONTAINS any
of the strings listed in that split's txt file.

Example:
    python split_images.py /data/images /data/splits /data/output
    python split_images.py /data/images /data/splits /data/output --move
"""

import argparse
import shutil
import sys
from pathlib import Path

SPLITS = ["train", "test", "validation"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def read_patterns(txt_path: Path) -> list[str]:
    """Read non-empty, stripped lines from a txt file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Split images into train/test/validation by substring match."
    )
    parser.add_argument("images_dir", type=Path, help="Directory containing the images")
    parser.add_argument(
        "txt_dir", type=Path,
        help="Directory containing train.txt, test.txt and validation.txt",
    )
    parser.add_argument("output_dir", type=Path, help="Directory to create the splits in")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument(
        "--recursive", action="store_true", help="Also search sub-directories of images_dir"
    )
    parser.add_argument(
        "--all-files", action="store_true",
        help="Consider every file, not just common image extensions",
    )
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        sys.exit(f"Images directory not found: {args.images_dir}")
    if not args.txt_dir.is_dir():
        sys.exit(f"Txt directory not found: {args.txt_dir}")

    # Load patterns for each split
    patterns = {}
    for split in SPLITS:
        txt_path = args.txt_dir / f"{split}.txt"
        if not txt_path.is_file():
            sys.exit(f"Missing file: {txt_path}")
        patterns[split] = read_patterns(txt_path)
        print(f"{split}: {len(patterns[split])} strings loaded")

    # Create output dirs
    for split in SPLITS:
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)

    # Collect images
    iterator = args.images_dir.rglob("*") if args.recursive else args.images_dir.iterdir()
    images = [
        p for p in iterator
        if p.is_file() and (args.all_files or p.suffix.lower() in IMAGE_EXTS)
    ]

    transfer = shutil.move if args.move else shutil.copy2
    counts = {s: 0 for s in SPLITS}
    unmatched = []
    multi = []

    for img in sorted(images):
        name = img.name
        matched = [s for s in SPLITS if any(p in name for p in patterns[s])]

        if not matched:
            unmatched.append(name)
            continue
        if len(matched) > 1:
            multi.append((name, matched))

        # Use first matching split (train > test > validation) if ambiguous
        split = matched[0]
        dest = args.output_dir / split / name
        if dest.exists():
            print(f"Warning: {dest} already exists, skipping {img}")
            continue
        transfer(str(img), str(dest))
        counts[split] += 1

    # Summary
    print("\n--- Summary ---")
    for s in SPLITS:
        print(f"{s}: {counts[s]} images")
    print(f"unmatched: {len(unmatched)} images")

    if multi:
        print(f"\nWarning: {len(multi)} image(s) matched multiple splits "
              f"(placed in the first match):")
        for name, m in multi[:20]:
            print(f"  {name} -> {', '.join(m)}")
        if len(multi) > 20:
            print(f"  ... and {len(multi) - 20} more")

    if unmatched:
        print("\nUnmatched examples:", ", ".join(unmatched[:10]))


if __name__ == "__main__":
    main()