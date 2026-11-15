#!/usr/bin/env python3
"""Unified Multipurpose Dataset Splitter.

Partitions dataset files and images into train / test / validation splits
according to split definition files (train.txt, test.txt, validation.txt).

Supports multiple file transfer modes:
  - symlink  (default): Fast, zero additional disk space, non-destructive.
  - hardlink : Hard links on the same filesystem (zero additional disk space).
  - copy     : Duplicates files into destination split directories.
  - move     : Moves files into destination split directories.

Usage Examples:
    # 1. Non-destructive symlink into train/, validation/, test/ (default)
    python3 scripts/split_dataset.py LunarSeg/all_images --output-dir LunarSeg/image --ext .png

    # 2. Positional format [split_dir] [output_dir]
    python3 scripts/split_dataset.py LunarSeg/all_images LunarSeg LunarSeg/image --copy
    python3 scripts/split_dataset.py LunarSeg/all_images LunarSeg LunarSeg/image --move

    # 3. Explicit mode selection
    python3 scripts/split_dataset.py /data/raw --output-dir LunarSeg/image --mode copy
    python3 scripts/split_dataset.py /data/raw --output-dir LunarSeg/image --mode move
    python3 scripts/split_dataset.py /data/raw --output-dir LunarSeg/image --mode hardlink

    # 4. Filter options
    python3 scripts/split_dataset.py /data/raw --output-dir /data/splits --all-files
    python3 scripts/split_dataset.py /data/raw --output-dir /data/splits --ext .png .jpg .npz
    python3 scripts/split_dataset.py /data/raw --output-dir /data/splits --exact
"""

import argparse
import os
from pathlib import Path
import shutil
import sys

DEFAULT_SPLITS = {"train": "train.txt", "validation": "validation.txt", "test": "test.txt"}
DEFAULT_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LUNAR_DIR = ROOT_DIR / "LunarSeg" if (ROOT_DIR / "LunarSeg").exists() else Path("LunarSeg")


def read_split_patterns(txt_path: Path) -> list[str]:
    """Read non-empty, stripped patterns from a split definition text file."""
    patterns = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(Path(line).name)
    return patterns


def collect_input_files(input_dir: Path, recursive: bool, exts: set[str] | None) -> list[Path]:
    """Collect candidate files from input directory, matching given extensions."""
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = []
    for p in iterator:
        if p.is_file():
            if exts is None or p.suffix.lower() in exts:
                files.append(p)
    return sorted(files)


def transfer_file(src: Path, dst: Path, mode: str, overwrite: bool = False) -> bool:
    """Transfer a single file using the specified mode.
    
    Returns:
        True if the file was newly transferred.
        False if skipped because dst already exists and overwrite=False.
    """
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        else:
            return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unsupported transfer mode: {mode}")

    return True


def main():
    ap = argparse.ArgumentParser(
        description="Unified multipurpose dataset splitter (symlink, hardlink, copy, move)."
    )
    ap.add_argument("input_dir", type=Path, help="Directory containing the input files/images")
    ap.add_argument("extra_args", type=Path, nargs="*", default=[],
                    help="Optional positional arguments: [split_dir] [output_dir]")
    ap.add_argument("--split-dir", type=Path, default=None,
                    help="Directory containing train.txt, validation.txt, test.txt (default: LunarSeg)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Destination directory to create train/, validation/, test/ in")
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy", "move"], default="symlink",
                    help="Transfer mode (default: symlink)")
    ap.add_argument("--copy", dest="mode_flag", action="store_const", const="copy",
                    help="Shortcut for --mode copy")
    ap.add_argument("--move", dest="mode_flag", action="store_const", const="move",
                    help="Shortcut for --mode move")
    ap.add_argument("--hardlink", dest="mode_flag", action="store_const", const="hardlink",
                    help="Shortcut for --mode hardlink")
    ap.add_argument("--symlink", dest="mode_flag", action="store_const", const="symlink",
                    help="Shortcut for --mode symlink")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", default=True,
                    help="Do not search input_dir recursively (default: recursive)")
    ap.add_argument("--recursive", dest="recursive", action="store_true", default=True,
                    help="Search input_dir recursively (enabled by default)")
    ap.add_argument("--all-files", action="store_true",
                    help="Process all file types, ignoring extension filter")
    ap.add_argument("--ext", nargs="+", default=DEFAULT_EXTS,
                    help="Extensions to filter (default: common image extensions)")
    ap.add_argument("--exact", action="store_true",
                    help="Require exact name or stem match instead of substring match")
    ap.add_argument("--on-conflict", choices=["skip", "overwrite"], default="skip",
                    help="Action if target file already exists in output folder (default: skip)")

    args = ap.parse_args()

    # Determine transfer mode (flag overrides --mode if explicitly provided)
    mode = args.mode_flag if args.mode_flag else args.mode

    # Resolve positional extra_args ([split_dir], [output_dir])
    split_dir = args.split_dir
    output_dir = args.output_dir

    if len(args.extra_args) >= 1:
        if split_dir is None:
            split_dir = args.extra_args[0]
    if len(args.extra_args) >= 2:
        if output_dir is None:
            output_dir = args.extra_args[1]
    if len(args.extra_args) > 2:
        sys.exit(f"Too many positional arguments provided: {args.extra_args}")

    if split_dir is None:
        split_dir = DEFAULT_LUNAR_DIR if DEFAULT_LUNAR_DIR.exists() else Path(".")
    if output_dir is None:
        output_dir = split_dir

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")
    if not split_dir.is_dir():
        sys.exit(f"Split directory not found: {split_dir}")

    # Load split patterns
    split_patterns: dict[str, list[str]] = {}
    for split_name, txt_name in DEFAULT_SPLITS.items():
        txt_path = split_dir / txt_name
        if not txt_path.is_file():
            print(f"[warning] Split file {txt_name} not found in {split_dir}, skipping '{split_name}'.")
            continue
        patterns = read_split_patterns(txt_path)
        split_patterns[split_name] = patterns
        print(f"[{split_name}] Loaded {len(patterns):,} sample IDs from {txt_name}")

    if not split_patterns:
        sys.exit(f"No valid split text files found in {split_dir}")

    # Collect input files
    ext_filter = None if args.all_files else {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    input_files = collect_input_files(args.input_dir, recursive=args.recursive, exts=ext_filter)
    if not input_files:
        sys.exit(f"No matching files found under {args.input_dir}")

    print(f"\nIndexed {len(input_files):,} candidate files under {args.input_dir}")
    print(f"Mode: {mode.upper()} -> Destination: {output_dir}\n")

    # Match files to splits
    transferred = {s: 0 for s in split_patterns}
    existing = {s: 0 for s in split_patterns}
    matched_patterns = {s: set() for s in split_patterns}
    unmatched_files = []
    multi_matched = []

    overwrite = (args.on_conflict == "overwrite")

    for file_path in input_files:
        fname = file_path.name
        fstem = file_path.stem

        matching_splits = []
        for s, patterns in split_patterns.items():
            for p in patterns:
                is_match = (p == fname or p == fstem) if args.exact else (p in fname)
                if is_match:
                    matching_splits.append(s)
                    matched_patterns[s].add(p)
                    break

        if not matching_splits:
            unmatched_files.append(fname)
            continue

        if len(matching_splits) > 1:
            multi_matched.append((fname, matching_splits))

        # Primary assignment to the first matching split
        primary_split = matching_splits[0]
        dst_path = output_dir / primary_split / fname

        success = transfer_file(file_path, dst_path, mode=mode, overwrite=overwrite)
        if success:
            transferred[primary_split] += 1
        else:
            existing[primary_split] += 1

    # Print summary
    print("=" * 60)
    print(f"SPLIT SUMMARY (Mode: {mode.upper()})")
    print("=" * 60)
    for s in split_patterns:
        total_in_split = transferred[s] + existing[s]
        missing_count = len(split_patterns[s]) - len(matched_patterns[s])
        print(f"Split '{s}': {transferred[s]:,} {mode}ed | {existing[s]:,} already present | total: {total_in_split:,}")
        if missing_count > 0:
            print(f"  └─ Missing from input: {missing_count:,} sample IDs defined in {DEFAULT_SPLITS[s]}")

    if unmatched_files:
        print(f"\nUnmatched files in input dir ({len(unmatched_files):,}):")
        for uf in unmatched_files[:5]:
            print(f"  - {uf}")
        if len(unmatched_files) > 5:
            print(f"  ... and {len(unmatched_files) - 5:,} more")

    if multi_matched:
        print(f"\nAmbiguous files matching multiple splits ({len(multi_matched):,}):")
        for mf, splits in multi_matched[:5]:
            print(f"  - {mf} -> {splits} (assigned to '{splits[0]}')")
        if len(multi_matched) > 5:
            print(f"  ... and {len(multi_matched) - 5:,} more")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
