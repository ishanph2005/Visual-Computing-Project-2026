#!/usr/bin/env python3
"""Gather all files with a given extension from an input dir into a folder.

Usage:
    python3 scripts/file_to_folder.py <input_dir> <output_folder> --ext png [--mode copy|move|symlink|hardlink]
"""
import argparse
import os
import shutil
import sys
from pathlib import Path


def unique_path(dst: Path) -> Path:
    if not (dst.exists() or dst.is_symlink()):
        return dst
    i = 1
    while True:
        cand = dst.with_name(f"{dst.stem}_{i}{dst.suffix}")
        if not (cand.exists() or cand.is_symlink()):
            return cand
        i += 1


def main():
    ap = argparse.ArgumentParser(description="Gather files of given extension(s) into one folder.")
    ap.add_argument("input_dir", type=Path, help="Directory to search")
    ap.add_argument("output_dir", type=Path, help="Folder to put the files in (created if missing)")
    ap.add_argument("--ext", nargs="+", required=True, help="Extension(s), with or without the dot (e.g. png .jpg)")
    ap.add_argument("--mode", choices=["copy", "move", "symlink", "hardlink"], default="copy",
                    help="How to add the files (default: copy)")
    ap.add_argument("--no-recursive", action="store_true", help="Only look in the top level of input_dir")
    ap.add_argument("--on-conflict", choices=["skip", "rename", "overwrite"], default="rename",
                    help="Conflict strategy: rename (default), skip, overwrite")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")

    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    it = args.input_dir.iterdir() if args.no_recursive else args.input_dir.rglob("*")
    files = sorted(p for p in it if p.is_file() and p.suffix.lower() in exts)
    files = [p for p in files if out_dir not in p.resolve().parents]
    if not files:
        sys.exit(f"No files with extension(s) {sorted(exts)} found in {args.input_dir}")

    done = skipped = 0
    for src in files:
        dst = out_dir / src.name
        if dst.exists() or dst.is_symlink():
            if args.on_conflict == "skip":
                skipped += 1
                continue
            if args.on_conflict == "rename":
                dst = unique_path(dst)
            else:
                dst.unlink()

        if args.mode == "copy":
            shutil.copy2(src, dst)
        elif args.mode == "move":
            shutil.move(str(src), str(dst))
        elif args.mode == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            os.link(src.resolve(), dst)
        done += 1

    print(f"{args.mode}: {done} file(s) -> {out_dir}" + (f", {skipped} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
