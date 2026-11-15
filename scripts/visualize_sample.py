#!/usr/bin/env python3
"""Visualize a single sample from the LunarSeg dataset.

Generates a 4-panel visual:
  [RGB Image] | [Class Label IDs (0, 1, 2)] | [Color Annotations] | [Depth Map]

Usage:
    python3 scripts/visualize_sample.py [--root LunarSeg] [--sample 00_0001] [--out results/dataset_sample.png]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Visualize a LunarSeg sample across modalities.")
    parser.add_argument("--root", default=None, help="Root path to LunarSeg dataset (default: auto-detect)")
    parser.add_argument("--sample", default="00_0001", help="Sample ID to visualize (default: 00_0001)")
    parser.add_argument("--out", default="results/dataset_sample.png", help="Output file path")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args()

    root = resolve_dataset_root(args.root)
    sample_name = args.sample

    rgb_p = find_file(root, "image", sample_name)
    label_p = find_file(root, "label", sample_name)
    label_color_p = find_file(root, "label_color", sample_name)
    depth_p = find_file(root, "depth", sample_name)

    panels = []
    if rgb_p and rgb_p.is_file():
        panels.append(("RGB Image", np.array(Image.open(rgb_p).convert("RGB")), None))
    else:
        print(f"[Warning] RGB image not found for sample '{sample_name}' under {root / 'image'}")

    if label_p and label_p.is_file():
        panels.append(("Class IDs (0, 1, 2)", np.array(Image.open(label_p)), "viridis"))
    else:
        print(f"[Warning] Label mask not found for sample '{sample_name}' under {root / 'label'}")

    if label_color_p and label_color_p.is_file():
        panels.append(("Color Annotations", np.array(Image.open(label_color_p).convert("RGB")), None))
    else:
        print(f"[Warning] Color label not found for sample '{sample_name}' under {root / 'label_color'}")

    if depth_p and depth_p.is_file():
        panels.append(("Depth Map", np.array(Image.open(depth_p)), "gray"))
    else:
        print(f"[Warning] Depth map not found for sample '{sample_name}' under {root / 'depth'}")

    if not panels:
        print(f"Error: No modalities found for sample '{sample_name}' in '{root}'.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, arr, cmap) in zip(axes, panels):
        if cmap:
            ax.imshow(arr, cmap=cmap)
        else:
            ax.imshow(arr)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved visualization to {out_path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
