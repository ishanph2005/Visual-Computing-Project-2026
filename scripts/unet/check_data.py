"""Verify LunarSeg dataset before training.

Usage:
    python scripts/unet/check_data.py [--root LunarSeg]
Prints split sizes, checks image/depth/label file presence (supporting both flat and split folders),
and summarizes class distributions.
"""
import argparse
import os
import random
import numpy as np
from PIL import Image


def resolve_root(root):
    if root and os.path.exists(os.path.join(root, "train.txt")):
        return root
    for c in ["LunarSeg", "../LunarSeg", "data", os.path.join(os.path.dirname(__file__), "..", "..", "LunarSeg")]:
        if os.path.exists(os.path.join(c, "train.txt")):
            return c
    return root or "LunarSeg"


p = argparse.ArgumentParser(description="Verify LunarSeg dataset setup.")
p.add_argument("--root", default=None, help="Path to LunarSeg dataset folder (default: auto-detects LunarSeg or data)")
p.add_argument("--n_stats", type=int, default=300, help="Number of training samples to profile for class statistics")
args = p.parse_args()
R = resolve_root(args.root)

# 1. splits and file matching
splits = {}
for s in ["train", "validation", "test"]:
    txt_path = os.path.join(R, f"{s}.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Split file '{s}.txt' not found in '{R}'.")
    with open(txt_path) as f:
        splits[s] = [l.strip() for l in f if l.strip()]
    print(f"{s:11s}: {len(splits[s])} names   e.g. {splits[s][:3]}")
allnames = sum(splits.values(), [])
print("total:", len(allnames), " unique:", len(set(allnames)))
overlap = set(splits["train"]) & (set(splits["validation"]) | set(splits["test"]))
print("train/val-test overlap:", len(overlap))


def find_file(folder, name, split):
    for path in [os.path.join(R, folder, f"{name}.png"), os.path.join(R, folder, split, f"{name}.png")]:
        if os.path.exists(path):
            return path
    return None


for folder in ["image", "depth", "label"]:
    missing = []
    for s, names in splits.items():
        missing.extend([n for n in names if find_file(folder, n, s) is None])
    print(f"missing in {folder}/: {len(missing)}", missing[:5] if missing else "[OK - all files present]")

# 2. one sample in detail
n0 = splits["train"][0]
print(f"\nInspection for sample '{n0}':")
for folder in ["image", "depth", "label"]:
    target = find_file(folder, n0, "train")
    if target and os.path.exists(target):
        im = Image.open(target)
        a = np.asarray(im)
        print(f"  {folder:6s} mode={im.mode} shape={a.shape} dtype={a.dtype} min={a.min()} max={a.max()}")
    else:
        print(f"  {folder:6s} not downloaded / not found on disk yet")

# 3. label values, class pixel ratios, depth range over a random subset
random.seed(0)
sample = random.sample(splits["train"], min(args.n_stats, len(splits["train"])))
counts = {}
imgs_with = {}
dmins, dmaxs, dzeros = [], [], []
for n in sample:
    lab_path = find_file("label", n, "train")
    if lab_path and os.path.exists(lab_path):
        lab = np.asarray(Image.open(lab_path))
        if lab.ndim == 3:
            lab = lab[..., 0]
        v, c = np.unique(lab, return_counts=True)
        for vi, ci in zip(v, c):
            counts[int(vi)] = counts.get(int(vi), 0) + int(ci)
            imgs_with[int(vi)] = imgs_with.get(int(vi), 0) + 1
    depth_path = find_file("depth", n, "train")
    if depth_path and os.path.exists(depth_path):
        d = np.asarray(Image.open(depth_path)).astype(np.float64)
        dmins.append(d.min())
        dmaxs.append(d.max())
        dzeros.append((d == 0).mean())

tot = sum(counts.values())
if tot > 0:
    print(f"\nlabel values over {len(sample)} train images:")
    for k in sorted(counts):
        print(f"  class {k}: {100 * counts[k] / tot:6.2f}% of pixels, present in {imgs_with[k]} images")
if dmins:
    print(f"depth min range [{min(dmins)}, {max(dmins)}], max range [{min(dmaxs)}, {max(dmaxs)}], "
          f"mean fraction of 0-pixels {np.mean(dzeros):.4f}")

# 4. condition hints from filenames (helps find the 4 illumination/terrain scenarios)
prefixes = {}
for n in allnames:
    key = n.split("_")[0] if "_" in n else n.rstrip("0123456789")
    prefixes[key] = prefixes.get(key, 0) + 1
print("\nfilename prefixes (first 20):", dict(list(prefixes.items())[:20]))

# 5. visual alignment check
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)
    picks = random.sample(splits["train"], 4)
    fig, ax = plt.subplots(4, 3, figsize=(12, 12))
    for r, n in enumerate(picks):
        img_p = find_file("image", n, "train")
        lab_p = find_file("label", n, "train")
        dep_p = find_file("depth", n, "train")
        if not (img_p and lab_p and dep_p):
            continue
        rgb = np.asarray(Image.open(img_p))[..., :3]
        lab = np.asarray(Image.open(lab_p))
        lab = lab[..., 0] if lab.ndim == 3 else lab
        d = np.asarray(Image.open(dep_p)).astype(np.float32)
        ov = rgb.copy().astype(np.float32)
        ov[lab == 1] = 0.5 * ov[lab == 1] + 0.5 * np.array([255, 0, 0])   # negative = red
        ov[lab == 2] = 0.5 * ov[lab == 2] + 0.5 * np.array([0, 255, 0])   # positive = green
        ax[r, 0].imshow(rgb); ax[r, 0].set_title(n)
        ax[r, 1].imshow(ov.astype(np.uint8)); ax[r, 1].set_title("red=crater, green=rock")
        ax[r, 2].imshow(d, cmap="viridis"); ax[r, 2].set_title("depth")
        for a in ax[r]: a.axis("off")
    plt.tight_layout(); plt.savefig("results/data_check.png", dpi=80)
    print("\nsaved results/data_check.png")
except ImportError:
    print("matplotlib missing, skipped figure")
