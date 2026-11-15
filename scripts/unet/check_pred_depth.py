"""Inspect Depth Anything predictions and score them against true depth.

Usage:
    python scripts/unet/check_pred_depth.py --pred_dir outputs/dav2_lunar_preds

1. Prints what is inside one .npz (keys, shape, dtype, range) and checks every
   image in the dataset has a prediction.
2. Checks the direction: is the prediction disparity (near = large) or depth?
3. Scores it against LunarSeg inverse depth on ground pixels (sky excluded), after
   a least-squares scale + shift fit per image (Depth Anything is relative, so
   scale is unknown). Reports AbsRel and delta<1.25 per condition (00..03) and
   separately on crater and rock pixels.
Saves results/pred_depth_check.png and results/pred_depth_metrics.json.
"""
import argparse, json, os, random
import numpy as np
from PIL import Image

import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from dataset import read_split, load_pred, gt_depth_to_inverse, pred_depth_to_disparity

p = argparse.ArgumentParser()
p.add_argument("--root", default="LunarSeg")
p.add_argument("--pred_dir", default="outputs/dav2_lunar_preds")
p.add_argument("--n", type=int, default=400, help="images to score")
p.add_argument("--invert", action="store_true", help="files hold depth (far = large): score 1/depth")
p.add_argument("--split", default=None, choices=["train", "validation", "test"], help="score only this split")
a = p.parse_args()

names = sum((read_split(a.root, s) for s in ["train", "validation", "test"]), [])
have = set()
for f in os.listdir(a.pred_dir):
    if f.endswith("_raw.npz"):
        have.add(f[:-8])
    elif f.endswith(".npy"):
        have.add(f[:-4])
missing = [n for n in names if n not in have]
print(f"predictions found: {len(have)}   missing for dataset images: {len(missing)}  {missing[:5]}")

# 1. contents of one file
n0 = sorted(have)[0]
fz = os.path.join(a.pred_dir, f"{n0}_raw.npz")
if os.path.exists(fz):
    z = np.load(fz)
    for k in z.files:
        v = z[k]
        print(f"{n0}_raw.npz['{k}']: shape={v.shape} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f}")
fy = os.path.join(a.pred_dir, f"{n0}.npy")
if os.path.exists(fy):
    v = np.load(fy)
    print(f"{n0}.npy: shape={v.shape} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f} "
          f"frac<=0={float((v <= 0).mean()):.3f}  top-row mean={v[..., :5, :].mean():.3f} bottom-row mean={v[..., -5:, :].mean():.3f}")

# 2+3. direction and accuracy
random.seed(0)
pool = [n for n in (read_split(a.root, a.split) if a.split else names) if n in have]
pick = random.sample(pool, min(a.n, len(pool)))
stats = {}
corrs = []
for n in pick:
    pred = load_pred(a.pred_dir, n)
    if a.invert:
        pred = pred_depth_to_disparity(pred)
    pred = pred.astype(np.float64)
    gt_raw = np.asarray(Image.open(os.path.join(a.root, "depth", f"{n}.png")))
    lab = np.asarray(Image.open(os.path.join(a.root, "label", f"{n}.png")))
    if pred.shape != gt_raw.shape:
        pred = np.asarray(Image.fromarray(pred.astype(np.float32)).resize(gt_raw.shape[::-1], Image.BILINEAR)).astype(np.float64)
    gt = gt_depth_to_inverse(gt_raw).astype(np.float64)
    ground = gt_raw < 255                     # exclude sky
    if ground.sum() < 100:
        continue
    corrs.append(np.corrcoef(pred[ground], gt[ground])[0, 1])
    # least-squares scale and shift on ground pixels
    A = np.stack([pred[ground], np.ones(ground.sum())], 1)
    s, t = np.linalg.lstsq(A, gt[ground], rcond=None)[0]
    al = s * pred + t
    cond = n.split("_")[0]
    for region, mask in [("ground", ground), ("crater", ground & (lab == 1)), ("rock", ground & (lab == 2))]:
        if mask.sum() == 0:
            continue
        g, q = gt[mask], np.clip(al[mask], 1e-4, None)
        absrel = np.mean(np.abs(q - g) / g)
        d1 = np.mean(np.maximum(q / g, g / q) < 1.25)
        for key in (f"{cond}/{region}", f"all/{region}"):
            stats.setdefault(key, []).append((absrel, d1))

mc = float(np.nanmean(corrs))
print(f"\nmean correlation with true inverse depth on ground: {mc:+.3f}")
print("  -> positive: prediction is disparity (near = large), matches our loader"
      if mc > 0 else "  -> NEGATIVE: prediction is depth (far = large); it must be inverted before training")

out = {"corr_with_inverse_depth": mc}
print(f"\n{'subset':16s} {'AbsRel':>8s} {'delta1':>8s} {'n':>5s}")
for k in sorted(stats):
    arr = np.array(stats[k])
    out[k] = {"AbsRel": float(arr[:, 0].mean()), "delta1": float(arr[:, 1].mean()), "n": len(arr)}
    print(f"{k:16s} {arr[:, 0].mean():8.3f} {arr[:, 1].mean():8.3f} {len(arr):5d}")
os.makedirs("results", exist_ok=True)
json.dump(out, open(f"results/pred_depth_metrics{'_' + a.split if a.split else ''}{'_inv' if a.invert else ''}.json", "w"), indent=2)

# visual
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = random.sample(pick, 4)
    fig, ax = plt.subplots(4, 3, figsize=(12, 12))
    for r, n in enumerate(show):
        ax[r, 0].imshow(np.asarray(Image.open(os.path.join(a.root, "image", f"{n}.png")))); ax[r, 0].set_title(n)
        ax[r, 1].imshow(gt_depth_to_inverse(np.asarray(Image.open(os.path.join(a.root, "depth", f"{n}.png")))), cmap="magma"); ax[r, 1].set_title("true inverse depth")
        pv = load_pred(a.pred_dir, n)
        ax[r, 2].imshow(pred_depth_to_disparity(pv) if a.invert else pv, cmap="magma"); ax[r, 2].set_title("prediction" + (" (inverted)" if a.invert else ""))
        for x in ax[r]: x.axis("off")
    plt.tight_layout(); plt.savefig("results/pred_depth_check.png", dpi=80)
    print("\nsaved results/pred_depth_check.png")
except ImportError:
    pass