#!/usr/bin/env python3
"""Compare metric Depth Anything V2 predictions (.npy, depth in meters) with LunarSeg GT depth PNGs.

Per image
  1. Load predicted metric depth (.npy) and min-max normalize it to [0, 1].
  2. Load GT depth PNG (OpenCV, unchanged) and divide by the dtype max (255 / 65535) -> [0, 1].
     With --norm-gt the GT is min-max normalized per image as well.
  3. Resize the prediction to the GT size if they differ.
  4. Valid mask: GT > 0 and finite prediction. The sky is included unless --sky-thresh is given.
  5. Optional --align scale_shift: least-squares s*pred + t ~= GT on valid pixels (default: none).
  6. Metrics (depth space): AbsRel, RMSE, delta1; Spearman rank correlation (no alignment).

Usage:
    python3 scripts/compare_metric_depth.py --pred-dir vis_metric_npy --gt-dir LunarSeg/depth/test [--csv results.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from scipy.stats import rankdata
except ImportError:
    rankdata = None

SUFFIXES = ("_raw_depth_meter", "_raw")
EPS = 1e-6


def minmax(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def load_gt_depth(path: Path) -> np.ndarray:
    """GT PNG -> float32 depth, single channel, normalized by dtype max."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read {path}")
    if arr.ndim == 3:
        arr = arr[..., :3]
        arr = arr[..., 0] if (np.array_equal(arr[..., 0], arr[..., 1]) and
                              np.array_equal(arr[..., 0], arr[..., 2])) \
            else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    max_val = 255.0 if arr.dtype == np.uint8 else 65535.0 if arr.dtype == np.uint16 else float(max(arr.max(), 1))
    return arr.astype(np.float32) / max_val


def load_pred(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and 1 in arr.shape:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D depth map, got shape {arr.shape}")
    return arr.astype(np.float32)


def spearman(a: np.ndarray, b: np.ndarray, max_n: int = 200_000, seed: int = 0) -> float:
    if rankdata is None or a.size < 3:
        return float("nan")
    if a.size > max_n:
        idx = np.random.default_rng(seed).choice(a.size, max_n, replace=False)
        a, b = a[idx], b[idx]
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(pred_metric: np.ndarray, gt_depth: np.ndarray, sky_thresh=None,
             align: str = "none", norm_gt: bool = False) -> dict:
    if pred_metric.shape != gt_depth.shape:
        pred_metric = cv2.resize(pred_metric, (gt_depth.shape[1], gt_depth.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

    pred = minmax(pred_metric)
    gt = minmax(gt_depth) if norm_gt else gt_depth.astype(np.float64)

    valid = (gt > 0) & np.isfinite(pred)
    if sky_thresh is not None:
        valid &= gt < sky_thresh
    if valid.sum() < 100:
        raise ValueError("too few valid pixels")

    g = gt[valid]
    p = pred[valid]
    s, t = 1.0, 0.0
    if align == "scale_shift":
        A = np.stack([p, np.ones_like(p)], axis=1)
        (s, t), *_ = np.linalg.lstsq(A, g, rcond=None)
        p = s * p + t
    p = np.clip(p, EPS, None)

    ratio = np.maximum(p / g, g / p)
    return {
        "valid_frac": float(valid.mean()),
        "scale": float(s),
        "shift": float(t),
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "spearman": spearman(p, g),
    }


def gt_stem(stem: str) -> str:
    for suf in SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def main():
    ap = argparse.ArgumentParser(description="Min-max normalize metric depth .npy files and compare with LunarSeg GT depth.")
    ap.add_argument("--pred-dir", type=Path, required=True, help="Directory of metric depth .npy files")
    ap.add_argument("--gt-dir", type=Path, required=True, help="Directory of GT depth PNGs")
    ap.add_argument("--align", choices=["none", "scale_shift"], default="none",
                    help="Optional per-image least-squares scale+shift after normalization (default: none)")
    ap.add_argument("--norm-gt", action="store_true", help="Also min-max normalize the GT per image")
    ap.add_argument("--sky-thresh", type=float, default=None,
                    help="Optional: mask out GT values >= this (e.g. 0.99). Default: sky included.")
    ap.add_argument("--csv", type=Path, help="Optional path for per-image CSV results")
    args = ap.parse_args()

    preds = sorted(args.pred_dir.glob("*.npy"))
    if not preds:
        sys.exit(f"No .npy files in {args.pred_dir}")

    rows = []
    for f in preds:
        stem = gt_stem(f.stem)
        gt_path = args.gt_dir / f"{stem}.png"
        if not gt_path.exists():
            print(f"[skip] no GT for {f.name} (expected {gt_path.name})")
            continue
        try:
            m = evaluate(load_pred(f), load_gt_depth(gt_path),
                         args.sky_thresh, args.align, args.norm_gt)
        except Exception as e:
            print(f"[skip] {stem}: {e}")
            continue
        m["name"] = stem
        rows.append(m)
        print(f"{stem}: AbsRel={m['absrel']:.4f} RMSE={m['rmse']:.4f} d1={m['delta1']:.4f} "
              f"Spearman={m['spearman']:.4f} valid={m['valid_frac']:.2f}")

    if not rows:
        sys.exit("Nothing evaluated.")

    keys = ["absrel", "rmse", "delta1", "spearman"]
    print(f"\nMean over {len(rows)} image(s) [align={args.align}, norm_gt={args.norm_gt}, "
          f"sky {'masked at >=' + str(args.sky_thresh) if args.sky_thresh is not None else 'included'}]:")
    for k in keys:
        print(f"  {k:10s} {np.nanmean([r[k] for r in rows]):.4f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "valid_frac", "scale", "shift"] + keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Per-image results written to {args.csv}")
        pd.set_option("display.max_columns", None)
        print(pd.read_csv(args.csv).describe())


if __name__ == "__main__":
    main()
