"""Compare trained runs on the TEST split.

    python -m src.compare --runs unet_r34_rgb unet_r34_rgbd
    python -m src.compare --runs unet_r34_rgb unet_r34_rgbpd unet_r34_rgbd   (later)

Outputs (results/compare/):
    summary.md      all tables, ready to paste into the report
    summary.json    same numbers, machine-readable
    disagree/       images where the first and last run disagree most on hazards

Measures
  1. pixel metrics: per-class IoU, precision, recall, mIoU (overall + per condition 00..03)
  2. hazard-level detection: each connected crater/rock region in the ground truth is one
     hazard; it counts as detected if >= --hit of its pixels are predicted as that class.
     Reported by size: small (<300 px, mostly far), medium (300-3000), large (>3000, near).
     Also false-alarm regions: predicted hazard blobs with no ground-truth overlap.
  3. paired bootstrap 95% CI of (last run - first run) for each IoU, resampling test images.
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import segmentation_models_pytorch as smp

from src.dataset import LunarSeg

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--split", default="test")
p.add_argument("--hit", type=float, default=0.5)
p.add_argument("--boot", type=int, default=1000)
p.add_argument("--min_area", type=int, default=20, help="ignore hazards / false-alarm blobs smaller than this (px)")
p.add_argument("--n_disagree", type=int, default=12)
p.add_argument("--out", default="results/compare")
a = p.parse_args()

CLS = {1: "crater", 2: "rock"}
SIZES = [("small", 0, 300), ("medium", 300, 3000), ("large", 3000, 10 ** 9)]
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.join(a.out, "disagree"), exist_ok=True)


def find_checkpoint_dir(name):
    candidates = [
        name,
        os.path.join("checkpoints", name),
        os.path.join("LunarHazardSeg", "checkpoints", name),
        os.path.join("runs", "unet", name),
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", name),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            return c
    return os.path.join("checkpoints", name)


def load(run):
    run_dir = find_checkpoint_dir(run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    m = smp.Unet(cfg["encoder"], encoder_weights=None,
                 in_channels=3 if cfg["mode"] == "rgb" else 4, classes=3)
    m.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
    return m.to(dev).eval(), cfg


def conf(gt, pr):
    return np.bincount(3 * gt.ravel() + pr.ravel(), minlength=9).reshape(3, 3)


def iou_from(cm):
    tp = np.diag(cm); d = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / d


def instances(gt, pr):
    """hazard-level hits and false alarms for one image."""
    rows, fa = [], {1: 0, 2: 0}
    for c in (1, 2):
        lab, n = ndimage.label(gt == c)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < a.min_area:
                continue
            rows.append((c, int(m.sum()), float((pr[m] == c).mean())))
        plab, pn = ndimage.label(pr == c)
        for k in range(1, pn + 1):
            pm = plab == k
            if pm.sum() >= a.min_area and not (gt[pm] == c).any():
                fa[c] += 1
    return rows, fa


# ---- run every model over the split, keep per-image results ----
per = {}   # run -> dict
names = None
for run in a.runs:
    model, cfg = load(run)
    ds = LunarSeg(cfg["root"], a.split, mode=cfg["mode"], pred_depth_dir=cfg.get("pred_depth_dir"), pred_invert=cfg.get("pred_invert", False), gt_linear=cfg.get("gt_linear", False))
    names = ds.names
    cms, inst, fas, preds_hz = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, _ = ds[i]
            pr = model(x[None].to(dev)).argmax(1)[0].cpu().numpy().astype(np.int64)
            gt = y.numpy()
            cms.append(conf(gt, pr))
            r, f = instances(gt, pr)
            inst.append(r); fas.append(f)
            preds_hz.append(pr.astype(np.uint8))  # uint8 keeps ~300 MB per run
            if (i + 1) % 200 == 0:
                print(f"{run}: {i + 1}/{len(ds)}")
    per[run] = {"cm": np.stack(cms), "inst": inst, "fa": fas, "pred": preds_hz}
    del model; torch.cuda.empty_cache() if dev == "cuda" else None

conds = np.array([n.split("_")[0] for n in names])
out = {}
md = [f"# Comparison on `{a.split}` split ({len(names)} images)\n"]


def pixel_table(idx, title):
    md.append(f"\n## {title}\n")
    md.append("| run | mIoU | IoU crater | IoU rock | recall crater | recall rock | precision crater | precision rock |")
    md.append("|---|---|---|---|---|---|---|---|")
    res = {}
    for run in a.runs:
        cm = per[run]["cm"][idx].sum(0).astype(np.float64)
        iou = iou_from(cm); rec = np.diag(cm) / cm.sum(1); pre = np.diag(cm) / cm.sum(0)
        res[run] = {"mIoU": float(np.nanmean(iou)), "IoU_crater": float(iou[1]), "IoU_rock": float(iou[2]),
                    "recall_crater": float(rec[1]), "recall_rock": float(rec[2]),
                    "precision_crater": float(pre[1]), "precision_rock": float(pre[2])}
        v = res[run]
        md.append(f"| {run} | {v['mIoU']:.4f} | {v['IoU_crater']:.4f} | {v['IoU_rock']:.4f} | "
                  f"{v['recall_crater']:.4f} | {v['recall_rock']:.4f} | {v['precision_crater']:.4f} | {v['precision_rock']:.4f} |")
    return res


all_idx = np.arange(len(names))
out["overall"] = pixel_table(all_idx, "1. Pixel metrics, all test images")
out["by_condition"] = {}
for c in sorted(set(conds)):
    idx = np.where(conds == c)[0]
    out["by_condition"][c] = pixel_table(idx, f"Condition {c} ({len(idx)} images)")

# ---- hazard-level detection ----
md.append(f"\n## 2. Hazard-level detection (hazards >= {a.min_area} px; found if >= {int(100 * a.hit)}% of its pixels predicted)\n")
md.append("| run | " + " | ".join(f"{CLS[c]} {s}" for c in (1, 2) for s, _, _ in SIZES) +
          " | false-alarm craters / img | false-alarm rocks / img |")
md.append("|---|" + "---|" * 8)
out["hazards"] = {}
for run in a.runs:
    rows = [r for img in per[run]["inst"] for r in img]
    cells, res = [], {}
    for c in (1, 2):
        for s, lo, hi in SIZES:
            sel = [h >= a.hit for cc, area, h in rows if cc == c and lo <= area < hi]
            rate = float(np.mean(sel)) if sel else float("nan")
            res[f"{CLS[c]}_{s}"] = {"detected": rate, "n": len(sel)}
            cells.append(f"{100 * rate:.1f}% (n={len(sel)})")
    fa_c = np.mean([f[1] for f in per[run]["fa"]]); fa_r = np.mean([f[2] for f in per[run]["fa"]])
    res["false_alarms_per_img"] = {"crater": float(fa_c), "rock": float(fa_r)}
    out["hazards"][run] = res
    md.append(f"| {run} | " + " | ".join(cells) + f" | {fa_c:.2f} | {fa_r:.2f} |")

# ---- paired bootstrap for every pair of runs ----
A, B = a.runs[0], a.runs[-1]
rng = np.random.default_rng(0)
idx_s = [rng.integers(0, len(names), len(names)) for _ in range(a.boot)]
out["bootstrap"] = {}
pairs = [(a.runs[i], a.runs[j]) for i in range(len(a.runs)) for j in range(i + 1, len(a.runs))]
if pairs:
    md.append(f"\n## 3. Paired bootstrap, IoU points, 95% CI ({a.boot} resamples of test images)\n")
    md.append("| comparison | metric | mean | 95% CI | significant? |"); md.append("|---|---|---|---|---|")
for P, Q in pairs:
    cp, cq = per[P]["cm"], per[Q]["cm"]
    diffs = []
    for s_ in idx_s:
        ip, iq = iou_from(cp[s_].sum(0).astype(float)), iou_from(cq[s_].sum(0).astype(float))
        diffs.append([np.nanmean(iq) - np.nanmean(ip), iq[1] - ip[1], iq[2] - ip[2]])
    diffs = np.array(diffs) * 100
    key = f"{Q} - {P}"
    out["bootstrap"][key] = {}
    for j, k in enumerate(["mIoU", "IoU_crater", "IoU_rock"]):
        lo, hi = np.percentile(diffs[:, j], [2.5, 97.5])
        sig = "yes" if lo > 0 or hi < 0 else "no"
        out["bootstrap"][key][k] = {"mean": float(diffs[:, j].mean()), "ci95": [float(lo), float(hi)]}
        md.append(f"| {key} | {k} | {diffs[:, j].mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {sig} |")
if A != B:
    # ---- disagreement images ----
    pal = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], np.uint8)
    score = []
    for i in range(len(names)):
        pa, pb = per[A]["pred"][i], per[B]["pred"][i]
        score.append(((pa > 0) != (pb > 0)).sum())
    root = json.load(open(os.path.join("checkpoints", A, "config.json")))["root"]
    for i in np.argsort(score)[::-1][:a.n_disagree]:
        n = names[i]
        rgb = np.asarray(Image.open(os.path.join(root, "image", f"{n}.png")))[..., :3]
        gt = np.asarray(Image.open(os.path.join(root, "label", f"{n}.png")))
        gt = gt[..., 0] if gt.ndim == 3 else gt
        row = np.concatenate([rgb, pal[gt], pal[per[A]["pred"][i]], pal[per[B]["pred"][i]]], 1)
        Image.fromarray(row).save(os.path.join(a.out, "disagree", f"{n}__rgb_gt_{A}_{B}.png"))
    md.append(f"\nDisagreement images (RGB | ground truth | {A} | {B}) saved in `{a.out}/disagree/`.")

open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md) + "\n")
json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
print("\n".join(md))