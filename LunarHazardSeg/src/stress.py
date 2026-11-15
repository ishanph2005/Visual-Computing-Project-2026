"""Thresholding, robustness and latency for trained runs (TEST split, no retraining).

    python -m src.stress --runs unet_r34_rgb unet_r34_rgbpd unet_r34_rgbd
    python -m src.stress --runs ... --parts threshold          (only one part)
    python -m src.stress --runs ... --limit 200                (quick check on 200 images)

Outputs in results/stress/:
  summary.md            all tables, ready for the report
  threshold.csv         per run, class, threshold: pixel recall / precision / false-positive rate,
                        hazard-level recall and false-alarm blobs per image
  threshold_curves.png  hazard recall vs false alarms as the threshold moves
  robustness.csv        per run and corruption: IoU crater / rock, recall, mIoU
  robustness.png        drop in rock and crater IoU under each corruption
  latency.csv           ms per 640x480 image (fp32 and fp16), params, peak GPU memory

1. THRESHOLD. Normal prediction = argmax of softmax. Here, a pixel is flagged as a hazard
   when P(crater) + P(rock) >= t ("any hazard"), or as class c when P(c) >= t, for t in
   --thresholds. Lower t = more cautious rover: more hazards caught, more false alarms.
   Hazard-level recall uses connected hazards >= --min_area px, found if >= 50% flagged.
2. ROBUSTNESS. The RGB input is corrupted (darker, gamma, sensor noise, blur); the depth
   channel is left untouched. For the predicted-depth model this is optimistic, because
   Depth Anything was run on the clean images (its depth would also degrade in reality).
3. LATENCY. Batch 1, 640x480, 20 warm-up passes, then the mean of 100 timed passes.
"""
import argparse, csv, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import segmentation_models_pytorch as smp

from src.dataset import LunarSeg

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--split", default="test")
p.add_argument("--parts", nargs="+", default=["threshold", "robustness", "latency"])
p.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
p.add_argument("--min_area", type=int, default=20)
p.add_argument("--limit", type=int, default=0, help="use only the first N images (quick check)")
p.add_argument("--out", default="results/stress")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
CLS = {1: "crater", 2: "rock"}
md = [f"# Thresholding, robustness and latency (`{a.split}` split)\n"]


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


def dataset(cfg, corrupt=None):
    ds = LunarSeg(cfg["root"], a.split, mode=cfg["mode"], pred_depth_dir=cfg.get("pred_depth_dir"), corrupt=corrupt, pred_invert=cfg.get("pred_invert", False), gt_linear=cfg.get("gt_linear", False))
    if a.limit:
        ds.names = ds.names[:a.limit]
    return ds


@torch.no_grad()
def probs(model, x):
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
        o = model(x[None].to(dev))
    return F.softmax(o.float(), 1)[0]


def iou_from(cm):
    tp = np.diag(cm); d = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / d


# ------------------------------------------------------------------ 1. threshold sweep
if "threshold" in a.parts:
    T = a.thresholds
    rows = []
    for run in a.runs:
        model, cfg = load(run)
        ds = dataset(cfg)
        # pixel counts per target ('hazard', 'crater', 'rock') per threshold: tp, fp, fn, tn
        cnt = {k: np.zeros((len(T), 4), np.int64) for k in ("hazard", "crater", "rock")}
        hits = np.zeros(len(T)); n_haz = 0; fa = np.zeros(len(T))
        for i in range(len(ds)):
            x, y, _ = ds[i]
            pr = probs(model, x)
            gt = y.to(dev)
            maps = {"hazard": (pr[1] + pr[2], gt > 0), "crater": (pr[1], gt == 1), "rock": (pr[2], gt == 2)}
            for k, (pm, g) in maps.items():
                for j, t in enumerate(T):
                    f = pm >= t
                    tp = (f & g).sum().item(); fp = (f & ~g).sum().item()
                    fn = (~f & g).sum().item(); cnt[k][j] += [tp, fp, fn, g.numel() - tp - fp - fn]
            # hazard-level (any hazard): connected GT hazards and predicted blobs
            gtn = (gt > 0).cpu().numpy()
            lab, n = ndimage.label(gtn)
            sizes = ndimage.sum(np.ones_like(gtn), lab, range(1, n + 1)) if n else np.array([])
            keep = [k + 1 for k in range(n) if sizes[k] >= a.min_area]
            n_haz += len(keep)
            hz = (pr[1] + pr[2]).cpu().numpy()
            for j, t in enumerate(T):
                f = hz >= t
                if keep:
                    frac = ndimage.mean(f, lab, keep)
                    hits[j] += (np.asarray(frac) >= 0.5).sum()
                plab, pn = ndimage.label(f)
                if pn:
                    psz = ndimage.sum(np.ones_like(f), plab, range(1, pn + 1))
                    over = ndimage.maximum(gtn, plab, range(1, pn + 1))
                    fa[j] += int(((np.asarray(psz) >= a.min_area) & (np.asarray(over) == 0)).sum())
            if (i + 1) % 200 == 0:
                print(f"threshold {run}: {i + 1}/{len(ds)}")
        for k in cnt:
            for j, t in enumerate(T):
                tp, fp, fn, tn = cnt[k][j]
                r = {"run": run, "target": k, "t": t,
                     "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                     "fpr": fp / max(fp + tn, 1), "iou": tp / max(tp + fp + fn, 1)}
                if k == "hazard":
                    r["hazard_recall"] = hits[j] / max(n_haz, 1)
                    r["false_alarms_per_img"] = fa[j] / len(ds)
                rows.append(r)
        del model
    keys = ["run", "target", "t", "recall", "precision", "fpr", "iou", "hazard_recall", "false_alarms_per_img"]
    with open(os.path.join(a.out, "threshold.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    md.append("\n## 1. Threshold sweep (any hazard = crater + rock)\n")
    md.append("Pixel recall = share of hazard pixels flagged; FPR = share of background pixels wrongly flagged. "
              f"Hazard recall counts hazards >= {a.min_area} px found (>= 50% flagged); false alarms = flagged blobs "
              f">= {a.min_area} px touching no hazard.\n")
    md.append("| run | t | pixel recall | pixel precision | FPR (%) | hazard IoU | hazard recall | false alarms / img |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["target"] == "hazard":
            md.append(f"| {r['run']} | {r['t']:.2f} | {r['recall']:.4f} | {r['precision']:.4f} | {100 * r['fpr']:.3f} | "
                      f"{r['iou']:.4f} | {100 * r['hazard_recall']:.1f}% | {r['false_alarms_per_img']:.3f} |")
    # operating points
    md.append("\n**Operating points.** Pixel recall at the threshold where FPR first drops to <= 0.1%, "
              "and the threshold with best hazard IoU:\n")
    md.append("| run | recall @ FPR <= 0.1% (t) | best hazard IoU (t) | crater recall @ t=0.5 vs t=0.2 | rock recall @ t=0.5 vs t=0.2 |")
    md.append("|---|---|---|---|---|")
    for run in a.runs:
        hz = [r for r in rows if r["run"] == run and r["target"] == "hazard"]
        ok = [r for r in hz if r["fpr"] <= 0.001]
        op = max(ok, key=lambda r: r["recall"]) if ok else None
        best = max(hz, key=lambda r: r["iou"])
        def rec(k, t):
            m = [r for r in rows if r["run"] == run and r["target"] == k and abs(r["t"] - t) < 1e-9]
            return f"{m[0]['recall']:.4f}" if m else "n/a"
        md.append(f"| {run} | {op['recall']:.4f} ({op['t']:.2f})" if op else f"| {run} | none",)
        md[-1] += f" | {best['iou']:.4f} ({best['t']:.2f}) | {rec('crater', 0.5)} vs {rec('crater', 0.2)} | {rec('rock', 0.5)} vs {rec('rock', 0.2)} |"
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for run in a.runs:
            hz = [r for r in rows if r["run"] == run and r["target"] == "hazard"]
            ax[0].plot([100 * r["fpr"] for r in hz], [r["recall"] for r in hz], "o-", label=run)
            ax[1].plot([r["false_alarms_per_img"] for r in hz], [100 * r["hazard_recall"] for r in hz], "o-", label=run)
        ax[0].set_xlabel("background pixels wrongly flagged (%)"); ax[0].set_ylabel("hazard pixel recall")
        ax[0].set_title("Pixel level: recall vs false positives"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].set_xlabel("false-alarm blobs per image"); ax[1].set_ylabel(f"hazards >= {a.min_area} px found (%)")
        ax[1].set_title("Hazard level: detection vs false alarms"); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "threshold_curves.png"), dpi=110)
    except ImportError:
        pass

# ------------------------------------------------------------------ 2. robustness
if "robustness" in a.parts:
    def make(kind, v):
        if kind == "clean":
            return None
        if kind == "darken":
            return lambda img, i: img * v
        if kind == "gamma":
            return lambda img, i: img ** v
        if kind == "noise":
            return lambda img, i: img + np.random.default_rng(i).normal(0, v / 255.0, img.shape)
        if kind == "blur":
            return lambda img, i: ndimage.gaussian_filter(img, sigma=(v, v, 0))
    CORR = [("clean", 0, "clean"), ("darken", 0.5, "darker x0.5"), ("darken", 0.25, "darker x0.25"),
            ("gamma", 2.0, "gamma 2.0 (deeper shadows)"), ("gamma", 0.5, "gamma 0.5 (washed out)"),
            ("noise", 10, "noise sigma 10"), ("noise", 25, "noise sigma 25"), ("blur", 2.0, "blur sigma 2")]
    rows = []
    for run in a.runs:
        model, cfg = load(run)
        for kind, v, label in CORR:
            ds = dataset(cfg, corrupt=make(kind, v))
            cm = np.zeros((3, 3), np.int64)
            for i in range(len(ds)):
                x, y, _ = ds[i]
                pr = probs(model, x).argmax(0).cpu().numpy()
                g = y.numpy()
                cm += np.bincount(3 * g.ravel() + pr.ravel(), minlength=9).reshape(3, 3)
            iou = iou_from(cm.astype(float)); rec = np.diag(cm) / cm.sum(1)
            rows.append({"run": run, "corruption": label, "mIoU": float(np.nanmean(iou)),
                         "IoU_crater": float(iou[1]), "IoU_rock": float(iou[2]),
                         "recall_crater": float(rec[1]), "recall_rock": float(rec[2])})
            print(f"robustness {run}: {label}: crater IoU {iou[1]:.4f} rock IoU {iou[2]:.4f}")
        del model
    with open(os.path.join(a.out, "robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md.append("\n## 2. Robustness to image corruption (RGB corrupted, depth channel untouched)\n")
    md.append("Change from the clean score in IoU points. The predicted-depth model keeps depth computed from clean "
              "images, so its numbers are optimistic.\n")
    md.append("| corruption | " + " | ".join(f"{r} crater / rock" for r in a.runs) + " |")
    md.append("|---|" + "---|" * len(a.runs))
    clean = {r["run"]: r for r in rows if r["corruption"] == "clean"}
    for _, _, label in CORR:
        cells = []
        for run in a.runs:
            r = next(z for z in rows if z["run"] == run and z["corruption"] == label)
            if label == "clean":
                cells.append(f"{r['IoU_crater']:.4f} / {r['IoU_rock']:.4f}")
            else:
                cells.append(f"{100 * (r['IoU_crater'] - clean[run]['IoU_crater']):+.2f} / "
                             f"{100 * (r['IoU_rock'] - clean[run]['IoU_rock']):+.2f}")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [c[2] for c in CORR[1:]]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        wdt = 0.8 / len(a.runs)
        for k, cls in enumerate(["IoU_crater", "IoU_rock"]):
            for j, run in enumerate(a.runs):
                vals = [100 * (next(z for z in rows if z["run"] == run and z["corruption"] == l)[cls] - clean[run][cls]) for l in labels]
                ax[k].bar(np.arange(len(labels)) + j * wdt, vals, wdt, label=run)
            ax[k].set_xticks(np.arange(len(labels)) + wdt * (len(a.runs) - 1) / 2)
            ax[k].set_xticklabels(labels, rotation=30, ha="right"); ax[k].axhline(0, color="k", lw=.8)
            ax[k].set_title(f"{cls.split('_')[1]} IoU change vs clean (points)"); ax[k].grid(axis="y", alpha=.3)
        ax[0].legend(); plt.tight_layout(); plt.savefig(os.path.join(a.out, "robustness.png"), dpi=110)
    except ImportError:
        pass

# ------------------------------------------------------------------ 3. latency
if "latency" in a.parts:
    rows = []
    for run in a.runs:
        model, cfg = load(run)
        in_ch = 3 if cfg["mode"] == "rgb" else 4
        x = torch.randn(1, in_ch, 480, 640, device=dev)
        for prec in ("fp32", "fp16"):
            if prec == "fp16" and dev != "cuda":
                continue
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=(prec == "fp16")):
                for _ in range(20):
                    model(x)
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(100):
                    model(x)
                if dev == "cuda": torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 10
            rows.append({"run": run, "precision": prec, "ms_per_image": round(ms, 2), "fps": round(1000 / ms, 1),
                         "params_M": round(sum(q.numel() for q in model.parameters()) / 1e6, 2),
                         "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev == "cuda" else None})
        del model
    with open(os.path.join(a.out, "latency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    md.append(f"\n## 3. Segmentation latency ({gpu}, batch 1, 640 x 480)\n")
    md.append("Excludes the depth source: a depth sensor for RGB-D, or one Depth Anything V2 pass per frame for "
              "predicted depth, which must be timed separately.\n")
    md.append("| run | precision | ms / image | FPS | params (M) | peak GPU memory (MB) |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['run']} | {r['precision']} | {r['ms_per_image']} | {r['fps']} | {r['params_M']} | {r['peak_mem_MB']} |")

open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))