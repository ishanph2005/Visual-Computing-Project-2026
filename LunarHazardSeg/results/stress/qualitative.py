"""Qualitative figure for the paper: test frames with predictions side by side.

    python -m src.qualitative
    python -m src.qualitative --segformer_dir path/to/segformer_masks        (adds a SegFormer column)
    python -m src.qualitative --names 00_0123 02_0456 03_0789               (choose frames yourself)

Columns: RGB | fine-tuned DA V2 depth | ground truth | U-Net RGB | U-Net + GT depth [| SegFormer RGB]
Masks are drawn over a dimmed copy of the image: red = crater, green = rock.

Frame choice (if --names is not given): runs the two U-Net models on the test split and
picks, from scenarios 00, 02 and one of 01/03, the frame where the two models disagree most
on rock pixels while ground truth has several rocks, so the figure shows where depth helps.
The script prints the chosen names; rerun with --names to keep the same frames.

--segformer_dir must hold one PNG per test frame, <name>.png, with class indices 0/1/2
(0 background, 1 crater, 2 rock), same size as the image.

--crop trims empty sky and near foreground to the band containing hazards, which makes
distant rocks visible at column width. Use --no-crop for full frames.

Outputs: results/qualitative.pdf and results/qualitative.png (full page width, figure*).
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

from src.dataset import LunarSeg, read_split, load_pred, pred_depth_to_disparity

p = argparse.ArgumentParser()
p.add_argument("--root", default="data")
p.add_argument("--rgb_run", default="unet_r34_rgb")
p.add_argument("--depth_run", default="unet_r34_rgbd_lin")
p.add_argument("--pred_depth_dir", default="data_pred_ft", help="fine-tuned DA V2 outputs (depth, far = large)")
p.add_argument("--segformer_dir", default=None)
p.add_argument("--names", nargs="+", default=None)
p.add_argument("--n", type=int, default=3)
p.add_argument("--crop", dest="crop", action="store_true", default=True)
p.add_argument("--no-crop", dest="crop", action="store_false")
p.add_argument("--out", default="results/qualitative")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
PAL = np.array([[0, 0, 0], [230, 40, 40], [40, 200, 70]], np.float32)


def load_run(run):
    cfg = json.load(open(os.path.join("checkpoints", run, "config.json")))
    m = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=3 if cfg["mode"] == "rgb" else 4, classes=3)
    m.load_state_dict(torch.load(os.path.join("checkpoints", run, "best.pth"), map_location=dev))
    ds = LunarSeg(cfg["root"], "test", mode=cfg["mode"], pred_depth_dir=cfg.get("pred_depth_dir"),
                  pred_invert=cfg.get("pred_invert", False), gt_linear=cfg.get("gt_linear", False))
    return m.to(dev).eval(), ds


@torch.no_grad()
def predict(model, ds, idx):
    x, y, _ = ds[idx]
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
        o = model(x[None].to(dev))
    return o.argmax(1)[0].cpu().numpy(), y.numpy()


m_rgb, ds_rgb = load_run(a.rgb_run)
m_d, ds_d = load_run(a.depth_run)
names = ds_rgb.names
index = {n: i for i, n in enumerate(names)}

# ---------- choose frames ----------
if a.names:
    chosen = a.names
else:
    score = {}
    for i, n in enumerate(names):
        pr, gt = predict(m_rgb, ds_rgb, i)
        pd, _ = predict(m_d, ds_d, i)
        rock = gt == 2
        if rock.sum() < 400:
            continue
        # rock pixels the depth model gets right and the RGB model gets wrong, minus the reverse
        score[n] = int(((pd == 2) & (pr != 2) & rock).sum() - ((pr == 2) & (pd != 2) & rock).sum())
        if (i + 1) % 200 == 0:
            print(f"scanning {i + 1}/{len(names)}")
    chosen = []
    for group in (["00"], ["02"], ["01", "03"]):
        cand = sorted([n for n in score if n.split("_")[0] in group], key=lambda n: -score[n])
        if cand:
            chosen.append(cand[0])
    chosen = chosen[:a.n]
print("frames:", " ".join(chosen))


def overlay(rgb, mask):
    base = rgb.astype(np.float32) * 0.55
    out = base.copy()
    for c in (1, 2):
        m = mask == c
        out[m] = 0.35 * base[m] + 0.65 * PAL[c]
    return out.clip(0, 255).astype(np.uint8)


def crop_rows(gt):
    rows = np.where((gt > 0).any(1))[0]
    if not a.crop or len(rows) == 0:
        return 0, gt.shape[0]
    lo, hi = max(rows.min() - 25, 0), min(rows.max() + 25, gt.shape[0])
    if hi - lo < 220:                                   # keep at least a 640 x 220 band
        mid = (lo + hi) // 2
        lo, hi = max(mid - 110, 0), min(mid + 110, gt.shape[0])
    return lo, hi


cols = ["RGB", "DA V2-FT depth", "Ground truth", "U-Net, RGB", "U-Net, RGB + GT depth"]
if a.segformer_dir:
    cols.append("SegFormer, RGB")
rows = []
for n in chosen:
    i = index[n]
    rgb = np.asarray(Image.open(os.path.join(a.root, "image", f"{n}.png")).convert("RGB"))
    pr, gt = predict(m_rgb, ds_rgb, i)
    pd, _ = predict(m_d, ds_d, i)
    dep = load_pred(a.pred_depth_dir, n).astype(np.float32)
    dep = pred_depth_to_disparity(dep) if dep.max() > 0 else dep        # show near = bright
    dep = (dep - dep.min()) / (np.ptp(dep) + 1e-8)
    panels = [rgb, dep, overlay(rgb, gt), overlay(rgb, pr), overlay(rgb, pd)]
    if a.segformer_dir:
        sf = np.asarray(Image.open(os.path.join(a.segformer_dir, f"{n}.png")))
        sf = sf[..., 0] if sf.ndim == 3 else sf
        panels.append(overlay(rgb, sf))
    lo, hi = crop_rows(gt)
    rows.append((n, [pnl[lo:hi] for pnl in panels]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})
nc, nr = len(cols), len(rows)
h_in = [r[1][0].shape[0] / r[1][0].shape[1] for r in rows]
fig_w = 7.16
cell_w = fig_w / nc
fig, axs = plt.subplots(nr, nc, figsize=(fig_w, sum(h_in) * cell_w * 1.06 + 0.25),
                        gridspec_kw={"height_ratios": h_in}, squeeze=False)
for r, (n, panels) in enumerate(rows):
    for c, pnl in enumerate(panels):
        ax = axs[r, c]
        ax.imshow(pnl, cmap="magma" if pnl.ndim == 2 else None)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.3)
        if r == 0:
            ax.set_title(cols[c], fontsize=7)
        if c == 0:
            scen = {"00": "low light, rough", "01": "low light, flat", "02": "high light, rough", "03": "high light, flat"}[n[:2]]
            ax.text(0.02, 0.96, f"{n} ({scen})", transform=ax.transAxes, fontsize=5.5, color="white",
                    va="top", ha="left", bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
top = 1 - 0.22 / fig.get_figheight()
plt.subplots_adjust(left=0.005, right=0.995, top=top, bottom=0.005, wspace=0.03, hspace=0.05)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=300)
print(f"saved {a.out}.pdf / .png")