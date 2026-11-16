#!/usr/bin/env python3
"""Run several U-Net / SegFormer models on ONE LunarSeg image and compare them side by side.

For every model the figure shows  RGB | depth input (only if the model uses depth) | prediction,  under a ground-truth
row. For that one image it also prints (and saves as CSV / JSON) the metrics of train_segformer_hf.py: pixel accuracy,
per-class IoU / precision / recall, mIoU, and the hazard-level analysis (hazards detected, false alarms) with the same
20 px / 50 % rule.

Run it from anywhere. Relative paths are looked up first from the folder you are in, then from the repository root, so
the defaults (LunarSeg/, runs/, checkpoints/) work without typing full paths.

Give one --model per model:   NAME=PATH[@DEPTH_DIR][#KIND]
  NAME        label used in the figure and tables (optional).
  PATH        any of:
                - a run name, e.g. segformer_b2_rgb or unet_r34_rgbd (searched in runs/, runs/unet/, checkpoints/ ...)
                - a SegFormer run folder (runs/segformer_b2_rgb) or its saved model folder (runs/segformer_b2_rgb/best)
                - a U-Net run folder (runs/unet/unet_r34_rgb) or its weights file (runs/unet/unet_r34_rgb/best.pth)
  DEPTH_DIR   folder with the depth maps this model takes as 4th channel, e.g. "LunarSeg/depth/{split}" ({split} is
              filled in). SegFormer RGB-D models need it. U-Net RGB-D models read it from their config.json
              (rgbd -> LunarSeg/depth, rgbpd -> pred_depth_dir) so you only add it to override that.
  KIND        unet | segformer. Normally detected automatically.

Example (from the repository root)
  python3 scripts/infer_compare.py \\
    --model "SegFormer RGB=segformer_b2_rgb" \\
    --model "SegFormer RGB-D=segformer_b2_rgbd@LunarSeg/depth/{split}" \\
    --model "U-Net RGB=unet_r34_rgb" \\
    --model "U-Net RGB-D=unet_r34_rgbd" \\
    --index 12            # or  --image 03_0123  or  --random --seed 1

Class ids: SegFormer was trained on label_color/ with 1 = rock, 2 = crater. The U-Net was trained on the id masks in
label/, where 1 = crater (negative obstacle) and 2 = rock (positive obstacle). This script puts every model and the
ground truth into the same order (background, rock, crater), so the comparison is fair.

Input pre-processing is copied from each training pipeline:
  SegFormer  RGB: /255, ImageNet mean/std.  Depth: 0-1 map, then (d - 0.5) / 0.25.
  U-Net      RGB: /255, ImageNet mean/std.  Depth (scripts/unet/dataset.py): rgbd -> 2 / max(depth, 2) (or depth / 255
             with gt_linear), rgbpd -> min-max normalised Depth Anything prediction.
"""
import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # never download anything
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent       # <repo>/scripts
REPO_ROOT = SCRIPTS_DIR.parent                      # <repo>

# SegFormer helpers (data reading, metrics, tables) from scripts/segformer/train_segformer_hf.py
sys.path.insert(0, str(SCRIPTS_DIR / "segformer"))
try:
    import train_segformer_hf as T
except ImportError as e:
    sys.exit(f"Could not import scripts/segformer/train_segformer_hf.py ({e}). Keep infer_compare.py in scripts/.")


def _load_unet_dataset():
    """scripts/unet/dataset.py, loaded under its own name so it cannot clash with other 'dataset' modules."""
    path = SCRIPTS_DIR / "unet" / "dataset.py"
    spec = importlib.util.spec_from_file_location("unet_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_unet_dataset()

# Common class order used for the ground truth, the figure and all metrics (same as the SegFormer runs).
CLASS_NAMES = ["background", "rock", "crater"]
CLASS_COLORS = [(0, 0, 0), (173, 69, 31), (0, 163, 226)]   # colours of label_color/ (checked against label/)
LABEL_ID_ORDER = ["background", "crater", "rock"]           # meaning of the ids in label/ (0, 1, 2)
UNET_CLASS_ORDER = ["background", "crater", "rock"]         # U-Net outputs: dataset.py says 1 = negative, 2 = positive

# Where a bare run name is searched (relative to the repository root). One or two extra folder levels below
# runs/ and checkpoints/ are searched too, so it does not matter exactly how the checkpoint zips were unpacked.
MODEL_SEARCH_DIRS = ["checkpoints", "runs", "runs/unet", "checkpoints/SegFormer", "checkpoints/U-Net"]

# The U-Net config.json files were written in the older LunarHazardSeg layout. Their folders map to this repository:
UNET_PATH_ALIASES = {
    "data": "LunarSeg",                                      # dataset root
    "data_pred": "depthmodel_results/npz_outputs/{split}",   # raw Depth Anything V2 predictions (<name>_raw.npz)
    "data_pred_ft": "finetuned/{split}",                     # fine-tuned Depth Anything V2 predictions (<name>.npy)
    "outputs/dav2_lunar_preds": "depthmodel_results/npz_outputs/{split}",
}


# ----------------------------------------------------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------------------------------------------------
def _probe(template):
    """The part of a path template before {split}, used to check whether the folder exists."""
    head = str(template).split("{split}")[0].rstrip("/\\")
    return Path(head or ".")


def resolve(template):
    """Resolve a (possibly relative) path or path template.

    Absolute paths are kept. Relative paths are tried from the current folder first, then from the repository root.
    If neither exists the repository-root version is returned, so error messages point at the expected place."""
    if template is None:
        return None
    p = Path(str(template)).expanduser()
    if p.is_absolute():
        return str(p)
    if _probe(p).exists():
        return str(p)
    return str(REPO_ROOT / p)


def find_file(template, name, split, exts=("png",)):
    """Find <name>.<ext> in a folder template. Works with per-split folders (x/{split}/) and flat folders (x/)."""
    filled = T.fill(template, split)
    folders = [filled]
    if "{split}" in str(template):
        folders.append(T.fill(str(template).replace("/{split}", "").replace("{split}", ""), split))
    else:
        folders.append(filled / split)
    for folder in folders:
        for ext in exts:
            f = folder / f"{name}.{ext}"
            if f.exists():
                return f
    return None


def locate_model(raw):
    """Turn what the user typed into (path to load, kind hint, run folder).

    A SegFormer is a Hugging Face folder (config.json + model.safetensors); a U-Net is a best.pth file."""
    p = Path(resolve(raw))
    if not p.exists() and len(Path(raw).parts) == 1:  # bare run name: search the usual places
        hits = [REPO_ROOT / d / raw for d in MODEL_SEARCH_DIRS if (REPO_ROOT / d / raw).exists()]
        for base in ("checkpoints", "runs"):
            for pattern in (f"*/{raw}", f"*/*/{raw}"):
                hits += sorted((REPO_ROOT / base).glob(pattern))
        if hits:
            p = hits[0]
    if not p.exists():
        sys.exit(f"Model '{raw}' not found. Looked for {p} and for a run named '{raw}' in: "
                 + ", ".join(MODEL_SEARCH_DIRS) + " and up to two folder levels below checkpoints/ and runs/ "
                 "(relative to the repository root). List what is there with: find checkpoints -maxdepth 3 -type d")
    if p.is_dir():
        if (p / "best.pth").exists():                       # U-Net run folder
            return p / "best.pth", "unet", p
        if (p / "best" / "config.json").exists():           # SegFormer run folder
            return p / "best", "segformer", p
        if (p / "config.json").exists() and any((p / f).exists() for f in ("model.safetensors", "pytorch_model.bin")):
            return p, "segformer", p.parent                 # SegFormer model folder (best/ or last/)
        sys.exit(f"{p} is a folder but holds neither best.pth (U-Net) nor a saved SegFormer model (config.json + "
                 f"model.safetensors, or a best/ sub-folder).")
    return p, None, p.parent                                # a weights file


# ----------------------------------------------------------------------------------------------------------------------
# model specs and loading
# ----------------------------------------------------------------------------------------------------------------------
def parse_spec(spec):
    """'NAME=PATH@DEPTH_DIR#KIND' -> dict(name, raw, depth_dir, kind). NAME, DEPTH_DIR and KIND are optional."""
    name, rest = spec.split("=", 1) if "=" in spec else (None, spec)
    kind = None
    if "#" in rest:
        rest, kind = rest.rsplit("#", 1)
        kind = kind.strip().lower()
        if kind not in ("unet", "segformer"):
            sys.exit(f"--model {spec!r}: KIND must be unet or segformer")
    depth_dir = None
    if "@" in rest:
        rest, depth_dir = rest.split("@", 1)
    raw = rest.strip()
    if name is None:
        p = Path(raw)
        name = p.parent.name if p.name in ("best", "last", "best.pth") else p.stem
    return {"name": name.strip(), "raw": raw, "depth_dir": depth_dir.strip() if depth_dir else None, "kind": kind}


def load_checkpoint(path):
    """.pth/.pt -> (module or None, state_dict or None). Only load files you trust (pickle)."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # old torch without weights_only
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], (dict, torch.nn.Module)):
                obj = obj[k]
                break
    if isinstance(obj, torch.nn.Module):
        return obj, None
    if not isinstance(obj, dict):
        sys.exit(f"{path}: not a state_dict or a model (got {type(obj).__name__})")
    return None, {re.sub(r"^(module\.|model\.)+", "", k): v for k, v in obj.items()}


def detect_kind(sd, path):
    keys = list(sd)
    if any(k.startswith(("segformer.", "decode_head.")) for k in keys):
        return "segformer"
    if any(k.startswith("encoder.") for k in keys) and any(k.startswith(("decoder.", "segmentation_head.")) for k in keys):
        return "unet"
    sys.exit(f"{path}: cannot tell whether this is a U-Net or a SegFormer from its weight names; add #unet or #segformer.")


def build_unet(sd, encoder):
    """segmentation_models_pytorch Unet, as in scripts/unet/train_unet.py. Channels and classes come from the weights."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        sys.exit("U-Net checkpoints need segmentation_models_pytorch: pip install segmentation-models-pytorch")
    in_ch = int(sd["encoder.conv1.weight"].shape[1])
    n = int(sd["segmentation_head.0.weight"].shape[0])
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=in_ch, classes=n)
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        sys.exit(f"The weights do not fit smp.Unet('{encoder}', in_channels={in_ch}, classes={n}):\n{str(e)[:600]}\n"
                 "Use --encoder for another backbone.")
    return model, in_ch, n


def build_segformer_from_state(sd):
    """MiT-B2 SegFormer from a plain state_dict (a saved folder is loaded with from_pretrained instead)."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    in_ch = int(sd["segformer.encoder.patch_embeddings.0.proj.weight"].shape[1])
    n = int(sd["decode_head.classifier.weight"].shape[0])
    dec = int(sd["decode_head.linear_fuse.weight"].shape[0])
    cfg = SegformerConfig(num_labels=n, num_channels=in_ch, depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
                          num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=dec)
    model = SegformerForSemanticSegmentation(cfg)
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        sys.exit(f"The weights do not fit a MiT-B2 SegFormer:\n{str(e)[:600]}\nEdit build_segformer_from_state() for other variants.")
    return model, in_ch, n


def unet_folder(folder, suffix):
    """Folder named in a U-Net config.json (+ suffix). Old LunarHazardSeg names are mapped to this repository."""
    direct = resolve(folder + suffix)
    if _probe(direct).exists():
        return direct
    alias = UNET_PATH_ALIASES.get(str(folder).rstrip("/"))
    if alias is None:
        return direct
    if not suffix and "{split}" in alias:      # prediction folder: alias already has {split}
        return resolve(alias)
    return resolve(alias.replace("/{split}", "") + suffix)


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


class Net:
    """One loaded model + how to feed it."""

    def __init__(self, spec, args, device):
        self.name = spec["name"]
        self.path, hint, run_dir = locate_model(spec["raw"])
        kind = spec["kind"] or hint
        self.cfg, self.unet_cfg = {}, {}
        if self.path.is_dir():  # SegFormer folder saved by train_segformer_hf.py
            from transformers import SegformerForSemanticSegmentation
            if kind == "unet":
                sys.exit(f"{self.path} is a SegFormer folder, not a U-Net checkpoint")
            self.kind = "segformer"
            self.model = SegformerForSemanticSegmentation.from_pretrained(str(self.path), local_files_only=True)
            self.in_ch, self.n = self.model.config.num_channels, self.model.config.num_labels
            self.cfg = read_json(run_dir / "run_config.json")
        else:
            module, sd = load_checkpoint(self.path)
            self.unet_cfg = read_json(self.path.parent / "config.json")  # written by train_unet.py
            if module is not None:  # a whole pickled model
                self.kind = kind or ("segformer" if "segformer" in type(module).__name__.lower() else "unet")
                self.model = module
                if self.kind == "segformer":
                    self.in_ch, self.n = module.config.num_channels, module.config.num_labels
                else:
                    first = next(p for p in module.parameters() if p.ndim == 4)
                    self.in_ch, self.n = int(first.shape[1]), int(args.num_classes or 3)
            else:
                self.kind = kind or detect_kind(sd, self.path)
                if self.kind == "unet":
                    encoder = args.encoder or self.unet_cfg.get("encoder", "resnet34")
                    self.model, self.in_ch, self.n = build_unet(sd, encoder)
                else:
                    self.model, self.in_ch, self.n = build_segformer_from_state(sd)
        self.model.to(device).eval()
        if self.in_ch not in (3, 4):
            sys.exit(f"{self.name}: unsupported number of input channels ({self.in_ch})")

        # class order of this model's outputs
        if self.kind == "segformer":
            names = self.cfg.get("class_names") or T.class_names(self.n)
        else:
            names = UNET_CLASS_ORDER if self.n == 3 else [f"class_{i}" for i in range(self.n)]
        self.class_order = list(names)

        # depth input
        self.unet_mode = None
        self.depth_dir = resolve(spec["depth_dir"]) if spec["depth_dir"] else None
        if self.kind == "unet":
            self.unet_mode = self.unet_cfg.get("mode") or ("rgb" if self.in_ch == 3 else None)
            if self.in_ch == 4 and self.unet_mode not in ("rgbd", "rgbpd"):
                # no config.json: guess from the depth folder (arrays -> predicted depth, PNGs -> ground truth)
                if not self.depth_dir:
                    sys.exit(f"{self.name}: no config.json next to {self.path.name}, so the depth input is unknown. "
                             f'Add @DEPTH_DIR, e.g. --model "{self.name}={spec["raw"]}@LunarSeg/depth/{{split}}"')
                probe = T.fill(self.depth_dir, args.split)
                arrays = any(probe.glob("*.npy")) or any(probe.glob("*.npz")) or any((probe / args.split).glob("*.np[yz]"))
                self.unet_mode = "rgbpd" if arrays else "rgbd"
                print(f"note: {self.name} has no config.json; treating it as '{self.unet_mode}' from its depth folder")
            if self.in_ch == 4 and not self.depth_dir:
                if self.unet_mode == "rgbd":
                    self.depth_dir = unet_folder(f"{self.unet_cfg.get('root') or 'LunarSeg'}", "/depth/{split}")
                else:
                    self.depth_dir = unet_folder(self.unet_cfg.get("pred_depth_dir") or "data_pred", "")
        if self.in_ch == 4 and not self.depth_dir:
            sys.exit(f'{self.name} takes RGB-D input: add @DEPTH_DIR, e.g. --model "{self.name}={spec["raw"]}@LunarSeg/depth/{{split}}"')
        if self.in_ch == 3 and self.depth_dir:
            print(f"note: {self.name} is RGB-only, its depth folder is ignored")
            self.depth_dir = None

        # input size
        self.h = args.height or int(self.cfg.get("height", 480))
        self.w = args.width or int(self.cfg.get("width", 640))
        if self.kind == "unet" and (self.h % 32 or self.w % 32):
            sys.exit(f"{self.name}: a U-Net needs height and width divisible by 32 (got {self.h}x{self.w})")

    # -- inputs ----------------------------------------------------------------------------------------------------
    def read_depth(self, name, split, shape):
        """Depth channel exactly as the training pipeline built it. Returns (H x W float at network size, file)."""
        if self.kind == "segformer":
            exts = [self.cfg.get("depth_ext") or "png"] + ["png", "npy"]
            f = find_file(self.depth_dir, name, split, exts)
            if f is None:
                sys.exit(f"{self.name}: missing depth map {name}.png/.npy in {T.fill(self.depth_dir, split)}")
            if f.suffix == ".png":
                raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if raw is not None and raw.ndim == 3 and not (raw[..., 0] == raw[..., 2]).all():
                    print(f"warning: {f} is a colour image; depth PNGs should be grayscale (is this a visualisation?)")
            d = T.read_depth(f)
            if d.shape != shape:
                sys.exit(f"depth {f} is {d.shape[::-1]} but the image is {shape[::-1]} (WxH)")
            return cv2.resize(d, (self.w, self.h), interpolation=cv2.INTER_LINEAR), f
        size = (self.w, self.h)  # U-Net: same steps as scripts/unet/dataset.py
        if self.unet_mode == "rgbd":
            f = find_file(self.depth_dir, name, split)
            if f is None:
                sys.exit(f"{self.name}: missing depth map {name}.png in {T.fill(self.depth_dir, split)}")
            img = Image.open(f)
            if img.size != size:
                img = img.resize(size, Image.NEAREST)
            d = np.asarray(img)
            d = d[..., 0] if d.ndim == 3 else d
            d = d.astype(np.float32) / 255.0 if self.unet_cfg.get("gt_linear") else U.gt_depth_to_inverse(d)
            return d.astype(np.float32), f
        folder = T.fill(self.depth_dir, split)
        try:
            d = U.load_pred(str(folder), name, split=split)
        except FileNotFoundError as e:
            sys.exit(f"{self.name}: {e}")
        if self.unet_cfg.get("pred_invert"):
            d = U.pred_depth_to_disparity(d)
        if d.shape != (self.h, self.w):
            d = np.asarray(Image.fromarray(d.astype(np.float32)).resize(size, Image.BILINEAR))
        return U.pred_disparity_normalise(d).astype(np.float32), folder / f"{name}.npy"

    def make_input(self, rgb, depth):
        """rgb: H x W x 3 uint8, depth: h x w float (already pre-processed) or None -> 1 x C x h x w tensor."""
        if self.kind == "segformer":
            img = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        else:
            img = np.asarray(Image.fromarray(rgb).resize((self.w, self.h), Image.BILINEAR)) if rgb.shape[:2] != (self.h, self.w) else rgb
        x = (img.astype(np.float32) / 255.0 - T.MEAN) / T.STD
        if self.in_ch == 4:
            dch = (depth - T.DEPTH_MEAN) / T.DEPTH_STD if self.kind == "segformer" else depth
            x = np.concatenate([x, dch[..., None]], axis=2)
        return torch.from_numpy(np.ascontiguousarray(x.astype(np.float32).transpose(2, 0, 1))).unsqueeze(0)

    def logits(self, x):
        return self.model(pixel_values=x).logits if self.kind == "segformer" else self.model(x)


# ----------------------------------------------------------------------------------------------------------------------
# ground truth
# ----------------------------------------------------------------------------------------------------------------------
def read_ground_truth(path, colors, mode):
    """Label image -> class ids in CLASS_NAMES order. Colour masks (label_color/) and id masks (label/) both work."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        sys.exit(f"cannot read label {path}")
    is_color = raw.ndim == 3 and not ((raw[..., 0] == raw[..., 1]).all() and (raw[..., 0] == raw[..., 2]).all())
    if mode == "auto":
        mode = "color" if is_color else "id"
    if mode == "color":
        if raw.ndim != 3:
            sys.exit(f"--label-mode color was given but {path} is a single-channel image")
        return T.ColorMap(colors).apply(raw)
    ids = raw[..., 0] if raw.ndim == 3 else raw
    lut = np.full(256, T.IGNORE, np.uint8)  # label/ ids -> common order
    for i, nm in enumerate(LABEL_ID_ORDER):
        lut[i] = CLASS_NAMES.index(nm)
    return lut[ids]


# ----------------------------------------------------------------------------------------------------------------------
# metrics (one image)
# ----------------------------------------------------------------------------------------------------------------------
def image_metrics(pred, y, n, names):
    """Confusion-matrix metrics for one image. mIoU = mean IoU over the classes present in this image's ground truth
    (a class absent from the ground truth but predicted has IoU 0 and is listed, but does not enter the mean)."""
    keep = y < n
    conf = np.bincount(y[keep].astype(np.int64) * n + pred[keep].astype(np.int64), minlength=n * n).reshape(n, n).astype(np.float64)
    tp, rows, cols = np.diag(conf), conf.sum(1), conf.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(rows + cols - tp > 0, tp / (rows + cols - tp), np.nan)
        prec = np.where(cols > 0, tp / cols, np.nan)
        rec = np.where(rows > 0, tp / rows, np.nan)
    present = rows > 0
    total = conf.sum()
    return {"pixel_acc": float(tp.sum() / total) if total else float("nan"),
            "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
            "classes": {names[c]: {"gt_pixels_pct": float(100 * rows[c] / total) if total else float("nan"),
                                   "iou": float(iou[c]), "precision": float(prec[c]), "recall": float(rec[c])}
                        for c in range(n)}}


def hazard_metrics(pred, y, n, names):
    """Hazard-level detection / false alarms with the codebase's HazardStats (same size bins, 20 px / 50 % rule)."""
    hs = T.HazardStats(n)
    hs.update(pred.astype(np.uint8), y)
    s = hs.summary(names)
    return {c: {"by_size": {b: d[b] for b in ("small", "medium", "large", "all", "tiny")},
                "false_alarms": s["false_alarms"][c]["count"],
                "false_alarms_any_size": s["false_alarms"][c]["count_any_size"]}
            for c, d in s["detection"].items()}


# ----------------------------------------------------------------------------------------------------------------------
# display
# ----------------------------------------------------------------------------------------------------------------------
def colorize(cls_map, rgb, palette, n, alpha, mask_only):
    pal = np.array([palette[i % len(palette)] for i in range(n)], np.float32)
    if mask_only:
        out = np.zeros(cls_map.shape + (3,), np.float32)
        for c in range(n):
            out[cls_map == c] = pal[c]
        return out.astype(np.uint8)
    out = rgb.astype(np.float32).copy()
    for c in range(1, n):  # background stays transparent
        m = cls_map == c
        out[m] = (1 - alpha) * out[m] + alpha * pal[c]
    return out.astype(np.uint8)


def make_figure(rgb, gt, results, names, palette, args, title):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    n = len(names)
    rows = 1 + len(results)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.1 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    gt_show = np.where(gt < n, gt, 0)
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB")
    axes[0, 2].imshow(colorize(gt_show, rgb, palette, n, args.alpha, args.mask_only)); axes[0, 2].set_title("Ground truth")
    for r, res in enumerate(results, start=1):
        axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"{res['name']}: RGB", fontsize=9)
        if res["depth"] is not None:
            axes[r, 1].imshow(res["depth"], cmap="magma"); axes[r, 1].set_title(f"depth input ({res['depth_name']})", fontsize=9)
        m = res["metrics"]
        sub = "  ".join(f"{nm} {m['classes'][nm]['iou']:.3f}" if m['classes'][nm]['iou'] == m['classes'][nm]['iou'] else f"{nm} n/a"
                        for nm in names[1:])
        axes[r, 2].imshow(colorize(res["pred"], rgb, palette, n, args.alpha, args.mask_only))
        axes[r, 2].set_title(f"{res['name']}: prediction\nmIoU {m['miou']:.3f} | IoU {sub}", fontsize=9)
    handles = [Patch(color=np.array(palette[i % len(palette)]) / 255.0, label=names[i]) for i in range(n)]
    fig.legend(handles=handles, loc="lower center", ncol=n, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig, plt


# ----------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH[@DEPTH_DIR][#KIND]",
                    help="repeat for every model to compare")
    ap.add_argument("--image-dir", default="LunarSeg/image/{split}", help="default: LunarSeg/image/{split}")
    ap.add_argument("--label-dir", default=None,
                    help="default: LunarSeg/label_color/{split} if it exists, else LunarSeg/label")
    ap.add_argument("--split-dir", default="LunarSeg", help="folder with <split>.txt (default: LunarSeg)")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--image", help="image name (file stem), as listed in <split>.txt, e.g. 00_0002")
    sel.add_argument("--index", type=int, help="position in <split>.txt (default 0)")
    sel.add_argument("--random", action="store_true", help="pick a random image (see --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-mode", choices=["auto", "id", "color"], default="auto",
                    help="auto (default): colour masks -> colour mode, single-channel masks -> label/ ids")
    ap.add_argument("--color-map", default=None, help='class colours "R,G,B;R,G,B;R,G,B" for background, rock, crater')
    ap.add_argument("--height", type=int, default=None, help="network input height (default: run_config.json or 480)")
    ap.add_argument("--width", type=int, default=None, help="network input width (default: run_config.json or 640)")
    ap.add_argument("--encoder", default=None, help="U-Net backbone (default: from config.json, else resnet34)")
    ap.add_argument("--num-classes", type=int, default=None, help="only for whole pickled U-Net models")
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=4,
                    help="connected-component connectivity for the hazard analysis (default 4, as in training)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--no-amp", action="store_true", help="disable half precision on CUDA")
    ap.add_argument("--alpha", type=float, default=0.55, help="overlay opacity of the hazard colours")
    ap.add_argument("--mask-only", action="store_true", help="show plain colour masks instead of overlays on the RGB image")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: results/infer_compare in the repository")
    ap.add_argument("--no-show", action="store_true", help="only save the figure, do not open a window")
    args = ap.parse_args()
    T.HAZARD_CONNECTIVITY = args.connectivity
    out_dir = args.out_dir.expanduser() if args.out_dir else REPO_ROOT / "results" / "infer_compare"

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"device: {device}" + ("  (CPU: this will be slow)" if device.type == "cpu" else ""))

    # ---- choose the image ------------------------------------------------------------------------------------------
    split_txt = Path(resolve(args.split_dir)) / f"{args.split}.txt"
    if not split_txt.exists():
        sys.exit(f"{split_txt} not found. Pass --split-dir with the folder holding train.txt / validation.txt / test.txt.")
    names_all = [Path(l.strip()).stem for l in open(split_txt) if l.strip()]
    if args.image:
        if args.image not in names_all:
            sys.exit(f"{args.image} is not listed in {split_txt}")
        name = args.image
    elif args.random:
        name = random.Random(args.seed).choice(names_all)
    else:
        idx = args.index or 0
        if not 0 <= idx < len(names_all):
            sys.exit(f"--index must be between 0 and {len(names_all) - 1}")
        name = names_all[idx]
    image_dir = resolve(args.image_dir)
    img_path = find_file(image_dir, name, args.split)
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if img_path else None
    if bgr is None:
        sys.exit(f"cannot read image {name}.png in {T.fill(image_dir, args.split)}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"image: {name}  ({rgb.shape[1]}x{rgb.shape[0]}, split {args.split})")

    # ---- ground truth ------------------------------------------------------------------------------------------------
    if args.label_dir:
        label_dir = resolve(args.label_dir)
    else:
        label_dir = resolve("LunarSeg/label_color/{split}")
        if not _probe(label_dir).exists():
            label_dir = resolve("LunarSeg/label")
    lab_path = find_file(label_dir, name, args.split)
    if lab_path is None:
        sys.exit(f"missing label {name}.png in {T.fill(label_dir, args.split)} (set --label-dir)")
    colors = T.parse_colors(args.color_map) if args.color_map else CLASS_COLORS
    y = read_ground_truth(lab_path, colors, args.label_mode)
    if y.shape != rgb.shape[:2]:
        sys.exit(f"label {y.shape[::-1]} and image {rgb.shape[1::-1]} sizes differ")
    n, class_nm = len(CLASS_NAMES), list(CLASS_NAMES)
    print(f"label: {lab_path}")

    # ---- run the models one after the other (keeps GPU memory low) ---------------------------------------------------
    results = []
    for sp in [parse_spec(s) for s in args.model]:
        net = Net(sp, args, device)
        if net.n != n:
            sys.exit(f"{net.name} predicts {net.n} classes, expected {n} ({', '.join(CLASS_NAMES)})")
        if sorted(net.class_order) != sorted(CLASS_NAMES):
            sys.exit(f"{net.name}: unknown class names {net.class_order}")
        perm = [net.class_order.index(c) for c in CLASS_NAMES]  # model output channels -> common order
        print(f"{net.name}: {net.kind}, {net.in_ch} channels, weights {net.path}"
              + (f", depth {net.depth_dir}" if net.in_ch == 4 else ""))
        depth = depth_name = None
        if net.in_ch == 4:
            depth, dfile = net.read_depth(name, args.split, rgb.shape[:2])
            parts = [p for p in Path(dfile).parent.parts if p not in ("train", "validation", "test")]
            depth_name = parts[-1] if parts else ""
        x = net.make_input(rgb, depth).to(device)
        with torch.no_grad():
            def fwd():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    return net.logits(x)
            fwd()  # warm-up
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            lg = fwd()
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            lg = lg[:, perm]
            up = F.interpolate(lg.float(), size=y.shape, mode="bilinear", align_corners=False)
            pred = up.argmax(1)[0].cpu().numpy().astype(np.uint8)
        results.append({"name": net.name, "kind": net.kind, "in_ch": net.in_ch, "input": f"{net.h}x{net.w}", "ms": ms,
                        "depth": depth, "depth_name": depth_name, "pred": pred,
                        "metrics": image_metrics(pred, y, n, class_nm), "hazard": hazard_metrics(pred, y, n, class_nm)})
        del net, x, lg, up
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- report ----------------------------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    num = T.num
    T.table(f"Pixel metrics for {name}  (mIoU over the classes present in this image)",
            ["model", "kind", "input", "pixel acc", "mIoU", "forward ms"],
            [[r["name"], f"{r['kind']}{'+depth' if r['in_ch'] == 4 else ''}", r["input"], num(r["metrics"]["pixel_acc"]),
              num(r["metrics"]["miou"]), f"{r['ms']:.1f}"] for r in results], out_dir / f"{name}_summary.csv")
    rows = []
    for r in results:
        for c in class_nm:
            m = r["metrics"]["classes"][c]
            rows.append([r["name"], c, num(m["gt_pixels_pct"]), num(m["iou"]), num(m["precision"]), num(m["recall"])])
    T.table(f"Per-class metrics for {name}  (n/a = class absent from the ground truth and/or the prediction)",
            ["model", "class", "GT pixels %", "IoU", "precision", "recall"], rows, out_dir / f"{name}_classes.csv")
    rows = []
    for r in results:
        for c, h in r["hazard"].items():
            a = h["by_size"]["all"]
            rows.append([r["name"], c, a["hazards"], a["detected"], num(a["rate"]) if a["rate"] is not None else None,
                         h["by_size"]["small"]["hazards"], h["by_size"]["small"]["detected"], h["false_alarms"]])
    T.table(f"Hazard-level detection for {name}  (regions >= {T.HAZARD_MIN_PX} px; detected if >= {int(100 * T.HAZARD_HIT_FRAC)}% "
            f"of the pixels are predicted as the class; false alarm = predicted region >= {T.HAZARD_MIN_PX} px touching no labelled hazard)",
            ["model", "class", "hazards", "detected", "rate", "small haz.", "small det.", "false alarms"],
            rows, out_dir / f"{name}_hazards.csv")
    print("\nA rate of n/a means this image has no hazard of that class. Statistics from a single image are noisy: "
          "use them to understand failure cases, not to rank models.")

    def clean(o):  # JSON-safe (NaN -> null, numpy scalars -> python)
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if o != o else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    dump = [{k: r[k] for k in ("name", "kind", "in_ch", "input", "ms", "metrics", "hazard")} for r in results]
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(clean({"image": name, "models": dump}), indent=2))

    fig, plt = make_figure(rgb, y, results, class_nm, colors, args, f"{name}  ({args.split})")
    png = out_dir / f"{name}_compare.png"
    fig.savefig(png, dpi=150)
    print(f"\nsaved {png}, and the tables as CSV / JSON in {out_dir}/")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
