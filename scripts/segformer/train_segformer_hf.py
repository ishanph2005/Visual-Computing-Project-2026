#!/usr/bin/env python3
"""Fine-tune and evaluate SegFormer (Hugging Face transformers) for semantic segmentation on LunarSeg.

No mmcv / mmsegmentation needed: this runs on a current PyTorch + transformers install.

Dataset layout (same idea as finetune_lunar.py): one folder per split
  <root>/image/{train,validation,test}/<name>.png     RGB
  <root>/label/{train,validation,test}/<name>.png     per-pixel class ids (0..N-1; 255 = ignore)
  <root>/train.txt, validation.txt, test.txt           one <name> per line (folder given with --split-dir)
Write the folders as --image-dir "<root>/image/{split}" --label-dir "<root>/label/{split}"; {split} is filled in
automatically (keep the quotes). Every listed file is checked before training starts.

Offline: the script never downloads anything (HF_HUB_OFFLINE is forced on). --model must be a local folder holding
the pre-downloaded encoder (config.json + weights), e.g. the output of `hf download nvidia/mit-b2 --local-dir ./mit-b2`.

Number of classes: read from the training labels (largest id + 1) unless --num-classes is given. The class ids
found and their pixel shares are printed at start-up, so check that they look right.

RGB or RGB-D input (--in-channels 3 | 4, default 3 = RGB only)
  --in-channels 4 adds the depth map as a 4th input channel. Also pass --depth-dir "<root>/depth/{split}" (same {split}
  rule as --image-dir). Depth files are <name>.png (8-bit -> /255, 16-bit -> /65535) or, with --depth-ext npy,
  <name>.npy (float, values must already be 0-1, e.g. the output of infer_finetuned.py). Depth is resized like the RGB
  image, flipped together with it, and is not colour-jittered. The only architectural change is the first patch-embedding
  convolution (3 -> 4 input channels): the pretrained RGB weights are kept and the new depth weights start at zero
  (--depth-init zero, so training starts from exactly the pretrained RGB behaviour) or at the mean of the RGB weights
  (--depth-init mean). The 4-channel setting is stored in the saved model folder, so eval picks it up automatically
  (it still needs --depth-dir); --in-channels on eval is only a safety check against the checkpoint.

Subcommands
  train   fine-tune, validate every epoch, save <out-dir>/best and <out-dir>/last (Hugging Face folders).
          At the end: training curves (training_curves.png), per-epoch tables (printed + epoch_*.csv + history.json),
          then the best model is run on the test split with the full report below (skip with --no-test).
  plot    re-draw training_curves.png and re-print the per-epoch tables from <run-dir>/history.json
          (no GPU / dataset needed; --run-dir defaults to the most recent runs/*/)
  eval    full report of a saved model folder on a split (default: test). With --output-dir it also writes, per image,
          <name>_mask.png (colour-coded predicted mask) and <name>.npy (uint8 predicted class ids 0,1,2,... at the
          original label resolution); the report CSV/JSON files go there too (else next to the checkpoint folder).

Metric definitions (all printed as tables at the end and saved as CSV / JSON)
  mIoU            mean IoU over all classes (LunarSeg / LuSeg convention; class ids 0 = background, 1 = rock,
                  2 = crater). Background is ~98% of the pixels and scores ~0.995 for every model, so mIoU hides most
                  differences: read the rock and crater IoUs.
  precision/recall  pixel level, per class, from the confusion matrix (macro = mean over the classes present)
  hazard detection  every connected region (4-connectivity by default, --connectivity 8 to join diagonals) of rock / crater in the label is one hazard. It is detected
                  if >= 50% of its pixels are predicted as that class. Regions < 20 px are ignored (listed as "tiny",
                  not in any total). Size in label pixels, which roughly tracks distance: small 20-300 px (far),
                  medium 300-3000, large > 3000 (close).
  false alarms/image  predicted rock / crater regions (connected) that overlap no labelled hazard pixel, per image.
                  Counted for predicted regions >= 20 px (same minimum as above); the any-size count is also reported.
  GFLOPs, GMACs   per image at the network input size, model forward only (1 MAC = 2 FLOPs; the same for every image)
  latency         per image, batch 1: forward + upsample to label size + argmax, GPU-synchronised, after warm-up;
                  excludes image loading, pre-processing and the host-to-device copy
  peak GPU memory torch.cuda.max_memory_allocated. Training: peak of each epoch's training loop. Inference: peak over
                  the whole evaluation loop (includes the model weights)

Examples (one line each):
  python3 train_segformer_hf.py plot                       # from the segformer folder: plots the latest run in runs/
  python3 train_segformer_hf.py plot --run-dir runs/segformer_b0
  python3 train_segformer_hf.py train --image-dir "../LunarSeg/image/{split}" --label-dir "../LunarSeg/label_color/{split}" --label-mode color --split-dir ../LunarSeg --out-dir runs/segformer_b2
  python3 train_segformer_hf.py eval --image-dir "../LunarSeg/image/{split}" --label-dir "../LunarSeg/label_color/{split}" --label-mode color --split-dir ../LunarSeg --ckpt runs/segformer_b0/best --split test
  python3 train_segformer_hf.py eval ... --ckpt runs/segformer_b0/best --split test --output-dir runs/segformer_b0/preds
  python3 train_segformer_hf.py train --label-mode color --image-dir "../LunarSeg/image/{split}" --label-dir "../LunarSeg/label_color/{split}" --split-dir ../LunarSeg --in-channels 4 --depth-dir "../LunarSeg/depth/{split}" --out-dir runs/segformer_b2_rgbd
  python3 train_segformer_hf.py eval  ... --ckpt runs/segformer_b2_rgbd/best --depth-dir "../LunarSeg/depth/{split}"
  python3 train_segformer_hf.py eval --ckpt runs/segformer_b2_rgbd_dav2metric/best --depth-dir "../LunarSeg/finetuned/{split}" --image-dir "../LunarSeg/image/{split}" --label-dir "../LunarSeg/label_color/{split}" --split-dir ../LunarSeg/
"""
import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"        # this script never downloads anything: models are loaded from local folders
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEPTH_MEAN, DEPTH_STD = 0.5, 0.25  # fixed constants for the 0-1 depth channel (not computed from the data)
IGNORE = 255
HAZARD_MIN_PX = 20        # hazards / predicted regions smaller than this are ignored ("tiny"), as in the reference tables
HAZARD_SMALL_MAX = 300    # 20 <= area <  300 px           -> small  (far)
HAZARD_LARGE_MIN = 3000   # area > 3000 px                 -> large  (close); 300..3000 -> medium
HAZARD_CONNECTIVITY = 4   # connected-component connectivity for hazards / predicted regions (--connectivity 4|8)
HAZARD_HIT_FRAC = 0.5     # a hazard is detected if >= 50% of its pixels are predicted as its class
SIZE_BINS = ("tiny", "small", "medium", "large")  # index = bin id used by HazardStats
SIZE_LABELS = {"tiny": "tiny (<20px, not in totals)", "small": "small (20-300px)", "medium": "medium (300-3000px)",
               "large": "large (>3000px)", "all": "all sizes (>=20px)"}
CLASS_NAMES = None  # optional override set with --class-names (or restored from run_config.json / history.json)


def set_class_names(names):
    global CLASS_NAMES
    CLASS_NAMES = list(names) if names else None


def class_names(n):
    """Class id -> name. LunarSeg ids: 0 = background, 1 = rock, 2 = crater (checked against the reference tables)."""
    if CLASS_NAMES and len(CLASS_NAMES) == n:
        return list(CLASS_NAMES)
    return ["background", "rock", "crater"] if n == 3 else [f"class_{i}" for i in range(n)]


def fill(path, split):
    return Path(str(path).replace("{split}", split)).expanduser()


class ColorMap:
    """Colour-coded label images -> class ids. colors[i] is the (R, G, B) of class i; unknown colours -> 255 (ignored)."""

    def __init__(self, colors):
        self.colors = [tuple(int(v) for v in c) for c in colors]
        codes = np.array([(r << 16) | (g << 8) | b for r, g, b in self.colors], dtype=np.int64)
        order = np.argsort(codes)
        self.sorted_codes, self.sorted_ids = codes[order], order.astype(np.uint8)

    def apply(self, img):  # img: H x W x (3 or 4), BGR(A) as read by OpenCV
        rgb = img[..., :3][..., ::-1].astype(np.int64)
        codes = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        idx = np.clip(np.searchsorted(self.sorted_codes, codes), 0, len(self.sorted_codes) - 1)
        return np.where(self.sorted_codes[idx] == codes, self.sorted_ids[idx], IGNORE).astype(np.uint8)


def parse_colors(text):
    """'0,0,0;200,80,30;0,150,230' -> [(0,0,0), (200,80,30), (0,150,230)]  (class id = position in the list)."""
    return [tuple(int(v) for v in part.split(",")) for part in text.split(";") if part.strip()]


def discover_colors(ds, max_files=200, max_colors=12):
    """Find the colours used in the training labels: black first (class 0), then the others by pixel count."""
    counts = {}
    for n in ds.names[:max_files]:
        img = cv2.imread(str(ds.label_dir / f"{n}.png"), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3:
            sys.exit(f"--label-mode color needs 3-channel colour label images, but {n}.png is not one.")
        rgb = img[..., :3][..., ::-1].reshape(-1, 3).astype(np.int64)
        u, c = np.unique((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2], return_counts=True)
        for code, k in zip(u.tolist(), c.tolist()):
            counts[code] = counts.get(code, 0) + k
    if len(counts) > max_colors:
        sys.exit(f"Found {len(counts)} distinct colours in the labels (more than {max_colors}). They look "
                 f"anti-aliased or compressed. Pass the exact class colours with --color-map \"R,G,B;R,G,B;...\".")
    order = sorted(counts, key=lambda k: (k != 0, -counts[k]))  # black (0,0,0) first, rest by frequency
    return ColorMap([((c >> 16) & 255, (c >> 8) & 255, c & 255) for c in order])


def read_label(path, cmap=None):
    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise IOError(f"Could not read label {path}")
    if lab.ndim == 3:
        if cmap is not None:
            return cmap.apply(lab)
        c = lab[..., :3]
        if (c[..., 0] == c[..., 1]).all() and (c[..., 0] == c[..., 2]).all():
            return c[..., 0]
        sys.exit(f"{path} is a colour-coded label image, not class ids. Re-run with --label-mode color "
                 f"(optionally --color-map \"R,G,B;R,G,B;...\", class id = position in the list).")
    if cmap is not None:
        sys.exit(f"--label-mode color was given but {path} is a single-channel image.")
    return lab


def read_depth(path):
    """Depth map as float32 H x W in 0-1. PNG: 8-bit /255, 16-bit /65535. NPY: must already be 0-1."""
    if Path(path).suffix == ".npy":
        d = np.load(path).astype(np.float32)
        if d.ndim == 3:
            d = d[..., 0] if d.shape[-1] in (1, 3) else d[0]
        if d.max() > 1.0 + 1e-3 or d.min() < -1e-3:
            sys.exit(f"{path}: values are in [{d.min():.3f}, {d.max():.3f}], but .npy depth must already be in 0-1 "
                     f"(normalise it first, e.g. with infer_finetuned.py)")
        return np.clip(d, 0, 1)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"Could not read depth {path}")
    if raw.ndim == 3:
        raw = raw[..., 0]
    return raw.astype(np.float32) / (65535.0 if raw.dtype == np.uint16 else 255.0)


class LunarSegDataset(Dataset):
    def __init__(self, image_dir, label_dir, split_dir, split, h, w, train, depth_dir=None, depth_ext="png"):
        self.image_dir, self.label_dir = fill(image_dir, split), fill(label_dir, split)
        self.depth_dir, self.depth_ext = (fill(depth_dir, split) if depth_dir else None), depth_ext
        self.names = [Path(l.strip()).stem for l in open(Path(split_dir).expanduser() / f"{split}.txt") if l.strip()]
        self.h, self.w, self.train = h, w, train
        self.cmap = None  # set to a ColorMap when the labels are colour images
        miss_img = [n for n in self.names if not (self.image_dir / f"{n}.png").exists()]
        miss_lab = [n for n in self.names if not (self.label_dir / f"{n}.png").exists()]
        miss_dep = ([n for n in self.names if not (self.depth_dir / f"{n}.{depth_ext}").exists()]
                    if self.depth_dir else [])
        if miss_img or miss_lab or miss_dep:
            def line(kind, folder, miss, ext="png"):
                if not miss:
                    return f"  {kind}: all {len(self.names)} found in {folder}"
                return (f"  {kind}: {len(miss)}/{len(self.names)} MISSING in {folder} "
                        f"(looked for <name>.{ext}, e.g. {miss[0]}.{ext})")
            sys.exit(f"[{split}] listed files not found\n" + line("images", self.image_dir, miss_img) + "\n"
                     + line("labels", self.label_dir, miss_lab) + "\n"
                     + (line("depth", self.depth_dir, miss_dep, depth_ext) + "\n" if self.depth_dir else "")
                     + f"  Check the folder names and the label file naming with: ls <folder> | head\n"
                     f"  For per-split folders pass e.g. --image-dir <root>/image/{{split}} --label-dir <root>/label/{{split}}"
                     + (f" --depth-dir <root>/depth/{{split}}" if self.depth_dir else ""))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        bgr = cv2.imread(str(self.image_dir / f"{name}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not read image {name}")
        label = read_label(self.label_dir / f"{name}.png", self.cmap)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        depth = None
        if self.depth_dir is not None:  # optional 4th channel, resized exactly like the RGB image
            depth = read_depth(self.depth_dir / f"{name}.{self.depth_ext}")
            if depth.shape != bgr.shape[:2]:
                raise ValueError(f"depth {name} is {depth.shape[::-1]} but the image is {bgr.shape[1::-1]} (WxH)")
            depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if self.train:
            label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            if random.random() < 0.5:  # horizontal flip, same for image, depth and label
                rgb, label = rgb[:, ::-1], label[:, ::-1]
                depth = depth[:, ::-1] if depth is not None else None
            rgb = np.clip(rgb * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)  # RGB only, not depth
        x = (np.ascontiguousarray(rgb) - MEAN) / STD
        if depth is not None:
            x = np.concatenate([x, ((np.ascontiguousarray(depth) - DEPTH_MEAN) / DEPTH_STD)[..., None]], axis=2)
        x = torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1).copy())
        y = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))
        return (x, y) if self.train else (x, y, name)


def scan_classes(ds, max_files=300):
    """Class ids (and pixel shares) in a sample of the training labels."""
    counts = {}
    for n in ds.names[:max_files]:
        ids, c = np.unique(read_label(ds.label_dir / f"{n}.png", ds.cmap), return_counts=True)
        for i, k in zip(ids.tolist(), c.tolist()):
            counts[i] = counts.get(i, 0) + k
    total = sum(counts.values())
    return {i: counts[i] / total for i in sorted(counts)}


def build_model(args, num_labels):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    id2label = dict(enumerate(class_names(num_labels)))
    extra = {"decoder_hidden_size": args.decoder_dim} if args.decoder_dim else {}
    if not args.random_init and not Path(args.model).expanduser().is_dir():
        sys.exit(f"--model must be a local folder with the pre-downloaded encoder (config.json + weights), "
                 f"but '{args.model}' is not a folder. Nothing is downloaded by this script.")
    if args.random_init:  # no download; used for smoke tests
        return SegformerForSemanticSegmentation(SegformerConfig(num_labels=num_labels, id2label=id2label,
                                                                num_channels=args.in_channels, **extra))
    model = SegformerForSemanticSegmentation.from_pretrained(  # always loaded as the pretrained 3-channel RGB model
        str(Path(args.model).expanduser()), local_files_only=True, num_labels=num_labels, id2label=id2label, **extra,
        label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True)
    if args.in_channels == 4:
        add_depth_channel(model, args.depth_init)
    return model


def add_depth_channel(model, init="zero"):
    """3 -> 4 input channels. Only the first patch-embedding convolution changes: the pretrained RGB weights are kept
    and the new depth-channel weights start at zero (the model then behaves exactly like the pretrained RGB model at
    step 0) or at the mean of the RGB weights. config.num_channels is updated so save_pretrained / from_pretrained
    rebuild the 4-channel model later."""
    # the first conv is the only Conv2d with 3 input channels (the module path differs between transformers versions:
    # segformer.encoder.patch_embeddings.0.proj in 4.x, segformer.stages.0.patch_embeddings.proj in 5.x)
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.Conv2d) and m.in_channels == model.config.num_channels]
    if not cands:
        sys.exit("Could not find the first patch-embedding convolution of this SegFormer to extend to 4 channels")
    path, old = cands[0]
    parent_path, attr = path.rsplit(".", 1)
    new = torch.nn.Conv2d(old.in_channels + 1, old.out_channels, old.kernel_size, old.stride, old.padding,
                          dilation=old.dilation, groups=old.groups, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :old.in_channels] = old.weight
        new.weight[:, old.in_channels:] = old.weight.mean(1, keepdim=True) if init == "mean" else 0.0
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(model.get_submodule(parent_path), attr, new.to(old.weight.device))
    model.config.num_channels = old.in_channels + 1


DEFAULT_PALETTE = [(0, 0, 0), (200, 80, 30), (0, 150, 230), (60, 180, 75), (255, 225, 25), (145, 30, 180),
                   (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255)]


def save_predictions(pred, names, out_dir, palette):
    """pred: B x H x W int tensor of class ids. Writes <name>.npy (uint8 ids) and <name>_mask.png (colour-coded)."""
    pal = np.array([palette[i % len(palette)] for i in range(max(len(palette), 256))], np.uint8)  # RGB lookup
    for p, name in zip(pred.cpu().numpy().astype(np.uint8), names):
        np.save(out_dir / f"{name}.npy", p)
        cv2.imwrite(str(out_dir / f"{name}_mask.png"), pal[p][..., ::-1])  # RGB -> BGR for OpenCV


def num(v):
    """float -> None when NaN (keeps the JSON valid and the tables readable)."""
    v = float(v)
    return None if v != v else v


def _f(v):
    return "n/a" if v is None else f"{v:.4f}"


class HazardStats:
    """Hazard-level detection and false alarms, accumulated image by image (classes 1..n-1 are hazards)."""

    def __init__(self, n):
        self.n, self.classes, self.images = n, list(range(1, n)), 0
        self.total = {c: np.zeros(4, np.int64) for c in self.classes}  # per size bin: tiny, small, medium, large
        self.hit = {c: np.zeros(4, np.int64) for c in self.classes}
        self.fa = {c: 0 for c in self.classes}      # false-alarm regions >= HAZARD_MIN_PX
        self.fa_any = {c: 0 for c in self.classes}  # false-alarm regions of any size

    def update(self, pred, y):  # pred: H x W uint8 class ids, y: H x W label ids (255 = ignore)
        self.images += 1
        any_haz = (y > 0) & (y < self.n)  # labelled hazard pixels of any class
        for c in self.classes:
            lab = y == c
            if lab.any():
                k, lbl, stats, _ = cv2.connectedComponentsWithStats(lab.astype(np.uint8), connectivity=HAZARD_CONNECTIVITY)
                area = stats[1:, cv2.CC_STAT_AREA]
                hit_px = np.bincount(lbl.ravel(), weights=(pred == c).ravel().astype(np.float64), minlength=k)[1:]
                detected = hit_px >= HAZARD_HIT_FRAC * area
                b = np.where(area < HAZARD_MIN_PX, 0,
                             np.where(area < HAZARD_SMALL_MAX, 1, np.where(area > HAZARD_LARGE_MIN, 3, 2)))
                self.total[c] += np.bincount(b, minlength=4)
                self.hit[c] += np.bincount(b[detected], minlength=4)
            prd = pred == c
            if prd.any():  # predicted regions touching no labelled hazard = false alarms
                k, plbl, pstats, _ = cv2.connectedComponentsWithStats(prd.astype(np.uint8), connectivity=HAZARD_CONNECTIVITY)
                parea = pstats[1:, cv2.CC_STAT_AREA]
                touched = np.bincount(plbl[any_haz], minlength=k)[1:] > 0
                fa = ~touched
                self.fa_any[c] += int(fa.sum())
                self.fa[c] += int((fa & (parea >= HAZARD_MIN_PX)).sum())

    def summary(self, names):
        def cell(t, h):
            return {"hazards": int(t), "detected": int(h), "rate": float(h) / float(t) if t else None}

        def row(t, h):  # "all" covers small + medium + large only; tiny is reported separately
            r = {bn: cell(t[i], h[i]) for i, bn in enumerate(SIZE_BINS) if bn != "tiny"}
            r["all"] = cell(t[1:].sum(), h[1:].sum())
            r["tiny"] = cell(t[0], h[0])
            return r
        det, tot, hit = {}, np.zeros(4, np.int64), np.zeros(4, np.int64)
        for c in self.classes:
            det[names[c]] = row(self.total[c], self.hit[c])
            tot, hit = tot + self.total[c], hit + self.hit[c]
        det["all_hazards"] = row(tot, hit)
        imgs = max(1, self.images)
        fa = {names[c]: {"count": self.fa[c], "per_image": self.fa[c] / imgs,
                         "count_any_size": self.fa_any[c], "per_image_any_size": self.fa_any[c] / imgs}
              for c in self.classes}
        n1, n2 = sum(self.fa.values()), sum(self.fa_any.values())
        fa["all_hazards"] = {"count": n1, "per_image": n1 / imgs, "count_any_size": n2, "per_image_any_size": n2 / imgs}
        return {"detection": det, "false_alarms": fa}


def measure_gflops(model, h, w, device):
    """GFLOPs (1 MAC = 2 FLOPs) of one forward pass at the network input size, batch 1. None if unavailable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(pixel_values=torch.zeros(1, model.config.num_channels, h, w, device=device))
        return fc.get_total_flops() / 1e9
    except Exception as e:  # old torch without FlopCounterMode, etc.
        print(f"warning: could not count FLOPs ({e}); needs torch >= 2.1")
        return None


def latency_stats(lat):
    if not lat:
        return None
    a = np.asarray(lat)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "std_ms": float(a.std()), "min_ms": float(a.min()), "max_ms": float(a.max()), "fps": float(1e3 / a.mean())}


@torch.no_grad()
def evaluate(model, loader, device, use_amp, n, out_dir=None, palette=None, full=False):
    """Pixel metrics (IoU, precision, recall, accuracy) and loss. full=True (batch size 1) also measures per-image
    latency and peak GPU memory and runs the hazard-level analysis."""
    model.eval()
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    conf = torch.zeros(n, n, dtype=torch.long, device=device)
    loss_sum, loss_cnt = torch.zeros((), device=device), 0
    lat, haz = [], (HazardStats(n) if full else None)
    pic, pnames = [], []  # per-image confusion matrices (for bootstrap CIs) and image names
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        palette = palette or DEFAULT_PALETTE
    if full and cuda:
        sync(); torch.cuda.reset_peak_memory_stats()
    for k, (x, y, names) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if full and k == 0:  # warm-up (cuDNN autotune, allocator, lazy init) so it does not pollute the timings
            for _ in range(3):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    model(pixel_values=x)
            sync()
        if full:
            sync(); t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(pixel_values=x).logits
        up = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = up.argmax(1)
        if full:
            sync(); lat.append((time.perf_counter() - t0) * 1e3)
        keep = (y >= 0) & (y < n)  # drops the ignore label (255) and any id outside the class range
        if keep.any():
            y_safe = torch.where(keep, y, torch.full_like(y, IGNORE))
            loss_sum += F.cross_entropy(up, y_safe, ignore_index=IGNORE) * x.shape[0]  # same loss as in training
            loss_cnt += x.shape[0]
        conf += torch.bincount(y[keep] * n + pred[keep], minlength=n * n).view(n, n)
        if out_dir is not None:
            save_predictions(pred, names, out_dir, palette)
        if full:
            for b in range(pred.shape[0]):
                kb = keep[b]
                pic.append(torch.bincount(y[b][kb] * n + pred[b][kb], minlength=n * n).view(n, n).cpu())
            pnames.extend(names)
            p_np, y_np = pred.cpu().numpy().astype(np.uint8), y.cpu().numpy()
            for b in range(p_np.shape[0]):
                haz.update(p_np[b], y_np[b])
    peak = torch.cuda.max_memory_allocated() / 2**30 if (full and cuda) else None
    conf = conf.double().cpu()
    diag, rows, cols = conf.diag(), conf.sum(1), conf.sum(0)
    iou = diag / (rows + cols - diag)
    iou[rows == 0] = float("nan")  # class absent from this split
    prec, rec = diag / cols, diag / rows  # NaN when the class is never predicted / absent
    res = {"miou": num(torch.nanmean(iou)), "pixel_acc": num(diag.sum() / conf.sum()),
           "iou": [num(v) for v in iou], "precision": [num(v) for v in prec], "recall": [num(v) for v in rec],
           "macro_precision": num(torch.nanmean(prec)), "macro_recall": num(torch.nanmean(rec)),
           "pixel_share": [num(v) for v in rows / rows.sum()],
           "val_loss": num(loss_sum.item() / loss_cnt) if loss_cnt else None}
    if full:
        res.update({"latency": latency_stats(lat), "peak_gpu_gib_inference": peak, "images": len(lat),
                    "hazards": haz.summary(class_names(n))})
        res["per_image_conf"] = torch.stack(pic).numpy()  # [N, n, n]; removed again in report_eval (not JSON)
        res["image_names"] = pnames
    return res


def full_eval(model, ds, device, use_amp, n, workers, out_dir=None, palette=None):
    """Everything: pixel metrics, hazard detection, GFLOPs, params, latency, peak GPU memory (inference)."""
    gflops = measure_gflops(model, ds.h, ds.w, device)  # before evaluate(), which resets the memory peak
    res = evaluate(model, DataLoader(ds, 1, shuffle=False, num_workers=workers), device, use_amp, n,
                   out_dir=out_dir, palette=palette, full=True)
    res.update({"gflops": gflops, "gmacs": None if gflops is None else gflops / 2,
                "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
                "input_size": f"{ds.h}x{ds.w}", "in_channels": model.config.num_channels, "precision_mode": "fp16 (AMP)" if use_amp else "fp32",
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})
    return res


def fmt(res, names=None):
    names = names or [str(i) for i in range(len(res["iou"]))]
    per = ", ".join(f"{nm}:{_f(v)}" for nm, v in zip(names, res["iou"]))
    return (f"loss {_f(res['val_loss'])} | mIoU {_f(res['miou'])} | pixel acc {_f(res['pixel_acc'])} | "
            f"IoU per class [{per}]")


def table(title, headers, rows, csv_path=None):
    """Print an aligned table (None -> n/a, floats 4 dp) and optionally save it as CSV."""
    cells = [[v if isinstance(v, str) else (_f(v) if isinstance(v, float) or v is None else str(v)) for v in r]
             for r in rows]
    w = [max([len(str(h))] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))
    print(f"\n{title}")
    print(line([str(h) for h in headers]))
    print("-" * (sum(w) + 2 * (len(w) - 1)))
    for r in cells:
        print(line(r))
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(headers)
            wr.writerows([["" if v is None else v for v in r] for r in rows])


def plot_training(history, names, path):
    """Left: train and val loss vs epoch. Right: val mIoU and per-class IoU vs epoch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: skipping the training plots (pip install matplotlib)")
        return
    nan = lambda v: np.nan if v is None else v
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(ep, [nan(h["train_loss"]) for h in history], "o-", label="train loss")
    ax[0].plot(ep, [nan(h["val_loss"]) for h in history], "s-", label="val loss")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Training and validation loss")
    for i, nm in enumerate(names):
        ax[1].plot(ep, [nan(h["iou"][i]) for h in history], "o-", ms=3, label=f"IoU {nm}")
    ax[1].plot(ep, [nan(h["val_miou"]) for h in history], "k--", lw=2.2, label="val mIoU")
    mi = [nan(h["val_miou"]) for h in history]
    if not np.all(np.isnan(mi)):
        best = int(np.nanargmax(mi))
        ax[1].axvline(ep[best], color="gray", ls=":", lw=1)
        ax[1].annotate(f"best epoch {ep[best]}", (ep[best], mi[best]), textcoords="offset points", xytext=(-78, -16))
    ax[1].set(xlabel="epoch", ylabel="IoU", title="Validation IoU per class and mIoU", ylim=(0, 1.02))
    from matplotlib.ticker import MaxNLocator
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))  # whole epochs only
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved training curves -> {path}")


def report_training(history, names, out, peak_train):
    out = Path(out)
    (out / "history.json").write_text(json.dumps({"peak_train_gpu_gib": peak_train, "class_names": names, "epochs": history}, indent=2))
    table("Per-epoch training / validation metrics",
          ["epoch", "time_s", "train_loss", "val_loss", "val_mIoU"] + [f"IoU_{m}" for m in names]
          + ["pixel_acc", "train_peak_GiB"],
          [[h["epoch"], h["epoch_time_s"], h["train_loss"], h["val_loss"], h["val_miou"]] + h["iou"]
           + [h["pixel_acc"], h["train_peak_gpu_gib"]] for h in history], out / "epoch_metrics.csv")
    table("Per-epoch validation precision / recall (pixel level)",
          ["epoch"] + [f"P_{m}" for m in names] + [f"R_{m}" for m in names] + ["macro_P", "macro_R"],
          [[h["epoch"]] + h["precision"] + h["recall"] + [h["macro_precision"], h["macro_recall"]] for h in history],
          out / "epoch_precision_recall.csv")
    print("\nPeak GPU memory during training: " +
          (f"{peak_train:.3f} GiB" if peak_train is not None else "n/a (CPU run)"))
    plot_training(history, names, out / "training_curves.png")


def iou_from_conf(c):
    """IoU per class from confusion matrices c[..., n, n] (rows = label, cols = prediction)."""
    c = c.astype(np.float64)
    diag, rows, cols = np.einsum("...ii->...i", c), c.sum(-1), c.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = diag / (rows + cols - diag)
    iou[rows == 0] = np.nan
    return iou


def bootstrap_iou(conf, conf_base=None, n_boot=1000, seed=0, chunk=100):
    """Bootstrap over test images. Each resample draws N images with replacement, sums their confusion matrices and
    computes the dataset-level IoU. With conf_base, the SAME resampled images are used for the baseline and the
    returned values are IoU(model) - IoU(baseline) (paired bootstrap). Returns an array [n_boot, n_classes]."""
    rng = np.random.default_rng(seed)
    N, out = conf.shape[0], []
    for s0 in range(0, n_boot, chunk):
        idx = rng.integers(0, N, size=(min(chunk, n_boot - s0), N))
        r = iou_from_conf(conf[idx].sum(1))
        if conf_base is not None:
            r = r - iou_from_conf(conf_base[idx].sum(1))
        out.append(r)
    return np.concatenate(out)


def report_bootstrap(res, names, tag, out, pic, inames, baseline, n_boot, seed):
    """Print / save 95% bootstrap CIs of the per-class IoU, and (with a baseline) of the IoU gain in points."""
    out = Path(out)
    np.savez_compressed(out / f"{tag}_per_image_conf.npz", conf=pic, names=np.array(inames))
    iou = iou_from_conf(pic.sum(0))
    bs = bootstrap_iou(pic, n_boot=n_boot, seed=seed)
    ci = lambda a: (np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0))
    lo, hi = ci(bs)
    m_lo, m_hi = ci(np.nanmean(bs, axis=1))
    miou = float(np.nanmean(iou))
    rows = [[nm, float(iou[i]), float(lo[i]), float(hi[i])] for i, nm in enumerate(names)]
    rows.append(["mIoU", miou, float(m_lo), float(m_hi)])
    table(f"IoU with 95% bootstrap CI [{tag}]  ({n_boot} resamples of {len(pic)} images, percentile method)",
          ["class", "IoU", "CI low", "CI high"], rows, out / f"{tag}_iou_ci.csv")
    res["bootstrap"] = {"n_boot": n_boot, "seed": seed, "n_images": len(pic),
                        "iou_ci95": {nm: [float(lo[i]), float(hi[i])] for i, nm in enumerate(names)},
                        "miou_ci95": [float(m_lo), float(m_hi)]}
    if baseline:
        base = np.load(baseline, allow_pickle=False)
        if list(base["names"]) != list(inames):
            sys.exit(f"--baseline-conf {baseline} was computed on different images / a different order; "
                     "a paired bootstrap needs the same test images in the same order")
        gain = (iou - iou_from_conf(base["conf"].sum(0))) * 100  # IoU points
        bg = bootstrap_iou(pic, base["conf"], n_boot=n_boot, seed=seed) * 100
        glo, ghi = ci(bg)
        mg = (miou - float(np.nanmean(iou_from_conf(base["conf"].sum(0))))) * 100
        mglo, mghi = ci(np.nanmean(bg, axis=1))
        rows = [[nm, float(gain[i]), float(glo[i]), float(ghi[i]),
                 "real (CI excludes 0)" if glo[i] > 0 or ghi[i] < 0 else "CI includes 0"]
                for i, nm in enumerate(names)]
        rows.append(["mIoU", mg, float(mglo), float(mghi), "real (CI excludes 0)" if mglo > 0 or mghi < 0 else "CI includes 0"])
        table(f"Gain over baseline in IoU points, paired bootstrap 95% CI [{tag}]  (baseline: {baseline})",
              ["class", "gain", "CI low", "CI high", "note"], rows, out / f"{tag}_gain_ci.csv")
        print("note: the CI covers test-set sampling only, not training randomness (compare with the seed-to-seed spread).")
        res["bootstrap"]["gain_points_vs_baseline"] = {
            "baseline": str(baseline),
            "per_class": {nm: [float(gain[i]), float(glo[i]), float(ghi[i])] for i, nm in enumerate(names)},
            "miou": [mg, float(mglo), float(mghi)]}


def report_eval(res, names, tag, out_dir, baseline=None, n_boot=1000, boot_seed=0):
    """Print the full report for one split as tables and save CSVs + <tag>_metrics.json into out_dir."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pic, inames = res.pop("per_image_conf", None), res.pop("image_names", None)
    if pic is not None:
        report_bootstrap(res, names, tag, out, pic, inames, baseline, n_boot, boot_seed)
    (out / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    n = len(names)
    print(f"\n{'=' * 24} {tag} report {'=' * 24}")
    rows = [[nm, None if res["pixel_share"][i] is None else 100 * res["pixel_share"][i], res["iou"][i],
             res["precision"][i], res["recall"][i]] for i, nm in enumerate(names)]
    rows.append(["mean (mIoU / macro P, R)", 100.0, res["miou"], res["macro_precision"], res["macro_recall"]])
    table(f"Pixel-level metrics per class [{tag}]  (pixel accuracy {_f(res['pixel_acc'])})",
          ["class", "gt_pixels_%", "IoU", "precision", "recall"], rows, out / f"{tag}_per_class.csv")
    print("note: mIoU is the mean over " + (", ".join(names[:-1]) + " and " + names[-1]) +
          ". Background is ~98% of pixels and scores ~0.995 for every model, so read the "
          + " and ".join(names[1:]) + " IoUs.")

    lt = res["latency"] or {}
    g = lambda k: lt.get(k)
    table(f"Efficiency [{tag}]  (per image, batch 1)", ["metric", "value"], [
        ["device", res["device"]], ["input size (HxW)", res["input_size"]],
        ["input channels", res.get("in_channels", 3)], ["precision", res["precision_mode"]],
        ["images timed", res["images"]], ["params (M)", res["params_m"]],
        ["GFLOPs / image (1 MAC = 2 FLOPs)", res["gflops"]], ["GMACs / image", res["gmacs"]],
        ["latency mean (ms)", g("mean_ms")], ["latency median (ms)", g("median_ms")], ["latency p95 (ms)", g("p95_ms")],
        ["latency std (ms)", g("std_ms")], ["latency min (ms)", g("min_ms")], ["latency max (ms)", g("max_ms")],
        ["throughput (FPS, 1/mean)", g("fps")],
        ["peak GPU memory, inference (GiB)", res["peak_gpu_gib_inference"]]], out / f"{tag}_efficiency.csv")
    print("note: latency = forward + upsample to label size + argmax, GPU-synchronised, after warm-up; excludes "
          "image loading / pre-processing. GFLOPs = model forward at the input size.")

    det = res["hazards"]["detection"]
    rows = [[cls, SIZE_LABELS[s], v["hazards"], v["detected"], v["rate"]]
            for cls, d in det.items() for s, v in d.items()]
    table(f"Hazard-level detection [{tag}]  (detected = >=50% of the region's pixels predicted as its class; "
          f"regions <{HAZARD_MIN_PX}px are listed as tiny and left out of the totals)",
          ["class", "size", "hazards", "detected", "detection_rate"], rows, out / f"{tag}_hazard_detection.csv")
    fa = res["hazards"]["false_alarms"]
    table(f"False alarms [{tag}]  (predicted regions touching no labelled hazard; main columns: regions >= "
          f"{HAZARD_MIN_PX}px)",
          ["predicted class", "regions (>=20px)", "per image (>=20px)", "regions (any size)", "per image (any size)"],
          [[cls, v["count"], v["per_image"], v["count_any_size"], v["per_image_any_size"]] for cls, v in fa.items()],
          out / f"{tag}_false_alarms.csv")
    print(f"\nReport files written to {out}/ ({tag}_metrics.json and {tag}_*.csv)")


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_train(args):
    from transformers import SegformerForSemanticSegmentation
    device = pick_device()
    use_amp = device.type == "cuda" and not args.no_amp
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.in_channels = args.in_channels or 3
    args.depth_ext = args.depth_ext or "png"
    if args.in_channels == 4 and not args.depth_dir:
        sys.exit('--in-channels 4 needs --depth-dir, e.g. --depth-dir "../LunarSeg/depth/{split}"')
    if args.in_channels == 3 and args.depth_dir:
        print("note: --depth-dir is ignored because --in-channels is 3 (RGB only)")
    depth_dir = args.depth_dir if args.in_channels == 4 else None
    mk = lambda split, train: LunarSegDataset(args.image_dir, args.label_dir, args.split_dir, split,
                                              args.height, args.width, train, depth_dir, args.depth_ext)
    train_ds, val_ds = mk("train", True), mk("validation", False)
    if args.max_samples:
        train_ds.names, val_ds.names = train_ds.names[:args.max_samples], val_ds.names[:args.max_samples]

    cmap = None
    if args.label_mode == "color":
        cmap = ColorMap(parse_colors(args.color_map)) if args.color_map else discover_colors(train_ds)
        train_ds.cmap = val_ds.cmap = cmap
        print("Colour -> class id: " + ", ".join(f"{i}: RGB{c}" for i, c in enumerate(cmap.colors)))
        print("  (class 0 is black / background when black is present; check the others match what you expect)")

    shares = scan_classes(train_ds)
    print("Class ids in the training labels (share of pixels): " +
          ", ".join(f"{i}: {s * 100:.1f}%" for i, s in shares.items() if i != IGNORE) +
          (f"  [ignore label {IGNORE}: {shares[IGNORE] * 100:.1f}%]" if IGNORE in shares else ""))
    n = args.num_classes or (max(i for i in shares if i != IGNORE) + 1)
    print(f"Training with {n} classes")
    print("Input: 3 channels (RGB only)" if args.in_channels == 3 else
          f"Input: 4 channels (RGB + depth from {train_ds.depth_dir}, .{args.depth_ext}; "
          f"depth weights of the first conv start at {'zero' if args.depth_init == 'zero' else 'the RGB mean'})")
    names = class_names(n)

    model = build_model(args, n).to(device)
    print(f"Model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
          f"(decoder width {model.config.decoder_hidden_size}), class names {names}")
    head = [p for k, p in model.named_parameters() if "decode_head" in k]
    body = [p for k, p in model.named_parameters() if "decode_head" not in k]
    groups = [{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}]
    base = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=device.type == "cuda")
    vl = DataLoader(val_ds, 1, shuffle=False, num_workers=args.workers)
    steps_per_epoch = max(1, len(tl) // args.accum)
    total = args.epochs * steps_per_epoch
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({"height": args.height, "width": args.width, "num_classes": n,
                                                      "model": str(args.model), "class_names": names,
                                                      "decoder_dim": model.config.decoder_hidden_size,
                                                      "in_channels": args.in_channels,
                                                      "depth_ext": args.depth_ext if args.in_channels == 4 else None,
                                                      "colors": cmap.colors if cmap else None}, indent=2))
    print(f"train {len(train_ds)} imgs, val {len(val_ds)} imgs, {steps_per_epoch} optimizer steps/epoch, amp={use_amp}")

    step, best = 0, -1.0
    history, loss, x, y = [], None, None, None
    cuda = device.type == "cuda"
    for epoch in range(args.epochs):
        if cuda:
            torch.cuda.reset_peak_memory_stats()  # peak GPU memory of this epoch's training loop
        model.train()
        t0, run, ep_loss, ep_n = time.time(), 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tl):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(pixel_values=x, labels=y).loss  # cross-entropy, ignores label 255
            scaler.scale(loss / args.accum).backward()
            lv = loss.item()
            run += lv; ep_loss += lv; ep_n += 1
            if (i + 1) % args.accum == 0:
                f = (1 - step / total)  # linear (poly power 1) decay
                for g, b in zip(opt.param_groups, base):
                    g["lr"] = b * f
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
            if (i + 1) % args.log_every == 0:
                print(f"epoch {epoch + 1}/{args.epochs} iter {i + 1}/{len(tl)} loss {run / args.log_every:.4f}")
                run = 0.0
        train_peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else None  # before validation
        res = evaluate(model, vl, device, use_amp, n)
        train_loss = ep_loss / ep_n if ep_n else None
        print(f"== epoch {epoch + 1} ({time.time() - t0:.0f}s) train loss {_f(train_loss)} | val {fmt(res, names)}")
        if cuda:
            print(f"   peak GPU memory (training): {train_peak:.2f} GiB")
        history.append({"epoch": epoch + 1, "epoch_time_s": time.time() - t0, "train_loss": train_loss,
                        "val_loss": res["val_loss"], "val_miou": res["miou"], "iou": res["iou"],
                        "precision": res["precision"], "recall": res["recall"],
                        "macro_precision": res["macro_precision"], "macro_recall": res["macro_recall"],
                        "pixel_acc": res["pixel_acc"], "train_peak_gpu_gib": train_peak})
        model.save_pretrained(out / "last")
        if res["miou"] is not None and res["miou"] > best:
            best = res["miou"]
            model.save_pretrained(out / "best")
            print(f"   new best (val mIoU {best:.4f}) -> {out / 'best'}")

    # Generate final training artifacts and evaluate the best checkpoint on the test set
    peaks = [h["train_peak_gpu_gib"] for h in history if h["train_peak_gpu_gib"] is not None]
    if history:
        report_training(history, names, out, max(peaks) if peaks else None)
    if args.no_test or not (out / "best").exists():
        return
    test_txt = Path(args.split_dir).expanduser() / "test.txt"
    if not test_txt.exists():
        print(f"\nNo {test_txt}: skipping the final test-split report (run the eval subcommand later).")
        return
    try:
        test_ds = mk("test", False)
    except SystemExit as e:
        print(f"\nSkipping the final test-split report: {e.code}")
        return
    test_ds.cmap = cmap
    if args.max_samples:
        test_ds.names = test_ds.names[:args.max_samples]
    # free the training state so the inference peak-memory number is not inflated by optimizer states / gradients
    del model, opt, scaler, groups, head, body, loss, x, y
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    best_model = SegformerForSemanticSegmentation.from_pretrained(out / "best", local_files_only=True).to(device)
    print(f"\nEvaluating the best model on the test split ({len(test_ds)} imgs) ...")
    res = full_eval(best_model, test_ds, device, use_amp, n, args.workers)
    print(f"test: {fmt(res, names)}")
    report_eval(res, names, "test", out, args.baseline_conf, args.n_boot, args.boot_seed)


def cmd_plot(args):
    """Re-create the training curves and per-epoch tables from a finished (or interrupted) run's history.json."""
    run = args.run_dir
    if run is None:  # newest runs/*/history.json, looked up in the current folder and root runs/
        _here = Path(__file__).resolve().parent
        _runs_root = _here.parent.parent / "runs" if _here.name == "segformer" else _here.parent / "runs"
        cands = [p for base in {Path("runs"), _runs_root}
                 for p in base.glob("*/history.json")]
        if not cands:
            sys.exit("No runs/*/history.json found (run from the folder that contains runs/, or pass --run-dir)")
        run = max(cands, key=lambda p: p.stat().st_mtime).parent
        print(f"Using the most recent run: {run}")
    hist_path = Path(run) / "history.json"
    if not hist_path.exists():
        sys.exit(f"{hist_path} not found - was this folder written by the train subcommand?")
    data = json.loads(hist_path.read_text())
    history = data["epochs"]
    if not history:
        sys.exit("history.json has no epochs")
    n = len(history[0]["iou"])
    if not args.class_names and len(data.get("class_names") or []) == n:
        set_class_names(data["class_names"])
    report_training(history, class_names(n), run, data.get("peak_train_gpu_gib"))


SCENARIOS = {"00": "low light, rough", "01": "low light, flat", "02": "high light, rough", "03": "high light, flat"}


def parse_floats(text):
    if not text or str(text).strip().lower() == "none":
        return []
    return [float(v) for v in str(text).split(",") if v.strip()]


def parse_corruptions(text):
    """"darker:0.5,gamma:2.0,noise:10,blur:2" -> [("darker", 0.5), ...]; "none" -> []"""
    if not text or text.strip().lower() == "none":
        return []
    out = []
    for item in text.split(","):
        kind, val = item.strip().split(":")
        if kind not in ("darker", "gamma", "noise", "blur"):
            sys.exit(f"unknown corruption '{kind}' (use darker, gamma, noise, blur)")
        out.append((kind, float(val)))
    return out


def gaussian_blur(img, sigma):
    r = int(np.ceil(3 * sigma))
    k = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-k ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = img.shape[1]
    img = F.pad(img, (r, r, r, r), mode="reflect")
    img = F.conv2d(img, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(img, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def corrupt_rgb(x, kind, val, gen):
    """x: normalised network input [B, C, H, W]. Corrupts only the RGB channels (in 0-1 image space, at network
    resolution); a depth channel, if present, is left unchanged. darker: x * val | gamma: x ** val (val > 1 darkens,
    val < 1 washes out) | noise: Gaussian, sigma = val on the 0-255 scale | blur: Gaussian, sigma = val pixels."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    rgb = (x[:, :3] * std + mean).clamp(0, 1)
    if kind == "darker":
        rgb = rgb * val
    elif kind == "gamma":
        rgb = rgb ** val
    elif kind == "noise":
        rgb = rgb + torch.randn(rgb.shape, generator=gen, device=rgb.device) * (val / 255.0)
    elif kind == "blur":
        rgb = gaussian_blur(rgb, val)
    rgb = rgb.clamp(0, 1)
    return torch.cat([(rgb - mean) / std, x[:, 3:]], 1)


@torch.no_grad()
def analyze(model, ds, device, use_amp, n, thresholds, haz_thresholds, corruptions, batch_size, workers, seed):
    """One pass over the split. Per batch: clean prediction (per-image confusion matrices for the per-scenario table,
    hazard-mask threshold sweep on P(crater)+P(rock), hazard-level detection at the chosen thresholds) and one
    forward pass per corruption of the RGB input (depth unchanged)."""
    model.eval()
    loader = DataLoader(ds, batch_size, shuffle=False, num_workers=workers)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    cnames = [f"{k} {v:g}" for k, v in corruptions]
    conf = {nm: torch.zeros(n, n, dtype=torch.long, device=device) for nm in ["clean"] + cnames}
    cnt = torch.zeros(len(thresholds), 3, dtype=torch.long, device=device)  # TP, FP, FN of the hazard mask
    bg = torch.zeros((), dtype=torch.long, device=device)                  # background pixels
    haz = {t: HazardStats(n) for t in haz_thresholds}
    pic, pnames = [], []

    def forward(xx, size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            lg = model(pixel_values=xx).logits
        return F.interpolate(lg.float(), size=size, mode="bilinear", align_corners=False)

    for k, (x, y, names) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        keep = (y >= 0) & (y < n)
        up = forward(x, y.shape[-2:])
        pred = up.argmax(1)
        conf["clean"] += torch.bincount(y[keep] * n + pred[keep], minlength=n * n).view(n, n)
        for b in range(pred.shape[0]):
            kb = keep[b]
            pic.append(torch.bincount(y[b][kb] * n + pred[b][kb], minlength=n * n).view(n, n).cpu())
        pnames.extend(names)
        prob = up.softmax(1)
        p_haz = prob[:, 1:].sum(1)                       # P(rock) + P(crater)
        gt = keep & (y > 0)
        for i, t in enumerate(thresholds):
            fl = (p_haz >= t) & keep
            cnt[i, 0] += (fl & gt).sum(); cnt[i, 1] += (fl & ~gt).sum(); cnt[i, 2] += ((~fl) & gt).sum()
        bg += (keep & ~gt).sum()
        if haz:
            cls = prob[:, 1:].argmax(1) + 1              # hazard class of a flagged pixel
            y_np = y.cpu().numpy()
            for t, hz in haz.items():
                pm = torch.where(p_haz >= t, cls, torch.zeros_like(cls)).cpu().numpy().astype(np.uint8)
                for b in range(pm.shape[0]):
                    hz.update(pm[b], y_np[b])
        for nm, (kind, val) in zip(cnames, corruptions):
            pc = forward(corrupt_rgb(x, kind, val, gen), y.shape[-2:]).argmax(1)
            conf[nm] += torch.bincount(y[keep] * n + pc[keep], minlength=n * n).view(n, n)
        if (k + 1) % 20 == 0:
            print(f"  analysed {min((k + 1) * batch_size, len(ds))}/{len(ds)} images")
    c = cnt.double().cpu().numpy()
    tp, fp, fn = c[:, 0], c[:, 1], c[:, 2]
    sweep = [{"t": float(t), "hazard_iou": tp[i] / (tp[i] + fp[i] + fn[i]), "hazard_recall": tp[i] / (tp[i] + fn[i]),
              "bg_flagged_pct": 100.0 * fp[i] / float(bg.item()), "fp_pixels": int(fp[i])}
             for i, t in enumerate(thresholds)]
    return {"conf": {nm: v.double().cpu().numpy() for nm, v in conf.items()}, "sweep": sweep,
            "hazard_level": {t: hz.summary(class_names(n)) for t, hz in haz.items()},
            "per_image_conf": torch.stack(pic).numpy(), "image_names": pnames}


def scenario_report(pic, pnames, names, baseline, out, tag):
    """IoU per class for each illumination / terrain scenario (filename prefix 00-03), and the gain over a baseline."""
    pre = np.array([nm[:2] for nm in pnames])
    base = None
    if baseline:
        base = np.load(baseline, allow_pickle=False)
        if list(base["names"]) != list(pnames):
            sys.exit(f"--baseline-conf {baseline} was computed on different images / a different order")
    rows, res = [], {}
    for sc in sorted(set(pre)):
        m = pre == sc
        iou = iou_from_conf(pic[m].sum(0))
        row = [f"{sc} {SCENARIOS.get(sc, '')}".strip(), int(m.sum())] + [float(iou[i]) for i in range(1, len(names))] + [float(np.nanmean(iou))]
        res[sc] = {"images": int(m.sum()), "iou": [float(v) for v in iou], "miou": float(np.nanmean(iou))}
        if base is not None:
            g = (iou - iou_from_conf(base["conf"][m].sum(0))) * 100
            row += [float(g[i]) for i in range(1, len(names))]
            res[sc]["gain_points"] = [float(v) for v in g]
        rows.append(row)
    heads = ["scenario", "images"] + [f"IoU {nm}" for nm in names[1:]] + ["mIoU"]
    if base is not None:
        heads += [f"gain {nm} (pts)" for nm in names[1:]]
    table(f"IoU per scenario [{tag}]" + (f"  (gain over baseline: {baseline})" if base is not None else ""),
          heads, rows, Path(out) / f"{tag}_scenario.csv")
    return res


def report_analysis(r, names, tag, out, baseline):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # decision threshold on P(rock)+P(crater)
    best = max(r["sweep"], key=lambda d: d["hazard_iou"])
    rows = []
    for d in r["sweep"]:
        hl = r["hazard_level"].get(d["t"])
        det = hl["detection"]["all_hazards"]["all"]["rate"] if hl else None
        fa = hl["false_alarms"]["all_hazards"]["per_image"] if hl else None
        rows.append([f"{d['t']:g}", d["hazard_iou"], 100 * d["hazard_recall"], d["bg_flagged_pct"], d["fp_pixels"],
                     None if det is None else 100 * det, fa])
    table(f"Decision threshold t on P(rock)+P(crater) [{tag}]  (best hazard IoU at t = {best['t']:g}: {best['hazard_iou']:.4f})",
          ["t", "hazard IoU", "pixel recall %", "bg flagged %", "bg FP pixels", "hazards detected %", "false alarms/img"],
          rows, out / f"{tag}_threshold.csv")
    print("note: hazard IoU / recall are for the union of rock and crater pixels; 'bg flagged' is the share of background "
          "pixels wrongly flagged. Hazard-level columns use the 20 px / 50% rule and are filled only for --haz-thresholds.")
    # robustness
    clean = iou_from_conf(r["conf"]["clean"])
    rob = {}
    if len(r["conf"]) > 1:
        rows = [["clean"] + [float(clean[i]) for i in range(1, len(names))] + [float(np.nanmean(clean))]
                + [0.0] * len(names)]
        for nm, c in r["conf"].items():
            if nm == "clean":
                continue
            iou = iou_from_conf(c)
            d = (iou - clean) * 100
            dm = (np.nanmean(iou) - np.nanmean(clean)) * 100
            rows.append([nm] + [float(iou[i]) for i in range(1, len(names))] + [float(np.nanmean(iou))]
                        + [float(d[i]) for i in range(1, len(names))] + [float(dm)])
            rob[nm] = {"iou": [float(v) for v in iou], "miou": float(np.nanmean(iou)),
                       "delta_points": [float(v) for v in d], "delta_miou_points": float(dm)}
        heads = (["input"] + [f"IoU {nm}" for nm in names[1:]] + ["mIoU"]
                 + [f"d {nm} (pts)" for nm in names[1:]] + ["d mIoU (pts)"])
        table(f"Robustness to RGB corruption [{tag}]  (depth channel unchanged; change in IoU points vs clean)",
              heads, rows, out / f"{tag}_robustness.csv")
    sc = scenario_report(r["per_image_conf"], r["image_names"], names, baseline, out, tag)
    (out / f"{tag}_analysis.json").write_text(json.dumps(
        {"threshold_sweep": r["sweep"], "best_threshold": best["t"], "robustness": rob, "scenarios": sc,
         "hazard_level_at_threshold": {str(t): v for t, v in r["hazard_level"].items()}}, indent=2))
    print(f"\nAnalysis files written to {out}/ ({tag}_analysis.json, {tag}_threshold.csv, {tag}_robustness.csv, {tag}_scenario.csv)")


def cmd_analyze(args):
    """Decision-threshold sweep, robustness to input corruption and per-scenario IoU for one checkpoint."""
    model, ds, device, use_amp, n, names, colors = _setup_eval(args)
    if n != 3:
        sys.exit("analyze assumes 3 classes (background, rock, crater)")
    r = analyze(model, ds, device, use_amp, n, parse_floats(args.thresholds), parse_floats(args.haz_thresholds),
                parse_corruptions(args.corruptions), args.batch_size, args.workers, args.corrupt_seed)
    report_analysis(r, names, args.split, args.output_dir or Path(args.ckpt).parent, args.baseline_conf)


def _setup_eval(args):
    """Load checkpoint + dataset exactly as eval does. Returns model, ds, device, use_amp, n, class names, colours."""
    from transformers import SegformerForSemanticSegmentation
    device = pick_device()
    use_amp = device.type == "cuda" and not args.no_amp
    cfgp = Path(args.ckpt).parent / "run_config.json"
    cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    h, w = args.height or cfg.get("height", 480), args.width or cfg.get("width", 640)
    model = SegformerForSemanticSegmentation.from_pretrained(args.ckpt, local_files_only=True).to(device)
    n = model.config.num_labels
    in_ch = model.config.num_channels  # 3 = RGB, 4 = RGB-D: stored in the checkpoint by train
    if args.in_channels is not None and args.in_channels != in_ch:
        sys.exit(f"--in-channels {args.in_channels} does not match the checkpoint, which takes {in_ch} input channels")
    if in_ch == 4 and not args.depth_dir:
        sys.exit('This checkpoint takes RGB-D input (4 channels): pass --depth-dir, e.g. --depth-dir "../LunarSeg/depth/{split}"')
    if in_ch == 3 and args.depth_dir:
        print("note: --depth-dir is ignored because this checkpoint takes RGB only (3 channels)")
    if not args.class_names and len(cfg.get("class_names") or []) == n:
        set_class_names(cfg["class_names"])  # names used when the model was trained
    ds = LunarSegDataset(args.image_dir, args.label_dir, args.split_dir, args.split, h, w, False,
                         args.depth_dir if in_ch == 4 else None, args.depth_ext or cfg.get("depth_ext") or "png")
    if getattr(args, "label_mode", None) == "id":
        colors = None
    else:
        colors = parse_colors(args.color_map) if args.color_map else cfg.get("colors")  # saved by train
    if colors:
        ds.cmap = ColorMap(colors)
    if args.max_samples:
        ds.names = ds.names[:args.max_samples]
    names = class_names(n)
    return model, ds, device, use_amp, n, names, colors


def cmd_eval(args):
    model, ds, device, use_amp, n, names, colors = _setup_eval(args)
    if args.output_dir:
        eval_out_dir = args.output_dir if getattr(args, "no_auto_split", False) else (args.output_dir / args.split)
        eval_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        eval_out_dir = None
    res = full_eval(model, ds, device, use_amp, n, args.workers,
                    out_dir=eval_out_dir, palette=colors)  # masks use the label colours when known
    print(f"{args.split} ({len(ds)} imgs): {fmt(res, names)}")
    if eval_out_dir:
        print(f"Saved {len(ds)} x (<name>_mask.png, <name>.npy) to {eval_out_dir}")
    report_eval(res, names, args.split, eval_out_dir or Path(args.ckpt).parent,
                args.baseline_conf, args.n_boot, args.boot_seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    _here = Path(__file__).resolve().parent
    _root = _here.parent.parent if _here.name == "segformer" else _here.parent
    _default_model = _root / "SegFormer" / "mit-b2" if (_root / "SegFormer" / "mit-b2").exists() else (Path("SegFormer/mit-b2") if Path("SegFormer/mit-b2").exists() else Path("./mit-b2"))
    _default_split = _root / "LunarSeg" if (_root / "LunarSeg").exists() else (Path("LunarSeg") if Path("LunarSeg").exists() else Path("."))

    def common(p):
        p.add_argument("--class-names", default=None,
                       help='Comma-separated names by class id, default "background,rock,crater" for 3 classes')
        p.add_argument("--image-dir", default="LunarSeg/image/{split}", help='RGB folder, e.g. "LunarSeg/image/{split}"')
        p.add_argument("--label-dir", default="LunarSeg/label_color/{split}", help='Label folder, e.g. "LunarSeg/label_color/{split}"')
        p.add_argument("--label-mode", choices=["id", "color"], default="color",
                       help="id: label PNGs hold class ids. color: label PNGs are colour-coded masks (default: color, matches label_color/ default)")
        p.add_argument("--color-map", default=None,
                       help='With --label-mode color: exact class colours "R,G,B;R,G,B;..." (id = position). '
                            'Default: found automatically (black = class 0, others by pixel count)')
        p.add_argument("--split-dir", default=str(_default_split), help="Folder with train.txt / validation.txt / test.txt (default: LunarSeg)")
        p.add_argument("--height", type=int, default=None, help="Network input height (default 480)")
        p.add_argument("--width", type=int, default=None, help="Network input width (default 640)")
        p.add_argument("--in-channels", type=int, choices=[3, 4], default=None,
                       help="3 = RGB only, 4 = RGB + depth (needs --depth-dir). train: default 3. eval: default is "
                            "what the checkpoint was trained with; if given, it must match the checkpoint")
        p.add_argument("--depth-dir", default="LunarSeg/depth/{split}",
                       help='Depth folder for 4-channel input, e.g. "LunarSeg/depth/{split}" ({split} is filled in)')
        p.add_argument("--depth-ext", choices=["png", "npy"], default=None,
                       help="Depth file type: png (8/16-bit, scaled to 0-1) or npy (float, already 0-1). "
                            "Default png (eval: the type used in training)")
        p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--max-samples", type=int, default=0, help="debug: limit images per split")
        p.add_argument("--baseline-conf", type=Path, default=None,
                       help="<tag>_per_image_conf.npz of a baseline run (e.g. RGB) on the same split: also report the "
                            "paired-bootstrap 95%% CI of the IoU gain over it")
        p.add_argument("--connectivity", type=int, choices=[4, 8], default=4,
                       help="Connected-component connectivity for the hazard-level analysis (default 4: edge "
                            "neighbours only, matches the U-Net reference counts; 8 also merges diagonal "
                            "neighbours, giving fewer, larger regions)")
        p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for the CIs (default 1000)")
        p.add_argument("--boot-seed", type=int, default=0, help="Seed of the bootstrap resampling (default 0)")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--model", default=_default_model, type=Path,
                   help="LOCAL folder with the pre-downloaded encoder (default: SegFormer/mit-b2). "
                        "mit-b2 gives ~24-25M parameters with --decoder-dim 256, comparable to U-Net + ResNet34 (~24M)")
    t.add_argument("--decoder-dim", type=int, default=256,
                   help="Width of the SegFormer decode head (default 256 = the standard B2 head, ~27.5M parameters "
                        "with mit-b2; 256 -> ~24.8M, closer to U-Net R34). The head is trained from scratch either way")
    t.add_argument("--depth-init", choices=["zero", "mean"], default="zero",
                   help="With --in-channels 4: initial weights of the new depth channel in the first conv. "
                        "zero = start exactly as the pretrained RGB model; mean = copy of the mean RGB weights")
    t.add_argument("--random-init", action="store_true", help="Do not load pretrained weights (smoke tests only)")
    t.add_argument("--num-classes", type=int, default=None)
    t.add_argument("--out-dir", type=Path, default=Path("runs/segformer"))
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--accum", type=int, default=4, help="gradient accumulation (effective batch = batch * accum)")
    t.add_argument("--lr", type=float, default=6e-5, help="encoder lr (decode head uses 10x)")
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no-test", action="store_true",
                   help="Skip the full report of the best model on the test split at the end of training")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval")
    common(e)
    e.add_argument("--ckpt", required=True, help="Saved model folder, e.g. runs/segformer_b0/best")
    e.add_argument("--split", default="test")
    e.add_argument("--output-dir", type=Path, default=None,
                   help="If set, save predicted masks here: <name>_mask.png (colour) and <name>.npy (class ids)")
    e.add_argument("--no-auto-split", action="store_true",
                   help="Save predictions directly in --output-dir without split subdirectory")
    e.set_defaults(fn=cmd_eval)

    an = sub.add_parser("analyze", help="Decision-threshold sweep, robustness to RGB corruption, per-scenario IoU")
    common(an)
    an.add_argument("--ckpt", required=True, help="Saved model folder, e.g. runs/segformer_b2_rgb/best")
    an.add_argument("--split", default="test")
    an.add_argument("--output-dir", type=Path, default=None, help="Where to write the analysis files (default: next to the checkpoint)")
    an.add_argument("--batch-size", type=int, default=4)
    an.add_argument("--thresholds", default="0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05",
                    help="Thresholds t on P(rock)+P(crater) for the pixel-level sweep")
    an.add_argument("--haz-thresholds", default="0.5,0.2",
                    help="Subset of the thresholds that also get hazard-level detection (slower; 'none' to skip)")
    an.add_argument("--corruptions", default="darker:0.5,darker:0.25,gamma:2.0,gamma:0.5,noise:10,noise:25,blur:2",
                    help="RGB corruptions applied at test time, kind:value (darker, gamma, noise sigma on 0-255, blur sigma "
                         "in pixels); 'none' to skip")
    an.add_argument("--corrupt-seed", type=int, default=0, help="Seed of the noise corruption")
    an.set_defaults(fn=cmd_analyze)

    pl = sub.add_parser("plot", help="Re-plot training curves + tables from <run-dir>/history.json")
    pl.add_argument("--class-names", default=None, help="Override the class names stored in history.json")
    pl.add_argument("--run-dir", type=Path, default=None,
                    help="Training output folder (default: the most recent runs/*/ that has a history.json)")
    pl.set_defaults(fn=cmd_plot)

    args = ap.parse_args()
    global HAZARD_CONNECTIVITY
    HAZARD_CONNECTIVITY = getattr(args, "connectivity", 4) or 4
    if getattr(args, "class_names", None):
        set_class_names([c.strip() for c in args.class_names.split(",")])
    if getattr(args, "height", None) is None and args.cmd == "train":
        args.height = 480
    if getattr(args, "width", None) is None and args.cmd == "train":
        args.width = 640
    args.fn(args)


if __name__ == "__main__":
    main()