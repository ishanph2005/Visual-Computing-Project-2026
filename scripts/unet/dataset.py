"""LunarSeg dataset loader for UNet hazard segmentation.

Folder layout flexibility:
    Supports both flat downloads (e.g. LunarSeg/image/<name>.png) and
    split-partitioned directories (e.g. LunarSeg/image/{split}/<name>.png).
    Classes: 0 = background (regolith), 1 = negative obstacle (crater), 2 = positive obstacle (rock).

Input Modes:
    rgb   -> 3-channel standard RGB imagery
    rgbd  -> 4-channel input (RGB + ground-truth depth)
    rgbpd -> 4-channel input (RGB + predicted depth from Depth Anything V2)

Depth Representations:
    Ground truth depth is typically inverted so that nearest terrain approaches 1.0
    and distant/sky regions approach 0.0, aligning directly with monocular relative
    disparity outputs from Depth Anything V2.
"""
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["background", "negative", "positive"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_dataset_root(root=None):
    """Auto-detects the LunarSeg dataset root if a generic or missing path is passed."""
    if root and os.path.isdir(root) and os.path.exists(os.path.join(root, "train.txt")):
        return root
    candidates = [
        root,
        "LunarSeg",
        "../LunarSeg",
        "data",
        os.path.join(os.path.dirname(__file__), "..", "..", "LunarSeg"),
        os.path.join(os.path.dirname(__file__), "..", "data"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "train.txt")):
            return c
    return root or "LunarSeg"


def read_split(root, split):
    split_file = os.path.join(root, f"{split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file '{split}.txt' not found under '{root}'. Check your dataset root path.")
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def gt_depth_to_inverse(depth):
    """LunarSeg ground-truth depth is stored in 8-bit (2..255 with sky ~255).
    Inverting depth spreads nearby terrain across the full [0, 1] interval instead
    of bunching it up, matching relative disparity from Depth Anything models."""
    d = np.maximum(depth.astype(np.float32), 2.0)
    return 2.0 / d


def load_pred(pred_dir, name, split=None):
    """Load a Depth Anything prediction saved as <name>.npy or <name>_raw.npz.
    
    Searches both flat prediction folders and split subdirectories (e.g. test/<name>.npy).
    """
    candidates = [
        os.path.join(pred_dir, f"{name}.npy"),
    ]
    if split:
        candidates.append(os.path.join(pred_dir, split, f"{name}.npy"))

    for p in candidates:
        if os.path.exists(p):
            return np.load(p).squeeze()

    npz_candidates = [
        os.path.join(pred_dir, f"{name}_raw.npz"),
        os.path.join(pred_dir, f"{name}.npz"),
    ]
    if split:
        npz_candidates.extend([
            os.path.join(pred_dir, split, f"{name}_raw.npz"),
            os.path.join(pred_dir, split, f"{name}.npz"),
        ])

    for zpath in npz_candidates:
        if os.path.exists(zpath):
            z = np.load(zpath)
            key = "depth" if "depth" in z.files else z.files[0]
            return z[key].squeeze()

    raise FileNotFoundError(f"Could not find predicted depth array for '{name}' under '{pred_dir}' (checked flat and split folders).")


def pred_depth_to_disparity(depth):
    """Converts metric depth predictions (where distant objects have large values)
    into disparity space. Near-zero spikes are clipped at the 99.5th percentile for stability."""
    d = depth.astype(np.float32)
    out = np.zeros_like(d)
    ok = d > 0
    out[ok] = 1.0 / d[ok]
    hi = np.percentile(out[ok], 99.5) if ok.any() else 0.0
    return np.clip(out, 0, hi) if hi > 0 else out


def pred_disparity_normalise(disp):
    """Min-max normalizes relative disparity to [0, 1] for consistent network inputs."""
    disp = disp.astype(np.float32)
    lo, hi = disp.min(), disp.max()
    return (disp - lo) / (hi - lo) if hi > lo else np.zeros_like(disp)


class LunarSeg(Dataset):
    def __init__(self, root, split, mode="rgb", pred_depth_dir=None,
                 augment=False, size=(640, 480), corrupt=None, pred_invert=False, gt_linear=False):
        assert mode in ("rgb", "rgbd", "rgbpd"), f"Invalid mode: {mode}. Choose from 'rgb', 'rgbd', 'rgbpd'."
        if mode == "rgbpd":
            assert pred_depth_dir, "rgbpd mode requires a valid --pred_depth_dir holding Depth Anything outputs."
        self.root = resolve_dataset_root(root)
        self.split = split
        self.mode = mode
        self.pred_depth_dir = pred_depth_dir
        self.augment = augment
        self.gt_linear = gt_linear  # True to feed depth directly as depth/255.0 without inversion
        self.pred_invert = pred_invert  # True if predicted depth files hold physical depth rather than disparity
        self.corrupt = corrupt  # Optional test-time corruption function
        self.size = size  # (W, H)
        self.names = read_split(self.root, split)

    def __len__(self):
        return len(self.names)

    def _find_file(self, folder, name, ext=".png"):
        candidates = [
            os.path.join(self.root, folder, f"{name}{ext}"),
            os.path.join(self.root, folder, self.split, f"{name}{ext}"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def _load(self, folder, name, resample):
        target_path = self._find_file(folder, name)
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Missing file for sample '{name}' in folder '{folder}' (checked both flat and '{self.split}/' subfolders in '{self.root}')."
            )
        img = Image.open(target_path)
        if img.size != self.size:
            img = img.resize(self.size, resample)
        return np.asarray(img)

    def __getitem__(self, i):
        name = self.names[i]
        rgb = self._load("image", name, Image.BILINEAR)[..., :3].astype(np.float32) / 255.0
        if self.corrupt is not None:
            rgb = np.clip(self.corrupt(rgb, i), 0.0, 1.0).astype(np.float32)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        label = self._load("label", name, Image.NEAREST).astype(np.int64)
        if label.ndim == 3:  # Safeguard if label masks are saved as 3-channel PNGs
            label = label[..., 0]

        chans = [rgb.transpose(2, 0, 1)]
        if self.mode == "rgbd":
            d = self._load("depth", name, Image.NEAREST)
            chans.append((d.astype(np.float32) / 255.0 if self.gt_linear else gt_depth_to_inverse(d))[None])
        elif self.mode == "rgbpd":
            d = load_pred(self.pred_depth_dir, name, split=self.split)
            if self.pred_invert:
                d = pred_depth_to_disparity(d)
            if d.shape != (self.size[1], self.size[0]):
                d = np.asarray(Image.fromarray(d.astype(np.float32)).resize(self.size, Image.BILINEAR))
            chans.append(pred_disparity_normalise(d)[None])
        x = np.concatenate(chans, axis=0)

        # Synchronized horizontal flip augmentation
        if self.augment and np.random.rand() < 0.5:
            x = x[:, :, ::-1]
            label = label[:, ::-1]

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(label)),
            name,
        )