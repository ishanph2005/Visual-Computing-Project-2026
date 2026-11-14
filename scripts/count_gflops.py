#!/usr/bin/env python3
"""Run Depth Anything V2 on one image and report how many GFLOPs the inference costs.

FLOPs are counted with torch.utils.flop_counter.FlopCounterMode (one multiply-add = 2 FLOPs).
Counts matrix multiplications, convolutions, and attention.

Usage:
    python3 scripts/count_gflops.py --img-path LunarSeg/image/test/00_0013.png --encoder vitl
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DAV2_REPO = ROOT_DIR / "Depth-Anything-V2"

CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def build_model(args):
    repo = Path(args.repo).expanduser()
    if not repo.exists() and DEFAULT_DAV2_REPO.exists():
        repo = DEFAULT_DAV2_REPO
    if args.variant == "metric":
        sys.path.insert(0, str(repo / "metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(**CONFIGS[args.encoder], max_depth=args.max_depth)
    else:
        sys.path.insert(0, str(repo))
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(**CONFIGS[args.encoder])
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print("Note: no --checkpoint given, using random weights (the FLOP count is the same).")
    return model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    default_repo = "Depth-Anything-V2" if DEFAULT_DAV2_REPO.exists() else "."
    ap = argparse.ArgumentParser(description="Depth Anything V2 inference + GFLOPs counter.")
    ap.add_argument("--repo", default=default_repo, help="Path to Depth-Anything-V2 repo")
    ap.add_argument("--img-path", required=True, type=Path, help="Input image")
    ap.add_argument("--encoder", choices=list(CONFIGS), default="vits")
    ap.add_argument("--checkpoint", help="Optional .pth weights matching --encoder / --variant")
    ap.add_argument("--variant", choices=["relative", "metric"], default="relative",
                    help="relative = standard model; metric = repo's metric_depth head")
    ap.add_argument("--max-depth", type=float, default=20.0, help="Only for --variant metric")
    ap.add_argument("--input-size", type=int, default=518,
                    help="Shorter side is resized to this (multiple of 14)")
    ap.add_argument("--runs", type=int, default=10, help="Timed forward passes for the latency / GFLOPS figure")
    ap.add_argument("--out-dir", type=Path, help="Optional: save the depth map (.npy) and an 8-bit preview (.png)")
    args = ap.parse_args()

    raw = cv2.imread(str(args.img_path))
    if raw is None:
        sys.exit(f"Could not read image: {args.img_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    x, (h, w) = model.image2tensor(raw, args.input_size)
    x = x.to(device)
    print(f"Image {w}x{h} -> network input {x.shape[-1]}x{x.shape[-2]} (WxH), "
          f"{args.encoder} / {args.variant}, {n_params / 1e6:.1f}M parameters, device: {device}")

    try:
        import depth_anything_v2.dinov2_layers.attention as _att
        if getattr(_att, "XFORMERS_AVAILABLE", False):
            print("WARNING: xFormers attention is active; its FLOPs are not counted, so the total is a slight underestimate.")
    except Exception:
        pass

    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        depth = model(x)
    total = fc.get_total_flops()
    print(f"\nTotal: {total / 1e9:.2f} GFLOPs  ({total / 2e9:.2f} GMACs) for one forward pass")
    try:
        dec = sum(v for k, d in fc.get_flop_counts().items() if k.endswith(".depth_head") for v in d.values())
        if dec:
            enc = total - dec
            print(f"  encoder (ViT): {enc / 1e9:.2f} GFLOPs ({100 * enc / total:.0f}%)   "
                  f"decoder (DPT head): {dec / 1e9:.2f} GFLOPs ({100 * dec / total:.0f}%)")
    except Exception:
        pass

    with torch.no_grad():
        for _ in range(2):
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(args.runs):
            model(x)
        sync(device)
    lat = (time.perf_counter() - t0) / args.runs
    print(f"Latency: {lat * 1000:.1f} ms per image ({args.runs} runs)  ->  achieved {total / 1e9 / lat:.1f} GFLOPS (GFLOPs/s)")

    d = torch.nn.functional.interpolate(depth[:, None].float(), (h, w), mode="bilinear", align_corners=True)[0, 0]
    d = d.cpu().numpy()
    print(f"Depth map {d.shape[1]}x{d.shape[0]}, range [{d.min():.3f}, {d.max():.3f}]"
          f" ({'depth from the metric head, bounded by --max-depth' if args.variant == 'metric' else 'relative disparity'})")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.out_dir / f"{args.img_path.stem}_depth.npy", d.astype(np.float32))
        vis = ((d - d.min()) / max(d.max() - d.min(), 1e-12) * 255).astype(np.uint8)
        cv2.imwrite(str(args.out_dir / f"{args.img_path.stem}_depth.png"), vis)
        print(f"Saved depth map to {args.out_dir}")


if __name__ == "__main__":
    main()
