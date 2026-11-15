#!/usr/bin/env python3
"""Plot training and validation curves from history.json."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot loss and IoU curves from history.json")
    parser.add_argument("--history", type=Path, default=Path("history.json"),
                        help="Path to history.json (default: history.json)")
    parser.add_argument("--output", type=Path, default=Path("training_curves.png"),
                        help="Output image path (default: training_curves.png)")
    parser.add_argument("--classes", default="background,rock,crater",
                        help="Comma-separated class names (default: background,rock,crater)")
    args = parser.parse_args()

    if not args.history.exists():
        # Try finding most recent history.json under runs/
        runs = sorted(Path("runs").glob("**/history.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            print(f"Note: {args.history} not found. Using most recent: {runs[0]}")
            args.history = runs[0]
        else:
            raise FileNotFoundError(f"Could not find {args.history}")

    data = json.load(open(args.history))["epochs"]
    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    nan = float("nan")
    ep = [h["epoch"] for h in data]
    v = lambda key: [nan if x is None else x for x in key]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(ep, v([h["train_loss"] for h in data]), "o-", label="train loss")
    ax[0].plot(ep, v([h["val_loss"] for h in data]), "s-", label="val loss")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Training and validation loss")

    for i, nm in enumerate(names):
        ax[1].plot(ep, v([h["iou"][i] for h in data]), "o-", ms=3, label=f"IoU {nm}")
    ax[1].plot(ep, v([h["val_miou"] for h in data]), "k--", lw=2.2, label="val mIoU")
    ax[1].set(xlabel="epoch", ylabel="IoU", title="Validation IoU per class and mIoU", ylim=(0, 1.02))

    for a in ax:
        a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
