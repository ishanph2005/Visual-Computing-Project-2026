from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

ROOT = Path.home() / "Documents" / "LunarSeg"

# Verify class IDs across the complete dataset.
counts = Counter()

for path in sorted((ROOT / "label").glob("*.png")):
    label = np.array(Image.open(path))
    values, frequencies = np.unique(label, return_counts=True)

    for value, frequency in zip(values, frequencies):
        counts[int(value)] += int(frequency)

print("=== Pixel counts across all labels ===")
total = sum(counts.values())

for class_id in sorted(counts):
    print(
        f"ID {class_id}: {counts[class_id]:,} pixels "
        f"({100 * counts[class_id] / total:.3f}%)"
    )

# Inspect one corresponding RGB, label, color-label and depth sample.
name = "00_0001.png"

for folder in ["image", "label", "label_color", "depth"]:
    path = ROOT / folder / name
    arr = np.array(Image.open(path))

    print(
        f"{folder}: shape={arr.shape}, dtype={arr.dtype}, "
        f"unique_values={len(np.unique(arr))}"
    )

print("\nDataset diagnostic complete.")
