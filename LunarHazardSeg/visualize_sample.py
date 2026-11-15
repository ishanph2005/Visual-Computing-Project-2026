from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path.home() / "Documents" / "LunarSeg"
NAME = "00_0001.png"

rgb = np.array(Image.open(ROOT / "image" / NAME).convert("RGB"))
label = np.array(Image.open(ROOT / "label" / NAME))
label_color = np.array(
    Image.open(ROOT / "label_color" / NAME).convert("RGB")
)
depth = np.array(Image.open(ROOT / "depth" / NAME))

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(rgb)
axes[0].set_title("RGB image")

axes[1].imshow(label, vmin=0, vmax=2, cmap="viridis")
axes[1].set_title("Label IDs: 0, 1, 2")

axes[2].imshow(label_color)
axes[2].set_title("Color annotations")

axes[3].imshow(depth, cmap="gray")
axes[3].set_title("Stored depth image")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = Path("results/dataset_sample.png")
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=160, bbox_inches="tight")
print(f"Saved visualization to {output.resolve()}")

plt.show()
