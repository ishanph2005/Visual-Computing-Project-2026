# LunarSeg Dataset - Technical Guide & Preprocessing Reference

The **LunarSeg** (LuSeg) dataset is a multi-modal lunar surface segmentation benchmark designed for hazard detection (rocks/boulders and impact craters) for planetary rovers and landers.

This directory contains the dataset annotations, split definitions, and dataset preparation utilities used by both the **Depth-Anything-V2** and **SegFormer** modules in this repository.

---

## 1. Directory Status & Missing Files

### A. Currently Present Components
- **`label/`**: Contains 5,143 single-channel ground-truth annotation PNGs (grayscale class IDs: `0`, `1`, `2`) in a flat directory structure.
- **`label_color/`**: Contains color-coded 3-channel RGB annotation masks organized into split subdirectories:
  - `train/`: 3,085 masks
  - `validation/`: 1,028 masks
  - `test/`: 1,030 masks
- **Split Definitions (`.txt`)**:
  - `train.txt`: 3,085 sample identifiers
  - `validation.txt`: 1,028 sample identifiers
  - `test.txt`: 1,030 sample identifiers
  - *Total: 5,143 samples*
- **Centralized Scripts**: All dataset preparation and conversion utilities live in `../scripts/` (`split_dataset.py`, `png_to_npz.py`, `normalize_npz.py`).

---

### B. Missing Pieces & How to Obtain Them

The following modalities are **not currently present** in this directory:

| Missing Component | Description | Expected Directory Path |
| :--- | :--- | :--- |
| **`image/`** | Photorealistic Lunar RGB images (`.png`) | `LunarSeg/image/{train,validation,test}/<name>.png` |
| **`depth/`** | Ground-truth physical metric/relative depth PNGs | `LunarSeg/depth/{train,validation,test}/<name>.png` |

#### How to Download:
1. **Source Repository:** The dataset is released by the NUDT Nubot research team as part of:
   > *LuSeg: Efficient Negative and Positive Obstacles Segmentation via Contrast-Driven Multi-Modal Feature Fusion on the Lunar Surface* (IROS).  
   > Official GitHub Repository: [https://github.com/nubot-nudt/LuSeg](https://github.com/nubot-nudt/LuSeg)
2. **Download Links:** Access the official Google Drive links provided in the **Dataset** section of the [LuSeg GitHub repository](https://github.com/nubot-nudt/LuSeg) to download the compressed `image` and `depth` archives.
3. **Placing Downloaded Data:** Extract the archives into `LunarSeg/`. If the downloaded archives are flat folders, organize them into split folders using `split_dataset.py` (see Section 3).

---

## 2. Dataset Schema & Class Definitions

### Semantic Classes
| Class ID | Class Name | Description | Approximate Pixel Share |
| :---: | :--- | :--- | :---: |
| **`0`** | `background` | Flat lunar regolith / terrain soil | ~98% |
| **`1`** | `rock` | Positive obstacle (boulders, rocks, stones) | ~1% |
| **`2`** | `crater` | Negative obstacle (impact craters, depressions) | ~1% |
| **`255`** | `ignore` | Ambiguous boundaries or masked sky regions | Ignored in loss & metrics |

### Expected Organization for Downstream Tasks
Downstream models (`train_segformer_hf.py` and `finetune_lunar.py`) expect the following folder layout:
```
LunarSeg/
├── image/
│   ├── train/          # 3,085 RGB PNGs
│   ├── validation/     # 1,028 RGB PNGs
│   └── test/           # 1,030 RGB PNGs
├── depth/
│   ├── train/          # 3,085 GT depth PNGs (or .npy)
│   ├── validation/     # 1,028 GT depth PNGs
│   └── test/           # 1,030 GT depth PNGs
├── label/
│   ├── train/          # 3,085 class ID PNGs (0, 1, 2)
│   ├── validation/     # 1,028 class ID PNGs
│   └── test/           # 1,030 class ID PNGs
├── label_color/
│   ├── train/          # 3,085 color-coded mask PNGs
│   ├── validation/     # 1,028 color-coded mask PNGs
│   └── test/           # 1,030 color-coded mask PNGs
├── train.txt
├── validation.txt
└── test.txt
```

---

## 3. Dependencies & Installation

The scripts in `LunarSeg` require standard scientific Python packages:
```bash
pip install numpy opencv-python
```
*(If you have already set up `segformer_env` or `dav2_env`, all dependencies are already installed).*

---

## 4. Scripts Reference & Exact Usage

All dataset utilities are centralized in `scripts/` at the repository root. Commands below can be executed from the project root or from `LunarSeg/`:

### 1. `split_dataset.py` - Multipurpose Dataset Splitter
Exposes or copies/moves images from a raw/flat directory into `train/`, `test/`, and `validation/` subdirectories according to `train.txt`, `test.txt`, and `validation.txt`. Supports multiple transfer modes: `symlink` (default), `hardlink`, `copy`, and `move`.

**Usage (from project root):**
```bash
# Symlink mode (default, saves disk space):
python3 scripts/split_dataset.py /path/to/raw_downloaded_images --output-dir LunarSeg/image --ext .png

# Hardlink mode:
python3 scripts/split_dataset.py /path/to/raw_downloaded_images --output-dir LunarSeg/image --hardlink

# Physical copy mode:
python3 scripts/split_dataset.py /path/to/raw_downloaded_images --output-dir LunarSeg/image --copy

# Organize flat label/ directory into train/, test/, validation/ subfolders inside a new label_split directory:
python3 scripts/split_dataset.py LunarSeg/label --output-dir LunarSeg/label_split --ext .png
```

**Key Arguments:**
- `input_dir`: Directory containing unsorted input files (searched recursively).
- `--output-dir`: Output destination where `train/`, `validation/`, and `test/` subfolders are created.
- `--mode {symlink,hardlink,copy,move}` or shortcut flags (`--symlink`, `--hardlink`, `--copy`, `--move`).
- `--ext`: Extensions to match (default: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, `.tiff`).
- `--all-files`: Consider all file types, ignoring extension filter.
- `--exact`: Require exact name/stem matching instead of substring matching.

---

### 2. `png_to_npz.py` - Convert Depth PNGs to Float32 `.npz` Arrays
Converts 8-bit or 16-bit depth PNG images into NumPy compressed `.npz` arrays storing single-channel float32 values normalized to $[0, 1]$ (divided by 255.0 for uint8 or 65535.0 for uint16).

**Usage:**
```bash
python3 scripts/png_to_npz.py LunarSeg/depth/test LunarSeg/depth_npz/test --key depth --compressed
```

**Arguments:**
- `input_dir`: Folder containing PNG depth images.
- `output_dir`: Target folder for `.npz` files.
- `--key`: Key name under which the array is stored inside the `.npz` archive (default: `depth`).
- `--compressed`: Use `np.savez_compressed` instead of uncompressed `np.savez`.

---

### 3. `normalize_npz.py` - Min-Max Normalizer for Depth Arrays
Takes raw depth `.npz` outputs (such as predictions exported by `Depth-Anything-V2/run.py`), applies per-image min-max normalization into $[0, 1]$:
$$\text{normalized} = \frac{d - d_{\min}}{d_{\max} - d_{\min}}$$
Supports value inversion (`--invert`) to transform disparity representations into depth representations.

**Usage:**
```bash
python3 scripts/normalize_npz.py outputs/depth_rel outputs/normalized_depth_npz --compressed
```

**Arguments:**
- `input_dir`: Folder containing input `.npz` files.
- `output_dir`: Target folder for normalized output `.npz` files.
- `--in-key`: Input array key name (default: `depth`).
- `--out-key`: Output array key name (default: `depth`).
- `--invert`: Flip normalized values ($1.0 - x$).
- `--keep-suffix`: Keep the `_raw` suffix in output filenames (stripped by default).
- `--compressed`: Save with compression (`np.savez_compressed`).

---

## 5. Summary Table of Scripts

| Script | Purpose | Key Inputs | Key Outputs |
| :--- | :--- | :--- | :--- |
| **`split_dataset.py`** | Multipurpose dataset partitioning (symlink, hardlink, copy, move) | Raw data folder, `train.txt`, `test.txt`, `validation.txt` | `train/`, `test/`, `validation/` split directories |
| **`png_to_npz.py`** | Converts 8/16-bit depth PNGs to float32 $[0, 1]$ `.npz` arrays | Depth PNG directory | Normalized `.npz` files (`depth` key) |
| **`normalize_npz.py`** | Min-max normalizes predicted `.npz` depth maps to $[0, 1]$ | Raw `.npz` depth directory | Normalized `.npz` depth maps |
