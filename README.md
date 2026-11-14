# Multi-Modal Lunar Terrain Hazard Segmentation & Depth Estimation

This repository implements an end-to-end multi-modal vision pipeline for lunar hazard detection (positive obstacles: **rocks and boulders**; negative obstacles: **craters**). It integrates **Depth Anything V2 (DAv2)** for monocular relative and metric depth estimation with **SegFormer** for semantic segmentation on the **LunarSeg** benchmark dataset.

The project is structured to keep third-party research frameworks unmodified as **Git submodules**, while all custom fine-tuning, inference, benchmarking, and dataset utilities are organized in the centralized **`scripts/`** directory.

---

## 1. Repository Structure

```
.
├── scripts/                         # Project-specific custom scripts & pipelines
│   ├── run_dav2_raw.py             # DAv2 relative depth inference + raw .npz array export
│   ├── finetune_lunar.py           # DAv2 metric head fine-tuning & evaluation on LunarSeg
│   ├── inference_dav2.py           # Batched inference with fine-tuned DAv2 checkpoints
│   ├── count_gflops.py             # FLOPs, GMACs, latency, and parameter profiler for DAv2
│   ├── compare_metric_depth.py     # Compares predicted depth against LunarSeg ground truth
│   ├── train_segformer_hf.py       # Hugging Face SegFormer pipeline (RGB & RGB-D, train/eval/analyze/plot)
│   ├── plot_results.py             # Visualizes loss and per-class IoU curves from history.json
│   ├── split_dataset.py            # Non-destructive dataset splitter via symlinks/hardlinks
│   ├── split_images.py             # Image file splitter based on substring matching
│   ├── png_to_npz.py               # Depth PNG to normalized float32 [0, 1] .npz converter
│   ├── normalize_npz.py            # Min-max normalizer for predicted depth maps
│   ├── npy_png_converter.py        # Converts raw depth .npy arrays to grayscale PNGs
│   └── file_to_folder.py           # File aggregation utility across nested subfolders
├── Depth-Anything-V2/              # Submodule: Upstream Depth Anything V2 repository
│   ├── checkpoints/                # Relative depth pretrained weights (depth_anything_v2_vitl.pth)
│   └── metric_depth/               # Metric depth estimation module & checkpoints
├── SegFormer/                       # Submodule: Upstream NVIDIA SegFormer repository
│   ├── mit-b2/                     # Pretrained SegFormer-B2 encoder weights & config
│   └── segformer_env/              # Configured Python virtual environment
├── LuSeg/                           # Submodule: Upstream LuSeg repository (IROS 2025)
├── LunarSeg/                        # Lunar terrain hazard dataset annotations & splits
│   ├── label/                      # 5,143 single-channel class ID masks (0, 1, 2)
│   ├── label_color/                # Color-coded annotation masks (train/validation/test)
│   ├── train.txt                   # 3,085 training sample IDs
│   ├── validation.txt              # 1,028 validation sample IDs
│   └── test.txt                    # 1,030 test sample IDs
├── .gitignore                      # Git ignore configuration (weights, envs, outputs)
└── README.md                       # Master project guide
```

---

## 2. Working with Git Submodules

### How Submodules Work
A Git submodule tracks a specific commit hash of an external Git repository via a `.gitmodules` file in your root project. By referencing official upstream repositories directly as submodules and keeping all project-specific logic in `scripts/`, you avoid forking third-party repositories and maintain clean code boundaries.

### Initial Setup (Adding Submodules from Scratch)
If you are initializing this repository or pushing it to a new remote:

```bash
# 1. Initialize root git repository
git init -b main

# 2. Add upstream repositories as submodules
git submodule add https://github.com/DepthAnything/Depth-Anything-V2.git Depth-Anything-V2
git submodule add https://github.com/NVlabs/SegFormer.git SegFormer
git submodule add https://github.com/nubot-nudt/LuSeg.git LuSeg

# 3. Verify .gitmodules was generated
cat .gitmodules

# 4. Commit and push
git add .gitmodules Depth-Anything-V2 SegFormer LuSeg scripts LunarSeg .gitignore README.md
git commit -m "Initialize project with Depth-Anything-V2, SegFormer, and LuSeg submodules"
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### Cloning a Repository with Submodules
To clone this project with all original submodule files and directory structures populated:
```bash
git clone --recurse-submodules https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

If the repository was cloned without `--recurse-submodules`, initialize and pull them at any time with:
```bash
git submodule update --init --recursive
```

### Pulling Upstream Updates
To pull the latest commits from the upstream repositories:
```bash
git submodule update --remote --merge
```

### Keeping Submodules Clean
Because all project-specific scripts are centralized in `scripts/`, submodules should remain untouched. If any local submodule files were accidentally modified or deleted, reset them to the clean upstream state:
```bash
cd Depth-Anything-V2 && git restore . && cd ..
cd SegFormer && git restore . && cd ..
```

---

## 3. Environment & Dependencies Installation

### Option A: Using the Configured Environment
The virtual environment in `SegFormer/segformer_env` is pre-configured with PyTorch 2.14.0+cu130, Hugging Face `transformers`, and imaging libraries:
```bash
source SegFormer/segformer_env/bin/activate
```

### Option B: Fresh Environment Setup (Python 3.10+)
To configure a unified root environment:
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA matching your GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install Transformers, Imaging & Web UI dependencies
pip install transformers opencv-python matplotlib numpy tqdm safetensors gradio gradio_imageslider

# 4. Install Metric Depth, Point Cloud & Analysis dependencies
pip install open3d scipy pandas tensorboard
```

---

## 4. Pretrained Model Weights

Checkpoints must be placed in their expected paths before running inference or fine-tuning:

| Model | Path | Source / Status |
| :--- | :--- | :--- |
| **Depth Anything V2 Large** | `Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth` | Pre-downloaded (335M params) |
| **Depth Anything V2 Metric Hypersim** | `Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth` | Pre-downloaded (indoor metric) |
| **SegFormer MIT-B2** | `SegFormer/mit-b2/` | Pre-downloaded (`pytorch_model.bin`, `config.json`) |

*For lightweight GPUs (e.g. 4–6 GB VRAM), download the `vits` checkpoint from Hugging Face:*
```bash
wget -P Depth-Anything-V2/checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

---

## 5. LunarSeg Dataset Setup

### A. Current Status & Missing Components
- **Present:**
  - `LunarSeg/label/`: 5,143 single-channel ground-truth annotation PNGs (`0 = background`, `1 = rock`, `2 = crater`).
  - `LunarSeg/label_color/`: Color-coded masks divided into `train/` (3,085), `validation/` (1,028), and `test/` (1,030).
  - `LunarSeg/train.txt`, `validation.txt`, `test.txt`: Split definition lists.
- **Missing Modalities:**
  - `LunarSeg/image/`: Photorealistic RGB surface images.
  - `LunarSeg/depth/`: Ground-truth depth maps.

### B. Downloading Missing Dataset Files
1. **Source:** Released under *LuSeg: Efficient Negative and Positive Obstacles Segmentation via Contrast-Driven Multi-Modal Feature Fusion on the Lunar Surface* (IROS).
2. **Download Repository:** Available via the official repository at **[nubot-nudt/LuSeg](https://github.com/nubot-nudt/LuSeg)**.
3. Access the Google Drive links in the **Dataset** section of the LuSeg repository to download the `image` and `depth` archives.
4. Extract the archives into `LunarSeg/`. If downloaded as flat folders, partition them into `train/`, `validation/`, and `test/` splits automatically using `scripts/split_dataset.py`.

---

## 6. End-to-End Workflow & Execution Commands

All commands below should be executed from the **repository root directory**.

### Phase 1: Dataset Partitioning & Preprocessing

#### 1. Split Raw Images into Train / Validation / Test (Symlinks)
Creates symbolic links into `train/`, `validation/`, and `test/` subfolders according to `train.txt`, `test.txt`, and `validation.txt` without duplicating disk storage:
```bash
python3 scripts/split_dataset.py /path/to/downloaded_images --output-dir LunarSeg/image --ext .png
```

#### 2. Alternative Splitter via Substring Matching
Copies or moves image files by substring matching against split text files:
```bash
python3 scripts/split_images.py /path/to/raw_images LunarSeg LunarSeg/image --recursive
```

#### 3. Convert Depth PNGs to Float32 `[0, 1]` `.npz` Archives
Normalizes 8-bit or 16-bit depth PNGs into compressed single-channel `.npz` arrays:
```bash
python3 scripts/png_to_npz.py LunarSeg/depth/test LunarSeg/depth_npz/test --compressed
```

#### 4. Flatten Files from Nested Folders
Aggregates all files of given extensions into a single flat directory:
```bash
python3 scripts/file_to_folder.py LunarSeg/image all_images --ext png --mode copy
```

---

### Phase 2: Monocular & Metric Depth Estimation (Depth Anything V2)

#### 1. Relative Depth Inference with Compressed `.npz` Export
Runs Depth Anything V2 inference on an image or directory. Exports side-by-side (or depth-only with `--pred-only`) PNG visualizations alongside compressed float32 raw depth arrays (`_raw.npz`):
```bash
python3 scripts/run_dav2_raw.py \
    --encoder vitl \
    --img-path LunarSeg/image/test \
    --outdir outputs/depth_rel \
    --pred-only
```

#### 2. Fine-Tuning Depth Anything V2 on Lunar Ground-Truth Depth
Adapts the metric DPT head (`sigmoid * max_depth` with `max_depth=1.0`) to predict normalized lunar depth $[0, 1]$ directly. Trains with L1 or SiLog loss, polynomial learning rate decay, and mixed precision:
```bash
python3 scripts/finetune_lunar.py train \
    --encoder vits \
    --pretrained Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
    --out-dir runs/dav2_lunar_vits \
    --epochs 20 --batch-size 2 --accum 8 \
    --lr 5e-6 --loss l1 --select all
```
*Tips: Add `--freeze-encoder` if GPU memory is constrained. Use `--select terrain` to select `best.pth` based on non-sky terrain AbsRel.*

#### 3. Evaluating Fine-Tuned Depth Checkpoint
Evaluates AbsRel, RMSE, and $\delta_1$ metrics on a split without per-image scale/shift fitting:
```bash
python3 scripts/finetune_lunar.py eval \
    --ckpt runs/dav2_lunar_vits/best.pth \
    --split test
```

#### 4. Batched Inference with Fine-Tuned Depth Model
Generates float32 depth `.npy` arrays and 8-bit grayscale `.png` previews:
```bash
python3 scripts/inference_dav2.py \
    --ckpt runs/dav2_lunar_vits/best.pth \
    --img-dir LunarSeg/image/test \
    --out-dir outputs/dav2_lunar_preds
```

#### 5. Benchmark Metric Predictions Against Lunar Ground Truth
Computes AbsRel, RMSE, $\delta_1$, and Spearman rank correlation between predictions and ground-truth PNGs, supporting least-squares scale-shift alignment (`--align scale_shift`) and CSV export:
```bash
python3 scripts/compare_metric_depth.py \
    --pred-dir outputs/dav2_lunar_preds \
    --gt-dir LunarSeg/depth/test \
    --align scale_shift \
    --csv outputs/depth_benchmark.csv
```

#### 6. Min-Max Normalizer for Predicted Depth `.npz` Maps
Normalizes raw `.npz` depth maps to $[0, 1]$ with optional value inversion (`--invert`):
```bash
python3 scripts/normalize_npz.py outputs/depth_rel outputs/depth_normalized --compressed
```

#### 7. Convert Depth `.npy` Files to Grayscale PNGs
Batch-converts float depth arrays into min-max normalized 8-bit grayscale PNGs:
```bash
python3 scripts/npy_png_converter.py -i outputs/dav2_lunar_preds -o outputs/dav2_lunar_pngs
```

#### 8. Computational Complexity Profiler (FLOPs & Latency)
Measures parameter count, forward-pass GFLOPs / GMACs via PyTorch `FlopCounterMode`, encoder vs. decoder FLOP breakdown, inference latency (ms), and achieved GFLOPS throughput:
```bash
python3 scripts/count_gflops.py \
    --img-path LunarSeg/image/test/00_0013.png \
    --encoder vitl \
    --variant relative \
    --runs 10
```

#### 9. Interactive Gradio Web Demo
Launches a local browser interface with an interactive slider comparing input images and predicted depth:
```bash
python3 Depth-Anything-V2/app.py
```

#### 10. Generate 3D Point Clouds
Projects 2D images and metric depth maps into colored 3D point clouds (`.ply` format via Open3D):
```bash
python3 Depth-Anything-V2/metric_depth/depth_to_pointcloud.py \
    --encoder vitl \
    --load-from Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --img-path LunarSeg/image/test/00_0013.png \
    --outdir outputs/pointclouds
```

---

### Phase 3: Semantic Hazard Segmentation (SegFormer)

The Hugging Face pipeline in `scripts/train_segformer_hf.py` runs natively on modern PyTorch + `transformers`, operating completely offline (`HF_HUB_OFFLINE=1`).

#### 1. RGB Mode Training (3 Input Channels)
Fine-tunes `mit-b2` on 3-channel RGB imagery with cross-entropy loss, polynomial decay, and validation checkpoints:
```bash
python3 scripts/train_segformer_hf.py train \
    --model SegFormer/mit-b2 \
    --in-channels 3 \
    --out-dir runs/segformer_b2_rgb \
    --epochs 30 --batch-size 4 --accum 4 \
    --lr 6e-5 --height 480 --width 640
```

#### 2. Multi-Modal RGB-D Training (4 Input Channels)
Expands the first patch-embedding layer to 4 channels ($3\text{ RGB} + 1\text{ Depth}$). Pretrained RGB weights are preserved, and new depth weights are initialized to zero (`--depth-init zero`):
```bash
python3 scripts/train_segformer_hf.py train \
    --model SegFormer/mit-b2 \
    --in-channels 4 \
    --depth-dir "LunarSeg/depth/{split}" \
    --label-mode color \
    --label-dir "LunarSeg/label_color/{split}" \
    --out-dir runs/segformer_b2_rgbd \
    --epochs 30 --batch-size 4 --accum 4
```

#### 3. Evaluation & Hazard Detection Analysis
Evaluates the model on `test.txt`. In addition to standard pixel metrics (mIoU, precision, recall), it computes **hazard-level detection rates** categorized by physical pixel size/distance:
- **Small (20–300 px):** Distant hazards
- **Medium (300–3000 px):** Intermediate hazards
- **Large (>3000 px):** Close hazards
- **False Alarms per Image:** Predicted hazard clusters that overlap zero ground-truth pixels.

```bash
python3 scripts/train_segformer_hf.py eval \
    --ckpt runs/segformer_b2_rgb/best \
    --split test \
    --output-dir runs/segformer_b2_rgb/preds
```
*(Exports color-coded predicted masks `<name>_mask.png` and uint8 class ID arrays `<name>.npy` to `--output-dir`).*

#### 4. Robustness to Test-Time Environmental Corruptions
Simulates lighting drops, sensor noise, and camera blur to evaluate out-of-distribution resilience:
```bash
python3 scripts/train_segformer_hf.py analyze \
    --ckpt runs/segformer_b2_rgb/best \
    --split test \
    --corruptions "darker:0.5,gamma:2.0,noise:25,blur:2"
```

#### 5. Plotting Training Curves
Re-generates loss curves and IoU progressions from `history.json`:
```bash
python3 scripts/plot_results.py \
    --history runs/segformer_b2_rgb/history.json \
    --output runs/segformer_b2_rgb/training_curves.png
```

---

## 7. Summary Mapping of Custom Scripts vs. Submodules

| Custom Script (`scripts/`) | Interacts With | Functionality | Primary Output |
| :--- | :--- | :--- | :--- |
| **`run_dav2_raw.py`** | `Depth-Anything-V2` | Relative depth inference + `.npz` raw export | Grayscale/Spectral `.png` & float32 `_raw.npz` |
| **`finetune_lunar.py`** | `Depth-Anything-V2`, `LunarSeg` | Fine-tune DAv2 metric head directly on LunarSeg | `best.pth`, `last.pth`, `config.json`, `.npy` |
| **`inference_dav2.py`** | `Depth-Anything-V2` | Batch inference with fine-tuned DAv2 weights | Float32 `[0, 1]` `.npy` & 8-bit preview `.png` |
| **`count_gflops.py`** | `Depth-Anything-V2` | Profiles GFLOPs, GMACs, latency, and throughput | Terminal benchmark breakdown |
| **`compare_metric_depth.py`** | `LunarSeg` | Compares depth predictions vs ground truth | AbsRel, RMSE, $\delta_1$, Spearman stats, `.csv` |
| **`train_segformer_hf.py`** | `SegFormer/mit-b2`, `LunarSeg` | SegFormer RGB/RGB-D fine-tuning, eval & corruptions | `best/`, `last/`, `history.json`, `_mask.png` |
| **`plot_results.py`** | Training runs | Plots training/validation loss and per-class IoU | `training_curves.png` |
| **`split_dataset.py`** | `LunarSeg` | Non-destructive split linking (symlinks/hardlinks) | `train/`, `validation/`, `test/` directories |
| **`split_images.py`** | `LunarSeg` | Splits files by substring matching | `train/`, `validation/`, `test/` directories |
| **`png_to_npz.py`** | `LunarSeg` | Converts depth PNGs to normalized float32 `.npz` | `.npz` archives with `depth` array |
| **`normalize_npz.py`** | `Depth-Anything-V2` | Min-max normalizes `.npz` depth arrays to `[0, 1]` | Normalized `.npz` files |
| **`npy_png_converter.py`** | Generic | Converts depth `.npy` arrays to 8-bit grayscale PNGs | Normalized grayscale `.png` images |
| **`file_to_folder.py`** | Generic | Flattens nested directories by file extension | Flat directory containing matched files |
