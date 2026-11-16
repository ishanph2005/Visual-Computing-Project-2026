# Lunar Hazard Segmentation: Inference

This branch contains the code for running our trained U-Net and SegFormer models on LunarSeg images. It can compare multiple models on the same image and show their predictions alongside the ground truth.

## 1. Clone the repository

```bash
git clone -b inference https://github.com/ishanph2005/Visual-Computing-Project-2026.git
cd Visual-Computing-Project-2026
```

No submodules are needed. Run the commands below from the repository root.

## 2. Set up the environment

Python 3.10 or newer is required. A GPU is recommended, but the script also works on CPU.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install PyTorch according to your system:

```bash
# NVIDIA GPU (CUDA 12.6 build, works with any recent NVIDIA driver)
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU only
# pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cpu
```

PyTorch must be version 2.6 or newer, because the current `transformers` refuses older versions. If you see
`SegformerForSemanticSegmentation requires the PyTorch library but it was not found` (or `PyTorch >= 2.6 is required`),
your PyTorch is too old. Upgrade it with the command above and add `--upgrade`.

Install the remaining packages:

```bash
pip install transformers safetensors segmentation-models-pytorch opencv-python matplotlib numpy pillow
```

You only need to activate the environment again when using the repository:

```bash
source venv/bin/activate
```

The inference script does not download model weights, so it can run offline once the required files are available.

## 3. Download the dataset and checkpoints

All required files are available in this [Google Drive folder](https://drive.google.com/drive/folders/1-TFIPXjvOhCb2KNSxU2zA8J4V2xV6FEg?usp=sharing).

It contains:

```text
visual-computing-project/
├── dataset/
│   ├── LunarSeg.zip
│   ├── finetuned.zip
│   └── depthmodel_results.zip
└── checkpoints/
    ├── SegFormer/checkpoints.zip
    └── U-Net/checkpoints.zip
```

Download all five ZIP files and place them in the repository root.

### Extract the files

```bash
unzip -q -o LunarSeg.zip
unzip -q -o finetuned.zip
unzip -q -o depthmodel_results.zip
```

For the checkpoints, create the folders first and extract the corresponding archives into them:

```bash
mkdir -p checkpoints/SegFormer checkpoints/U-Net

unzip -q -o SegFormer_checkpoints.zip -d checkpoints/SegFormer
unzip -q -o UNet_checkpoints.zip -d checkpoints/U-Net
```

Use the actual names of the downloaded checkpoint ZIP files in these commands.

The extracted folders should contain the following:

```text
LunarSeg/
├── image/{train,validation,test}/
├── depth/{train,validation,test}/
├── label/
├── label_color/{train,validation,test}/
└── train.txt, validation.txt, test.txt

finetuned/{train,validation,test}/

depthmodel_results/npz_outputs/{train,validation,test}/

checkpoints/
├── SegFormer/
│   └── segformer_b2_rgb/
│       ├── run_config.json
│       └── best/
│           ├── config.json
│           └── model.safetensors
└── U-Net/
    └── unet_r34_rgb/
        ├── best.pth
        └── config.json
```

There may be additional model folders and files depending on which checkpoints you downloaded.

You can check whether the main files are in place using:

```bash
ls LunarSeg/image/test | head -3
find checkpoints -name best.pth
find checkpoints -name run_config.json
```

The first command should list test images. The other two should find the U-Net and SegFormer checkpoints.

Some ZIP files may contain an extra folder level after extraction. This is fine; the script searches up to two additional levels below `checkpoints/`. Avoid renaming or deleting files inside the model folders.

## 4. Run inference

For example, the following command compares three SegFormer models and two U-Net models on test image `00_0002`:

```bash
python3 scripts/infer_compare.py \
    --model "SegFormer RGB=segformer_b2_rgb" \
    --model "SegFormer RGB-D=segformer_b2_rgbd@LunarSeg/depth/{split}" \
    --model "SegFormer DAv2-FT=segformer_b2_rgbd_dav2_ft@finetuned/{split}" \
    --model "U-Net RGB=unet_r34_rgb" \
    --model "U-Net DAv2-FT=unet_r34_rgbpd_ft" \
    --image 00_0002
```

A window will open showing the predictions, and the results will also be printed in the terminal.

### Running each model separately

Each command below runs a single model on test image `00_0002`. Change `--image` to try another image (see [Choosing an image](#choosing-an-image)).

**SegFormer, RGB**

```bash
python3 scripts/infer_compare.py --model "SegFormer RGB=segformer_b2_rgb" --image 00_0002
```

**SegFormer, RGB + ground truth depth**

```bash
python3 scripts/infer_compare.py --model "SegFormer RGB-D=segformer_b2_rgbd@LunarSeg/depth/{split}" --image 00_0002
```

**SegFormer, RGB + fine-tuned DAv2 depth**

```bash
python3 scripts/infer_compare.py --model "SegFormer DAv2-FT=segformer_b2_rgbd_dav2_ft@finetuned/{split}" --image 00_0002
```

**U-Net, RGB**

```bash
python3 scripts/infer_compare.py --model "U-Net RGB=unet_r34_rgb" --image 00_0002
```

**U-Net, RGB (second seed)**

```bash
python3 scripts/infer_compare.py --model "U-Net RGB s1=unet_r34_rgb_s1" --image 00_0002
```

**U-Net, RGB + ground truth depth**

```bash
python3 scripts/infer_compare.py --model "U-Net RGB-D=unet_r34_rgbd" --image 00_0002
```

**U-Net, RGB + ground truth depth (linear scaling)**

```bash
python3 scripts/infer_compare.py --model "U-Net RGB-D lin=unet_r34_rgbd_lin" --image 00_0002
```

**U-Net, RGB + original DAv2 depth**

```bash
python3 scripts/infer_compare.py --model "U-Net DAv2=unet_r34_rgbpd" --image 00_0002
```

**U-Net, RGB + fine-tuned DAv2 depth**

```bash
python3 scripts/infer_compare.py --model "U-Net DAv2-FT=unet_r34_rgbpd_ft" --image 00_0002
```

**U-Net, RGB + fine-tuned DAv2 depth (inverted)**

```bash
python3 scripts/infer_compare.py --model "U-Net DAv2-FT inv=unet_r34_rgbpd_ft_inv" --image 00_0002
```

Notes:

- The text before `=` is only the label shown in the output and can be changed freely. The text after it is the checkpoint folder name.
- SegFormer RGB-D models need the depth folder after `@`, as shown above. Keep `{split}` in the path; the script replaces it with the selected split.
- U-Net models read their depth folder from their own `config.json`, so no `@` part is needed.
- Use the same type of depth input that the model was trained with. The coloured images inside `depthmodel_results/png_outputs/` are only previews; use the grayscale depth maps instead.
- To compare models, pass several `--model` options in one command, as in the example above.
- You can also give a full checkpoint path instead of a folder name, e.g. `--model "U-Net RGB=checkpoints/U-Net/unet_r34_rgb"`.

### Choosing an image

The script can use a specific image, an index from the test list or a random image.

```bash
--image 00_0002
--index 12
--random --seed 3
--split validation
```

By default, it uses the first image in the test set.

A few other useful options:

| Option | Purpose |
|---|---|
| `--no-show` | Save the figure without opening a window |
| `--out-dir my_results` | Change the output folder |
| `--mask-only` | Show colour masks without overlaying them on the image |
| `--device cpu` | Run inference on CPU |

## 5. Results

For each image, the script produces a figure and three CSV tables.

The figure shows the input image and ground truth at the top, followed by the predictions from each model. For depth-based models, the depth input is also displayed.

Rocks are shown in brown and craters in blue. Each prediction includes its mIoU, rock IoU and crater IoU.

The three tables contain:

1. **Pixel metrics:** pixel accuracy, mIoU and inference time.
2. **Per-class metrics:** IoU, precision and recall for background, rocks and craters.
3. **Hazard detection:** detected rocks and craters, along with false alarms. A hazard is counted as detected when at least 50% of its pixels are correctly predicted. Regions smaller than 20 pixels are ignored.

The files are saved in `results/infer_compare/`:

```text
00_0002_compare.png
00_0002_summary.csv
00_0002_classes.csv
00_0002_hazards.csv
00_0002_metrics.json
```

These results are for the selected image only, so they are mainly useful for looking at individual predictions and comparing where the models make mistakes. They should not be treated as overall test-set performance.

## 6. A few notes

- The script can also be run from inside the `scripts/` folder. For example, `python3 infer_compare.py` works there.
- Image resizing, normalization and depth preprocessing are handled by the script according to each model's configuration.
- Ground truth labels are read from `LunarSeg/label_color/
