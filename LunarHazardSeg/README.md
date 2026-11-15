# LunarHazardSeg

Setup (inside ~/LunarHazardSeg, existing venv activated):
    pip install segmentation-models-pytorch matplotlib
    ln -sfn ~/Documents/LunarSeg data

Phase 0  python -m src.check_data
Phase 1  python -m src.train --mode rgb  --name unet_r34_rgb
Phase 2  python -m src.train --mode rgbd --name unet_r34_rgbd
Eval     python -m src.evaluate --name unet_r34_rgb      (test split, run once at the end)
