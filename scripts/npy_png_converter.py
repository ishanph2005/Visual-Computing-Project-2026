#!/usr/bin/env python3
"""Convert raw depth .npy arrays to normalized grayscale PNG images using OpenCV."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def convert_npy_to_png(input_dir, output_dir):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = list(in_dir.glob('*.npy'))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    for file_path in npy_files:
        try:
            depth_data = np.load(file_path)
            min_val = np.min(depth_data)
            max_val = np.max(depth_data)

            if max_val - min_val == 0:
                normalized = np.zeros_like(depth_data, dtype=np.uint8)
            else:
                normalized = ((depth_data - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

            out_file = out_dir / f"{file_path.stem}.png"
            cv2.imwrite(str(out_file), normalized)
            print(f"Converted: {file_path.name} -> {out_file.name}")
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw depth .npy to grayscale PNG using OpenCV")
    parser.add_argument("-i", "--input", required=True, help="Input directory path")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    args = parser.parse_args()
    convert_npy_to_png(args.input, args.output)
