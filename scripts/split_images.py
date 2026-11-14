#!/usr/bin/env python3
"""Compatibility wrapper for split_dataset.py.

Note: `split_images.py` has been merged into the multipurpose `split_dataset.py`.
This script forwards all calls directly to `split_dataset.py` with `--mode copy`
(or `--mode move` if `--move` is passed).
"""

import sys
from pathlib import Path

# Insert scripts folder to import split_dataset
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import split_dataset

if __name__ == "__main__":
    # If no mode was explicitly given and not --move, default split_images legacy behavior is copy
    args = sys.argv[1:]
    if "--copy" not in args and "--move" not in args and "--mode" not in args and "--symlink" not in args and "--hardlink" not in args:
        # Default legacy split_images.py behavior is copy
        sys.argv.append("--copy")
    split_dataset.main()
