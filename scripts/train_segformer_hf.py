#!/usr/bin/env python3
"""Compatibility forwarder for scripts/segformer/train_segformer_hf.py."""
import sys
from pathlib import Path

_SEGFORMER_DIR = Path(__file__).resolve().parent / "segformer"
if str(_SEGFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_SEGFORMER_DIR))

from train_segformer_hf import main

if __name__ == "__main__":
    main()
