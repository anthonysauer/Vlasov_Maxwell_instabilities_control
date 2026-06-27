"""Small helpers for loading packaged data."""

from __future__ import annotations

from pathlib import Path
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"
REFERENCE_FIGURE_DIR = DATA_DIR / "reference_figures"


def load_npz(name: str, allow_pickle: bool = False):
    return np.load(PRECOMPUTED_DIR / name, allow_pickle=allow_pickle)
