"""Tiny conversion helpers so metrics accept torch tensors *or* numpy arrays.

Vendored from the team's merged `change-detect` monorepo (originally
authored as part of `metrics-integration-kusal`), with `_squeeze_mask`
folded in here from `metrics/segmentation.py` so this repo doesn't need
to carry the rest of that (unrelated, Member 3-owned) file just for one
helper. Not this repo's own contribution — included only so
`calibration.py` / `temperature.py` / `swap.py` / `directional.py` run
standalone.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(x: Any) -> np.ndarray:
    if x is None:
        raise TypeError("expected array/tensor, got None")
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return mask >= threshold


def valid_mask(gt: np.ndarray) -> np.ndarray:
    """Pixels with label >= 0 are evaluated; -1 is ignore (frozen dataset contract)."""
    return gt >= 0


def _squeeze_mask(x: np.ndarray) -> np.ndarray:
    """[B,1,H,W] or [B,H,W] or [H,W] -> [B,H,W]."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"expected mask-like array with 2-4 dims, got {x.shape}")
    return x
