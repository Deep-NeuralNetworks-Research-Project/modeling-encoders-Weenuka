"""MODEL_REGISTRY — top-level model registry for complete CD models.

This is separate from ENCODER_REGISTRY (which holds backbone encoders only).
MODEL_REGISTRY holds complete models (baselines + proposed) that implement
the full forward(img1, img2) -> dict contract.

Vendored from the team's merged `change-detect` monorepo, needed as a
dependency of `models/proposed.py`.
"""

from cdlib.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
