"""Per-pixel confidence head. Emitted as ``outputs['confidence']``.

Vendored from the team's merged `change-detect` monorepo, needed as a
dependency of `models/proposed.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceHead(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
