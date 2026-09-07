"""Minimal model builder — STARTER STUB, not P3-owned long-term.

CLAUDE.md's contract says `build_model(cfg)` is shared infra (P2 builds
the real Hydra-driven version) and P3 must never edit it once it exists.
This file exists only so the P3 component set (encoders/fusion/decoders/
siamese_resnet18) is runnable and testable standalone before that shared
scaffold lands. When P2's real `build.py` arrives, delete this file's
contents in favour of theirs — don't keep extending this switch
statement with new models.
"""
from __future__ import annotations

import torch.nn as nn

from cdlib.models.baselines.siamese_resnet18 import SiameseResNet18

_MODELS = {
    "siamese_resnet18": SiameseResNet18,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(_MODELS)}")
    return _MODELS[name](**kwargs)
