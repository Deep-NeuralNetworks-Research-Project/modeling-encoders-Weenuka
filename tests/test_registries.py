"""Registry integrity tests, per research/06's test strategy: no
duplicate keys, every entry importable and constructible against a
dummy input. Scoped to the registries this repo owns/creates
(ENCODER_REGISTRY, FUSION_REGISTRY, DECODER_REGISTRY) — the generic
`Registry` class itself already refuses duplicate keys at import time
(see `cdlib.utils.registry`), so a real duplicate would fail collection
entirely rather than surface as a test failure here; these tests instead
guard against the keys/kwargs silently drifting out of sync with what
the rest of the P3 checklist assumes.
"""
from __future__ import annotations

import torch

from cdlib.models.decoders import DECODER_REGISTRY
from cdlib.models.encoders import ENCODER_REGISTRY
from cdlib.models.fusion import FUSION_REGISTRY


def test_encoder_registry_has_expected_keys():
    assert set(ENCODER_REGISTRY.keys()) == {"resnet18", "efficientnet_b0", "efficientnet_b2"}


def test_fusion_registry_has_expected_keys():
    assert set(FUSION_REGISTRY.keys()) == {"absdiff", "signed_fusion"}


def test_decoder_registry_has_expected_keys():
    assert set(DECODER_REGISTRY.keys()) == {"unet_decoder"}


def test_unknown_key_raises_with_available_keys_listed():
    try:
        ENCODER_REGISTRY.build("not_a_real_encoder")
        assert False, "expected KeyError"
    except KeyError as e:
        msg = str(e)
        assert "resnet18" in msg  # error message should help, not just say "not found"


def test_every_encoder_builds_and_runs_dummy_forward():
    # .eval() deliberately: at 32x32, B2's 5 stride-2 stages collapse
    # spatial dims to 1x1, and BatchNorm in *train* mode can't compute
    # stats from a single value per channel ("Expected more than 1 value
    # per channel when training"). eval() uses running stats instead, so
    # this stays a pure build/forward/shape smoke test; train-mode
    # correctness at a safe resolution is covered by test_shapes.py.
    for key in ENCODER_REGISTRY.keys():
        enc = ENCODER_REGISTRY.build(key, pretrained=False).eval()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            feats = enc.forward(x)
        assert isinstance(feats, list) and len(feats) > 0, f"{key}: forward() must return a list"


def test_every_fusion_builds_and_runs_dummy_forward():
    for key in FUSION_REGISTRY.keys():
        kwargs = {"mode": "signed"} if key == "signed_fusion" else {}
        fusion = FUSION_REGISTRY.build(key, in_channels=[8], **kwargs)
        a = [torch.rand(1, 8, 4, 4)]
        b = [torch.rand(1, 8, 4, 4)]
        out = fusion(a, b)
        assert isinstance(out, list) and len(out) == 1
