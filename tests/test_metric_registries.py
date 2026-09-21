"""Registry integrity + frozen metric contract on dummy tensors.

This repo's METRIC_REGISTRY is deliberately trimmed to Member 5's track
(calibration, swap_consistency, swap_calibration,
directional_equivariance) — see cdlib/metrics/registry.py's docstring.
The full team registry also has segmentation/boundary/region/robustness
(Member 3's evaluation suite), not vendored here.
"""
from __future__ import annotations

import torch

from cdlib.metrics.base import Metric
from cdlib.metrics.registry import METRIC_REGISTRY, build_metric


def _dummy(b=2, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "outputs": {
            "logits": torch.randn(b, 1, h, w, generator=g),
            "confidence": None,
            "aux": {},
        },
        "batch": {
            "img1": torch.rand(b, 3, h, w, generator=g),
            "img2": torch.rand(b, 3, h, w, generator=g),
            "mask": (torch.rand(b, 1, h, w, generator=g) > 0.9).float(),
            "nuisance_label": torch.zeros(b, dtype=torch.int64),
        },
    }


def test_expected_keys_present():
    for key in ("calibration", "swap_consistency", "swap_calibration", "directional_equivariance"):
        assert key in METRIC_REGISTRY


def test_unknown_key_is_helpful():
    try:
        METRIC_REGISTRY.get("not_a_metric")
        raise AssertionError("should have raised")
    except KeyError as e:
        assert "calibration" in str(e)


_NEEDS_SPECIAL_INPUT = {"swap_consistency", "swap_calibration", "directional_equivariance"}


def test_every_metric_update_compute_reset():
    dummy = _dummy()
    for key in METRIC_REGISTRY.keys():
        if key in _NEEDS_SPECIAL_INPUT:
            continue  # needs logits_swapped and/or aux directional_logits(_swapped)
        m = build_metric(key)
        assert isinstance(m, Metric)
        m.update(dummy["outputs"], dummy["batch"])
        out = m.compute()
        assert isinstance(out, dict)
        assert out, f"{key} returned empty dict"
        for k, v in out.items():
            assert isinstance(k, str)
            assert isinstance(v, float), f"{key}.{k} is {type(v)}"
        m.reset()
        z = m.compute()
        assert isinstance(z, dict)


def test_swap_consistency_contract():
    dummy = _dummy()
    dummy["outputs"]["logits_swapped"] = dummy["outputs"]["logits"].flip(0)
    m = build_metric("swap_consistency")
    m.update(dummy["outputs"], dummy["batch"])
    out = m.compute()
    assert 0.0 <= out["swap_consistency"] <= 1.0


def test_swap_calibration_contract():
    dummy = _dummy()
    dummy["outputs"]["logits_swapped"] = dummy["outputs"]["logits"].flip(0)
    m = build_metric("swap_calibration")
    m.update(dummy["outputs"], dummy["batch"])
    out = m.compute()
    for k in ("ece_fwd", "ece_rev", "delta_ece_swap", "sce_prob_mean", "aurc_sce_prob", "aurc_msr"):
        assert k in out and isinstance(out[k], float)
    assert out["sce_prob_mean"] >= 0.0
    m.reset()
    z = m.compute()
    assert z["delta_ece_swap"] == 0.0


def test_directional_equivariance_contract():
    dummy = _dummy()
    b, h, w = 2, 32, 32
    g = torch.Generator().manual_seed(1)
    dummy["outputs"]["aux"] = {
        "directional_logits": torch.randn(b, 2, h, w, generator=g),
        "directional_logits_swapped": torch.randn(b, 2, h, w, generator=g),
    }
    m = build_metric("directional_equivariance")
    m.update(dummy["outputs"], dummy["batch"])
    out = m.compute()
    assert 0.0 <= out["directional_equivariance_error"] <= 1.0
