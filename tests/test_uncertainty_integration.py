"""End-to-end wiring check: real ProposedModel output -> Member 5's new
metrics, not just hand-built dicts. This is the integration point that
matters once real checkpoints exist — confirms the metrics consume
exactly what ProposedModel emits in train mode, with no adapter glue
needed.
"""
from __future__ import annotations

import torch

from cdlib.metrics.directional import DirectionalEquivarianceMetric
from cdlib.metrics.registry import build_metric
from cdlib.metrics.swap import SwapCalibrationMetric
from cdlib.models.proposed import ProposedModel

B, H, W = 2, 64, 64


def _tiny_proposed(**kwargs):
    defaults = dict(
        encoder={"name": "resnet18", "pretrained": False},
        fusion={"name": "signed_fusion", "mode": "signed"},
        alignment={"name": "identity"},
        decoder={"name": "unet_decoder"},
        heads={"confidence": True, "directional": True},
        compute_swap=True,
    )
    defaults.update(kwargs)
    return ProposedModel(**defaults)


def _batch():
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, :, W // 2 :] = 1.0
    return img1, img2, {"mask": mask}


def test_swap_calibration_consumes_real_proposed_model_output():
    torch.manual_seed(0)
    model = _tiny_proposed()
    model.train()
    img1, img2, batch = _batch()
    out = model(img1, img2)

    m = build_metric("swap_calibration")
    m.update(out, batch)
    result = m.compute()
    assert 0.0 <= result["delta_ece_swap"]
    assert 0.0 <= result["sce_prob_mean"]


def test_directional_equivariance_consumes_real_proposed_model_output():
    torch.manual_seed(0)
    model = _tiny_proposed()
    model.train()
    img1, img2, batch = _batch()
    out = model(img1, img2)

    m = build_metric("directional_equivariance")
    m.update(out, batch)
    result = m.compute()
    assert 0.0 <= result["directional_equivariance_error"] <= 1.0


def test_symmetric_fusion_proposed_model_has_zero_sce_prob():
    """The identity that matters for the paper: abs-diff fusion should
    give exactly the same guarantee inside the real composed model, not
    just in isolated synthetic tensors.
    """
    torch.manual_seed(0)
    model = _tiny_proposed(fusion={"name": "absdiff"}, heads={"confidence": False, "directional": False})
    model.eval()  # eval() so BN doesn't perturb fwd vs rev batch statistics differently
    img1, img2, batch = _batch()
    with torch.no_grad():
        fwd_only = model(img1, img2)
        rev_only = model(img2, img1)

    m = SwapCalibrationMetric()
    m.update({"logits": fwd_only["logits"], "logits_swapped": rev_only["logits"], "aux": {}}, batch)
    result = m.compute()
    assert result["sce_prob_mean"] < 1e-5
    assert result["delta_ece_swap"] < 1e-5
