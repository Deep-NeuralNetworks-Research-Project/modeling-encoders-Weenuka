"""Correctness tests for the directional-head equivariance metric
(docs/member-tracks/05, Part C).

``PairOrderConsistencyLoss`` trains the directional head so that swapping
(I1,I2) permutes the appeared/disappeared channels; this metric measures
whether that property holds, independent of whether the model was
actually trained with that loss. Built on synthetic logits since no
trained checkpoint exists yet.
"""
from __future__ import annotations

import numpy as np
import torch

from cdlib.metrics.directional import (
    DirectionalEquivarianceMetric,
    directional_equivariance_error,
)


def _one_hot_logits(classes: np.ndarray) -> torch.Tensor:
    """[B,H,W] int class (0/1) -> [B,2,H,W] logits that argmax to it."""
    b, h, w = classes.shape
    logits = np.full((b, 2, h, w), -6.0, dtype=np.float32)
    for c in (0, 1):
        m = classes == c
        logits[:, c, :, :][m] = 6.0
    return torch.from_numpy(logits)


def test_perfectly_equivariant_head_gives_zero_error():
    b, h, w = 2, 8, 8
    rng = np.random.default_rng(0)
    class_fwd = rng.integers(0, 2, size=(b, h, w))
    # Construct rev so that permute(rev) == fwd exactly: rev = 1 - fwd.
    class_rev = 1 - class_fwd

    d_fwd = _one_hot_logits(class_fwd)
    d_rev = _one_hot_logits(class_rev)

    err = directional_equivariance_error(d_fwd.numpy(), d_rev.numpy())
    assert err == 0.0

    m = DirectionalEquivarianceMetric()
    m.update(
        {"aux": {"directional_logits": d_fwd, "directional_logits_swapped": d_rev}},
        {"mask": torch.ones(b, 1, h, w)},
    )
    out = m.compute()
    assert out["directional_equivariance_error"] == 0.0


def test_invariant_not_equivariant_head_gives_full_error():
    """A head that is merely *invariant* (predicts the same class under
    swap instead of swapping) should score maximal error under this
    metric — that's the whole point: invariance is the wrong property
    for a directional task, equivariance is required.
    """
    b, h, w = 2, 8, 8
    rng = np.random.default_rng(1)
    class_fwd = rng.integers(0, 2, size=(b, h, w))
    class_rev = class_fwd.copy()  # invariant: identical, not permuted

    d_fwd = _one_hot_logits(class_fwd)
    d_rev = _one_hot_logits(class_rev)

    err = directional_equivariance_error(d_fwd.numpy(), d_rev.numpy())
    assert err == 1.0


def test_partial_disagreement_is_fractional():
    b, h, w = 1, 4, 4
    class_fwd = np.zeros((b, h, w), dtype=int)
    class_rev = np.zeros((b, h, w), dtype=int)
    # Correct (permuted) on 12 of 16 pixels, wrong on 4.
    class_rev[0, :, :] = 1  # permute(1) = 0 == fwd everywhere first
    class_rev[0, 0, :4] = 0  # these 4 pixels: permute(0) = 1 != fwd(0) -> wrong

    d_fwd = _one_hot_logits(class_fwd)
    d_rev = _one_hot_logits(class_rev)
    err = directional_equivariance_error(d_fwd.numpy(), d_rev.numpy())
    assert err == 4 / 16


def test_valid_mask_restricts_the_average():
    b, h, w = 1, 1, 4
    class_fwd = np.array([[[0, 0, 0, 0]]])
    class_rev = np.array([[[0, 1, 1, 1]]])  # wrong at index 0, correct elsewhere
    valid = np.array([[True, True, True, False]]).reshape(1, 1, 4)

    err_all = directional_equivariance_error(
        _one_hot_logits(class_fwd).numpy(), _one_hot_logits(class_rev).numpy()
    )
    err_valid = directional_equivariance_error(
        _one_hot_logits(class_fwd).numpy(), _one_hot_logits(class_rev).numpy(), valid=valid
    )
    assert err_all == 1 / 4  # wrong only at index 0, out of 4
    assert err_valid == 1 / 3  # index 3 excluded, so 1 wrong out of 3
