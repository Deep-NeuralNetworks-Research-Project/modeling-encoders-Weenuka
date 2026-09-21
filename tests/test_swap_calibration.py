"""Correctness tests for Δ-ECE_swap and SCE_prob (docs/member-tracks/05).

No trained checkpoints exist yet anywhere in this project (paper/sections
/results.tex is still all TBD), so these are built on synthetic
logits/masks — the same approach Phase 1 used for the overfit-one-batch
sanity check. They check the two claims the track doc stakes out:

1. Architecturally symmetric fusion (forward logits == reversed logits,
   which is what |a-b| guarantees regardless of trunk) must give
   Δ-ECE_swap == 0 and SCE_prob == 0 *exactly* — a free correctness
   check on the taxonomy, independent of any trained weights.
2. SCE_prob (label-free) should correlate with actual pixel error when
   disagreement and error are constructed to coincide — the thing that
   would make it a genuine deployment-time confidence signal.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cdlib.metrics.swap import (
    SwapCalibrationMetric,
    sce_prob_pair_score,
    swap_prob_disagreement,
)


def _mask(b, h, w, change_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    m = (rng.random((b, 1, h, w)) < change_frac).astype(np.float32)
    return torch.from_numpy(m)


def test_symmetric_logits_give_zero_sce_prob_and_zero_delta_ece():
    """Simulates an abs-diff-style model: forward and reversed logits are
    identical by construction (the guarantee doesn't depend on training).
    """
    b, h, w = 2, 16, 16
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(b, 1, h, w, generator=g)
    mask = _mask(b, h, w)

    d = swap_prob_disagreement(logits.numpy(), logits.numpy())
    assert np.allclose(d, 0.0)

    m = SwapCalibrationMetric(n_bins=4)
    m.update(
        {"logits": logits, "logits_swapped": logits.clone(), "aux": {}},
        {"mask": mask},
    )
    out = m.compute()
    assert out["delta_ece_swap"] == 0.0
    assert out["sce_prob_mean"] == 0.0


def test_asymmetric_logits_give_nonzero_sce_prob():
    b, h, w = 2, 16, 16
    g = torch.Generator().manual_seed(1)
    fwd = torch.randn(b, 1, h, w, generator=g)
    rev = torch.randn(b, 1, h, w, generator=torch.Generator().manual_seed(2))
    mask = _mask(b, h, w)

    m = SwapCalibrationMetric(n_bins=4)
    m.update({"logits": fwd, "logits_swapped": rev, "aux": {}}, {"mask": mask})
    out = m.compute()
    assert out["sce_prob_mean"] > 0.0


def test_sce_prob_pair_score_is_one_minus_mean_disagreement():
    b, h, w = 1, 8, 8
    fwd = torch.zeros(b, 1, h, w)
    rev = torch.zeros(b, 1, h, w)
    rev[0, 0, 0, 0] = 10.0  # one pixel flips from p~0.5 to p~1.0 under reversal
    score = sce_prob_pair_score(fwd.numpy(), rev.numpy())
    d = swap_prob_disagreement(fwd.numpy(), rev.numpy())
    assert score == pytest.approx(1.0 - float(d.mean()), abs=1e-6)
    assert 0.0 <= score <= 1.0


def test_sce_prob_predicts_error_when_constructed_to_correlate():
    """Build a case where the forward prediction is wrong exactly where
    swap-disagreement is high, and confirm the correlation shows up —
    this is the Part B result the track doc calls out as the genuine
    deployment-time contribution, if it holds.
    """
    h, w = 1, 64
    y = torch.zeros(1, 1, h, w)
    y[0, 0, 0, : w // 2] = 1.0  # left half changed, right half unchanged

    fwd = torch.zeros(1, 1, h, w)
    fwd[0, 0, 0, : w // 2] = 6.0   # confident + correct on the left half
    fwd[0, 0, 0, w // 2 :] = -6.0  # confident + correct on the right half
    # Make the model wrong on a contiguous block, and disagree with the
    # reversed pass on exactly that same block.
    wrong_start, wrong_end = 10, 20
    fwd[0, 0, 0, wrong_start:wrong_end] = -6.0  # now confidently *wrong* there

    rev = fwd.clone()
    rev[0, 0, 0, wrong_start:wrong_end] = 6.0  # disagrees with fwd only on the wrong block
    # everywhere else, rev matches fwd -> zero disagreement there

    m = SwapCalibrationMetric(n_bins=4)
    m.update({"logits": fwd, "logits_swapped": rev, "aux": {}}, {"mask": y})
    out = m.compute()

    assert out["sce_prob_error_corr"] > 0.5  # strong positive correlation
    assert out["aurc_sce_prob"] <= out["aurc_msr"] + 1e-9  # SCE_prob ranks this pair correctly


def test_ece_fwd_matches_standalone_calibration_metric():
    """ece_fwd must agree with plain CalibrationMetric on the forward pass
    alone — Δ-ECE_swap should be a strict extension, not a different
    formula in disguise.
    """
    from cdlib.metrics.calibration import CalibrationMetric

    b, h, w = 2, 16, 16
    g = torch.Generator().manual_seed(3)
    fwd = torch.randn(b, 1, h, w, generator=g)
    rev = torch.randn(b, 1, h, w, generator=torch.Generator().manual_seed(4))
    mask = _mask(b, h, w, seed=5)

    m_swap = SwapCalibrationMetric(n_bins=8)
    m_swap.update({"logits": fwd, "logits_swapped": rev, "aux": {}}, {"mask": mask})
    out = m_swap.compute()

    m_plain = CalibrationMetric(n_bins=8)
    m_plain.update({"logits": fwd}, {"mask": mask})
    plain = m_plain.compute()

    assert out["ece_fwd"] == plain["ece_pooled"]
