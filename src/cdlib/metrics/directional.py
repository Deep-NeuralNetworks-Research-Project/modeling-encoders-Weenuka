"""Directional-head evaluation: equivariance under swap (Member 5, Part C).

The directional head predicts 2 channels, ``[appeared, disappeared]``
(``heads/directional_head.py``). Under swap (I1,I2) -> (I2,I1), the
correct behaviour is **equivariance, not invariance**: channel 0 and
channel 1 should exchange. This is exactly what
``losses.pair_order_consistency.PairOrderConsistencyLoss`` trains for
(it permutes ``d_rev``'s channels before comparing to ``d_fwd``); this
metric measures whether that property actually holds, on any model —
trained with the loss or not — the same way ``swap.SwapConsistencyMetric``
measures the binary (invariant) case.

Also reports per-class accuracy against ground-truth directional labels
when they are available. Real datasets don't ship these (see the
Member 5 track doc's "annotation problem" note): callers pass
``batch['directional_label']`` only when derived/pseudo/synthetic labels
exist. When absent, only the label-free equivariance numbers are
reported.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.base import Metric


def _argmax_channel(logits: np.ndarray) -> np.ndarray:
    """[B,2,H,W] -> [B,H,W] hard class (0=appeared, 1=disappeared)."""
    return np.argmax(np.asarray(logits), axis=1)


def directional_equivariance_error(
    d_fwd: np.ndarray, d_rev: np.ndarray, valid: np.ndarray | None = None
) -> float:
    """Fraction of valid pixels whose predicted class does NOT correctly
    swap under reversal. 0.0 = perfectly equivariant.

    Correct behaviour: class_fwd == permute(class_rev), where permute
    swaps 0<->1 (appeared<->disappeared). This is the exact
    per-pixel-class analogue of what ``PairOrderConsistencyLoss``
    penalises at the logit level.
    """
    class_fwd = _argmax_channel(d_fwd)
    class_rev = _argmax_channel(d_rev)
    permuted_rev = 1 - class_rev
    disagree = class_fwd != permuted_rev
    if valid is not None:
        v = np.asarray(valid).astype(bool)
        if v.sum() == 0:
            return 0.0
        return float(disagree[v].mean())
    return float(disagree.mean())


class DirectionalEquivarianceMetric(Metric):
    """Requires ``aux['directional_logits']`` and
    ``aux['directional_logits_swapped']`` (``ProposedModel`` emits both
    in train mode when its directional head is enabled).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._disagree = 0
        self._n = 0
        self._correct_fwd = 0
        self._n_labelled = 0

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        aux = outputs.get("aux") or {}
        d_fwd = aux.get("directional_logits")
        d_rev = aux.get("directional_logits_swapped")
        if d_fwd is None or d_rev is None:
            raise KeyError(
                "DirectionalEquivarianceMetric needs "
                "outputs['aux']['directional_logits'] and "
                "outputs['aux']['directional_logits_swapped']"
            )
        d_fwd = as_numpy(d_fwd)
        d_rev = as_numpy(d_rev)

        gt_mask = batch.get("mask")
        if gt_mask is not None:
            v = valid_mask(as_numpy(gt_mask))
            if v.ndim == 4 and v.shape[1] == 1:
                v = v[:, 0]
        else:
            v = np.ones(d_fwd.shape[0:1] + d_fwd.shape[2:], dtype=bool)

        class_fwd = _argmax_channel(d_fwd)
        class_rev = _argmax_channel(d_rev)
        permuted_rev = 1 - class_rev
        disagree = class_fwd != permuted_rev
        self._disagree += int(disagree[v].sum())
        self._n += int(v.sum())

        label = batch.get("directional_label")
        if label is not None:
            y = as_numpy(label)
            if y.ndim == 4 and y.shape[1] == 1:
                y = y[:, 0]
            labelled = v & (y >= 0)
            if labelled.any():
                self._correct_fwd += int((class_fwd[labelled] == y[labelled]).sum())
                self._n_labelled += int(labelled.sum())

    def compute(self) -> dict[str, float]:
        out = {
            "directional_equivariance_error": (self._disagree / self._n) if self._n else 0.0,
        }
        if self._n_labelled:
            out["directional_accuracy"] = self._correct_fwd / self._n_labelled
        return out
