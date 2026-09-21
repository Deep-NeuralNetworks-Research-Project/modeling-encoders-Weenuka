"""Pair-swap consistency for ablation B7.

P4 owns the training loss; P5 runs the *metric* on every model, including
ones not trained with the loss. Score = fraction of valid pixels whose
hard prediction is unchanged under (I1,I2) ↔ (I2,I1).

This file also carries the two swap-based diagnostics that are specific
to Member 5's uncertainty track (docs/member-tracks/05):

- ``swap_prob_disagreement`` / ``SCE_prob``: the *continuous* per-pixel
  companion to the hard agreement above. Needs no ground truth, so it is
  available at deployment time as a label-free confidence signal
  (Part B) — test whether pixels with high disagreement have higher
  error rates.
- ``SwapCalibrationMetric``: per-ordering ECE (forward vs. reversed) and
  the gap between them, ``Δ-ECE_swap`` — the calibration analogue of
  swap-consistency. For any architecturally symmetric fusion (abs-diff,
  the even head), this must come out exactly 0, the same free
  correctness check ``SwapConsistencyMetric`` gives on the hard side.

``SwapConsistencyMetric`` itself was authored in the team's
`metrics-integration-kusal` slice; ``swap_prob_disagreement``,
``sce_prob_pair_score`` and ``SwapCalibrationMetric`` are this session's
Member 5 additions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cdlib.metrics._tensor import as_numpy, valid_mask, _squeeze_mask
from cdlib.metrics.base import Metric
from cdlib.metrics.calibration import equal_mass_ece, risk_coverage_from_scores


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def swap_agreement(logits_fwd: np.ndarray, logits_rev: np.ndarray, threshold: float = 0.5) -> float:
    pf = (1 / (1 + np.exp(-np.clip(_squeeze_mask(np.asarray(logits_fwd)), -80, 80)))) >= threshold
    pr = (1 / (1 + np.exp(-np.clip(_squeeze_mask(np.asarray(logits_rev)), -80, 80)))) >= threshold
    return float((pf == pr).mean())


class SwapConsistencyMetric(Metric):
    """Requires the caller to pass ``outputs['logits_swapped']`` from a second forward."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self._agree = 0
        self._n = 0

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        if "logits_swapped" not in outputs:
            raise KeyError("SwapConsistencyMetric needs outputs['logits_swapped']")
        fwd = _squeeze_mask(as_numpy(outputs["logits"]))
        rev = _squeeze_mask(as_numpy(outputs["logits_swapped"]))
        gt = _squeeze_mask(as_numpy(batch["mask"]))
        valid = valid_mask(gt)
        pf = (1 / (1 + np.exp(-np.clip(fwd, -80, 80)))) >= self.threshold
        pr = (1 / (1 + np.exp(-np.clip(rev, -80, 80)))) >= self.threshold
        self._agree += int(((pf == pr) & valid).sum())
        self._n += int(valid.sum())

    def compute(self) -> dict[str, float]:
        return {"swap_consistency": (self._agree / self._n) if self._n else 1.0}


def score_model_swap(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor, threshold: float = 0.5) -> float:
    model.eval()
    with torch.no_grad():
        fwd = model(img1, img2)["logits"]
        rev = model(img2, img1)["logits"]
    return swap_agreement(as_numpy(fwd), as_numpy(rev), threshold=threshold)


def swap_prob_disagreement(logits_fwd: np.ndarray, logits_rev: np.ndarray) -> np.ndarray:
    """Per-pixel ``|p_fwd - p_rev|`` — ``SCE_prob`` in the paper draft.

    Unlike ``swap_agreement`` (hard, 0/1), this is continuous and needs no
    ground truth: it can be computed on unlabelled data at deployment
    time. Identically 0 everywhere for architecturally symmetric fusion
    (``|a-b|``, the even head) — same guarantee as the hard metric, for
    the same reason (h(a-b) = h(b-a) by construction).
    """
    pf = _sigmoid(_squeeze_mask(np.asarray(logits_fwd)))
    pr = _sigmoid(_squeeze_mask(np.asarray(logits_rev)))
    return np.abs(pf - pr)


def sce_prob_pair_score(logits_fwd: np.ndarray, logits_rev: np.ndarray) -> float:
    """Pair-level, GT-free confidence score derived from SCE_prob:
    ``1 - mean(SCE_prob)`` over the pair. High swap-agreement -> high
    score. Directly comparable to ``calibration.msr_pair_score`` /
    ``soft_dice_confidence`` on the same risk-coverage axis (Part B) —
    with the explicit caveat that it is identically 1.0 (perfect,
    uninformative) for symmetric fusion, so it is only a useful ranking
    signal for the models that actually need it.
    """
    d = swap_prob_disagreement(logits_fwd, logits_rev)
    return float(1.0 - d.mean())


class SwapCalibrationMetric(Metric):
    """Δ-ECE_swap and the SCE_prob-vs-error relationship.

    Requires ``outputs['logits_swapped']`` (``ProposedModel`` emits this
    in train mode; for an eval-time pass, call the model twice — once per
    ordering — and stitch it in, exactly as ``score_model_swap`` does for
    the hard metric).
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = int(n_bins)
        self.reset()

    def reset(self) -> None:
        self._p_fwd: list[np.ndarray] = []
        self._p_rev: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._sce_prob: list[np.ndarray] = []
        self._err_fwd: list[np.ndarray] = []
        self._pair_sce_scores: list[float] = []
        self._pair_msr_scores: list[float] = []
        self._pair_errors: list[float] = []

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        if "logits_swapped" not in outputs:
            raise KeyError("SwapCalibrationMetric needs outputs['logits_swapped']")
        fwd = _squeeze_mask(as_numpy(outputs["logits"]))
        rev = _squeeze_mask(as_numpy(outputs["logits_swapped"]))
        gt_raw = _squeeze_mask(as_numpy(batch["mask"]))
        valid = valid_mask(gt_raw)
        y = (gt_raw > 0.5) & valid

        pf = _sigmoid(fwd)
        pr = _sigmoid(rev)
        sce = np.abs(pf - pr)

        for i in range(fwd.shape[0]):
            v = valid[i]
            self._p_fwd.append(pf[i][v].astype(np.float64))
            self._p_rev.append(pr[i][v].astype(np.float64))
            self._y.append(y[i][v].astype(np.float64))
            self._sce_prob.append(sce[i][v].astype(np.float64))
            pred_fwd = pf[i][v] >= 0.5
            self._err_fwd.append((pred_fwd != y[i][v]).astype(np.float64))

            # pair-level: does SCE_prob rank pairs by error as well as MSR?
            full_pf = pf[i]
            self._pair_sce_scores.append(sce_prob_pair_score(fwd[i], rev[i]))
            conf = np.maximum(full_pf, 1.0 - full_pf)
            self._pair_msr_scores.append(float(conf.mean()))
            pred_full = full_pf >= 0.5
            tp = float((pred_full & y[i] & v).sum())
            fp = float((pred_full & ~y[i] & v).sum())
            fn = float((~pred_full & y[i] & v).sum())
            f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
            self._pair_errors.append(1.0 - f1)

    def compute(self) -> dict[str, float]:
        if not self._p_fwd:
            return {
                "ece_fwd": 0.0,
                "ece_rev": 0.0,
                "delta_ece_swap": 0.0,
                "sce_prob_mean": 0.0,
                "sce_prob_error_corr": 0.0,
                "aurc_sce_prob": 0.0,
                "aurc_msr": 0.0,
            }
        p_fwd = np.concatenate(self._p_fwd)
        p_rev = np.concatenate(self._p_rev)
        y = np.concatenate(self._y)
        sce = np.concatenate(self._sce_prob)
        err_fwd = np.concatenate(self._err_fwd)

        conf_fwd = np.where(p_fwd >= 0.5, p_fwd, 1.0 - p_fwd)
        correct_fwd = (1.0 - err_fwd)
        ece_fwd, *_ = equal_mass_ece(conf_fwd, correct_fwd, self.n_bins)

        conf_rev = np.where(p_rev >= 0.5, p_rev, 1.0 - p_rev)
        correct_rev = ((p_rev >= 0.5).astype(np.float64) == y).astype(np.float64)
        ece_rev, *_ = equal_mass_ece(conf_rev, correct_rev, self.n_bins)

        delta = abs(ece_fwd - ece_rev)

        # Does high SCE_prob predict high pixel error? Point-biserial
        # correlation between the continuous disagreement signal and the
        # binary "was the forward prediction wrong" outcome.
        if sce.size > 1 and sce.std() > 0 and err_fwd.std() > 0:
            corr = float(np.corrcoef(sce, err_fwd)[0, 1])
        else:
            corr = 0.0

        aurc_sce, _, _, _ = risk_coverage_from_scores(
            np.asarray(self._pair_sce_scores), np.asarray(self._pair_errors)
        )
        aurc_msr, _, _, _ = risk_coverage_from_scores(
            np.asarray(self._pair_msr_scores), np.asarray(self._pair_errors)
        )

        return {
            "ece_fwd": float(ece_fwd),
            "ece_rev": float(ece_rev),
            "delta_ece_swap": float(delta),
            "sce_prob_mean": float(sce.mean()),
            "sce_prob_error_corr": corr,
            "aurc_sce_prob": float(aurc_sce),
            "aurc_msr": float(aurc_msr),
        }
