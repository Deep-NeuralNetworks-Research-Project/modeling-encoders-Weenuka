"""METRIC_REGISTRY — one new metric file + one register() line.

Trimmed to the metrics this repo actually carries (Member 5's uncertainty
track). The team's full `change-detect` monorepo also registers
`segmentation`, `boundary`, `region`, `robustness` (Member 3's evaluation
suite, not vendored here) — this file intentionally doesn't reference
them so nothing dangles.
"""
from __future__ import annotations

from cdlib.metrics.base import Metric
from cdlib.utils.registry import Registry

METRIC_REGISTRY: Registry = Registry("metric")


def build_metric(key: str, **kwargs) -> Metric:
    return METRIC_REGISTRY.build(key, **kwargs)


def _register_builtins() -> None:
    # Imported here to avoid circular imports at module load.
    from cdlib.metrics.calibration import CalibrationMetric
    from cdlib.metrics.directional import DirectionalEquivarianceMetric
    from cdlib.metrics.swap import SwapCalibrationMetric, SwapConsistencyMetric

    METRIC_REGISTRY.register("calibration")(CalibrationMetric)
    METRIC_REGISTRY.register("swap_consistency")(SwapConsistencyMetric)
    METRIC_REGISTRY.register("swap_calibration")(SwapCalibrationMetric)
    METRIC_REGISTRY.register("directional_equivariance")(DirectionalEquivarianceMetric)


_register_builtins()
