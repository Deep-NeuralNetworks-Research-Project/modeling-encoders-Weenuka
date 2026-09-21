"""Member 5 track (uncertainty, calibration, selective prediction, directional
equivariance). Contract: reset() / update(outputs, batch) / compute() -> dict[str, float].

`calibration.py` / `temperature.py` were originally authored in the
team's `metrics-integration-kusal` slice under the old P5 role; carried
here since Part A of Member 5's track builds directly on them.
`swap.py`'s `SwapCalibrationMetric`/`SCE_prob` additions and all of
`directional.py` are this session's work.
"""

from cdlib.metrics.base import Metric
from cdlib.metrics.calibration import CalibrationMetric
from cdlib.metrics.directional import DirectionalEquivarianceMetric
from cdlib.metrics.registry import METRIC_REGISTRY, build_metric
from cdlib.metrics.swap import SwapCalibrationMetric, SwapConsistencyMetric
from cdlib.metrics.temperature import TemperatureScaler

__all__ = [
    "METRIC_REGISTRY",
    "CalibrationMetric",
    "DirectionalEquivarianceMetric",
    "Metric",
    "SwapCalibrationMetric",
    "SwapConsistencyMetric",
    "TemperatureScaler",
    "build_metric",
]
