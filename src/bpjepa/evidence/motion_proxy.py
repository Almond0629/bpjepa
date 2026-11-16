"""Unwindowed native-frame phase descriptor shared with alignment fallback."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class PhaseMeasurement:
    result: tuple[float, float, float] | None
    displacement: float
    response: float
    bin: int  # 0:[0,.005), 1:[.005,.02), 2:[.02,infinity), 3:UNAVAILABLE


def phase_measurement(previous: np.ndarray, current: np.ndarray) -> PhaseMeasurement:
    if (previous.shape != current.shape or previous.ndim != 2
            or previous.dtype != np.float32 or current.dtype != np.float32):
        raise ValueError("matching float32 grayscale frames required")
    try:
        (dx, dy), response = cv2.phaseCorrelate(previous, current)
    except cv2.error:
        return PhaseMeasurement(None, math.nan, math.nan, 3)
    result = float(dx), float(dy), float(response)
    if not all(math.isfinite(value) for value in result):
        return PhaseMeasurement(result, math.nan, result[2], 3)
    displacement = math.hypot(dx / previous.shape[1], dy / previous.shape[0])
    if not math.isfinite(displacement):
        return PhaseMeasurement(result, math.nan, result[2], 3)
    if response < .02:
        return PhaseMeasurement(result, displacement, result[2], 3)
    return PhaseMeasurement(result, displacement, result[2],
                            0 if displacement < .005 else 1 if displacement < .02 else 2)
