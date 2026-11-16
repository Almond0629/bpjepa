"""Cut/wipe detection and camera alignment (ECC with phase-correlation fallback)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MotionConfig:
    grayscale_method: str = "channel_mean"
    ecc_iterations: int = 100
    ecc_epsilon: float = 1e-6
    ecc_min_confidence: float = 0.40
    phase_min_response: float = 0.02
    cut_histogram_correlation: float = 0.20
    wipe_changed_fraction: float = 0.85
    wipe_pixel_difference: float = 0.35
    lower_percentile: float = 0.05
    upper_percentile: float = 0.95
    delta: float = 1e-6

    def validate(self) -> None:
        if self.grayscale_method != "channel_mean":
            raise ValueError("unsupported grayscale method")
        if self.ecc_iterations <= 0 or self.ecc_epsilon <= 0:
            raise ValueError("ECC iteration and epsilon settings must be positive")
        for value in (
            self.ecc_min_confidence,
            self.phase_min_response,
            self.cut_histogram_correlation,
            self.wipe_changed_fraction,
            self.wipe_pixel_difference,
        ):
            if not 0 <= value <= 1:
                raise ValueError("motion thresholds must be in [0,1]")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 1:
            raise ValueError("invalid change percentile bounds")
        if self.delta <= 0:
            raise ValueError("change normalization delta must be positive")


@dataclass(frozen=True)
class AlignmentResult:
    status: str
    transform: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    confidence: float | None
    reason: str | None = None
    ecc_status: str | None = None
    ecc_confidence: float | None = None
    phase_response: float | None = None
    affine_degeneracy: tuple[bool, bool, bool, bool] = (False, False, False, False)

    def matrix(self) -> np.ndarray:
        if self.transform is None:
            raise ValueError("alignment result has no transform")
        matrix = np.asarray(self.transform, dtype=np.float32)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError("alignment transform is invalid")
        return matrix


CutDetector = Callable[[np.ndarray, np.ndarray, MotionConfig], str | None]
Aligner = Callable[[np.ndarray, np.ndarray, MotionConfig], AlignmentResult]


def _gray_frames(frames: torch.Tensor | np.ndarray) -> list[np.ndarray]:
    values = frames.detach().cpu().numpy() if torch.is_tensor(frames) else np.asarray(frames)
    if values.ndim == 4 and values.shape[-1] == 3:
        values = values.mean(-1)
    elif values.ndim == 4 and values.shape[1] == 3:
        values = values.mean(1)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("motion frames must be finite [T,H,W] or RGB frames")
    minimum, maximum = float(values.min()), float(values.max())
    if minimum < 0 or maximum > 1:
        raise ValueError("motion frames must be scaled to [0,1]")
    if min(values.shape[1:]) <= 1:
        raise ValueError("motion frames must have spatial dimensions greater than one")
    return [np.ascontiguousarray(frame, dtype=np.float32) for frame in values]


def measure_cut_wipe(previous: np.ndarray, current: np.ndarray,
                     config: MotionConfig) -> tuple[float, float, bool, bool]:
    """Return both frozen detectors and their raw measurements, without short-circuiting."""
    hist_previous = cv2.calcHist([(previous * 255).astype(np.uint8)], [0], None, [32], [0, 256])
    hist_current = cv2.calcHist([(current * 255).astype(np.uint8)], [0], None, [32], [0, 256])
    correlation = float(cv2.compareHist(hist_previous, hist_current, cv2.HISTCMP_CORREL))
    changed = np.abs(current - previous) >= config.wipe_pixel_difference
    fraction = float(changed.mean())
    return (correlation, fraction,
            not np.isfinite(correlation) or correlation < config.cut_histogram_correlation,
            fraction >= config.wipe_changed_fraction)


def detect_cut_or_wipe(previous: np.ndarray, current: np.ndarray, config: MotionConfig) -> str | None:
    _, _, cut, wipe = measure_cut_wipe(previous, current, config)
    if cut:
        return "cut"
    if wipe:
        return "replay_wipe"
    return None


def _tuple_matrix(matrix: np.ndarray):
    return tuple(tuple(float(value) for value in row) for row in matrix)


def affine_degeneracy(matrix: np.ndarray) -> tuple[bool, bool, bool, bool]:
    """Nonfinite, nonpositive determinant, excessive condition, failed condition."""
    matrix = np.asarray(matrix, dtype=np.float64)
    nonfinite = matrix.shape != (2, 3) or not np.isfinite(matrix).all()
    if nonfinite:
        return True, False, False, False
    linear = matrix[:, :2]
    nonpositive = float(np.linalg.det(linear)) <= 0
    try:
        excessive = bool(np.linalg.cond(linear) > 1e6)
        failed = False
    except (ValueError, np.linalg.LinAlgError):
        excessive, failed = False, True
    return False, nonpositive, excessive, failed


_PHASE_NOT_SUPPLIED = object()


def estimate_alignment(previous: np.ndarray, current: np.ndarray, config: MotionConfig,
                       *, phase_result=_PHASE_NOT_SUPPLIED) -> AlignmentResult:
    """Estimate previous-to-current affine motion, falling back to translation."""
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_iterations,
        config.ecc_epsilon,
    )
    ecc_reason = None
    ecc_status = "error"
    ecc_confidence = None
    degeneracy = (False, False, False, False)
    try:
        confidence, matrix = cv2.findTransformECC(
            current, previous, identity.copy(), cv2.MOTION_AFFINE, criteria, None, 1
        )
        ecc_confidence = float(confidence)
        degeneracy = affine_degeneracy(matrix)
        if any(degeneracy):
            ecc_status = "degenerate"
            ecc_reason = f"degenerate ECC affine ({degeneracy})"
        elif not np.isfinite(confidence) or confidence < config.ecc_min_confidence:
            ecc_status = "low_confidence"
            ecc_reason = f"invalid/low-confidence ECC ({confidence})"
        else:
            response = phase_result[2] if isinstance(phase_result, tuple) else None
            return AlignmentResult("ecc", _tuple_matrix(matrix), float(confidence),
                                   ecc_status="accepted", ecc_confidence=float(confidence),
                                   phase_response=response)
    except cv2.error as error:
        ecc_reason = f"ECC failed: {error.code}"

    if phase_result is _PHASE_NOT_SUPPLIED:
        try:
            (dx, dy), response = cv2.phaseCorrelate(previous, current)
            phase_result = (dx, dy, response)
        except cv2.error as error:
            return AlignmentResult("failure", None, None,
                                   f"{ecc_reason}; phase correlation failed: {error.code}",
                                   ecc_status, ecc_confidence, None, degeneracy)
    if phase_result is None:
        return AlignmentResult("failure", None, None, f"{ecc_reason}; phase correlation failed",
                               ecc_status, ecc_confidence, None, degeneracy)
    dx, dy, response = phase_result
    try:
        matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        if np.isfinite(matrix).all() and np.isfinite(response) and response >= config.phase_min_response:
            return AlignmentResult(
                "phase_correlation", _tuple_matrix(matrix), float(response), ecc_reason,
                ecc_status, ecc_confidence, float(response), degeneracy
            )
        return AlignmentResult("failure", None, float(response),
                               f"{ecc_reason}; invalid phase correlation", ecc_status,
                               ecc_confidence, float(response), degeneracy)
    except (TypeError, ValueError, OverflowError):
        return AlignmentResult("failure", None, None, f"{ecc_reason}; invalid phase correlation",
                               ecc_status, ecc_confidence, None, degeneracy)


def _scaled_matrix(
    result: AlignmentResult,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> np.ndarray:
    matrix = result.matrix().astype(np.float64)
    source_height, source_width = source_grid
    target_height, target_width = target_grid
    if min(source_height, source_width, target_height, target_width) <= 1:
        raise ValueError("alignment grids must have dimensions greater than one")
    scale_x = source_width / target_width
    scale_y = source_height / target_height
    source_scale = np.array(
        [
            [scale_x, 0.0, (scale_x - 1.0) / 2.0],
            [0.0, scale_y, (scale_y - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target_scale = np.linalg.inv(source_scale)
    homogeneous = np.vstack((matrix, (0.0, 0.0, 1.0)))
    scaled = (target_scale @ homogeneous @ source_scale)[:2]
    if not np.isfinite(scaled).all():
        raise ValueError("scaled alignment transform is invalid")
    return scaled.astype(np.float32)


def _warp_features(
    previous: torch.Tensor,
    result: AlignmentResult,
    *,
    motion_grid: tuple[int, int] | None = None,
) -> torch.Tensor:
    height, width, channels = previous.shape
    source = previous.detach().cpu().numpy().astype(np.float32)
    warped = np.empty_like(source)
    matrix = _scaled_matrix(result, motion_grid or (height, width), (height, width))
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if result.status == "ecc" else 0)
    for channel in range(channels):
        warped[..., channel] = cv2.warpAffine(
            source[..., channel],
            matrix,
            (width, height),
            flags=flags,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return torch.from_numpy(warped).to(previous.device, previous.dtype)
