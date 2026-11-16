"""Raw change and pooling over fully in-bounds affine-warp tokens.

Legacy Phase-2 reflected-border maps are deliberately not consumed here.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .change import AlignmentResult, _scaled_matrix, affine_degeneracy
from .transition_grid import INVALID, VALID, Reason

MIN_TOP_TOKENS = 4
RHO_CANDIDATES = (0.05, 0.10)


def top_count(count: int, rho: float) -> int:
    if rho not in RHO_CANDIDATES or count < 0:
        raise ValueError("unsupported pooling candidate/count")
    denominator = 20 if rho == 0.05 else 10
    return (count + denominator - 1) // denominator


def enough_overlap(count: int, *, min_top_tokens: int = MIN_TOP_TOKENS) -> bool:
    if min_top_tokens not in (4, 6, 8):
        raise ValueError("support floor must be one of 4, 6, 8")
    return top_count(count, min(RHO_CANDIDATES)) >= min_top_tokens


def mask_only(alignment: AlignmentResult, *, motion_grid: tuple[int, int],
              feature_grid: tuple[int, int] = (14, 14)) -> tuple[np.ndarray, int]:
    """Full bilinear footprint without allocating a feature tensor."""
    if alignment.status not in {"ecc", "phase_correlation"} or alignment.transform is None:
        raise ValueError("successful alignment required")
    if any(affine_degeneracy(np.asarray(alignment.transform, dtype=np.float64))):
        raise ValueError("degenerate/singular affine transform")
    height, width = feature_grid
    matrix = _scaled_matrix(alignment, motion_grid, feature_grid).astype(np.float64)
    # ECC already supplies destination-to-source; phase correlation supplies the inverse.
    inverse = matrix if alignment.status == "ecc" else cv2.invertAffineTransform(matrix)
    yy, xx = np.mgrid[:height, :width]
    sx = inverse[0,0]*xx + inverse[0,1]*yy + inverse[0,2]
    sy = inverse[1,0]*xx + inverse[1,1]*yy + inverse[1,2]
    mask = (sx >= 0) & (sx <= width-1) & (sy >= 0) & (sy <= height-1)
    return mask, int(mask.sum())


def warp_in_bounds(previous: torch.Tensor, alignment: AlignmentResult, *,
                   motion_grid: tuple[int, int] | None = None) -> tuple[torch.Tensor, np.ndarray]:
    """Use the same scaled affine/direction as the feature warp, full bilinear support."""
    if previous.ndim != 3 or not bool(torch.isfinite(previous).all()):
        raise ValueError("finite feature grid required")
    height, width, channels = previous.shape
    mask, _ = mask_only(alignment, motion_grid=motion_grid or (height, width),
                        feature_grid=(height, width))
    matrix = _scaled_matrix(alignment, motion_grid or (height, width), (height, width)).astype(np.float64)
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if alignment.status == "ecc" else 0)
    source = previous.detach().float().cpu().numpy()
    warped = np.stack([cv2.warpAffine(source[...,c], matrix, (width,height), flags=flags,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                       for c in range(channels)], axis=-1)
    return torch.from_numpy(warped).to(previous.device), mask


@dataclass(frozen=True)
class RawTransition:
    raw_c: np.ndarray
    in_bounds: np.ndarray
    state: int
    reason: int


def raw_transition(previous: torch.Tensor, current: torch.Tensor,
                   alignment: AlignmentResult, *, motion_grid=None,
                   min_top_tokens: int = MIN_TOP_TOKENS) -> RawTransition:
    """Pure raw cosine change; no median subtraction or clip-relative normalization.

    The caller merges cut/graphics and other producer flags; this is the warp component.
    NaNs outside support are deliberate sentinels, never calibration/pooling samples.
    """
    if previous.shape != current.shape or not bool(torch.isfinite(current).all()):
        raise ValueError("MISSING_NONFINITE_FEATURES: mismatched/nonfinite features")
    if not bool(torch.isfinite(previous).all()):
        raise ValueError("MISSING_NONFINITE_FEATURES: nonfinite source features")
    warped, mask = warp_in_bounds(F.normalize(previous.float(), dim=-1), alignment,
                                  motion_grid=motion_grid)
    raw = 1 - (F.normalize(current.float(), dim=-1) *
               F.normalize(warped, dim=-1)).sum(-1).clamp(-1,1)
    values = raw.detach().cpu().numpy().copy()
    values[~mask] = np.nan
    usable = enough_overlap(int(mask.sum()), min_top_tokens=min_top_tokens)
    return RawTransition(values, mask, VALID if usable else INVALID,
                         0 if usable else int(Reason.WARP_OVERLAP))


def masked_topmean(calibrated: np.ndarray, in_bounds: np.ndarray, rho: float,
                   *, min_top_tokens: int = MIN_TOP_TOKENS) -> float:
    """Select TopMean only from usable calibrated tokens; the mask is mandatory."""
    values, mask = np.asarray(calibrated), np.asarray(in_bounds)
    if mask.dtype != np.bool_ or values.shape != mask.shape or values.ndim != 2:
        raise ValueError("calibrated map and boolean per-token mask must match")
    selected = values[mask]
    if not enough_overlap(len(selected), min_top_tokens=min_top_tokens):
        raise ValueError("WARP_OVERLAP: insufficient selected tokens at rho=0.05")
    if not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1)):
        raise ValueError("usable calibrated tokens must be finite in [0,1]")
    k = top_count(len(selected), rho)
    return float(np.sort(selected)[-k:].mean())
