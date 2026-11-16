"""Canonical half-frame transition grid shared by E, M, and validity audits.

Transition j joins zero-based frames j and j+1. Tubelet t covers 2t,2t+1.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math
from types import MappingProxyType

import numpy as np


class Reason(IntFlag):
    CUT = 1
    REPLAY_GRAPHICS = 2
    ALIGNMENT = 4
    WARP_OVERLAP = 8
    MISSING_NONFINITE_FEATURES = 16
    DECODE = 32
    UNKNOWN = 64


REASON_PRIORITY = (Reason.DECODE, Reason.MISSING_NONFINITE_FEATURES,
                   Reason.WARP_OVERLAP, Reason.ALIGNMENT, Reason.CUT,
                   Reason.REPLAY_GRAPHICS, Reason.UNKNOWN)
VALID, INVALID, UNKNOWN = 1, 0, -1
GRID_VERSION = "full_span_frame_transitions_v1"


def validate_states_reasons(states, reasons):
    """Validate original integers before any lossy conversion."""
    states, reasons = np.asarray(states), np.asarray(reasons)
    if states.dtype.kind not in "iu" or reasons.dtype.kind not in "iu":
        raise ValueError("transition states and reasons require integer dtypes")
    if states.shape != reasons.shape or not np.isin(states, [UNKNOWN, INVALID, VALID]).all():
        raise ValueError("invalid transition state or reason shape")
    if np.any(reasons < 0) or np.any(reasons > sum(int(r) for r in REASON_PRIORITY)):
        raise ValueError("reason bits outside the declared vocabulary")
    if np.any((states == VALID) & (reasons != 0)) or np.any((states == INVALID) & (reasons == 0)):
        raise ValueError("valid transitions have no reasons; invalid ones require a reason")
    unknown_reason = (reasons & int(Reason.UNKNOWN)) != 0
    if np.any((states == UNKNOWN) != unknown_reason):
        raise ValueError("unknown state and reason must agree")
    return states.astype(np.int8, copy=True), reasons.astype(np.uint8, copy=True)


def primary_reason(flags: int) -> Reason | None:
    return next((reason for reason in REASON_PRIORITY if flags & reason), None)


def transition_interval(t: int, u: int, frame_count: int) -> tuple[int, int]:
    """Inclusive interval J(t,u), including both endpoint-internal transitions."""
    if frame_count < 4 or frame_count % 2 or min(t, u) < 0 or max(t, u) >= frame_count // 2 or t == u:
        raise ValueError("invalid distinct tubelet pair or clip frame count")
    return 2 * min(t, u), 2 * max(t, u)


def interval_length(t: int, u: int, frame_count: int) -> int:
    lo, hi = transition_interval(t, u, frame_count)
    return hi - lo + 1


def legacy_destination_to_transitions(change: np.ndarray) -> np.ndarray:
    """Drop only the synthetic first-frame placeholder; destination j+1 is transition j."""
    values = np.asarray(change)
    if values.ndim < 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("legacy frame-indexed maps must be finite [frames, ...]")
    if np.any(values[0] != 0):
        raise ValueError("legacy first frame must be the synthetic zero placeholder")
    return values[1:].copy()


def topmean(change: np.ndarray, rho: float) -> np.ndarray:
    """Mean the largest ceil(rho*N) finite tokens per frame transition."""
    values = np.asarray(change, dtype=np.float32)
    if values.ndim < 2 or not 0 < rho <= 1 or not np.isfinite(values).all():
        raise ValueError("topmean requires finite transition token maps and rho in (0,1]")
    flattened = values.reshape(values.shape[0], -1)
    k = max(1, math.ceil(rho * flattened.shape[1]))
    # The selected values and their mean are tie-invariant. No extra normalization.
    return np.sort(flattened, axis=1)[:, -k:].mean(axis=1, dtype=np.float32)


@dataclass(frozen=True)
class TransitionGrid:
    states: np.ndarray  # 1 valid, 0 known-invalid, -1 unknown
    reasons: np.ndarray  # bitwise Reason; multiple flags allowed
    evidence: np.ndarray | None = None  # one scalar per transition
    provenance: dict | None = None  # half-artifact identity; absent on synthetic/clip grids

    def __post_init__(self) -> None:
        states, reasons = validate_states_reasons(self.states, self.reasons)
        if states.ndim != 1 or len(states) < 3 or reasons.shape != states.shape:
            raise ValueError("transition states/reasons must be equally sized vectors")
        if self.evidence is not None:
            evidence = np.array(self.evidence, dtype=np.float32, copy=True)
            if evidence.shape != states.shape or not np.isfinite(evidence).all():
                raise ValueError("transition evidence must be finite; invalid slots use zero plus flags")
            object.__setattr__(self, "evidence", evidence)
            evidence.flags.writeable = False
        states.flags.writeable = reasons.flags.writeable = False
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "reasons", reasons)
        if self.provenance is not None:
            object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def frame_count(self) -> int:
        return len(self.states) + 1

    def require_production_ready(self) -> None:
        if np.any(self.states == UNKNOWN):
            raise RuntimeError("unresolved unknown transitions block production")
        if np.any(self.reasons & int(Reason.DECODE | Reason.MISSING_NONFINITE_FEATURES)):
            raise RuntimeError("decode or missing-feature infrastructure failures block production")
        if self.evidence is None or not np.isfinite(self.evidence[self.states == VALID]).all():
            raise RuntimeError("usable evidence is required for production")

    def clip(self, start_frame: int, frame_count: int = 128) -> TransitionGrid:
        if start_frame < 0 or start_frame + frame_count > self.frame_count:
            raise ValueError("clip falls outside canonical half")
        region = slice(start_frame, start_frame + frame_count - 1)
        values = None if self.evidence is None else self.evidence[region]
        return TransitionGrid(self.states[region], self.reasons[region], values)

    def pair_valid(self, t: int, u: int) -> bool:
        lo, hi = transition_interval(t, u, self.frame_count)
        return bool(np.all(self.states[lo:hi + 1] == VALID))

    def key_validity(self) -> tuple[np.ndarray, np.ndarray]:
        """Same predicate for positive/negative keys: internal transition 2t.

        Decode/missing-feature reasons already block production globally;
        unknown is never usable. Runtime teacher nonfinites are hard failures.
        """
        if self.frame_count % 2:
            raise ValueError("key validity requires an even clip frame count")
        return (self.states[::2] == VALID), self.reasons[::2].copy()

    def pair_max(self, t: int, u: int) -> float:
        if not self.pair_valid(t, u) or self.evidence is None:
            raise ValueError("pair has invalid/missing evidence")
        lo, hi = transition_interval(t, u, self.frame_count)
        return float(np.max(self.evidence[lo:hi + 1]))

    def pair_table(self, offsets: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Directed validity and raw maxima; invalid maxima are NaN."""
        if not offsets or len(set(offsets)) != len(offsets) or 0 in offsets:
            raise ValueError("explicit nonzero offsets required")
        temporal = self.frame_count // 2
        valid = np.zeros((temporal, temporal), dtype=bool)
        maxima = np.full((temporal, temporal), np.nan, dtype=np.float32)
        invalid_prefix = np.r_[0, np.cumsum(self.states != VALID)]
        for t in range(temporal):
            for delta in offsets:
                u = t + delta
                if not 0 <= u < temporal:
                    continue
                lo, hi = transition_interval(t, u, self.frame_count)
                if invalid_prefix[hi + 1] != invalid_prefix[lo]:
                    continue
                valid[t, u] = True
                if self.evidence is not None:
                    maxima[t, u] = np.max(self.evidence[lo:hi + 1])
        keys, _ = self.key_validity()
        if np.any(valid & ~(keys[:, None] & keys[None, :])):
            raise RuntimeError("pair validity implies valid endpoint keys")
        return valid, maxima

    def length_conditioned_maxima(self, offsets: tuple[int, ...]) -> dict[int, np.ndarray]:
        """Reference-ready usable directed maxima, separate for each J length."""
        if self.evidence is None:
            raise ValueError("transition evidence required for pair references")
        valid, maxima = self.pair_table(offsets)
        temporal = self.frame_count // 2
        result = {}
        for distance in sorted({abs(delta) for delta in offsets}):
            length = 2 * distance + 1
            values = [maxima[t, u] for t in range(temporal)
                      for u in (t - distance, t + distance)
                      if 0 <= u < temporal and valid[t, u]]
            result[length] = np.asarray(values, dtype=np.float32)
        return result

    def reason_summary(self) -> dict:
        totals = {reason.name.lower(): int(np.count_nonzero(self.reasons & reason))
                  for reason in REASON_PRIORITY}
        primary = {reason.name.lower(): int(sum(primary_reason(int(flags)) == reason
                                               for flags in self.reasons))
                   for reason in REASON_PRIORITY}
        return {"valid": int(np.count_nonzero(self.states == VALID)),
                "invalid": int(np.count_nonzero(self.states == INVALID)),
                "unknown": int(np.count_nonzero(self.states == UNKNOWN)),
                "reason_flags": totals, "primary_reasons": primary}


def displaced_offsets(length: int, offsets: tuple[int, ...] = (1, 2, 4)) -> tuple[int, ...]:
    if length < 0:
        raise ValueError("negative segment length")
    displacement = length // 2
    return tuple(delta for delta in offsets if displacement > 2 * abs(delta) + 1)


def rotate_valid_segments(grid: TransitionGrid, *, delta_max: int, seed: int,
                          half_states: np.ndarray | None = None,
                          clip_start_frame: int = 0) -> tuple[np.ndarray, list[dict]]:
    """One seeded circular transition shift per contiguous clip-valid segment."""
    if grid.evidence is None or delta_max < 1:
        raise ValueError("M requires scalar evidence and positive delta_max")
    if half_states is not None and (clip_start_frame < 0 or
                                    clip_start_frame + len(grid.states) > len(half_states)):
        raise ValueError("clip/half transition mapping mismatch")
    result = grid.evidence.copy()
    rng = np.random.default_rng(seed)
    records = []
    starts = np.flatnonzero(np.r_[True, grid.states[1:] != grid.states[:-1]])
    ends = np.r_[starts[1:], len(grid.states)]
    for lo, hi in zip(starts, ends):
        if grid.states[lo] != VALID:
            continue
        length = int(hi - lo)
        legal = [d for d in range(1, length) if min(d, length - d) > 2 * delta_max + 1]
        # Positive np.roll is the frozen odd-length fallback direction.
        shift = int(rng.choice(legal)) if legal else length // 2
        if length > 1:
            result[lo:hi] = np.roll(grid.evidence[lo:hi], shift)
        absolute_lo, absolute_hi = clip_start_frame + lo, clip_start_frame + hi
        left = bool(half_states is not None and absolute_lo > 0 and
                    half_states[absolute_lo - 1] == VALID)
        right = bool(half_states is not None and absolute_hi < len(half_states) and
                     half_states[absolute_hi] == VALID)
        records.append({"start_transition": int(lo), "end_transition_exclusive": int(hi),
                        "half_start_transition": int(absolute_lo),
                        "half_end_transition_exclusive": int(absolute_hi),
                        "length": length, "shift": shift,
                        "displacement": min(shift, length - shift),
                        "left_clip_truncated": left, "right_clip_truncated": right})
    return result, records
