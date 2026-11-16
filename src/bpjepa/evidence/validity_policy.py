"""Pure floor-parameter validity policy over persisted V1 measurements."""
import numpy as np

from .transition_grid import (INVALID, UNKNOWN, VALID, REASON_PRIORITY, Reason,
                              primary_reason, validate_states_reasons)


def derive_validity(measurements: dict[str, np.ndarray], floor: int,
                    *, feature_nonfinite: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(floor) is not int or floor not in {4, 6, 8}:
        raise ValueError("floor must be one of 4, 6, 8")
    decoded = np.asarray(measurements["frame_decode"])
    if decoded.dtype != np.bool_ or decoded.ndim != 1 or len(decoded) < 2:
        raise ValueError("frame_decode must be a boolean frame vector")
    length = len(decoded) - 1
    for name in ("cut", "wipe", "processing_unknown"):
        value = np.asarray(measurements[name])
        if value.dtype != np.bool_ or value.shape != (length,):
            raise ValueError(f"{name} must be a boolean transition vector")
    alignment = np.asarray(measurements["alignment_status"])
    support = np.asarray(measurements["n_valid"])
    if (alignment.dtype.kind not in "iu" or support.dtype.kind not in "iu"
            or alignment.shape != (length,) or support.shape != (length,)
            or not np.isin(alignment, (0, 1, 2)).all()
            or np.any((support < -1) | (support > 196))):
        raise ValueError("invalid alignment/support integers")
    missing = np.zeros(length, dtype=bool) if feature_nonfinite is None else np.asarray(feature_nonfinite)
    if missing.dtype != np.bool_ or missing.shape != (length,):
        raise ValueError("feature_nonfinite must be a boolean transition vector")

    states = np.ones(length, dtype=np.int64)
    reasons = np.zeros(length, dtype=np.int64)
    for j in range(length):
        if not (decoded[j] and decoded[j + 1]):
            reasons[j] |= int(Reason.DECODE)
        else:
            if measurements["cut"][j]:
                reasons[j] |= int(Reason.CUT)
            if measurements["wipe"][j]:
                reasons[j] |= int(Reason.REPLAY_GRAPHICS)
            if measurements["processing_unknown"][j]:
                reasons[j] |= int(Reason.UNKNOWN)
            elif alignment[j] == 0:
                reasons[j] |= int(Reason.ALIGNMENT)
            elif (int(support[j]) + 19) // 20 < floor:
                reasons[j] |= int(Reason.WARP_OVERLAP)
            if missing[j]:
                reasons[j] |= int(Reason.MISSING_NONFINITE_FEATURES)
        states[j] = (UNKNOWN if reasons[j] & int(Reason.UNKNOWN)
                     else INVALID if reasons[j] else VALID)
    states, reasons = validate_states_reasons(states, reasons)
    labels = np.asarray([primary_reason(int(flags)).name if flags else "VALID"
                         for flags in reasons], dtype="U32")
    return states, reasons, labels
