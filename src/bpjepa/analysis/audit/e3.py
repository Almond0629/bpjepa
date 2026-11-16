"""Event/control population and scoring of the concordance audit (exact routines of the audit run)."""
import math

import numpy as np

from bpjepa.common import canonical_digest

DURATIONS = (1, 2.5)
RHO = (.05, .1)


def event_ids(half: str, annotations: list[dict]) -> list[list]:
    counts, result = {}, []
    for event in annotations:
        ms, label = event['position_ms'], event['class']
        if type(ms) is not int or ms < 0 or not isinstance(label, str) or not label:
            raise ValueError('event positions/classes must retain original label-file order')
        key = (ms, label)
        result.append([half, ms, label, counts.get(key, 0)])
        counts[key] = counts.get(key, 0) + 1
    return result


def window(center: float, d: float, frame_count: int):
    if not math.isfinite(center) or d not in DURATIONS or frame_count < 2:
        raise ValueError('bad E3 window coordinates')
    if center - d < 0 or center + d > (frame_count - 1) / 2:
        return None
    lo, hi = math.ceil(2 * (center - d)), math.floor(2 * (center + d))
    return lo, hi + 1  # exclusive frame end; transitions [lo, hi)


def controls(event_id: list, d: float, span: tuple[int, int], valid: np.ndarray,
             all_event_times: np.ndarray) -> list[int]:
    """Same frame/VALID-transition counts, every frame >=10s from every label."""
    if d not in DURATIONS:
        raise ValueError('unadopted duration')
    valid = np.asarray(valid)
    times = np.asarray(all_event_times, dtype=np.float64)
    if valid.ndim != 1 or valid.dtype != np.bool_ or not np.isfinite(times).all():
        raise ValueError('malformed validity/event positions')
    lo, end = span
    n = end - lo
    if not 0 <= lo < end <= len(valid) + 1 or n < 2:
        raise ValueError('event window outside half')
    count = int(valid[lo:end - 1].sum())
    safe = np.ones(len(valid) + 1, dtype=np.bool_)
    frame_times = np.arange(len(safe)) / 2
    for time in times:
        safe &= np.abs(frame_times - time) >= 10
    vp = np.r_[0, valid.cumsum()]
    fp = np.r_[0, (~safe).cumsum()]
    eligible = [a for a in range(len(safe) - n + 1)
                if fp[a + n] == fp[a] and vp[a + n - 1] - vp[a] == count]
    key = lambda a: (canonical_digest({'purpose': 'e3_controls', 'seed': 20260927,
                                       'half_id': event_id[0], 'event_id': event_id,
                                       'd': int(d) if d == 1 else 2.5, 'a': a}), a)
    return sorted(eligible, key=key)[:5]


def modal_motion(bins) -> int | None:
    bins = np.asarray(bins)
    if bins.ndim != 1 or bins.dtype.kind not in 'iu':
        raise ValueError('frozen motion bins must be integer (3 or -1 unavailable)')
    if np.any(bins < -1) or np.any(bins > 3):
        raise ValueError('unknown motion-bin code')
    bins = bins[(bins >= 0) & (bins < 3)]  # calibration policy: code 3 = UNAVAILABLE
    if not len(bins):
        return None
    values, counts = np.unique(bins, return_counts=True)
    return int(values[counts == counts.max()].max())


def normalize_labels(document: dict, half_number: int) -> list[dict]:
    """Preserve SoccerNet annotation order and every class in this half."""
    if half_number not in (1, 2) or not isinstance(document.get("annotations"), list):
        raise ValueError("SoccerNet half and annotations required")
    result = []
    for annotation in document["annotations"]:
        game_time = annotation["gameTime"]
        half = int(game_time.split(" - ", 1)[0])
        if half not in (1, 2):
            raise ValueError("unknown label half")
        if half == half_number:
            position = annotation["position"]
            if isinstance(position, bool) or not str(position).isdigit():
                raise ValueError("label position must be integer half-relative milliseconds")
            result.append({"position_ms": int(position), "class": annotation["label"]})
    event_ids("normalized", result)
    return result


def clean_region(center: float, cut_times_s) -> bool:
    """Closed +/-2.5s around the event/control center, using frozen detector times.

    The evidence-owner's input adapter must bind the detector's transition-time
    convention; this function never invents times from frame/reason indices.
    """
    times = np.asarray(cut_times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or not math.isfinite(center):
        raise ValueError('nonfinite cut/replay timing')
    return not bool(np.any(np.abs(times - center) <= 2.5))


def score_half(half: str, match: str, annotations: list[dict], states, evidence: dict,
               motion_bins, cut_times_s, d: float) -> tuple[list[dict], list[dict]]:
    """One common validity/control population for BOTH rho candidates."""
    states, motion_bins = np.asarray(states), np.asarray(motion_bins)
    if (states.ndim != 1 or states.dtype.kind not in 'iu' or not np.isin(states, [-1, 0, 1]).all()
            or motion_bins.shape != states.shape or set(evidence) != set(RHO)):
        raise ValueError('final validity/motion/rho population mismatch')
    valid = states == 1
    for e in evidence.values():
        e = np.asarray(e)
        if e.shape != states.shape or not np.isfinite(e[valid]).all() or np.any((e[valid] < 0) | (e[valid] > 1)):
            raise ValueError('usable finite transition evidence in [0,1] required')
    ids = event_ids(half, annotations)
    times = np.asarray([eid[1] / 1000 for eid in ids])
    records, omitted = [], []
    for eid, time in zip(ids, times):
        span = window(float(time), d, len(states) + 1)
        reason = None
        if span is None:
            reason = 'HALF_BOUNDARY'
        else:
            lo, end = span
            nvalid = int(valid[lo:end - 1].sum())
            if not nvalid or nvalid < 0.5 * (end - lo - 1):
                reason = 'VALID_SUPPORT'
            else:
                starts = controls(eid, d, span, valid, times)
                if len(starts) != 5:
                    reason = 'CONTROL_SHORTAGE'
        if reason:
            omitted.append({'event_id': eid, 'match': match, 'class': eid[2], 'reason': reason})
            continue
        n = end - lo
        spans = [span] + [(a, a + n) for a in starts]
        centers = [float(time)] + [(2 * a + n - 1) / 4 for a in starts]
        scores = {}
        for rho in RHO:
            e = np.asarray(evidence[rho])
            values = [float(e[a:z - 1][valid[a:z - 1]].max()) for a, z in spans]
            wins = [1.0 if values[0] > v else 0.5 if values[0] == v else 0.0 for v in values[1:]]
            scores[str(rho)] = {'concordance': float(np.mean(wins)), 'event_score': values[0],
                                'control_scores': values[1:]}
        motions = [modal_motion(motion_bins[a:z - 1]) for a, z in spans]
        records.append({'event_id': eid, 'half_id': half, 'match': match, 'class': eid[2],
                        'd': d, 'span': list(span), 'controls': starts, 'control_centers': centers[1:],
                        'valid_transitions': nvalid, 'nominal_transitions': n - 1,
                        'motion_bin': motions[0], 'control_motion_bins': motions[1:],
                        'clean': all(clean_region(c, cut_times_s) for c in centers), 'scores': scores})
    return records, omitted



def associated(start_frame, offsets, event_times):
    """Closed covered-frame span intersects the closed event +/-1s region."""
    times = np.sort(np.asarray(event_times, dtype=np.float64))
    if not np.isfinite(times).all() or start_frame < 0:
        raise ValueError('finite nominal half-relative coordinates required')
    t = np.arange(64)[:, None]
    u = t + np.asarray(offsets)[None, :]
    left = start_frame / 2 + np.minimum(t, u) - 1
    right = start_frame / 2 + np.maximum(t, u) + .5 + 1
    return np.searchsorted(times, right, side='right') > np.searchsorted(times, left, side='left')

