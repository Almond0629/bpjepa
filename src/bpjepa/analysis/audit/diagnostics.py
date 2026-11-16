"""Boundary statistics and summaries used by the evidence audit."""
import math

import numpy as np

REASONS = ['CUT', 'REPLAY_GRAPHICS', 'ALIGNMENT', 'WARP_OVERLAP',
           'MISSING_NONFINITE_FEATURES', 'DECODE', 'UNKNOWN', 'HALF_BOUNDARY']
RHOS = (.05, .1)
LENGTHS = (3, 5, 9)


def topmean(values, mask, rho):
    n = mask.sum(1)
    k = np.maximum(1, np.ceil(rho * n).astype(int))
    sorted_values = np.sort(np.where(mask, values, -np.inf), axis=1)[:, ::-1]
    out = np.where(np.arange(values.shape[1])[None, :] < k[:, None], sorted_values, 0).sum(1) / k
    out[n == 0] = 0
    return out


def sides(values, support):
    values = np.where(support, values, 0)
    n = len(values)
    ix = np.arange(n)
    p = np.concatenate((np.zeros_like(values[:1]), np.cumsum(values, axis=0)), axis=0)
    c = np.concatenate((np.zeros_like(support[:1], dtype=np.int64), np.cumsum(support, axis=0)), axis=0)
    left, right = np.maximum(ix - 4, 0), np.minimum(ix + 5, n)
    lc, rc = c[ix] - c[left], c[right] - c[ix + 1]
    pre = (p[ix] - p[left]) / np.maximum(lc, 1)
    post = (p[right] - p[ix + 1]) / np.maximum(rc, 1)
    assert pre.min() > -1e-8 and post.min() > -1e-8
    return np.maximum(pre, 0), np.maximum(post, 0), support & (lc >= 2) & (rc >= 2)


def boundary(pre, post, support, statistic):
    if statistic == 'absolute':
        score = np.abs(post - pre)
    else:
        assert statistic == 'log_ratio'
        score = np.abs(np.log((post + 1e-6) / (pre + 1e-6)))
    assert np.isfinite(score[support]).all()
    return np.where(support, score, 0)


def token_boundary(raw, mask, valid, statistic):
    pre, post, support = sides(raw, mask & valid[:, None])
    supported = valid & (support.sum(1) >= 61)
    support &= supported[:, None]
    return boundary(pre, post, support, statistic), support, supported


def scalar_boundary(raw, mask, valid, statistic, rho):
    pooled = topmean(raw, mask, rho)
    pre, post, support = sides(pooled, valid)
    return boundary(pre, post, support, statistic), support


def maximum_rows(rows, score, support):
    values, counts, widths = [], [], []
    for _, row in rows:
        lo, end = row['span']
        width = end - lo - 1
        starts = [lo] + row['controls']
        values.append([float(np.max(score[s:s + width], initial=0)) for s in starts])
        counts.append([int(support[s:s + width].sum()) for s in starts])
        widths.append(width)
    return np.array(values), np.array(counts), np.array(widths)


def wins(values):
    return ((values[:, :1] > values[:, 1:]) + .5 * (values[:, :1] == values[:, 1:])).mean(1)


def context_reasons(reasons):
    n = len(reasons)
    output = np.zeros(n, np.uint16)
    for shift in range(-4, 5):
        indices = np.arange(n) + shift
        ok = (indices >= 0) & (indices < n)
        output[ok] |= reasons[indices[ok]].astype(np.uint16)
        output[~ok] |= 128
    return output


def pools(state):
    a, m, o = state
    return [('L0', a, m, o), ('L1', a, o), ('L2',)]


def keyname(key):
    return '|'.join(map(str, key))


def tail(reference, values):
    p = (1 + len(reference) - np.searchsorted(reference, values, side='left')) / (len(reference) + 1)
    return np.maximum(0., 1 - p / .02)


def pair_starts(valid, clips, length):
    possible = np.zeros(max(0, len(valid) - length + 1), bool)
    for lo, end in clips:
        assert end - lo == 128 and lo % 2 == 0
        possible[lo:end - length:2] = True
    starts = np.flatnonzero(possible)
    bad = np.r_[0, np.cumsum(~valid)]
    return starts[bad[starts + length] == bad[starts]]


def pair_values(score, valid, clips, length):
    starts = pair_starts(valid, clips, length)
    return np.lib.stride_tricks.sliding_window_view(score, length)[starts].max(1)


def subset_summary(rows, values, keep, confidence=True):
    records = [row for row, retained in zip(rows, keep) if retained]
    values = values[keep]
    if not len(records):
        return {'events': 0, 'matches': 0, 'concordance': None, 'event_zero_max': None, 'control_zero_max': None}
    score = wins(values)
    matches = sorted({r['match'] for r in records})
    output = {'events': len(records), 'matches': len(matches), 'concordance': float(score.mean()),
              'event_zero_max': float(np.mean(values[:, 0] == 0)),
              'control_zero_max': float(np.mean(values[:, 1:] == 0))}
    if confidence:
        sums = np.array([sum(v for r, v in zip(records, score) if r['match'] == m) for m in matches])
        nums = np.array([sum(r['match'] == m for r in records) for m in matches])
        draw = np.random.Generator(np.random.PCG64(20260927)).integers(0, len(matches), (2000, len(matches)))
        bs = sums[draw].sum(1) / nums[draw].sum(1)
        output['bootstrap_95_interval'] = np.quantile(bs, [.025, .975]).tolist()
        output['B4_lower_100th'] = float(np.sort(bs)[99])
        output['B4'] = 'PASS' if len(records) >= 500 and len(matches) >= 30 and score.mean() >= .60 and output['B4_lower_100th'] > .55 else 'FAIL_OR_INSUFFICIENT'
    return output


def summarize(rows, values, counts, widths, selected):
    enough = counts >= (widths[:, None] / 2)
    event_ok = enough[:, 0]
    controls_ok = enough[:, 1:].all(1)
    matched = (counts[:, 1:] == counts[:, :1]).all(1)
    keep = event_ok & controls_ok & matched
    clean = np.array([r['clean'] for r in rows])
    output = {'original_events': len(rows), 'retained': subset_summary(rows, values, keep),
              'cut_free': subset_summary(rows, values, keep & clean),
              'fixed_population_zero_placeholder': subset_summary(rows, values, np.ones(len(rows), bool), False),
              'fixed_population_clean_zero_placeholder': subset_summary(rows, values, clean, False),
              'attrition_sequential': {'event_support': int((~event_ok).sum()),
                  'control_support': int((event_ok & ~controls_ok).sum()),
                  'valid_count_mismatch': int((event_ok & controls_ok & ~matched).sum())},
              'events_dropped': int((~keep).sum()), 'event_drop_fraction': float((~keep).mean()),
              'event_no_supported_center_fraction': float((counts[:, 0] == 0).mean()),
              'control_no_supported_center_fraction': float((counts[:, 1:] == 0).mean())}
    selected_mask = np.array([r['match'] in selected for r in rows])
    output['original_20_match_fixed_population'] = subset_summary(rows, values, selected_mask, False)
    output['original_20_match_retained'] = subset_summary(rows, values, keep & selected_mask, False)
    output['classes'] = []
    for cls in sorted({r['class'] for r in rows}):
        ix = keep & np.array([r['class'] == cls for r in rows])
        sub = subset_summary(rows, values, ix, False)
        sub['class'] = cls
        sub['eligible'] = sub['events'] >= 30 and sub['matches'] >= 10
        output['classes'].append(sub)
    classes = [c for c in output['classes'] if c['eligible']]
    output['B5'] = 'PASS' if len(classes) >= 5 and sum(c['concordance'] > .5 for c in classes) >= math.ceil(len(classes) / 2) else 'FAIL_OR_INSUFFICIENT'
    cf = output['cut_free']
    output['B7'] = 'PASS' if cf['events'] >= 300 and cf['matches'] >= 30 and cf['concordance'] >= .55 else 'FAIL_OR_INSUFFICIENT'
    motion = []
    total = int(keep.sum())
    for b in range(3):
        original = keep & np.array([r['motion_bin'] == b for r in rows])
        ix = original & np.array([all(v == b for v in r['control_motion_bins']) for r in rows])
        sub = subset_summary(rows, values, ix, False)
        major = original.sum() >= 200 and original.sum() / max(1, total) >= .1
        sub.update(bin=b, major=bool(major), original_events=int(original.sum()))
        sub['status'] = 'NOT_MAJOR' if not major else ('NONBLOCKING_NOT_ASSESSED' if sub['events'] < 200 or sub['matches'] < 10 else ('PASS' if sub['concordance'] > .52 else 'FAIL'))
        motion.append(sub)
    output['B6_bins'] = motion
    output['B6'] = 'FAIL' if any(m['status'] == 'FAIL' for m in motion) else 'NO_ASSESSED_FAILURE'
    return output, keep
